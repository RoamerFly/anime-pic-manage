//! Workflow templates and parameter binding.
//!
//! Templates are stored as our own envelope so the ComfyUI prompt stays a valid
//! API-format graph:
//!
//! ```json
//! {
//!   "version": 1,
//!   "name": "txt2img",
//!   "description": "…",
//!   "bindings": { "checkpoint": ["4", "inputs", "ckpt_name"], … },
//!   "prompt": { "4": { "class_type": "CheckpointLoaderSimple", … } }
//! }
//! ```

use super::types::{ComfyError, ComfyGenerateRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;

/// ComfyUI's API rejects a negative seed (`KSampler.seed` has `min: 0`); the
/// `-1` convention only exists in its own frontend. A negative request from our
/// UI therefore means "pick a random seed", which we resolve here so the value
/// shown afterwards is the one that was actually used.
pub fn resolved_seed(seed: i64) -> i64 {
    if seed >= 0 {
        return seed;
    }
    // uuid is already a dependency and gives a fresh random value per call;
    // stay below 2^31 so the number survives JavaScript round-tripping.
    (uuid::Uuid::new_v4().as_u128() % 2_147_483_647) as i64
}

pub const TEMPLATE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub version: u32,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub bindings: Map<String, Value>,
    pub prompt: Map<String, Value>,
}

impl WorkflowTemplate {
    pub fn from_file(path: &Path) -> Result<Self, ComfyError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| ComfyError::Io(format!("{}: {error}", path.display())))?;
        let template: Self = serde_json::from_str(&raw)
            .map_err(|error| ComfyError::Template(format!("{}: {error}", path.display())))?;
        if template.version != TEMPLATE_VERSION {
            return Err(ComfyError::Template(format!(
                "{}: 不支持的模板版本 {}",
                path.display(),
                template.version
            )));
        }
        Ok(template)
    }

    fn binding(&self, key: &str) -> Option<Vec<String>> {
        let raw = self.bindings.get(key)?;
        let parts: Vec<String> = raw
            .as_array()?
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect();
        if parts.len() < 2 {
            None
        } else {
            Some(parts)
        }
    }

    /// Replace one bound input, e.g. `["4", "inputs", "ckpt_name"]`.
    fn set_bound(&mut self, key: &str, value: Value) {
        let Some(parts) = self.binding(key) else {
            return;
        };
        let mut node = match self.prompt.get_mut(&parts[0]) {
            Some(node) => node,
            None => return,
        };
        for part in &parts[1..parts.len() - 1] {
            match node.get_mut(part) {
                Some(next) => node = next,
                None => return,
            }
        }
        if let Some(object) = node.as_object_mut() {
            object.insert(parts[parts.len() - 1].clone(), value);
        }
    }

    pub fn supports_lora(&self) -> bool {
        self.binding("lora_name").is_some()
    }

    /// Drop the LoRA node and reconnect its consumers to the upstream model/clip.
    ///
    /// ComfyUI validates every node input before execution, even for nodes in
    /// bypass mode, so an empty `lora_name` is rejected. Removing the node is the
    /// only way to build a valid graph without a LoRA selected.
    fn remove_lora_node(&mut self) -> Result<(), ComfyError> {
        let Some(parts) = self.binding("lora_name") else {
            return Ok(());
        };
        let node_id = parts[0].clone();
        let Some(node) = self.prompt.remove(&node_id) else {
            return Ok(());
        };
        let inputs = node.get("inputs").cloned().unwrap_or(Value::Null);
        let (Some(model), Some(clip)) = (inputs.get("model"), inputs.get("clip")) else {
            // A template without upstream references cannot be rewired safely.
            self.prompt.insert(node_id, node);
            return Err(ComfyError::Template(
                "LoRA 节点缺少 model/clip 输入，无法在不选择 LoRA 时旁路".to_string(),
            ));
        };
        let mut replacements: HashMap<u64, Value> = HashMap::new();
        replacements.insert(0, model.clone());
        replacements.insert(1, clip.clone());
        for node in self.prompt.values_mut() {
            let Some(inputs) = node.get_mut("inputs").and_then(Value::as_object_mut) else {
                continue;
            };
            for value in inputs.values_mut() {
                Self::rewire_reference(value, &node_id, &replacements);
            }
        }
        Ok(())
    }

    fn rewire_reference(value: &mut Value, node_id: &str, replacements: &HashMap<u64, Value>) {
        let Some(reference) = value.as_array() else {
            return;
        };
        if reference.len() < 2 || reference[0].as_str() != Some(node_id) {
            return;
        }
        let Some(output_index) = reference[1].as_u64() else {
            return;
        };
        if let Some(replacement) = replacements.get(&output_index) {
            *value = replacement.clone();
        }
    }

    /// Build the ComfyUI prompt (API format) for one request.
    pub fn build_prompt(&self, request: &ComfyGenerateRequest) -> Result<Value, ComfyError> {
        let mut template = self.clone();
        template.set_bound("checkpoint", json!(request.checkpoint));
        template.set_bound("positive", json!(request.positive));
        template.set_bound("negative", json!(request.negative));
        template.set_bound("steps", json!(request.steps));
        template.set_bound("cfg", json!(request.cfg));
        template.set_bound("sampler", json!(request.sampler));
        template.set_bound("scheduler", json!(request.scheduler));
        template.set_bound("width", json!(request.width));
        template.set_bound("height", json!(request.height));
        template.set_bound("batch", json!(request.batch));
        template.set_bound("seed", json!(resolved_seed(request.seed)));
        let prefix = request
            .filename_prefix
            .clone()
            .unwrap_or_else(|| "anime-pic-manage".to_string());
        template.set_bound("filename_prefix", json!(prefix));

        if template.supports_lora() {
            match request
                .lora
                .as_ref()
                .filter(|selection| !selection.name.trim().is_empty())
            {
                Some(selection) => {
                    template.set_bound("lora_name", json!(selection.name.trim()));
                    template.set_bound("lora_strength_model", json!(selection.strength));
                    template.set_bound("lora_strength_clip", json!(selection.strength));
                    // ComfyUI node modes: 0 = active, 4 = bypass.
                    template.set_bound("lora_mode", json!(0));
                }
                None => template.remove_lora_node()?,
            }
        } else if request.lora.is_some() {
            return Err(ComfyError::Template(
                "该工作流模板没有 LoRA 加载节点".to_string(),
            ));
        }

        let prompt = Value::Object(template.prompt);
        if prompt.as_object().map(Map::is_empty).unwrap_or(true) {
            return Err(ComfyError::Template("模板缺少 prompt 节点图".to_string()));
        }
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comfy::types::ComfyLoraSelection;

    fn template() -> WorkflowTemplate {
        let raw = json!({
            "version": 1,
            "name": "txt2img",
            "description": "test",
            "bindings": {
                "checkpoint": ["4", "inputs", "ckpt_name"],
                "positive": ["6", "inputs", "text"],
                "negative": ["7", "inputs", "text"],
                "steps": ["3", "inputs", "steps"],
                "cfg": ["3", "inputs", "cfg"],
                "sampler": ["3", "inputs", "sampler_name"],
                "scheduler": ["3", "inputs", "scheduler"],
                "width": ["5", "inputs", "width"],
                "height": ["5", "inputs", "height"],
                "batch": ["5", "inputs", "batch_size"],
                "seed": ["3", "inputs", "seed"],
                "filename_prefix": ["9", "inputs", "filename_prefix"],
                "lora_name": ["10", "inputs", "lora_name"],
                "lora_strength_model": ["10", "inputs", "strength_model"],
                "lora_strength_clip": ["10", "inputs", "strength_clip"],
                "lora_mode": ["10", "mode"]
            },
            "prompt": {
                "3": {"class_type": "KSampler", "inputs": {"steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "seed": 1, "model": ["10", 0]}},
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old.safetensors"}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["10", 1]}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["10", 1]}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI"}},
                "10": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["4", 1], "lora_name": "", "strength_model": 1.0, "strength_clip": 1.0}, "mode": 4}
            }
        });
        serde_json::from_value(raw).expect("template should parse")
    }

    fn request() -> ComfyGenerateRequest {
        ComfyGenerateRequest {
            template: "txt2img".to_string(),
            checkpoint: "illustriousXL_v01.safetensors".to_string(),
            lora: Some(ComfyLoraSelection {
                name: "character_alpha.safetensors".to_string(),
                strength: 0.8,
            }),
            positive: "score_9, character_alpha".to_string(),
            negative: "worst quality".to_string(),
            steps: 28,
            cfg: 6.5,
            sampler: "dpmpp_2m".to_string(),
            scheduler: "karras".to_string(),
            width: 1024,
            height: 1024,
            batch: 2,
            seed: 12345,
            filename_prefix: Some("anime".to_string()),
        }
    }

    #[test]
    fn negative_seeds_become_random_valid_seeds() {
        // ComfyUI's API validates `seed >= 0`; -1 only means "random" in its
        // own frontend, so the value must never reach /prompt unchanged.
        let negative = ComfyGenerateRequest {
            seed: -1,
            ..request()
        };
        let built = template()
            .build_prompt(&negative)
            .expect("prompt builds with a random seed");
        let seed = built["3"]["inputs"]["seed"]
            .as_i64()
            .expect("seed is an integer");

        assert!((0..2_147_483_647).contains(&seed));
        assert_eq!(resolved_seed(4242), 4242);
        assert_ne!(resolved_seed(-1), resolved_seed(-1));
    }

    #[test]
    fn binds_every_parameter_into_the_prompt() {
        let built = template().build_prompt(&request()).expect("prompt builds");

        assert_eq!(
            built["4"]["inputs"]["ckpt_name"],
            "illustriousXL_v01.safetensors"
        );
        assert_eq!(built["6"]["inputs"]["text"], "score_9, character_alpha");
        assert_eq!(built["7"]["inputs"]["text"], "worst quality");
        assert_eq!(built["3"]["inputs"]["steps"], 28);
        assert_eq!(built["3"]["inputs"]["cfg"], 6.5);
        assert_eq!(built["3"]["inputs"]["sampler_name"], "dpmpp_2m");
        assert_eq!(built["3"]["inputs"]["scheduler"], "karras");
        assert_eq!(built["3"]["inputs"]["seed"], 12345);
        assert_eq!(built["5"]["inputs"]["width"], 1024);
        assert_eq!(built["5"]["inputs"]["batch_size"], 2);
        assert_eq!(built["9"]["inputs"]["filename_prefix"], "anime");
        assert_eq!(
            built["10"]["inputs"]["lora_name"],
            "character_alpha.safetensors"
        );
        assert_eq!(built["10"]["inputs"]["strength_model"], 0.8);
        assert_eq!(built["10"]["mode"], 0);
    }

    #[test]
    fn without_a_lora_the_loader_is_removed_and_rewired() {
        let mut request = request();
        request.lora = None;

        let built = template().build_prompt(&request).expect("prompt builds");

        // ComfyUI validates bypassed node inputs, so the node must disappear
        // and its consumers must point straight at the checkpoint.
        assert!(built.get("10").is_none());
        assert_eq!(built["3"]["inputs"]["model"], json!(["4", 0]));
        assert_eq!(built["6"]["inputs"]["clip"], json!(["4", 1]));
        assert_eq!(built["7"]["inputs"]["clip"], json!(["4", 1]));
    }

    #[test]
    fn an_empty_lora_name_is_treated_as_no_lora() {
        let mut request = request();
        request.lora = Some(ComfyLoraSelection {
            name: "   ".to_string(),
            strength: 0.8,
        });

        let built = template().build_prompt(&request).expect("prompt builds");

        assert!(built.get("10").is_none());
        assert_eq!(built["3"]["inputs"]["model"], json!(["4", 0]));
    }

    #[test]
    fn missing_prompt_graph_is_rejected() {
        let mut template = template();
        template.prompt.clear();

        assert!(matches!(
            template.build_prompt(&request()),
            Err(ComfyError::Template(_))
        ));
    }

    #[test]
    fn every_shipped_template_loads_and_builds_a_prompt() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .expect("crate lives under apps/desktop/src-tauri")
            .join("resources")
            .join("comfy-workflows");
        let mut checked = 0;
        for entry in std::fs::read_dir(&directory).expect("template directory exists") {
            let path = entry.expect("directory entry").path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let template = WorkflowTemplate::from_file(&path).expect("shipped template parses");
            for lora in [
                None,
                Some(crate::comfy::types::ComfyLoraSelection {
                    name: "character.safetensors".to_string(),
                    strength: 0.75,
                }),
            ] {
                let mut request = request();
                request.template = template.name.clone();
                request.lora = lora;
                if !template.supports_lora() && request.lora.is_some() {
                    continue;
                }
                let prompt = template
                    .build_prompt(&request)
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                assert!(
                    prompt
                        .as_object()
                        .map(|nodes| !nodes.is_empty())
                        .unwrap_or(false),
                    "{} produced an empty prompt",
                    path.display()
                );
            }
            checked += 1;
        }
        assert!(checked >= 3, "expected shipped templates, found {checked}");
    }
}
