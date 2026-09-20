//! LoRA training dataset export (recognition labels → kohya-ready folder).

use crate::app_settings::AppSettings;
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraExportOptions {
    /// Identity (character) to export; its annotated boxes are the source data.
    pub identity_id: String,
    #[serde(default)]
    pub trigger: Option<String>,
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default = "default_crop_mode")]
    pub crop_mode: String,
    #[serde(default = "default_resolution")]
    pub resolution: u32,
    #[serde(default = "default_max_images")]
    pub max_images: u32,
    #[serde(default = "default_duplicate_distance")]
    pub duplicate_distance: u32,
    #[serde(default = "default_quality_preset")]
    pub quality_preset: String,
    #[serde(default)]
    pub extra_tags: String,
    #[serde(default)]
    pub remove_tags: String,
    #[serde(default = "default_true")]
    pub drop_character_tags: bool,
    #[serde(default = "default_general_threshold")]
    pub general_threshold: f64,
    #[serde(default = "default_character_threshold")]
    pub character_threshold: f64,
    #[serde(default = "default_repeats")]
    pub repeats: u32,
    #[serde(default)]
    pub keep_tokens: u32,
    #[serde(default = "default_true")]
    pub shuffle_caption: bool,
    #[serde(default)]
    pub base_model: String,
}

fn default_crop_mode() -> String {
    "person".to_string()
}
fn default_resolution() -> u32 {
    1024
}
fn default_max_images() -> u32 {
    40
}
fn default_duplicate_distance() -> u32 {
    4
}
fn default_quality_preset() -> String {
    "pony".to_string()
}
fn default_repeats() -> u32 {
    10
}
fn default_true() -> bool {
    true
}
fn default_general_threshold() -> f64 {
    0.35
}
fn default_character_threshold() -> f64 {
    0.85
}

#[derive(Debug, Clone, Serialize)]
pub struct LoraExportCandidates {
    pub identities: Vec<crate::database::IdentitySampleSummary>,
    pub default_output_dir: String,
}

pub(crate) fn default_dataset_dir(app: &AppHandle, settings: &AppSettings) -> String {
    // Products stay next to ComfyUI (that is where kohya and the user already
    // work), but with a `-manage` suffix so the app's exports never mix with
    // hand-made datasets.
    if !settings.comfy_root.trim().is_empty() {
        if let Ok(paths) = crate::comfy::resolve_paths(
            &settings.comfy_root,
            &settings.comfy_output_dir,
            settings.comfy_port,
        ) {
            let root = PathBuf::from(&paths.root);
            if let Some(parent) = root.parent() {
                return parent
                    .join(MANAGED_DATASET_DIR)
                    .to_string_lossy()
                    .into_owned();
            }
        }
    }
    // Without ComfyUI everything falls back to the app's own `output\datasets`.
    let data_dir = PathBuf::from(&app.state::<AppState>().data_dir);
    crate::portable::PortableLayout::from_data_dir(&data_dir)
        .dataset_dir()
        .to_string_lossy()
        .into_owned()
}

/// Folder the app exports training sets into, next to ComfyUI.
pub(crate) const MANAGED_DATASET_DIR: &str = "lora-datasets-manage";
/// Folder the app writes trained LoRA files into, next to ComfyUI.
pub(crate) const MANAGED_LORA_DIR: &str = "lora-models-manage";

/// Sibling directory for trained LoRA files.
///
/// `lora-datasets-manage` -> `lora-models-manage`, keeping whatever suffix the
/// caller configured; anything else falls back to the package's `output\loras`.
pub(crate) fn lora_dir_for(dataset_dir: &Path) -> Option<PathBuf> {
    let name = dataset_dir.file_name()?.to_string_lossy();
    if !name.starts_with("lora-datasets") {
        return None;
    }
    let lora_name = name.replacen("lora-datasets", "lora-models", 1);
    Some(dataset_dir.parent()?.join(lora_name))
}

fn load_settings(app: &AppHandle) -> Result<AppSettings, String> {
    let state = app.state::<AppState>();
    let database = state
        .database
        .lock()
        .map_err(|_| "本地数据库状态锁暂时不可用。".to_string())?;
    crate::app_settings::load_app_settings(&database)
}

#[tauri::command]
pub fn get_lora_export_candidates(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<LoraExportCandidates> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "dataset.candidates",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let identities = {
        let state = app.state::<AppState>();
        let database = match state.database.lock() {
            Ok(database) => database,
            Err(_) => {
                return failure(
                    "dataset.candidates",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATABASE_LOCK_FAILED",
                        "本地数据库状态锁暂时不可用。",
                        None,
                        true,
                    ),
                )
            }
        };
        match database.list_identity_sample_summaries() {
            Ok(rows) => rows,
            Err(error) => {
                return failure(
                    "dataset.candidates",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATASET_QUERY_FAILED",
                        "无法读取已标注角色列表。",
                        Some(error.to_string()),
                        true,
                    ),
                )
            }
        }
    };
    success(
        "dataset.candidates",
        request_id,
        LoraExportCandidates {
            identities,
            default_output_dir: default_dataset_dir(&app, &settings),
        },
    )
}

#[tauri::command]
pub async fn export_lora_dataset(
    app: AppHandle,
    options: LoraExportOptions,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "dataset.export",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let (summary, samples) = {
        let state = app.state::<AppState>();
        let database = match state.database.lock() {
            Ok(database) => database,
            Err(_) => {
                return failure(
                    "dataset.export",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATABASE_LOCK_FAILED",
                        "本地数据库状态锁暂时不可用。",
                        None,
                        true,
                    ),
                )
            }
        };
        let summary = match database.list_identity_sample_summaries() {
            Ok(rows) => rows
                .into_iter()
                .find(|row| row.identity_id == options.identity_id),
            Err(error) => {
                return failure(
                    "dataset.export",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATASET_QUERY_FAILED",
                        "无法读取该角色的标注样本。",
                        Some(error.to_string()),
                        true,
                    ),
                )
            }
        };
        let samples = match database.list_identity_samples(&options.identity_id) {
            Ok(samples) => samples,
            Err(error) => {
                return failure(
                    "dataset.export",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATASET_QUERY_FAILED",
                        "无法读取该角色的标注样本。",
                        Some(error.to_string()),
                        true,
                    ),
                )
            }
        };
        (summary, samples)
    };
    let Some(summary) = summary else {
        return failure(
            "dataset.export",
            request_id.clone(),
            core_error(
                &request_id,
                "IDENTITY_NOT_FOUND",
                "该角色没有已标注的图片，无法导出训练集。",
                None,
                false,
            ),
        );
    };
    if samples.is_empty() {
        return failure(
            "dataset.export",
            request_id.clone(),
            core_error(
                &request_id,
                "DATASET_EMPTY",
                "该角色没有已标注的图片，无法导出训练集。",
                None,
                false,
            ),
        );
    }

    let trigger = options
        .trigger
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| summary.label_name.to_lowercase().replace(' ', "_"));
    let output_dir = options
        .output_dir
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_dataset_dir(&app, &settings));
    let images: Vec<Value> = samples
        .iter()
        .map(|sample| {
            json!({
                "path": sample.path,
                "bbox": [sample.bbox_x, sample.bbox_y, sample.bbox_width, sample.bbox_height],
                "source": sample.source,
                "manually_adjusted": sample.manually_adjusted,
            })
        })
        .collect();

    let payload = json!({
        "identity_label": summary.label_name,
        "trigger": trigger,
        "output_dir": output_dir,
        "images": images,
        "crop_mode": options.crop_mode,
        "resolution": options.resolution,
        "max_images": options.max_images,
        "duplicate_distance": options.duplicate_distance,
        "quality_preset": options.quality_preset,
        "extra_tags": options.extra_tags,
        "remove_tags": options.remove_tags,
        "drop_character_tags": options.drop_character_tags,
        "general_threshold": options.general_threshold,
        "character_threshold": options.character_threshold,
        "repeats": options.repeats,
        "keep_tokens": options.keep_tokens,
        "shuffle_caption": options.shuffle_caption,
        "base_model": options.base_model,
    });

    let worker = Arc::clone(&app.state::<AppState>().worker);
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        manager
            .request("dataset.export", payload)
            .map_err(|error| error.to_string())
    })
    .await;

    match result {
        Ok(Ok(payload)) => success("dataset.export", request_id, payload),
        Ok(Err(message)) => failure(
            "dataset.export",
            request_id.clone(),
            core_error(
                &request_id,
                "DATASET_EXPORT_FAILED",
                "训练集导出失败。",
                Some(message),
                true,
            ),
        ),
        Err(error) => failure(
            "dataset.export",
            request_id.clone(),
            core_error(
                &request_id,
                "DATASET_EXPORT_TASK_FAILED",
                "训练集导出任务异常中止。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_products_are_namespaced_next_to_comfyui() {
        assert_eq!(MANAGED_DATASET_DIR, "lora-datasets-manage");
        assert_eq!(MANAGED_LORA_DIR, "lora-models-manage");
    }

    #[test]
    fn trained_loras_land_in_the_sibling_of_the_managed_datasets() {
        let datasets = PathBuf::from(r"E:\comfy\lora-datasets-manage");

        assert_eq!(
            lora_dir_for(&datasets),
            Some(PathBuf::from(r"E:\comfy\lora-models-manage"))
        );
        // The package fallback (`…\output\datasets`) has no sibling naming rule,
        // so callers keep using `output\loras`.
        assert_eq!(lora_dir_for(Path::new(r"D:\app\output\datasets")), None);
    }
}
