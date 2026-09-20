//! Types shared by the ComfyUI integration.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_COMFY_PORT: u16 = 8188;
pub const MIN_COMFY_PORT: u16 = 1024;

#[derive(Debug, Error)]
pub enum ComfyError {
    #[error("未配置 ComfyUI 目录")]
    NotConfigured,
    #[error("ComfyUI 目录无效：{0}")]
    InvalidRoot(String),
    #[error("ComfyUI 未运行：{0}")]
    NotRunning(String),
    #[error("ComfyUI 请求失败：{0}")]
    Request(String),
    #[error("ComfyUI 返回错误：{0}")]
    Response(String),
    #[error("工作流模板无效：{0}")]
    Template(String),
    #[error("文件操作失败：{0}")]
    Io(String),
}

/// Paths derived from the configured ComfyUI root.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComfyPaths {
    /// Directory that contains `ComfyUI/`, `python/` and the launcher.
    pub root: String,
    pub app_dir: String,
    pub python: String,
    pub main_script: String,
    pub models_dir: String,
    pub loras_dir: String,
    pub output_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComfyStatus {
    pub configured: bool,
    pub installed: bool,
    pub running: bool,
    pub owned: bool,
    pub root: Option<String>,
    pub python: Option<String>,
    pub version: Option<String>,
    pub port: u16,
    pub output_dir: Option<String>,
    pub device: Option<String>,
    pub vram_gb: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfyModelList {
    pub checkpoints: Vec<String>,
    pub loras: Vec<String>,
    pub samplers: Vec<String>,
    pub schedulers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfyLoraSelection {
    pub name: String,
    #[serde(default = "default_lora_strength")]
    pub strength: f64,
}

fn default_lora_strength() -> f64 {
    0.8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfyGenerateRequest {
    pub template: String,
    pub checkpoint: String,
    #[serde(default)]
    pub lora: Option<ComfyLoraSelection>,
    pub positive: String,
    #[serde(default)]
    pub negative: String,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_cfg")]
    pub cfg: f64,
    #[serde(default = "default_sampler")]
    pub sampler: String,
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_size")]
    pub width: u32,
    #[serde(default = "default_size")]
    pub height: u32,
    #[serde(default = "default_batch")]
    pub batch: u32,
    /// `-1` asks ComfyUI to pick a random seed.
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default)]
    pub filename_prefix: Option<String>,
}

fn default_steps() -> u32 {
    24
}
fn default_cfg() -> f64 {
    7.0
}
fn default_sampler() -> String {
    "euler_ancestral".to_string()
}
fn default_scheduler() -> String {
    "normal".to_string()
}
fn default_size() -> u32 {
    832
}
fn default_batch() -> u32 {
    1
}
fn default_seed() -> i64 {
    -1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComfyGeneratedImage {
    pub filename: String,
    pub subfolder: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfyGenerateResult {
    pub prompt_id: String,
    pub images: Vec<ComfyGeneratedImage>,
    pub output_dir: String,
    pub elapsed_ms: u64,
    pub seed: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComfyProgress {
    pub phase: String,
    pub current: usize,
    pub total: usize,
    pub message: String,
    pub prompt_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_request_defaults_are_anime_friendly() {
        let request: ComfyGenerateRequest = serde_json::from_value(serde_json::json!({
            "template": "txt2img",
            "checkpoint": "model.safetensors",
            "positive": "1girl"
        }))
        .expect("minimal request should deserialize");

        assert_eq!(request.steps, 24);
        assert_eq!(request.width, 832);
        assert_eq!(request.height, 832);
        assert_eq!(request.batch, 1);
        assert_eq!(request.seed, -1);
        assert!(request.lora.is_none());
    }
}
