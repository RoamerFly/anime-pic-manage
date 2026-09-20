use crate::database::DatabaseHealth;
use crate::worker_runtime::WorkerCapabilitiesInfo;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, Serialize)]
pub struct WorkerHealth {
    pub status: String,
    pub message: String,
    pub version: Option<String>,
    pub model_count: Option<u64>,
    pub runtime_mode: String,
    pub runtime_mode_label: String,
    pub runtime_path: Option<String>,
    pub compute_device: String,
    pub compute_device_label: String,
    pub capabilities: Option<WorkerCapabilitiesInfo>,
    pub capability_error: Option<String>,
    pub checked_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerRuntimeSettings {
    pub mode: String,
    pub mode_label: String,
    pub resolved_path: Option<String>,
    pub compute_device: String,
    pub compute_device_label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub app_version: String,
    pub data_dir: String,
    pub database: DatabaseHealth,
    pub worker: WorkerHealth,
}

#[derive(Debug, Clone, Serialize)]
pub struct LibrarySelection {
    pub path: String,
    pub display_name: String,
}

pub const PREVIEW_IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "gif"];

pub fn display_name_for_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or_default()
        .to_string()
}

pub fn is_supported_preview_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            PREVIEW_IMAGE_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        })
        .unwrap_or(false)
}

pub fn allow_asset_directory(app: &AppHandle, directory: &Path) -> Result<(), String> {
    app.asset_protocol_scope()
        .allow_directory(directory, true)
        .map_err(|error| error.to_string())
}

pub fn allow_asset_file(app: &AppHandle, file: &Path) -> Result<(), String> {
    app.asset_protocol_scope()
        .allow_file(file)
        .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanProgress {
    pub phase: String,
    pub current: usize,
    pub total: usize,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanError {
    pub path: String,
    pub code: String,
    pub message: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanLibraryResult {
    pub directory: String,
    pub total_discovered: usize,
    pub processed: usize,
    pub cancelled: bool,
    pub results: Vec<Value>,
    pub errors: Vec<ScanError>,
    pub model_version: Option<i64>,
    pub model_name: Option<String>,
    pub skip_annotated: Option<bool>,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovedFileItem {
    pub source: String,
    pub destination: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveFileErrorItem {
    pub source: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMoveResult {
    pub success_count: usize,
    pub failed_count: usize,
    pub moved: Vec<MovedFileItem>,
    pub errors: Vec<MoveFileErrorItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanCancellation {
    pub requested: bool,
    pub running: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateInfo {
    pub status: String,
    pub current_version: String,
    pub version: Option<String>,
    pub date: Option<String>,
    pub body: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateProgress {
    pub phase: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub percent: Option<u64>,
    pub message: String,
}
