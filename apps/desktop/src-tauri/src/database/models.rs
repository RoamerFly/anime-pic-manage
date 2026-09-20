use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

pub const CURRENT_SCHEMA_VERSION: i64 = 5;

/// Version of the similarity feature payload. Bumped when extraction starts
/// recording more than one frame, so stale animation caches are refreshed.
pub const SIMILARITY_FEATURE_VERSION: i64 = 2;
pub const PERSONAL_EMBEDDING_DIMENSION: usize = 2048;
pub const MIN_PERSONAL_TRAINING_SAMPLES: usize = 3;
pub const MIN_PERSONAL_SAMPLES_PER_CLASS: usize = 3;
pub const MIN_PERSONAL_TRAINING_CLASSES: usize = 2;
pub const WORKER_RUNTIME_MODE_SETTING: &str = "worker.runtime_mode";
pub const WORKER_COMPUTE_DEVICE_SETTING: &str = "worker.compute_device";
pub const WORKER_ONNX_THREADS_SETTING: &str = "worker.onnx_threads";
pub const WORKER_SKIP_ANNOTATED_SETTING: &str = "recognition.skip_annotated";
pub const SIMILARITY_WORKERS_SETTING: &str = "similarity.workers";
pub const SIMILARITY_THRESHOLD_SETTING: &str = "similarity.threshold";
pub const SIMILARITY_INCLUDE_SUBFOLDERS_SETTING: &str = "similarity.include_subfolders";
pub const SIMILARITY_ARCHIVE_DIR_SETTING: &str = "similarity.archive_dir";
pub const UI_FONT_SIZE_SETTING: &str = "ui.font_size";
pub const COMFY_ROOT_SETTING: &str = "comfy.root";
pub const COMFY_PORT_SETTING: &str = "comfy.port";
pub const COMFY_LOW_VRAM_SETTING: &str = "comfy.low_vram";
pub const COMFY_AUTO_START_SETTING: &str = "comfy.auto_start";
pub const COMFY_OUTPUT_DIR_SETTING: &str = "comfy.output_dir";
pub const LORA_TRAINER_ROOT_SETTING: &str = "lora_trainer.root";
pub const LORA_TRAINER_PYTHON_SETTING: &str = "lora_trainer.python";
pub const LORA_TRAINER_BASE_MODEL_SETTING: &str = "lora_trainer.base_model";
pub const LORA_TRAINER_OUTPUT_DIR_SETTING: &str = "lora_trainer.output_dir";
/// Optional folder holding CUDA 12 / cuDNN 9 DLLs for GPU inference.
pub const CUDA_RUNTIME_DIR_SETTING: &str = "worker.cuda_runtime_dir";
/// Character recognizer in use; empty means "the installed default".
pub const RECOGNIZER_MODEL_SETTING: &str = "recognition.recognizer_model";
/// `below_normal` (default) keeps scans from starving the desktop; `normal`
/// lets long jobs take all the CPU they can get.
pub const BACKGROUND_PRIORITY_SETTING: &str = "worker.background_priority";
/// Whether scans should consult the reference-image library.
pub const REFERENCE_MATCHING_SETTING: &str = "recognition.reference_matching";
/// Similarity backend for the reference library: `ccip` or `embedding`.
pub const REFERENCE_BACKEND_SETTING: &str = "recognition.reference_backend";
/// HTTP(S) proxy used by CUDA/model downloads and the AI Worker.
pub const NETWORK_PROXY_SETTING: &str = "network.proxy";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageSimilarityFeatureRecord {
    pub path: String,
    pub file_size: i64,
    pub width: i64,
    pub height: i64,
    pub format: String,
    pub clarity_score: f64,
    pub phash: String,
    pub dhash: String,
    pub color_hist_json: String,
    /// JSON array with the per-frame features of animated files.
    pub frames_json: Option<String>,
    pub frame_count: i64,
    pub feature_version: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityGroupItem {
    pub path: String,
    pub file_size: i64,
    pub dimensions: [i64; 2],
    pub format: String,
    pub clarity_score: f64,
    pub is_recommended: bool,
    pub recommend_reason: Option<String>,
    pub decision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityGroup {
    pub group_id: String,
    pub group_type: String,
    pub average_similarity: f64,
    pub items: Vec<SimilarityGroupItem>,
}

#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("无法创建应用数据目录: {0}")]
    CreateDirectory(#[from] std::io::Error),
    #[error("无法打开本地数据库: {0}")]
    Open(#[from] rusqlite::Error),
    #[error("标注版本冲突，当前版本为 {current_revision}")]
    RevisionConflict { current_revision: i64 },
    #[error("无法序列化个人特征向量: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct Database {
    pub connection: Connection,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatabaseHealth {
    pub status: String,
    pub path: String,
    pub schema_version: i64,
    pub table_count: i64,
    pub message: String,
}

/// A normalized rectangle in image coordinates.  The UI and Worker share
/// this representation so annotations remain independent of preview size.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct AnnotationBbox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AnnotationInput {
    pub id: Option<String>,
    pub identity_id: Option<String>,
    pub label_name: String,
    pub bbox: AnnotationBbox,
    pub source: String,
    pub manually_adjusted: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnnotationRecord {
    pub id: String,
    pub identity_id: String,
    pub label_name: String,
    pub bbox: AnnotationBbox,
    pub source: String,
    pub manually_adjusted: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ImageAnnotations {
    pub path: String,
    pub image_size: [u32; 2],
    pub revision: i64,
    pub annotations: Vec<AnnotationRecord>,
    pub learned_samples: usize,
    pub training_recommended: bool,
    pub verified_sample_count: usize,
    pub verified_class_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CharacterIdentity {
    pub id: String,
    pub display_name: String,
    pub created_at: String,
    pub updated_at: String,
}

/// One annotated person box, used to build LoRA training datasets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentitySample {
    pub path: String,
    pub label_name: String,
    pub image_width: u32,
    pub image_height: u32,
    pub bbox_x: f64,
    pub bbox_y: f64,
    pub bbox_width: f64,
    pub bbox_height: f64,
    pub source: String,
    pub manually_adjusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentitySampleSummary {
    pub identity_id: String,
    pub display_name: String,
    pub label_name: String,
    pub image_count: usize,
    pub manual_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PersonalPrototype {
    pub identity_id: String,
    pub display_name: String,
    pub embedding: Vec<f32>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PersonalTrainingSettings {
    pub min_total_samples: usize,
    pub min_samples_per_class: usize,
    pub auto_activate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingClassProgress {
    pub identity_id: String,
    pub display_name: String,
    pub sample_count: usize,
    pub min_required: usize,
    pub eligible: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TrainingReadiness {
    pub verified_sample_count: usize,
    pub verified_class_count: usize,
    pub eligible_class_count: usize,
    pub ready: bool,
    pub reasons: Vec<String>,
    pub class_progress: Vec<TrainingClassProgress>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PersonalModelVersionRecord {
    pub id: String,
    pub version: i64,
    pub status: String,
    pub sample_count: usize,
    pub class_count: usize,
    pub algorithm: String,
    pub artifact: Option<serde_json::Value>,
    pub metrics: serde_json::Value,
    pub warnings: Vec<String>,
    pub eligible_for_activation: bool,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub activated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PersonalTrainingStatus {
    pub status: String,
    pub message: String,
    pub readiness: TrainingReadiness,
    pub settings: PersonalTrainingSettings,
    pub active_version: Option<i64>,
    pub versions: Vec<PersonalModelVersionRecord>,
    pub updated_at: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub annotation_id: String,
    pub identity_id: String,
    pub display_name: String,
    pub image_path: String,
    pub revision: i64,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TrainedPersonalModel {
    pub artifact: serde_json::Value,
    pub metrics: serde_json::Value,
    pub algorithm: String,
    pub warnings: Vec<String>,
    pub eligible_for_activation: bool,
    pub sample_count: usize,
    pub class_count: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingSample {
    pub annotation_id: String,
    pub identity_id: String,
    pub embedding: Vec<f32>,
    pub manually_verified: bool,
}
