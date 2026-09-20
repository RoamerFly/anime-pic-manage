//! Persisted, user-tunable application settings.
//!
//! Values are stored as individual rows in the `settings` table so older
//! versions keep working when new keys are added. Every read falls back to the
//! documented default instead of failing the whole settings payload.

use crate::database::{
    Database, BACKGROUND_PRIORITY_SETTING, COMFY_AUTO_START_SETTING, COMFY_LOW_VRAM_SETTING,
    COMFY_OUTPUT_DIR_SETTING, COMFY_PORT_SETTING, COMFY_ROOT_SETTING, CUDA_RUNTIME_DIR_SETTING,
    LORA_TRAINER_BASE_MODEL_SETTING, LORA_TRAINER_OUTPUT_DIR_SETTING, LORA_TRAINER_PYTHON_SETTING,
    LORA_TRAINER_ROOT_SETTING, NETWORK_PROXY_SETTING, RECOGNIZER_MODEL_SETTING,
    REFERENCE_BACKEND_SETTING, REFERENCE_MATCHING_SETTING, SIMILARITY_ARCHIVE_DIR_SETTING,
    SIMILARITY_INCLUDE_SUBFOLDERS_SETTING, SIMILARITY_THRESHOLD_SETTING,
    SIMILARITY_WORKERS_SETTING, UI_FONT_SIZE_SETTING, WORKER_ONNX_THREADS_SETTING,
    WORKER_SKIP_ANNOTATED_SETTING,
};
use serde::{Deserialize, Serialize};

pub const MIN_SIMILARITY_WORKERS: u32 = 1;
pub const MAX_SIMILARITY_WORKERS: u32 = 32;
pub const MIN_ONNX_THREADS: u32 = 1;
pub const MAX_ONNX_THREADS: u32 = 16;
pub const MIN_SIMILARITY_THRESHOLD: f64 = 0.50;
pub const MAX_SIMILARITY_THRESHOLD: f64 = 0.99;
pub const MAX_ARCHIVE_DIR_NAME: usize = 64;
pub const UI_FONT_BASE_SIZE: f64 = 14.0;
pub const MIN_UI_FONT_SIZE: f64 = 12.0;
pub const MAX_UI_FONT_SIZE: f64 = 20.0;
pub const UI_FONT_SIZE_STEP: f64 = 0.5;
pub const MIN_COMFY_PORT: u16 = 1024;
pub const MAX_COMFY_PORT: u16 = 65535;
pub const DEFAULT_COMFY_PORT: u16 = 8188;
pub const DEFAULT_NETWORK_PROXY: &str = "127.0.0.1:7890";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppSettings {
    /// ONNX Runtime intra-op threads used by CPU recognition inference.
    pub recognition_onnx_threads: u32,
    /// Default for the recognition scan "skip already annotated" switch.
    pub recognition_skip_annotated: bool,
    /// Threads used to decode and hash images during a similarity scan.
    pub similarity_workers: u32,
    /// Default match sensitivity for new similarity scans.
    pub similarity_threshold: f64,
    /// Default for the similarity "include subfolders" switch.
    pub similarity_include_subfolders: bool,
    /// Default archive subfolder name used by the similarity governance bar.
    pub similarity_archive_dir: String,
    /// UI font size in pixels against a 14px design baseline.
    pub ui_font_size: f64,
    /// Root of the user's own ComfyUI portable package (never bundled).
    pub comfy_root: String,
    pub comfy_port: u16,
    /// Start ComfyUI with `--lowvram`, recommended below 8 GB of VRAM.
    pub comfy_low_vram: bool,
    /// Start ComfyUI automatically together with generation requests.
    pub comfy_auto_start: bool,
    /// Empty means `<comfy_root>/../Images/generated`.
    pub comfy_output_dir: String,
    /// Root of the user's own kohya_ss / sd-scripts install (never bundled).
    pub lora_trainer_root: String,
    /// Optional python.exe override for the trainer environment.
    pub lora_trainer_python: String,
    /// Base checkpoint used for LoRA training; must match the LoRA family.
    pub lora_trainer_base_model: String,
    /// Empty means a `lora-models` folder next to the exported datasets.
    pub lora_trainer_output_dir: String,
    /// Optional folder with CUDA 12 / cuDNN 9 DLLs for GPU inference; empty
    /// means the Worker uses the normal system search path.
    pub cuda_runtime_dir: String,
    /// Character recognizer model id; empty uses the installed default.
    pub recognition_recognizer_model: String,
    /// `below_normal` keeps background inference polite; `normal` runs flat out.
    pub background_priority: String,
    /// Whether scans consult the reference-image library.
    pub reference_matching_enabled: bool,
    /// Similarity backend of that library: `ccip` or `embedding`.
    pub reference_backend: String,
    /// Optional HTTP(S) proxy for model, CUDA and Hugging Face downloads.
    /// Stored without a scheme for a compact UI value; runtime adds `http://`.
    pub network_proxy: String,
}

pub fn default_similarity_workers() -> u32 {
    std::thread::available_parallelism()
        .map(|value| value.get() as u32)
        .unwrap_or(4)
        .clamp(MIN_SIMILARITY_WORKERS, 8)
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            recognition_onnx_threads: 4,
            recognition_skip_annotated: true,
            similarity_workers: default_similarity_workers(),
            similarity_threshold: 0.85,
            similarity_include_subfolders: true,
            similarity_archive_dir: "_duplicates".to_string(),
            ui_font_size: UI_FONT_BASE_SIZE,
            comfy_root: String::new(),
            comfy_port: DEFAULT_COMFY_PORT,
            comfy_low_vram: true,
            comfy_auto_start: false,
            comfy_output_dir: String::new(),
            lora_trainer_root: String::new(),
            lora_trainer_python: String::new(),
            lora_trainer_base_model: String::new(),
            lora_trainer_output_dir: String::new(),
            cuda_runtime_dir: String::new(),
            recognition_recognizer_model: String::new(),
            background_priority: "below_normal".to_string(),
            reference_matching_enabled: false,
            reference_backend: "ccip".to_string(),
            network_proxy: DEFAULT_NETWORK_PROXY.to_string(),
        }
    }
}

/// Convert the user-facing proxy value into a URL accepted by reqwest and
/// proxy-aware Python libraries. Empty input disables proxying.
pub fn proxy_url(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = reqwest::Url::parse(&candidate).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return None;
    }
    Some(candidate)
}

impl AppSettings {
    /// Clamp and normalize every field, returning a user-facing error message
    /// when a value cannot be repaired.
    pub fn validated(mut self) -> Result<Self, String> {
        if !(MIN_ONNX_THREADS..=MAX_ONNX_THREADS).contains(&self.recognition_onnx_threads) {
            return Err(format!(
                "识别并行线程数必须在 {MIN_ONNX_THREADS} 到 {MAX_ONNX_THREADS} 之间。"
            ));
        }
        if !(MIN_SIMILARITY_WORKERS..=MAX_SIMILARITY_WORKERS).contains(&self.similarity_workers) {
            return Err(format!(
                "相似度并行度必须在 {MIN_SIMILARITY_WORKERS} 到 {MAX_SIMILARITY_WORKERS} 之间。"
            ));
        }
        if !self.similarity_threshold.is_finite()
            || !(MIN_SIMILARITY_THRESHOLD..=MAX_SIMILARITY_THRESHOLD)
                .contains(&self.similarity_threshold)
        {
            return Err(format!(
                "默认相似度阈值必须在 {MIN_SIMILARITY_THRESHOLD} 到 {MAX_SIMILARITY_THRESHOLD} 之间。"
            ));
        }
        self.similarity_threshold = (self.similarity_threshold * 100.0).round() / 100.0;
        let archive = self.similarity_archive_dir.trim();
        if archive.is_empty() || archive.chars().count() > MAX_ARCHIVE_DIR_NAME {
            return Err(format!(
                "归档文件夹名不能为空，且不超过 {MAX_ARCHIVE_DIR_NAME} 个字符。"
            ));
        }
        if archive.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) {
            return Err("归档文件夹名不能包含 / \\ : * ? \" < > | 等字符。".to_string());
        }
        if archive == "." || archive == ".." {
            return Err("归档文件夹名不能是 . 或 ..".to_string());
        }
        self.similarity_archive_dir = archive.to_string();
        if !self.ui_font_size.is_finite() {
            self.ui_font_size = UI_FONT_BASE_SIZE;
        }
        // Snap to the 0.5px steps the settings control offers.
        self.ui_font_size = ((self.ui_font_size / UI_FONT_SIZE_STEP).round() * UI_FONT_SIZE_STEP)
            .clamp(MIN_UI_FONT_SIZE, MAX_UI_FONT_SIZE);
        if !(MIN_COMFY_PORT..=MAX_COMFY_PORT).contains(&self.comfy_port) {
            return Err(format!(
                "ComfyUI 端口必须在 {MIN_COMFY_PORT} 到 {MAX_COMFY_PORT} 之间。"
            ));
        }
        self.comfy_root = self.comfy_root.trim().to_string();
        self.comfy_output_dir = self.comfy_output_dir.trim().to_string();
        self.lora_trainer_root = self.lora_trainer_root.trim().to_string();
        self.lora_trainer_python = self.lora_trainer_python.trim().to_string();
        self.lora_trainer_base_model = self.lora_trainer_base_model.trim().to_string();
        self.lora_trainer_output_dir = self.lora_trainer_output_dir.trim().to_string();
        self.cuda_runtime_dir = self.cuda_runtime_dir.trim().to_string();
        self.network_proxy = self.network_proxy.trim().to_string();
        if !self.network_proxy.is_empty() && proxy_url(&self.network_proxy).is_none() {
            return Err("网络代理必须是 host:port 或 http(s)://host:port 格式。".to_string());
        }
        self.recognition_recognizer_model = self.recognition_recognizer_model.trim().to_string();
        let priority = self.background_priority.trim().to_ascii_lowercase();
        self.background_priority = if priority == "normal" {
            "normal".to_string()
        } else {
            "below_normal".to_string()
        };
        let backend = self.reference_backend.trim().to_ascii_lowercase();
        self.reference_backend = if backend == "embedding" {
            "embedding".to_string()
        } else {
            "ccip".to_string()
        };
        // Trainer paths are deliberately not validated here: files move, and a
        // hard error would make `load_app_settings` fall back to defaults and
        // silently reset every other setting. The status probe reports missing
        // paths instead.
        Ok(self)
    }
}

fn parse_bool(value: Option<String>, fallback: bool) -> bool {
    match value.as_deref() {
        Some("true") => true,
        Some("false") => false,
        Some("1") => true,
        Some("0") => false,
        _ => fallback,
    }
}

pub fn load_app_settings(database: &Database) -> Result<AppSettings, String> {
    let defaults = AppSettings::default();
    let onnx_threads = database
        .get_setting_string(WORKER_ONNX_THREADS_SETTING)
        .map_err(|error| error.to_string())?
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| (MIN_ONNX_THREADS..=MAX_ONNX_THREADS).contains(value))
        .unwrap_or(defaults.recognition_onnx_threads);
    let similarity_workers = database
        .get_setting_string(SIMILARITY_WORKERS_SETTING)
        .map_err(|error| error.to_string())?
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| (MIN_SIMILARITY_WORKERS..=MAX_SIMILARITY_WORKERS).contains(value))
        .unwrap_or(defaults.similarity_workers);
    let similarity_threshold = database
        .get_setting_string(SIMILARITY_THRESHOLD_SETTING)
        .map_err(|error| error.to_string())?
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| {
            value.is_finite()
                && (MIN_SIMILARITY_THRESHOLD..=MAX_SIMILARITY_THRESHOLD).contains(value)
        })
        .unwrap_or(defaults.similarity_threshold);
    let archive_dir = database
        .get_setting_string(SIMILARITY_ARCHIVE_DIR_SETTING)
        .map_err(|error| error.to_string())?
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| defaults.similarity_archive_dir.clone());
    let ui_font_size = database
        .get_setting_string(UI_FONT_SIZE_SETTING)
        .map_err(|error| error.to_string())?
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && (MIN_UI_FONT_SIZE..=MAX_UI_FONT_SIZE).contains(value))
        .unwrap_or(defaults.ui_font_size);
    let comfy_root = database
        .get_setting_string(COMFY_ROOT_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let comfy_port = database
        .get_setting_string(COMFY_PORT_SETTING)
        .map_err(|error| error.to_string())?
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|value| (MIN_COMFY_PORT..=MAX_COMFY_PORT).contains(value))
        .unwrap_or(defaults.comfy_port);
    let comfy_output_dir = database
        .get_setting_string(COMFY_OUTPUT_DIR_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let lora_trainer_root = database
        .get_setting_string(LORA_TRAINER_ROOT_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let lora_trainer_python = database
        .get_setting_string(LORA_TRAINER_PYTHON_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let lora_trainer_base_model = database
        .get_setting_string(LORA_TRAINER_BASE_MODEL_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let lora_trainer_output_dir = database
        .get_setting_string(LORA_TRAINER_OUTPUT_DIR_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let cuda_runtime_dir = database
        .get_setting_string(CUDA_RUNTIME_DIR_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let recognition_recognizer_model = database
        .get_setting_string(RECOGNIZER_MODEL_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    let background_priority = database
        .get_setting_string(BACKGROUND_PRIORITY_SETTING)
        .map_err(|error| error.to_string())?
        .filter(|value| value == "normal" || value == "below_normal")
        .unwrap_or_else(|| defaults.background_priority.clone());
    let reference_backend = database
        .get_setting_string(REFERENCE_BACKEND_SETTING)
        .map_err(|error| error.to_string())?
        .filter(|value| value == "ccip" || value == "embedding")
        .unwrap_or_else(|| defaults.reference_backend.clone());
    let network_proxy = database
        .get_setting_string(NETWORK_PROXY_SETTING)
        .map_err(|error| error.to_string())?
        .unwrap_or_else(|| defaults.network_proxy.clone());

    AppSettings {
        recognition_onnx_threads: onnx_threads,
        recognition_skip_annotated: parse_bool(
            database
                .get_setting_string(WORKER_SKIP_ANNOTATED_SETTING)
                .map_err(|error| error.to_string())?,
            defaults.recognition_skip_annotated,
        ),
        similarity_workers,
        similarity_threshold,
        similarity_include_subfolders: parse_bool(
            database
                .get_setting_string(SIMILARITY_INCLUDE_SUBFOLDERS_SETTING)
                .map_err(|error| error.to_string())?,
            defaults.similarity_include_subfolders,
        ),
        similarity_archive_dir: archive_dir,
        ui_font_size,
        comfy_root,
        comfy_port,
        comfy_low_vram: parse_bool(
            database
                .get_setting_string(COMFY_LOW_VRAM_SETTING)
                .map_err(|error| error.to_string())?,
            defaults.comfy_low_vram,
        ),
        comfy_auto_start: parse_bool(
            database
                .get_setting_string(COMFY_AUTO_START_SETTING)
                .map_err(|error| error.to_string())?,
            defaults.comfy_auto_start,
        ),
        comfy_output_dir,
        lora_trainer_root,
        lora_trainer_python,
        lora_trainer_base_model,
        lora_trainer_output_dir,
        cuda_runtime_dir,
        recognition_recognizer_model,
        background_priority,
        reference_matching_enabled: parse_bool(
            database
                .get_setting_string(REFERENCE_MATCHING_SETTING)
                .map_err(|error| error.to_string())?,
            defaults.reference_matching_enabled,
        ),
        reference_backend,
        network_proxy,
    }
    .validated()
    .or(Ok(defaults))
}

/// Persist every setting in one transaction.
///
/// Writing row by row used to leave a half-saved state when anything aborted
/// mid-way (which is exactly how a database ended up with only the first seven
/// keys and no `comfy.*` rows at all).
pub fn save_app_settings(database: &mut Database, settings: &AppSettings) -> Result<(), String> {
    let settings = settings.clone().validated()?;
    let entries: [(&str, String); 22] = [
        (
            WORKER_ONNX_THREADS_SETTING,
            settings.recognition_onnx_threads.to_string(),
        ),
        (
            WORKER_SKIP_ANNOTATED_SETTING,
            settings.recognition_skip_annotated.to_string(),
        ),
        (
            SIMILARITY_WORKERS_SETTING,
            settings.similarity_workers.to_string(),
        ),
        (
            SIMILARITY_THRESHOLD_SETTING,
            format!("{:.2}", settings.similarity_threshold),
        ),
        (
            SIMILARITY_INCLUDE_SUBFOLDERS_SETTING,
            settings.similarity_include_subfolders.to_string(),
        ),
        (
            SIMILARITY_ARCHIVE_DIR_SETTING,
            settings.similarity_archive_dir.clone(),
        ),
        (
            UI_FONT_SIZE_SETTING,
            format!("{:.1}", settings.ui_font_size),
        ),
        (COMFY_ROOT_SETTING, settings.comfy_root.clone()),
        (COMFY_PORT_SETTING, settings.comfy_port.to_string()),
        (COMFY_LOW_VRAM_SETTING, settings.comfy_low_vram.to_string()),
        (
            COMFY_AUTO_START_SETTING,
            settings.comfy_auto_start.to_string(),
        ),
        (COMFY_OUTPUT_DIR_SETTING, settings.comfy_output_dir.clone()),
        (
            LORA_TRAINER_ROOT_SETTING,
            settings.lora_trainer_root.clone(),
        ),
        (
            LORA_TRAINER_PYTHON_SETTING,
            settings.lora_trainer_python.clone(),
        ),
        (
            LORA_TRAINER_BASE_MODEL_SETTING,
            settings.lora_trainer_base_model.clone(),
        ),
        (
            LORA_TRAINER_OUTPUT_DIR_SETTING,
            settings.lora_trainer_output_dir.clone(),
        ),
        (CUDA_RUNTIME_DIR_SETTING, settings.cuda_runtime_dir.clone()),
        (
            RECOGNIZER_MODEL_SETTING,
            settings.recognition_recognizer_model.clone(),
        ),
        (
            BACKGROUND_PRIORITY_SETTING,
            settings.background_priority.clone(),
        ),
        (
            REFERENCE_MATCHING_SETTING,
            settings.reference_matching_enabled.to_string(),
        ),
        (
            REFERENCE_BACKEND_SETTING,
            settings.reference_backend.clone(),
        ),
        (NETWORK_PROXY_SETTING, settings.network_proxy.clone()),
    ];
    let transaction = database
        .connection
        .transaction()
        .map_err(|error| error.to_string())?;
    for (key, value) in entries {
        let value_json = serde_json::to_string(&value).map_err(|error| error.to_string())?;
        transaction
            .execute(
                "INSERT INTO settings(key, value_json, value_version, updated_at)
                 VALUES (?1, ?2, 1, CURRENT_TIMESTAMP)
                 ON CONFLICT(key) DO UPDATE SET
                     value_json = excluded.value_json,
                     value_version = settings.value_version + 1,
                     updated_at = CURRENT_TIMESTAMP",
                rusqlite::params![key, value_json],
            )
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saving_writes_every_key_in_one_go() {
        let mut database = Database::open_in_memory().expect("in-memory database");
        let settings = AppSettings {
            comfy_root: "E:/ComfyUI-aki-v3".to_string(),
            comfy_port: 8199,
            reference_backend: "embedding".to_string(),
            ..AppSettings::default()
        };

        save_app_settings(&mut database, &settings).expect("save settings");

        // A row-by-row save used to leave a half-written settings table (only
        // the first few keys landed), which silently dropped `comfy.*`.
        assert_eq!(
            database
                .get_setting_string(COMFY_ROOT_SETTING)
                .unwrap()
                .as_deref(),
            Some("E:/ComfyUI-aki-v3")
        );
        assert_eq!(
            database
                .get_setting_string(COMFY_PORT_SETTING)
                .unwrap()
                .as_deref(),
            Some("8199")
        );
        assert_eq!(
            database
                .get_setting_string(REFERENCE_BACKEND_SETTING)
                .unwrap()
                .as_deref(),
            Some("embedding")
        );
        let stored: i64 = database
            .connection
            .query_row("SELECT COUNT(*) FROM settings", [], |row| row.get(0))
            .unwrap();
        assert_eq!(stored, 22, "every setting must be persisted");
    }

    #[test]
    fn settings_round_trip_through_the_settings_table() {
        let mut database = Database::open_in_memory().expect("in-memory database");
        let defaults = AppSettings::default();
        assert_eq!(
            load_app_settings(&database).expect("load defaults"),
            defaults
        );

        let custom = AppSettings {
            recognition_onnx_threads: 8,
            recognition_skip_annotated: false,
            similarity_workers: 6,
            similarity_threshold: 0.92,
            similarity_include_subfolders: false,
            similarity_archive_dir: "dupes".to_string(),
            ui_font_size: 15.5,
            comfy_root: "E:/freetime/AI_Draw/ComfyUI-aki/ComfyUI-aki-v3".to_string(),
            comfy_port: 8199,
            comfy_low_vram: true,
            comfy_auto_start: true,
            comfy_output_dir: "E:/generated".to_string(),
            lora_trainer_root: "E:/kohya_ss".to_string(),
            lora_trainer_python: String::new(),
            lora_trainer_base_model: String::new(),
            lora_trainer_output_dir: "E:/lora-models".to_string(),
            cuda_runtime_dir: "E:/cuda-runtime".to_string(),
            recognition_recognizer_model: "camie-initial".to_string(),
            background_priority: "normal".to_string(),
            reference_matching_enabled: true,
            reference_backend: "ccip".to_string(),
            network_proxy: "http://127.0.0.1:8080".to_string(),
        };
        save_app_settings(&mut database, &custom).expect("save settings");

        assert_eq!(load_app_settings(&database).expect("reload"), custom);
    }

    #[test]
    fn comfy_settings_round_trip_and_validate() {
        let mut database = Database::open_in_memory().expect("in-memory database");
        let custom = AppSettings {
            comfy_root: "  E:/freetime/AI_Draw/ComfyUI-aki/ComfyUI-aki-v3  ".to_string(),
            comfy_port: 8188,
            comfy_low_vram: false,
            comfy_auto_start: true,
            comfy_output_dir: "  E:/generated  ".to_string(),
            ..AppSettings::default()
        };
        save_app_settings(&mut database, &custom).expect("save comfy settings");

        let loaded = load_app_settings(&database).expect("reload");

        // Whitespace is trimmed when settings are validated.
        assert_eq!(
            loaded.comfy_root,
            "E:/freetime/AI_Draw/ComfyUI-aki/ComfyUI-aki-v3"
        );
        assert_eq!(loaded.comfy_output_dir, "E:/generated");
        assert_eq!(loaded.comfy_port, 8188);
        assert!(!loaded.comfy_low_vram);
        assert!(loaded.comfy_auto_start);
    }

    #[test]
    fn invalid_comfy_port_is_rejected() {
        let bad_port = AppSettings {
            comfy_port: 80,
            ..AppSettings::default()
        };
        assert!(bad_port.validated().is_err());
    }

    #[test]
    fn proxy_value_is_normalized_and_validated() {
        assert_eq!(
            proxy_url("127.0.0.1:7890").as_deref(),
            Some("http://127.0.0.1:7890")
        );
        assert_eq!(
            proxy_url("https://proxy.example:443").as_deref(),
            Some("https://proxy.example:443")
        );
        assert!(AppSettings {
            network_proxy: "not a proxy".to_string(),
            ..Default::default()
        }
        .validated()
        .is_err());
    }

    #[test]
    fn validation_rejects_out_of_range_values() {
        let too_many_threads = AppSettings {
            recognition_onnx_threads: MAX_ONNX_THREADS + 1,
            ..AppSettings::default()
        };
        assert!(too_many_threads.validated().is_err());

        let too_parallel = AppSettings {
            similarity_workers: MAX_SIMILARITY_WORKERS + 1,
            ..AppSettings::default()
        };
        assert!(too_parallel.validated().is_err());

        let bad_threshold = AppSettings {
            similarity_threshold: 0.2,
            ..AppSettings::default()
        };
        assert!(bad_threshold.validated().is_err());

        let bad_archive = AppSettings {
            similarity_archive_dir: "..\\escape".to_string(),
            ..AppSettings::default()
        };
        assert!(bad_archive.validated().is_err());
    }

    #[test]
    fn corrupted_setting_values_fall_back_to_defaults() {
        let database = Database::open_in_memory().expect("in-memory database");
        database
            .set_setting_string(SIMILARITY_WORKERS_SETTING, "not-a-number")
            .expect("write raw setting");
        database
            .set_setting_string(SIMILARITY_THRESHOLD_SETTING, "5.0")
            .expect("write raw setting");

        let loaded = load_app_settings(&database).expect("load");

        assert_eq!(loaded.similarity_workers, default_similarity_workers());
        assert_eq!(loaded.similarity_threshold, 0.85);
    }

    #[test]
    fn ui_font_size_is_clamped_and_snapped_to_half_pixels() {
        let too_small = AppSettings {
            ui_font_size: 4.0,
            ..AppSettings::default()
        };
        assert_eq!(
            too_small.validated().unwrap().ui_font_size,
            MIN_UI_FONT_SIZE
        );

        let too_large = AppSettings {
            ui_font_size: 99.0,
            ..AppSettings::default()
        };
        assert_eq!(
            too_large.validated().unwrap().ui_font_size,
            MAX_UI_FONT_SIZE
        );

        let in_between = AppSettings {
            ui_font_size: 14.26,
            ..AppSettings::default()
        };
        assert_eq!(in_between.validated().unwrap().ui_font_size, 14.5);
    }
}
