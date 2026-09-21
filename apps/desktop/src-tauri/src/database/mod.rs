pub mod annotations;
pub mod models;
pub mod personal_models;
pub mod schema;
pub mod settings;
pub mod similarity;

#[cfg(test)]
mod tests;

use rusqlite::{params, Connection};
use std::fs;
use std::path::{Path, PathBuf};

#[allow(unused_imports)]
pub use models::{
    AnnotationBbox, AnnotationInput, AnnotationRecord, CharacterIdentity, Database, DatabaseError,
    DatabaseHealth, EmbeddingSample, IdentitySample, IdentitySampleSummary, ImageAnnotations,
    ImageSimilarityFeatureRecord, PersonalModelVersionRecord, PersonalPrototype,
    PersonalTrainingSettings, PersonalTrainingStatus, SimilarityGroup, SimilarityGroupItem,
    TrainedPersonalModel, TrainingClassProgress, TrainingReadiness, TrainingSample,
    BACKGROUND_PRIORITY_SETTING, COMFY_AUTO_START_SETTING, COMFY_LOW_VRAM_SETTING,
    COMFY_OUTPUT_DIR_SETTING, COMFY_PORT_SETTING, COMFY_ROOT_SETTING, CUDA_RUNTIME_DIR_SETTING,
    CURRENT_SCHEMA_VERSION, LORA_TRAINER_BASE_MODEL_SETTING, LORA_TRAINER_OUTPUT_DIR_SETTING,
    LORA_TRAINER_PYTHON_SETTING, LORA_TRAINER_ROOT_SETTING, MIN_PERSONAL_SAMPLES_PER_CLASS,
    MIN_PERSONAL_TRAINING_CLASSES, MIN_PERSONAL_TRAINING_SAMPLES, NETWORK_PROXY_SETTING,
    PERSONAL_EMBEDDING_DIMENSION, RECOGNIZER_MODEL_SETTING, REFERENCE_BACKEND_SETTING,
    REFERENCE_MATCHING_SETTING, SIMILARITY_ARCHIVE_DIR_SETTING, SIMILARITY_FEATURE_VERSION,
    SIMILARITY_INCLUDE_SUBFOLDERS_SETTING, SIMILARITY_THRESHOLD_SETTING,
    SIMILARITY_WORKERS_SETTING, UI_FONT_SIZE_SETTING, WORKER_COMPUTE_DEVICE_SETTING,
    WORKER_ONNX_THREADS_SETTING, WORKER_RUNTIME_MODE_SETTING, WORKER_SKIP_ANNOTATED_SETTING,
};
#[allow(unused_imports)]
pub use personal_models::valid_personal_model_artifact;

impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DatabaseError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&path)?;
        connection.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;",
        )?;
        let database = Self { connection, path };
        database.migrate()?;
        Ok(database)
    }

    pub fn open_in_memory() -> Result<Self, DatabaseError> {
        let connection = Connection::open_in_memory()?;
        connection.execute_batch("PRAGMA foreign_keys = ON;")?;
        let database = Self {
            connection,
            path: PathBuf::from(":memory:"),
        };
        database.migrate()?;
        Ok(database)
    }

    pub fn migrate(&self) -> Result<(), rusqlite::Error> {
        schema::run_migrations(&self.connection)
    }

    pub fn health(&self) -> Result<DatabaseHealth, rusqlite::Error> {
        let schema_version: i64 = self
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))?;
        let table_count: i64 = self.connection.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get(0),
        )?;
        Ok(DatabaseHealth {
            status: if schema_version >= CURRENT_SCHEMA_VERSION {
                "ready"
            } else {
                "initializing"
            }
            .to_string(),
            path: self.path.to_string_lossy().into_owned(),
            schema_version,
            table_count,
            message: format!("SQLite 已迁移至 v{schema_version}，包含 {table_count} 张核心表"),
        })
    }

    pub fn table_exists(&self, table: &str) -> Result<bool, rusqlite::Error> {
        self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
            params![table],
            |row| row.get(0),
        )
    }
}
