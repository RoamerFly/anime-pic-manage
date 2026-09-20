use crate::database::models::{
    Database, PersonalModelVersionRecord, PersonalPrototype, PersonalTrainingSettings,
    PersonalTrainingStatus, TrainedPersonalModel, TrainingClassProgress, TrainingReadiness,
    TrainingSample, MIN_PERSONAL_SAMPLES_PER_CLASS, MIN_PERSONAL_TRAINING_CLASSES,
    MIN_PERSONAL_TRAINING_SAMPLES, PERSONAL_EMBEDDING_DIMENSION,
};
use rusqlite::{params, OptionalExtension};

pub fn valid_personal_model_artifact(value: &serde_json::Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let schema_valid = matches!(
        object.get("schema_version"),
        Some(serde_json::Value::String(value)) if value == "1.0" || value == "1"
    ) || matches!(object.get("schema_version"), Some(serde_json::Value::Number(value)) if value.as_i64() == Some(1));
    if !schema_valid
        || object.get("algorithm").and_then(serde_json::Value::as_str)
            != Some("normalized_centroid_v1")
        || object
            .get("version")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
    {
        return false;
    }
    for key in ["strict_threshold", "min_margin"] {
        if object
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return false;
        }
    }
    let Some(prototypes) = object
        .get("prototypes")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    if prototypes.is_empty() {
        return false;
    }
    let mut identities = std::collections::HashSet::with_capacity(prototypes.len());
    for prototype in prototypes {
        let Some(prototype) = prototype.as_object() else {
            return false;
        };
        let Some(identity_id) = prototype
            .get("identity_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            return false;
        };
        if !identities.insert(identity_id)
            || prototype
                .get("display_name")
                .and_then(serde_json::Value::as_str)
                .is_none()
        {
            return false;
        }
        let Some(embedding) = prototype
            .get("embedding")
            .and_then(serde_json::Value::as_array)
        else {
            return false;
        };
        if embedding.len() != PERSONAL_EMBEDDING_DIMENSION
            || embedding
                .iter()
                .any(|value| value.as_f64().is_none_or(|number| !number.is_finite()))
            || prototype
                .get("sample_count")
                .and_then(serde_json::Value::as_u64)
                .is_none_or(|value| value == 0)
        {
            return false;
        }
    }
    true
}

impl Database {
    /// Return normalized centroids of all currently visible, manually verified
    /// samples. Historical vectors (including unverified model output) remain
    /// available for audit/retraining but are not used for automatic fusion.
    pub fn personal_prototypes(&self) -> Result<Vec<PersonalPrototype>, rusqlite::Error> {
        let mut statement = self.connection.prepare(
            "SELECT e.identity_id, i.display_name, e.embedding_json
             FROM personal_embeddings e
             JOIN annotation_images ai
               ON ai.path = e.image_path AND ai.current_revision = e.revision
             JOIN character_identities i ON i.id = e.identity_id AND i.deleted_at IS NULL
             JOIN image_annotations_current c
               ON c.image_path = e.image_path AND c.annotation_id = e.annotation_id
              AND c.identity_id = e.identity_id AND c.deleted = 0
             WHERE e.dimension = ?1 AND e.manually_verified = 1",
        )?;
        let mut sums: std::collections::BTreeMap<String, (String, Vec<f64>, usize)> =
            std::collections::BTreeMap::new();
        for row in statement.query_map(params![PERSONAL_EMBEDDING_DIMENSION as i64], |row| {
            let identity_id: String = row.get(0)?;
            let display_name: String = row.get(1)?;
            let embedding_json: String = row.get(2)?;
            Ok((identity_id, display_name, embedding_json))
        })? {
            let (identity_id, display_name, embedding_json) = row?;
            let embedding: Vec<f64> = serde_json::from_str(&embedding_json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    embedding_json.len(),
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
            if embedding.len() != PERSONAL_EMBEDDING_DIMENSION
                || embedding.iter().any(|value| !value.is_finite())
            {
                return Err(rusqlite::Error::InvalidQuery);
            }
            let entry = sums
                .entry(identity_id)
                .or_insert_with(|| (display_name, vec![0.0; PERSONAL_EMBEDDING_DIMENSION], 0));
            for (sum, value) in entry.1.iter_mut().zip(embedding) {
                *sum += value;
            }
            entry.2 += 1;
        }
        Ok(sums
            .into_iter()
            .filter_map(|(identity_id, (display_name, sum, sample_count))| {
                let norm = sum.iter().map(|value| value * value).sum::<f64>().sqrt();
                if norm <= f64::EPSILON {
                    return None;
                }
                Some(PersonalPrototype {
                    identity_id,
                    display_name,
                    embedding: sum.into_iter().map(|value| (value / norm) as f32).collect(),
                    sample_count,
                })
            })
            .collect())
    }

    /// Return the current, visible, manually verified samples.  Embeddings
    /// from previous annotation revisions remain in the database for audit,
    /// but are deliberately excluded from training and automatic fusion.
    pub(crate) fn verified_training_samples(&self) -> Result<Vec<TrainingSample>, rusqlite::Error> {
        let mut statement = self.connection.prepare(
            "SELECT e.annotation_id, e.identity_id, i.display_name,
                    e.image_path, e.revision, e.embedding_json
             FROM personal_embeddings e
             JOIN annotation_images ai
               ON ai.path = e.image_path AND ai.current_revision = e.revision
             JOIN image_annotations_current c
               ON c.image_path = e.image_path AND c.annotation_id = e.annotation_id
              AND c.identity_id = e.identity_id AND c.deleted = 0
             JOIN character_identities i
               ON i.id = e.identity_id AND i.deleted_at IS NULL
             WHERE e.dimension = ?1 AND e.manually_verified = 1
             ORDER BY e.identity_id, e.image_path, e.annotation_id",
        )?;
        let samples = statement
            .query_map(params![PERSONAL_EMBEDDING_DIMENSION as i64], |row| {
                let embedding_json: String = row.get(5)?;
                let embedding: Vec<f32> =
                    serde_json::from_str(&embedding_json).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            embedding_json.len(),
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?;
                if embedding.len() != PERSONAL_EMBEDDING_DIMENSION
                    || embedding.iter().any(|value| !value.is_finite())
                {
                    return Err(rusqlite::Error::InvalidQuery);
                }
                Ok(TrainingSample {
                    annotation_id: row.get(0)?,
                    identity_id: row.get(1)?,
                    display_name: row.get(2)?,
                    image_path: row.get(3)?,
                    revision: row.get(4)?,
                    embedding,
                })
            })?
            .collect();
        samples
    }

    pub fn personal_training_settings(&self) -> Result<PersonalTrainingSettings, rusqlite::Error> {
        let (min_total_samples, min_samples_per_class, auto_activate): (i64, i64, i64) =
            self.connection.query_row(
                "SELECT min_total_samples, min_samples_per_class, auto_activate
                 FROM personal_training_settings WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        Ok(PersonalTrainingSettings {
            min_total_samples: usize::try_from(min_total_samples.max(1))
                .unwrap_or(MIN_PERSONAL_TRAINING_SAMPLES),
            min_samples_per_class: usize::try_from(min_samples_per_class.max(1))
                .unwrap_or(MIN_PERSONAL_SAMPLES_PER_CLASS),
            auto_activate: auto_activate != 0,
        })
    }

    pub fn training_readiness(&self) -> Result<TrainingReadiness, rusqlite::Error> {
        let settings = self.personal_training_settings()?;
        let mut statement = self.connection.prepare(
            "SELECT e.identity_id, i.display_name, COUNT(*)
             FROM personal_embeddings e
             JOIN annotation_images ai
               ON ai.path = e.image_path AND ai.current_revision = e.revision
             JOIN image_annotations_current c
               ON c.image_path = e.image_path AND c.annotation_id = e.annotation_id
              AND c.identity_id = e.identity_id AND c.deleted = 0
             JOIN character_identities i
               ON i.id = e.identity_id AND i.deleted_at IS NULL
             WHERE e.dimension = ?1 AND e.manually_verified = 1
             GROUP BY e.identity_id, i.display_name
             ORDER BY COUNT(*) DESC, i.display_name ASC",
        )?;
        let mut class_progress = Vec::new();
        let mut counts = Vec::new();
        for row in statement.query_map(params![PERSONAL_EMBEDDING_DIMENSION as i64], |row| {
            let identity_id: String = row.get(0)?;
            let display_name: String = row.get(1)?;
            let count = row.get::<_, i64>(2)?.max(0) as usize;
            Ok((identity_id, display_name, count))
        })? {
            let (identity_id, display_name, count) = row?;
            let eligible = count >= settings.min_samples_per_class;
            counts.push(count);
            class_progress.push(TrainingClassProgress {
                identity_id,
                display_name,
                sample_count: count,
                min_required: settings.min_samples_per_class,
                eligible,
            });
        }
        let verified_sample_count = counts.iter().sum();
        let verified_class_count = counts.len();
        let eligible_class_count = counts
            .iter()
            .filter(|count| **count >= settings.min_samples_per_class)
            .count();
        let mut reasons = Vec::new();
        if verified_sample_count < settings.min_total_samples {
            reasons.push(format!(
                "至少需要 {} 个已验证样本，当前为 {} 个",
                settings.min_total_samples, verified_sample_count
            ));
        }
        if verified_class_count < MIN_PERSONAL_TRAINING_CLASSES {
            if verified_class_count == 0 {
                reasons.push("尚未有可用于训练的分类".to_string());
            } else {
                reasons.push(format!(
                    "至少需要 {} 个分类，当前为 {} 个",
                    MIN_PERSONAL_TRAINING_CLASSES, verified_class_count
                ));
            }
        }
        if verified_class_count > 0 && eligible_class_count != verified_class_count {
            reasons.push(format!(
                "每个分类至少需要 {} 个样本，当前 {}/{} 个分类达标",
                settings.min_samples_per_class, eligible_class_count, verified_class_count
            ));
        }
        Ok(TrainingReadiness {
            verified_sample_count,
            verified_class_count,
            eligible_class_count,
            ready: reasons.is_empty(),
            reasons,
            class_progress,
        })
    }

    pub fn personal_training_status(&self) -> Result<PersonalTrainingStatus, rusqlite::Error> {
        let readiness = self.training_readiness()?;
        let settings = self.personal_training_settings()?;
        let (status, message, last_error, updated_at) = self.connection.query_row(
            "SELECT status, message, error_message, updated_at
             FROM personal_training_state WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        let active_version = self
            .connection
            .query_row(
                "SELECT version FROM personal_model_versions
                 WHERE status = 'active' ORDER BY version DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let mut statement = self.connection.prepare(
            "SELECT id, version, status, sample_count, class_count, algorithm,
                    artifact_json, metrics_json, warnings_json,
                    eligible_for_activation, error_code, error_message,
                    created_at, updated_at, activated_at
             FROM personal_model_versions ORDER BY version DESC",
        )?;
        let versions = statement
            .query_map([], |row| {
                let artifact_json: Option<String> = row.get(6)?;
                let metrics_json: String = row.get(7)?;
                let warnings_json: String = row.get(8)?;
                Ok(PersonalModelVersionRecord {
                    id: row.get(0)?,
                    version: row.get(1)?,
                    status: row.get(2)?,
                    sample_count: row.get::<_, i64>(3)?.max(0) as usize,
                    class_count: row.get::<_, i64>(4)?.max(0) as usize,
                    algorithm: row.get(5)?,
                    artifact: artifact_json.and_then(|value| serde_json::from_str(&value).ok()),
                    metrics: serde_json::from_str(&metrics_json)
                        .unwrap_or_else(|_| serde_json::json!({})),
                    warnings: serde_json::from_str(&warnings_json).unwrap_or_default(),
                    eligible_for_activation: row.get::<_, i64>(9)? != 0,
                    error_code: row.get(10)?,
                    error_message: row.get(11)?,
                    created_at: row.get(12)?,
                    updated_at: row.get(13)?,
                    activated_at: row.get(14)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PersonalTrainingStatus {
            status,
            message,
            readiness,
            settings,
            active_version,
            versions,
            updated_at,
            last_error,
        })
    }

    pub(crate) fn active_personal_model_artifact(
        &self,
    ) -> Result<Option<serde_json::Value>, rusqlite::Error> {
        let raw: Option<String> = self
            .connection
            .query_row(
                "SELECT artifact_json FROM personal_model_versions
                 WHERE status = 'active' AND eligible_for_activation = 1
                   AND artifact_json IS NOT NULL
                   AND json_valid(artifact_json) = 1
                   AND json_type(artifact_json) = 'object'
                   AND json_type(json_extract(artifact_json, '$.prototypes')) = 'array'
                   AND json_array_length(json_extract(artifact_json, '$.prototypes')) > 0
                 ORDER BY version DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(raw.and_then(|value| {
            serde_json::from_str(&value)
                .ok()
                .filter(valid_personal_model_artifact)
        }))
    }

    pub(crate) fn personal_model_artifact_by_version(
        &self,
        version: i64,
    ) -> Result<Option<serde_json::Value>, rusqlite::Error> {
        let raw: Option<String> = self
            .connection
            .query_row(
                "SELECT artifact_json FROM personal_model_versions
                 WHERE version = ?1
                   AND artifact_json IS NOT NULL
                   AND json_valid(artifact_json) = 1
                   AND json_type(artifact_json) = 'object'
                   AND json_type(json_extract(artifact_json, '$.prototypes')) = 'array'
                   AND json_array_length(json_extract(artifact_json, '$.prototypes')) > 0
                 LIMIT 1",
                params![version],
                |row| row.get(0),
            )
            .optional()?;
        Ok(raw.and_then(|value| {
            serde_json::from_str(&value)
                .ok()
                .filter(valid_personal_model_artifact)
        }))
    }

    pub(crate) fn mark_training_started(&self) -> Result<(), rusqlite::Error> {
        self.connection.execute(
            "UPDATE personal_training_state SET status = 'running',
                    message = '正在训练个人模型。', error_code = NULL,
                    error_message = NULL, started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            [],
        )?;
        Ok(())
    }

    pub(crate) fn next_personal_model_version(&self) -> Result<i64, rusqlite::Error> {
        self.connection.query_row(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM personal_model_versions",
            [],
            |row| row.get(0),
        )
    }

    pub(crate) fn personal_model_version_exists(
        &self,
        version: i64,
    ) -> Result<bool, rusqlite::Error> {
        let count: i64 = self.connection.query_row(
            "SELECT COUNT(1) FROM personal_model_versions WHERE version = ?1",
            rusqlite::params![version],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    pub(crate) fn mark_training_failed(
        &self,
        code: &str,
        message: &str,
    ) -> Result<(), rusqlite::Error> {
        self.connection.execute(
            "UPDATE personal_training_state SET status = 'failed',
                    message = ?1, error_code = ?2, error_message = ?1,
                    updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP
             WHERE id = 1",
            params![message, code],
        )?;
        Ok(())
    }

    pub(crate) fn save_trained_personal_model(
        &mut self,
        model: &TrainedPersonalModel,
        version: i64,
        auto_activate: bool,
        is_overwrite: bool,
    ) -> Result<PersonalTrainingStatus, rusqlite::Error> {
        if !valid_personal_model_artifact(&model.artifact) {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let expected_artifact_version = format!("personal-v{version}");
        if model
            .artifact
            .get("version")
            .and_then(serde_json::Value::as_str)
            != Some(expected_artifact_version.as_str())
        {
            return Err(rusqlite::Error::InvalidQuery);
        }
        if model
            .artifact
            .get("algorithm")
            .and_then(serde_json::Value::as_str)
            != Some(model.algorithm.as_str())
        {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let artifact_json = serde_json::to_string(&model.artifact)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        let metrics_json = serde_json::to_string(&model.metrics)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        let warnings_json = serde_json::to_string(&model.warnings)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        let transaction = self.connection.transaction()?;
        let activate = auto_activate && model.eligible_for_activation;
        let status = if activate { "active" } else { "draft" };

        if is_overwrite {
            // Cannot overwrite the initial base model (version 1)
            if version < 2 {
                return Err(rusqlite::Error::InvalidQuery);
            }
            let count: i64 = transaction.query_row(
                "SELECT COUNT(*) FROM personal_model_versions WHERE version = ?1",
                params![version],
                |row| row.get(0),
            )?;
            if count == 0 {
                return Err(rusqlite::Error::InvalidQuery);
            }
            if activate {
                transaction.execute(
                    "UPDATE personal_model_versions SET status = 'archived',
                            updated_at = CURRENT_TIMESTAMP WHERE status = 'active' AND version != ?1",
                    params![version],
                )?;
            }
            transaction.execute(
                "UPDATE personal_model_versions
                 SET status = ?1, sample_count = ?2, class_count = ?3, algorithm = ?4,
                     artifact_json = ?5, metrics_json = ?6, warnings_json = ?7,
                     eligible_for_activation = ?8, updated_at = CURRENT_TIMESTAMP,
                     activated_at = CASE WHEN ?9 = 1 THEN CURRENT_TIMESTAMP ELSE activated_at END
                 WHERE version = ?10",
                params![
                    status,
                    model.sample_count as i64,
                    model.class_count as i64,
                    model.algorithm,
                    artifact_json,
                    metrics_json,
                    warnings_json,
                    model.eligible_for_activation as i64,
                    activate as i64,
                    version,
                ],
            )?;
            transaction.execute(
                "UPDATE personal_training_state SET status = 'idle',
                        message = CASE WHEN ?1 = 1 THEN '个人模型更新完成并已激活。'
                                       ELSE '个人模型更新完成，等待激活。' END,
                        last_version = ?2, error_code = NULL, error_message = NULL,
                        updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP
                 WHERE id = 1",
                params![activate as i64, version],
            )?;
        } else {
            let next_version: i64 = transaction.query_row(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM personal_model_versions",
                [],
                |row| row.get(0),
            )?;
            if version != next_version {
                return Err(rusqlite::Error::InvalidQuery);
            }
            if activate {
                transaction.execute(
                    "UPDATE personal_model_versions SET status = 'archived',
                            updated_at = CURRENT_TIMESTAMP WHERE status = 'active'",
                    [],
                )?;
            }
            transaction.execute(
                "INSERT INTO personal_model_versions(
                    id, version, status, sample_count, class_count, algorithm,
                    artifact_json, metrics_json, warnings_json, eligible_for_activation,
                    created_at, updated_at, activated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                         CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                         CASE WHEN ?11 = 1 THEN CURRENT_TIMESTAMP ELSE NULL END)",
                params![
                    uuid::Uuid::new_v4().to_string(),
                    version,
                    status,
                    model.sample_count as i64,
                    model.class_count as i64,
                    model.algorithm,
                    artifact_json,
                    metrics_json,
                    warnings_json,
                    model.eligible_for_activation as i64,
                    activate as i64,
                ],
            )?;
            transaction.execute(
                "UPDATE personal_training_state SET status = 'idle',
                        message = CASE WHEN ?1 = 1 THEN '个人模型训练完成并已激活。'
                                       ELSE '个人模型训练完成，等待激活。' END,
                        last_version = ?2, error_code = NULL, error_message = NULL,
                        updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP
                 WHERE id = 1",
                params![activate as i64, version],
            )?;
        }
        transaction.commit()?;
        self.personal_training_status()
    }

    pub(crate) fn activate_personal_model(
        &mut self,
        version: i64,
    ) -> Result<PersonalTrainingStatus, rusqlite::Error> {
        let transaction = self.connection.transaction()?;
        let valid_artifact: Option<String> = transaction
            .query_row(
                "SELECT artifact_json FROM personal_model_versions
                 WHERE version = ?1 AND status IN ('draft', 'archived', 'active')
                   AND eligible_for_activation = 1
                   AND artifact_json IS NOT NULL
                   AND json_valid(artifact_json) = 1
                   AND json_type(artifact_json) = 'object'
                   AND json_type(json_extract(artifact_json, '$.prototypes')) = 'array'
                   AND json_array_length(json_extract(artifact_json, '$.prototypes')) > 0",
                params![version],
                |row| row.get(0),
            )
            .optional()?;
        let valid = valid_artifact
            .as_deref()
            .and_then(|artifact| serde_json::from_str(artifact).ok())
            .is_some_and(|artifact| valid_personal_model_artifact(&artifact));
        if !valid {
            return Err(rusqlite::Error::QueryReturnedNoRows);
        }
        transaction.execute(
            "UPDATE personal_model_versions SET status = 'archived',
                    updated_at = CURRENT_TIMESTAMP WHERE status = 'active' AND version <> ?1",
            params![version],
        )?;
        transaction.execute(
            "UPDATE personal_model_versions SET status = 'active',
                    activated_at = COALESCE(activated_at, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP WHERE version = ?1",
            params![version],
        )?;
        transaction.execute(
            "UPDATE personal_training_state SET status = 'idle',
                    message = '个人模型已激活。', last_version = ?1,
                    updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            params![version],
        )?;
        transaction.commit()?;
        self.personal_training_status()
    }

    pub(crate) fn rollback_personal_model(
        &mut self,
        version: Option<i64>,
    ) -> Result<PersonalTrainingStatus, rusqlite::Error> {
        let target = if let Some(version) = version {
            Some(version)
        } else {
            self.connection
                .query_row(
                    "SELECT version FROM personal_model_versions
                     WHERE status = 'archived' AND eligible_for_activation = 1
                       AND artifact_json IS NOT NULL
                       AND json_valid(artifact_json) = 1
                       AND json_type(artifact_json) = 'object'
                       AND json_type(json_extract(artifact_json, '$.prototypes')) = 'array'
                       AND json_array_length(json_extract(artifact_json, '$.prototypes')) > 0
                     ORDER BY version DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()?
        };
        let Some(target) = target else {
            return Err(rusqlite::Error::QueryReturnedNoRows);
        };
        self.activate_personal_model(target)
    }

    pub(crate) fn delete_personal_model_version(
        &mut self,
        version: i64,
    ) -> Result<PersonalTrainingStatus, rusqlite::Error> {
        let is_active: bool = self.connection.query_row(
            "SELECT COUNT(*) > 0 FROM personal_model_versions WHERE version = ?1 AND status = 'active'",
            params![version],
            |row| row.get(0),
        )?;
        if is_active {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let rows_deleted = self.connection.execute(
            "DELETE FROM personal_model_versions WHERE version = ?1",
            params![version],
        )?;
        if rows_deleted == 0 {
            return Err(rusqlite::Error::QueryReturnedNoRows);
        }
        self.personal_training_status()
    }
}
