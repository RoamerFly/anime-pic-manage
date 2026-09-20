use crate::database::{
    PersonalTrainingSettings, PersonalTrainingStatus, TrainedPersonalModel, TrainingReadiness,
    TrainingSample, MIN_PERSONAL_TRAINING_CLASSES, PERSONAL_EMBEDDING_DIMENSION,
};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope, IpcError};
use crate::state::AppState;
use crate::worker_runtime;
use serde_json::{json, Value};
use tauri::State;

fn training_sample_payload(sample: &TrainingSample) -> Value {
    json!({
        "annotation_id": &sample.annotation_id,
        "identity_id": &sample.identity_id,
        "display_name": &sample.display_name,
        "image_path": &sample.image_path,
        "revision": sample.revision,
        "embedding": &sample.embedding,
        "embedding_dimension": sample.embedding.len(),
    })
}

fn worker_training_payload(
    samples: &[TrainingSample],
    settings: &PersonalTrainingSettings,
    version: i64,
) -> Value {
    let mut classes = std::collections::BTreeSet::new();
    for sample in samples {
        classes.insert(sample.identity_id.as_str());
    }
    json!({
        "samples": samples.iter().map(training_sample_payload).collect::<Vec<_>>(),
        "sample_count": samples.len(),
        "class_count": classes.len(),
        "min_total_samples": settings.min_total_samples,
        "min_samples_per_class": settings.min_samples_per_class,
        "embedding_dimension": PERSONAL_EMBEDDING_DIMENSION,
        "version": format!("personal-v{version}"),
        "config": {
            "min_classes": MIN_PERSONAL_TRAINING_CLASSES,
            "min_samples_per_class": settings.min_samples_per_class,
            "min_validation_count": 1,
            "min_accuracy": 0.90,
        },
    })
}

fn value_as_usize(value: Option<&Value>, field: &str) -> Result<usize, String> {
    let value = value
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} 必须是非负整数"))?;
    usize::try_from(value).map_err(|_| format!("{field} 超出本机整数范围"))
}

pub(crate) fn validate_training_artifact(artifact: &Value) -> Result<(), String> {
    let object = artifact
        .as_object()
        .ok_or_else(|| "artifact 必须是 JSON 对象".to_string())?;
    let prototypes = object
        .get("prototypes")
        .and_then(Value::as_array)
        .ok_or_else(|| "artifact 必须包含 prototypes 数组".to_string())?;
    if prototypes.is_empty() {
        return Err("artifact.prototypes 不能为空".to_string());
    }
    let mut identity_ids = std::collections::HashSet::with_capacity(prototypes.len());
    for (index, prototype) in prototypes.iter().enumerate() {
        let object = prototype
            .as_object()
            .ok_or_else(|| format!("artifact.prototypes[{index}] 必须是对象"))?;
        let identity_id = object
            .get("identity_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("artifact.prototypes[{index}].identity_id 无效"))?;
        if !identity_ids.insert(identity_id) {
            return Err(format!(
                "artifact.prototypes 包含重复 identity_id: {identity_id}"
            ));
        }
        if object.get("display_name").and_then(Value::as_str).is_none() {
            return Err(format!(
                "artifact.prototypes[{index}].display_name 必须是字符串"
            ));
        }
        let embedding = object
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("artifact.prototypes[{index}].embedding 必须是数组"))?;
        if embedding.len() != PERSONAL_EMBEDDING_DIMENSION
            || embedding
                .iter()
                .any(|value| value.as_f64().is_none_or(|number| !number.is_finite()))
        {
            return Err(format!(
                "artifact.prototypes[{index}].embedding 必须是有限的 2048 维向量"
            ));
        }
        let sample_count = value_as_usize(
            object.get("sample_count"),
            &format!("artifact.prototypes[{index}].sample_count"),
        )?;
        if sample_count == 0 {
            return Err(format!(
                "artifact.prototypes[{index}].sample_count 必须大于 0"
            ));
        }
    }
    Ok(())
}

fn parse_worker_personal_model(
    payload: Value,
    readiness: &TrainingReadiness,
) -> Result<TrainedPersonalModel, worker_runtime::WorkerRuntimeError> {
    let protocol_error = |message: String| worker_runtime::WorkerRuntimeError::Protocol(message);
    let artifact = payload
        .get("artifact")
        .cloned()
        .ok_or_else(|| protocol_error("personal_model.train 缺少 artifact".to_string()))?;
    validate_training_artifact(&artifact).map_err(protocol_error)?;
    let metrics = payload
        .get("metrics")
        .cloned()
        .ok_or_else(|| protocol_error("personal_model.train 缺少 metrics".to_string()))?;
    if !metrics.is_object() {
        return Err(protocol_error(
            "personal_model.train.metrics 必须是对象".to_string(),
        ));
    }
    let sample_count = value_as_usize(
        payload
            .get("sample_count")
            .or_else(|| metrics.get("sample_count")),
        "sample_count",
    )
    .map_err(protocol_error)?;
    let class_count = value_as_usize(
        payload
            .get("class_count")
            .or_else(|| metrics.get("class_count")),
        "class_count",
    )
    .map_err(protocol_error)?;
    if sample_count != readiness.verified_sample_count {
        return Err(protocol_error(format!(
            "Worker 返回 sample_count={}，当前已验证样本为 {}",
            sample_count, readiness.verified_sample_count
        )));
    }
    if class_count != readiness.verified_class_count {
        return Err(protocol_error(format!(
            "Worker 返回 class_count={}，当前已验证分类为 {}",
            class_count, readiness.verified_class_count
        )));
    }
    let algorithm = payload
        .get("algorithm")
        .or_else(|| artifact.get("algorithm"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| protocol_error("personal_model.train 缺少非空 algorithm".to_string()))?
        .to_string();
    let eligible_for_activation = payload
        .get("eligible_for_activation")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            protocol_error("personal_model.train 缺少 eligible_for_activation 布尔值".to_string())
        })?;
    if eligible_for_activation && !readiness.ready {
        return Err(protocol_error(
            "Worker 在本地 readiness 未达标时返回可激活模型".to_string(),
        ));
    }
    let warnings = match payload.get("warnings") {
        None => Vec::new(),
        Some(value) => value
            .as_array()
            .ok_or_else(|| protocol_error("warnings 必须是字符串数组".to_string()))?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .filter(|warning| !warning.trim().is_empty())
                    .map(str::to_string)
                    .ok_or_else(|| protocol_error(format!("warnings[{index}] 必须是非空字符串")))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok(TrainedPersonalModel {
        artifact,
        metrics,
        algorithm,
        warnings,
        eligible_for_activation,
        sample_count,
        class_count,
    })
}

fn training_database_error(
    request_id: &str,
    code: &str,
    message: &str,
    error: impl std::fmt::Display,
) -> IpcError {
    core_error(request_id, code, message, Some(error.to_string()), true)
}

#[tauri::command]
pub fn get_personal_training_status(
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<PersonalTrainingStatus> {
    let request_id = normalize_request_id(request_id);
    let database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "personal.training.status",
                request_id.clone(),
                core_error(
                    &request_id,
                    "DATABASE_LOCK_FAILED",
                    "本地数据库暂时被占用。",
                    None,
                    true,
                ),
            )
        }
    };
    match database.personal_training_status() {
        Ok(status) => success("personal.training.status", request_id, status),
        Err(error) => failure(
            "personal.training.status",
            request_id.clone(),
            training_database_error(
                &request_id,
                "PERSONAL_TRAINING_STATUS_FAILED",
                "读取个人模型训练状态失败。",
                error,
            ),
        ),
    }
}

#[tauri::command]
pub fn train_personal_model(
    state: State<'_, AppState>,
    auto_activate: Option<bool>,
    overwrite_version: Option<i64>,
    request_id: Option<String>,
) -> IpcEnvelope<PersonalTrainingStatus> {
    let request_id = normalize_request_id(request_id);
    let _training_guard = match state.training_control.try_start() {
        Some(guard) => guard,
        None => {
            return failure(
                "personal.training.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "TRAINING_ALREADY_RUNNING",
                    "已有个人模型训练任务正在运行，请等待完成。",
                    None,
                    true,
                ),
            )
        }
    };
    let (samples, readiness, settings, target_model_version, is_overwrite) = {
        let database = match state.database.lock() {
            Ok(database) => database,
            Err(_) => {
                return failure(
                    "personal.training.start",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATABASE_LOCK_FAILED",
                        "本地数据库暂时被占用。",
                        None,
                        true,
                    ),
                )
            }
        };
        let readiness = match database.training_readiness() {
            Ok(readiness) => readiness,
            Err(error) => {
                return failure(
                    "personal.training.start",
                    request_id.clone(),
                    training_database_error(
                        &request_id,
                        "PERSONAL_TRAINING_READINESS_FAILED",
                        "读取个人模型训练样本失败。",
                        error,
                    ),
                )
            }
        };
        let settings = match database.personal_training_settings() {
            Ok(settings) => settings,
            Err(error) => {
                return failure(
                    "personal.training.start",
                    request_id.clone(),
                    training_database_error(
                        &request_id,
                        "PERSONAL_TRAINING_SETTINGS_FAILED",
                        "读取个人模型训练设置失败。",
                        error,
                    ),
                )
            }
        };
        if !readiness.ready {
            return failure(
                "personal.training.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "PERSONAL_TRAINING_NOT_READY",
                    "已验证样本尚不足以训练个人模型。",
                    Some(readiness.reasons.join("；")),
                    false,
                ),
            );
        }
        let samples = match database.verified_training_samples() {
            Ok(samples) => samples,
            Err(error) => {
                return failure(
                    "personal.training.start",
                    request_id.clone(),
                    training_database_error(
                        &request_id,
                        "PERSONAL_TRAINING_SAMPLES_FAILED",
                        "读取个人模型训练样本失败。",
                        error,
                    ),
                )
            }
        };
        let (target_model_version, is_overwrite) = match overwrite_version {
            Some(v) => {
                if v < 2 {
                    return failure(
                        "personal.training.start",
                        request_id.clone(),
                        core_error(
                            &request_id,
                            "PROTECTED_BASE_MODEL",
                            "系统基础模型(v1)受保护，无法覆盖。请选择保存为新模型或选择您自己训练的个人模型(v2及以上)。",
                            None,
                            false,
                        ),
                    );
                }
                let exists = match database.personal_model_version_exists(v) {
                    Ok(exists) => exists,
                    Err(error) => {
                        return failure(
                            "personal.training.start",
                            request_id.clone(),
                            training_database_error(
                                &request_id,
                                "PERSONAL_TRAINING_VERSION_FAILED",
                                "无法检查指定的个人模型版本。",
                                error,
                            ),
                        );
                    }
                };
                if !exists {
                    return failure(
                        "personal.training.start",
                        request_id.clone(),
                        core_error(
                            &request_id,
                            "PERSONAL_MODEL_NOT_FOUND",
                            "指定的个人模型版本不存在，无法覆盖。",
                            None,
                            false,
                        ),
                    );
                }
                (v, true)
            }
            None => {
                let next_version = match database.next_personal_model_version() {
                    Ok(version) => version,
                    Err(error) => {
                        return failure(
                            "personal.training.start",
                            request_id.clone(),
                            training_database_error(
                                &request_id,
                                "PERSONAL_TRAINING_VERSION_FAILED",
                                "无法分配个人模型版本号。",
                                error,
                            ),
                        );
                    }
                };
                (next_version, false)
            }
        };
        if let Err(error) = database.mark_training_started() {
            return failure(
                "personal.training.start",
                request_id.clone(),
                training_database_error(
                    &request_id,
                    "PERSONAL_TRAINING_STATE_FAILED",
                    "无法记录个人模型训练状态。",
                    error,
                ),
            );
        }
        (
            samples,
            readiness,
            settings,
            target_model_version,
            is_overwrite,
        )
    };

    let worker_payload = {
        let mut worker = match state.worker.lock() {
            Ok(worker) => worker,
            Err(_) => {
                if let Ok(database) = state.database.lock() {
                    let _ = database
                        .mark_training_failed("WORKER_LOCK_FAILED", "AI Worker 状态暂时不可用。");
                }
                return failure(
                    "personal.training.start",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "WORKER_LOCK_FAILED",
                        "AI Worker 状态暂时不可用。",
                        None,
                        true,
                    ),
                );
            }
        };
        match worker.request(
            "personal_model.train",
            worker_training_payload(&samples, &settings, target_model_version),
        ) {
            Ok(payload) => payload,
            Err(error) => {
                if let Ok(database) = state.database.lock() {
                    let _ =
                        database.mark_training_failed("WORKER_TRAINING_FAILED", &error.to_string());
                }
                return failure(
                    "personal.training.start",
                    request_id.clone(),
                    match error {
                        worker_runtime::WorkerRuntimeError::WorkerResponse { code, message } => {
                            core_error(&request_id, &code, &message, None, false)
                        }
                        error => core_error(
                            &request_id,
                            "PERSONAL_TRAINING_FAILED",
                            "个人模型训练失败。",
                            Some(error.to_string()),
                            true,
                        ),
                    },
                );
            }
        }
    };
    let model = match parse_worker_personal_model(worker_payload, &readiness) {
        Ok(model) => model,
        Err(error) => {
            if let Ok(database) = state.database.lock() {
                let _ =
                    database.mark_training_failed("WORKER_PROTOCOL_INVALID", &error.to_string());
            }
            return failure(
                "personal.training.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "WORKER_PROTOCOL_INVALID",
                    "个人模型训练结果不符合协议，未保存模型。",
                    Some(error.to_string()),
                    false,
                ),
            );
        }
    };
    let mut database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "personal.training.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "DATABASE_LOCK_FAILED",
                    "本地数据库暂时被占用，训练结果未保存。",
                    None,
                    true,
                ),
            )
        }
    };
    let activate = auto_activate.unwrap_or(true);
    match database.save_trained_personal_model(&model, target_model_version, activate, is_overwrite)
    {
        Ok(status) => success("personal.training.start", request_id, status),
        Err(error) => {
            let _ = database.mark_training_failed("PERSONAL_MODEL_SAVE_FAILED", &error.to_string());
            failure(
                "personal.training.start",
                request_id.clone(),
                training_database_error(
                    &request_id,
                    "PERSONAL_MODEL_SAVE_FAILED",
                    "个人模型训练结果保存失败。",
                    error,
                ),
            )
        }
    }
}

#[tauri::command]
pub fn activate_personal_model(
    state: State<'_, AppState>,
    version: i64,
    request_id: Option<String>,
) -> IpcEnvelope<PersonalTrainingStatus> {
    let request_id = normalize_request_id(request_id);
    let mut database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "personal.training.activate",
                request_id.clone(),
                core_error(
                    &request_id,
                    "DATABASE_LOCK_FAILED",
                    "本地数据库暂时被占用。",
                    None,
                    true,
                ),
            )
        }
    };
    match database.activate_personal_model(version) {
        Ok(status) => success("personal.training.activate", request_id, status),
        Err(rusqlite::Error::QueryReturnedNoRows) => failure(
            "personal.training.activate",
            request_id.clone(),
            core_error(
                &request_id,
                "PERSONAL_MODEL_NOT_ACTIVATABLE",
                "指定的个人模型不存在、无效或尚未通过验证。",
                None,
                false,
            ),
        ),
        Err(error) => failure(
            "personal.training.activate",
            request_id.clone(),
            training_database_error(
                &request_id,
                "PERSONAL_MODEL_ACTIVATE_FAILED",
                "激活个人模型失败，当前模型未改变。",
                error,
            ),
        ),
    }
}

#[tauri::command]
pub fn rollback_personal_model(
    state: State<'_, AppState>,
    version: Option<i64>,
    request_id: Option<String>,
) -> IpcEnvelope<PersonalTrainingStatus> {
    let request_id = normalize_request_id(request_id);
    let mut database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "personal.training.rollback",
                request_id.clone(),
                core_error(
                    &request_id,
                    "DATABASE_LOCK_FAILED",
                    "本地数据库暂时被占用。",
                    None,
                    true,
                ),
            )
        }
    };
    match database.rollback_personal_model(version) {
        Ok(status) => success("personal.training.rollback", request_id, status),
        Err(rusqlite::Error::QueryReturnedNoRows) => failure(
            "personal.training.rollback",
            request_id.clone(),
            core_error(
                &request_id,
                "PERSONAL_MODEL_NO_ROLLBACK_TARGET",
                "没有可回滚的有效个人模型版本。",
                None,
                false,
            ),
        ),
        Err(error) => failure(
            "personal.training.rollback",
            request_id.clone(),
            training_database_error(
                &request_id,
                "PERSONAL_MODEL_ROLLBACK_FAILED",
                "回滚个人模型失败，当前模型未改变。",
                error,
            ),
        ),
    }
}

#[tauri::command]
pub fn delete_personal_model(
    state: State<'_, AppState>,
    version: i64,
    request_id: Option<String>,
) -> IpcEnvelope<PersonalTrainingStatus> {
    let request_id = normalize_request_id(request_id);
    let mut database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "personal.training.delete",
                request_id.clone(),
                core_error(
                    &request_id,
                    "DATABASE_LOCK_FAILED",
                    "本地数据库暂时被占用。",
                    None,
                    true,
                ),
            )
        }
    };
    match database.delete_personal_model_version(version) {
        Ok(status) => success("personal.training.delete", request_id, status),
        Err(rusqlite::Error::QueryReturnedNoRows) => failure(
            "personal.training.delete",
            request_id.clone(),
            core_error(
                &request_id,
                "PERSONAL_MODEL_VERSION_NOT_FOUND",
                "找不到指定的个人模型版本，无法删除。",
                None,
                false,
            ),
        ),
        Err(rusqlite::Error::InvalidQuery) => failure(
            "personal.training.delete",
            request_id.clone(),
            core_error(
                &request_id,
                "CANNOT_DELETE_ACTIVE_MODEL",
                "不能直接删除当前激活的个人模型。请先将其他版本设为激活，或使用基础模型后再删除。",
                None,
                false,
            ),
        ),
        Err(error) => failure(
            "personal.training.delete",
            request_id.clone(),
            training_database_error(
                &request_id,
                "PERSONAL_MODEL_DELETE_FAILED",
                "删除个人模型版本失败。",
                error,
            ),
        ),
    }
}
