use crate::database::{
    AnnotationBbox, AnnotationInput, AnnotationRecord, CharacterIdentity, Database, DatabaseError,
    EmbeddingSample, ImageAnnotations, PERSONAL_EMBEDDING_DIMENSION,
};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope, IpcError};
use crate::models::is_supported_preview_image;
use crate::path_utils::canonicalize_for_user;
use crate::state::AppState;
use crate::worker_runtime;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tauri::State;

pub(crate) const MAX_ANNOTATIONS_PER_IMAGE: usize = 100;

#[derive(Debug, Clone, Deserialize)]
struct AnnotationWorkerEmbedding {
    annotation_id: String,
    embedding: Vec<f32>,
}

type AnnotationPathError = (&'static str, &'static str, Option<String>);

fn validate_annotation_image_path(path: &str) -> Result<PathBuf, AnnotationPathError> {
    let candidate = Path::new(path);
    if !candidate.is_absolute() {
        return Err(("INVALID_IMAGE_PATH", "图片路径必须是绝对路径。", None));
    }
    let file = canonicalize_for_user(candidate).map_err(|error| {
        (
            "INVALID_IMAGE_PATH",
            "无法访问指定图片。",
            Some(error.to_string()),
        )
    })?;
    if !file.is_file() {
        return Err(("INVALID_IMAGE_PATH", "指定路径不是普通图片文件。", None));
    }
    if !is_supported_preview_image(&file) {
        return Err((
            "UNSUPPORTED_IMAGE_FORMAT",
            "当前仅支持 PNG、JPG、JPEG、WEBP 和 GIF 图片。",
            None,
        ));
    }
    Ok(file)
}

fn validate_image_size(image_size: [u32; 2]) -> Result<(), String> {
    if image_size[0] == 0
        || image_size[1] == 0
        || image_size[0] > 100_000
        || image_size[1] > 100_000
    {
        return Err("image_size 必须是 1 到 100000 之间的正整数宽高。".to_string());
    }
    Ok(())
}

fn validate_bbox(bbox: AnnotationBbox) -> Result<(), String> {
    let values = [bbox.x, bbox.y, bbox.width, bbox.height];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("bbox 坐标必须是有限数字。".to_string());
    }
    if bbox.x < 0.0
        || bbox.y < 0.0
        || bbox.width <= 0.0
        || bbox.height <= 0.0
        || bbox.x > 1.0
        || bbox.y > 1.0
        || bbox.width > 1.0
        || bbox.height > 1.0
        || bbox.x + bbox.width > 1.0 + f64::EPSILON
        || bbox.y + bbox.height > 1.0 + f64::EPSILON
    {
        return Err("bbox 必须位于 0..1，且右下边界不能超出图片。".to_string());
    }
    Ok(())
}

fn validate_annotation_input(annotation: &AnnotationInput) -> Result<(), String> {
    let label = annotation.label_name.trim();
    if label.is_empty() || label.chars().count() > 256 {
        return Err("label_name 必须是 1 到 256 个字符。".to_string());
    }
    if label.chars().any(char::is_control) {
        return Err("label_name 不能包含控制字符。".to_string());
    }
    if !matches!(annotation.source.as_str(), "model" | "manual") {
        return Err("source 必须是 model 或 manual。".to_string());
    }
    if let Some(id) = annotation.id.as_deref() {
        uuid::Uuid::parse_str(id).map_err(|_| "annotation id 必须是 UUID。".to_string())?;
    }
    if let Some(id) = annotation.identity_id.as_deref() {
        uuid::Uuid::parse_str(id).map_err(|_| "identity_id 必须是 UUID。".to_string())?;
    }
    validate_bbox(annotation.bbox)
}

fn normalize_label_name(value: &str) -> String {
    value.trim().to_string()
}

pub(crate) fn annotation_records(
    database: &Database,
    annotations: Vec<AnnotationInput>,
) -> Result<Vec<AnnotationRecord>, String> {
    if annotations.len() > MAX_ANNOTATIONS_PER_IMAGE {
        return Err(format!(
            "单张图片最多保存 {MAX_ANNOTATIONS_PER_IMAGE} 个识别框。"
        ));
    }
    let mut ids = std::collections::HashSet::with_capacity(annotations.len());
    let mut identities_by_name = std::collections::HashMap::with_capacity(annotations.len());
    let mut records = Vec::with_capacity(annotations.len());
    for annotation in annotations {
        validate_annotation_input(&annotation)?;
        let id = annotation
            .id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if !ids.insert(id.clone()) {
            return Err(format!("annotation id 重复: {id}"));
        }
        let label_name = normalize_label_name(&annotation.label_name);
        let normalized_name = label_name.to_lowercase();
        let identity_id = if let Some(identity_id) = annotation.identity_id {
            if let Some(previous) = identities_by_name.get(&normalized_name) {
                if previous != &identity_id {
                    return Err(format!(
                        "同一批次中同名分类对应了多个 identity_id: {label_name}"
                    ));
                }
            }
            identities_by_name.insert(normalized_name, identity_id.clone());
            identity_id
        } else if let Some(identity_id) = identities_by_name.get(&normalized_name) {
            identity_id.clone()
        } else {
            let identity_id = database
                .identity_id_by_name(&normalized_name)
                .map_err(|error| format!("读取分类失败: {error}"))?
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            identities_by_name.insert(normalized_name, identity_id.clone());
            identity_id
        };
        records.push(AnnotationRecord {
            id,
            identity_id,
            label_name,
            bbox: annotation.bbox,
            source: annotation.source,
            manually_adjusted: annotation.manually_adjusted,
        });
    }
    Ok(records)
}

pub(crate) fn worker_annotation_payload(annotation: &AnnotationRecord) -> Value {
    json!({
        "annotation_id": &annotation.id,
        "identity_id": &annotation.identity_id,
        "label_name": &annotation.label_name,
        "bbox": {
            "x1": annotation.bbox.x,
            "y1": annotation.bbox.y,
            "x2": annotation.bbox.x + annotation.bbox.width,
            "y2": annotation.bbox.y + annotation.bbox.height,
        }
    })
}

fn parse_worker_embeddings(
    payload: Value,
    records: &[AnnotationRecord],
) -> Result<Vec<EmbeddingSample>, worker_runtime::WorkerRuntimeError> {
    let raw = payload
        .get("embeddings")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            worker_runtime::WorkerRuntimeError::Protocol(
                "recognition.embedding payload 缺少 embeddings 数组".to_string(),
            )
        })?;
    if raw.len() != records.len() {
        return Err(worker_runtime::WorkerRuntimeError::Protocol(format!(
            "recognition.embedding 返回 {} 个向量，预期 {} 个",
            raw.len(),
            records.len()
        )));
    }
    let mut parsed = Vec::with_capacity(raw.len());
    let mut seen = std::collections::HashSet::with_capacity(raw.len());
    for value in raw {
        let item: AnnotationWorkerEmbedding =
            serde_json::from_value(value.clone()).map_err(|error| {
                worker_runtime::WorkerRuntimeError::Protocol(format!(
                    "recognition.embedding 返回项无效: {error}"
                ))
            })?;
        if item.embedding.len() != PERSONAL_EMBEDDING_DIMENSION
            || item.embedding.iter().any(|value| !value.is_finite())
        {
            return Err(worker_runtime::WorkerRuntimeError::Protocol(
                "recognition.embedding 必须返回有限的 2048 维向量".to_string(),
            ));
        }
        if !seen.insert(item.annotation_id.clone()) {
            return Err(worker_runtime::WorkerRuntimeError::Protocol(
                "recognition.embedding 返回了重复 annotation_id".to_string(),
            ));
        }
        let Some(record) = records
            .iter()
            .find(|record| record.id == item.annotation_id)
        else {
            return Err(worker_runtime::WorkerRuntimeError::Protocol(format!(
                "recognition.embedding 返回未知 annotation_id: {}",
                item.annotation_id
            )));
        };
        parsed.push(EmbeddingSample {
            annotation_id: item.annotation_id,
            identity_id: record.identity_id.clone(),
            embedding: item.embedding,
            manually_verified: record.source == "manual" || record.manually_adjusted,
        });
    }
    if parsed.len() != records.len() {
        return Err(worker_runtime::WorkerRuntimeError::Protocol(
            "recognition.embedding 未返回所有 annotation_id".to_string(),
        ));
    }
    Ok(parsed)
}

fn worker_error_as_core(request_id: &str, error: worker_runtime::WorkerRuntimeError) -> IpcError {
    match error {
        worker_runtime::WorkerRuntimeError::WorkerResponse { code, message } => {
            core_error(request_id, &code, &message, None, false)
        }
        other => core_error(
            request_id,
            "EMBEDDING_FAILED",
            "无法提取人工标注的个人特征，标注未保存。",
            Some(other.to_string()),
            true,
        ),
    }
}

#[tauri::command]
pub fn list_image_annotations(
    state: State<'_, AppState>,
    path: String,
    request_id: Option<String>,
) -> IpcEnvelope<ImageAnnotations> {
    let request_id = normalize_request_id(request_id);
    let path = match validate_annotation_image_path(&path) {
        Ok(path) => path,
        Err((code, message, detail)) => {
            return failure(
                "annotation.list",
                request_id.clone(),
                core_error(&request_id, code, message, detail, false),
            );
        }
    };
    let path = path.to_string_lossy().into_owned();
    let database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "annotation.list",
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
    match database.list_image_annotations(&path) {
        Ok(payload) => success("annotation.list", request_id, payload),
        Err(error) => failure(
            "annotation.list",
            request_id.clone(),
            core_error(
                &request_id,
                "ANNOTATION_LIST_FAILED",
                "读取图片标注失败。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub fn list_character_identities(
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<Vec<CharacterIdentity>> {
    let request_id = normalize_request_id(request_id);
    let database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "identity.list",
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
    match database.list_character_identities() {
        Ok(payload) => success("identity.list", request_id, payload),
        Err(error) => failure(
            "identity.list",
            request_id.clone(),
            core_error(
                &request_id,
                "IDENTITY_LIST_FAILED",
                "读取分类列表失败。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub fn save_image_annotations(
    state: State<'_, AppState>,
    path: String,
    image_size: [u32; 2],
    annotations: Vec<AnnotationInput>,
    base_revision: Option<i64>,
    request_id: Option<String>,
) -> IpcEnvelope<ImageAnnotations> {
    let request_id = normalize_request_id(request_id);
    let path = match validate_annotation_image_path(&path) {
        Ok(path) => path,
        Err((code, message, detail)) => {
            return failure(
                "annotation.save",
                request_id.clone(),
                core_error(&request_id, code, message, detail, false),
            );
        }
    };
    if let Err(message) = validate_image_size(image_size) {
        return failure(
            "annotation.save",
            request_id.clone(),
            core_error(&request_id, "INVALID_IMAGE_SIZE", &message, None, false),
        );
    }
    let path = path.to_string_lossy().into_owned();
    let records = {
        let database = match state.database.lock() {
            Ok(database) => database,
            Err(_) => {
                return failure(
                    "annotation.save",
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
        match annotation_records(&database, annotations) {
            Ok(records) => records,
            Err(message) => {
                return failure(
                    "annotation.save",
                    request_id.clone(),
                    core_error(&request_id, "INVALID_ANNOTATIONS", &message, None, false),
                )
            }
        }
    };

    let embedding_samples = if records.is_empty() {
        Vec::new()
    } else {
        let worker_annotations: Vec<Value> =
            records.iter().map(worker_annotation_payload).collect();
        let payload = {
            let mut worker = match state.worker.lock() {
                Ok(worker) => worker,
                Err(_) => {
                    return failure(
                        "annotation.save",
                        request_id.clone(),
                        core_error(
                            &request_id,
                            "WORKER_LOCK_FAILED",
                            "AI Worker 状态暂时不可用，标注未保存。",
                            None,
                            true,
                        ),
                    )
                }
            };
            match worker.request(
                "recognition.embedding",
                json!({"path": &path, "annotations": worker_annotations}),
            ) {
                Ok(payload) => payload,
                Err(error) => {
                    return failure(
                        "annotation.save",
                        request_id.clone(),
                        worker_error_as_core(&request_id, error),
                    )
                }
            }
        };
        match parse_worker_embeddings(payload, &records) {
            Ok(embeddings) => embeddings,
            Err(error) => {
                return failure(
                    "annotation.save",
                    request_id.clone(),
                    worker_error_as_core(&request_id, error),
                )
            }
        }
    };

    let mut database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return failure(
                "annotation.save",
                request_id.clone(),
                core_error(
                    &request_id,
                    "DATABASE_LOCK_FAILED",
                    "本地数据库暂时被占用，标注未保存。",
                    None,
                    true,
                ),
            )
        }
    };
    match database.save_image_annotations(
        &path,
        image_size,
        &records,
        &embedding_samples,
        base_revision,
    ) {
        Ok(payload) => success("annotation.save", request_id, payload),
        Err(DatabaseError::RevisionConflict { current_revision }) => failure(
            "annotation.save",
            request_id.clone(),
            core_error(
                &request_id,
                "ANNOTATION_REVISION_CONFLICT",
                "图片标注已被其他操作更新，请重新加载后再保存。",
                Some(format!("current_revision={current_revision}")),
                true,
            ),
        ),
        Err(error) => failure(
            "annotation.save",
            request_id.clone(),
            core_error(
                &request_id,
                "ANNOTATION_SAVE_FAILED",
                "保存标注失败，未留下半成品。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}
