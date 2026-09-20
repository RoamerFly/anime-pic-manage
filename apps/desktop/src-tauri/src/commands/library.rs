use crate::commands::personal_model::validate_training_artifact;
use crate::database::{Database, PersonalPrototype, PERSONAL_EMBEDDING_DIMENSION};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::models::{
    allow_asset_directory, allow_asset_file, display_name_for_path, is_supported_preview_image,
    BatchMoveResult, LibrarySelection, MoveFileErrorItem, MovedFileItem, ScanCancellation,
    ScanError, ScanLibraryResult, ScanProgress,
};
use crate::path_utils::{canonicalize_for_user, display_path};
use crate::result_store::{self, ResultKind};
use crate::state::{AppState, ScanControl, ScanControlResponse};
use crate::worker_runtime::{self, WorkerManager};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

/// If limit is None or Some(0), it means no upper bound (scan all images in directory).
/// If limit is Some(v > 0), preserve the requested limit.
pub(crate) fn clamp_scan_limit(limit: Option<u32>) -> Option<usize> {
    match limit {
        None | Some(0) => None,
        Some(val) => Some(val as usize),
    }
}

fn scan_error_from_worker(path: &str, error: worker_runtime::WorkerRuntimeError) -> ScanError {
    match error {
        worker_runtime::WorkerRuntimeError::WorkerResponse { code, message } => ScanError {
            path: path.to_string(),
            code,
            message,
            detail: None,
        },
        other => ScanError {
            path: path.to_string(),
            code: "RECOGNITION_FAILED".to_string(),
            message: "图片识别失败，已跳过此文件。".to_string(),
            detail: Some(other.to_string()),
        },
    }
}

fn emit_scan_progress(
    app: &AppHandle,
    phase: &str,
    current: usize,
    total: usize,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    let _ = app.emit(
        "scan-progress",
        ScanProgress {
            phase: phase.to_string(),
            current,
            total,
            path: path.into(),
            message: message.into(),
        },
    );
}

fn emit_control_progress(
    app: &AppHandle,
    control: &ScanControl,
    phase: &str,
    current: usize,
    total: usize,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    let path = path.into();
    let message = message.into();
    control.update_progress(current, total, &path, &message);
    emit_scan_progress(app, phase, current, total, path, message);
}

#[allow(clippy::too_many_arguments)]
fn emit_scan_result(
    app: &AppHandle,
    directory: &Path,
    total_discovered: usize,
    processed: usize,
    result: Option<&Value>,
    error: Option<&ScanError>,
    model_version: Option<i64>,
    model_name: Option<&str>,
    skip_annotated: bool,
) {
    let _ = app.emit(
        "library://scan-result",
        json!({
            "directory": display_path(directory),
            "total_discovered": total_discovered,
            "processed": processed,
            "result": result,
            "error": error,
            "model_version": model_version,
            "model_name": model_name,
            "skip_annotated": skip_annotated,
        }),
    );
}

pub(crate) fn total_discovered_from_payload(payload: &Value, image_count: usize) -> usize {
    payload
        .get("total_count")
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or(image_count)
}

#[derive(Debug, Clone)]
enum PersonalScanContext {
    Model(Value),
    Prototypes(Value),
}

/// Merge the current verified prototypes into an activated artifact for a scan.
///
/// The activated model's version and calibrated thresholds remain authoritative,
/// while a live prototype replaces an older prototype with the same identity and
/// new identities are appended.  This keeps newly verified samples effective on
/// the next scan without sending both personal-model representations to Worker.
pub(crate) fn merge_personal_model_artifact(
    artifact: Value,
    live_prototypes: &[PersonalPrototype],
) -> Value {
    if live_prototypes.is_empty() || validate_training_artifact(&artifact).is_err() {
        return artifact;
    }

    let mut live_by_identity = std::collections::BTreeMap::new();
    for prototype in live_prototypes {
        if prototype.identity_id.trim().is_empty()
            || prototype.display_name.trim().is_empty()
            || prototype.embedding.len() != PERSONAL_EMBEDDING_DIMENSION
            || prototype.embedding.iter().any(|value| !value.is_finite())
            || prototype.sample_count == 0
            || live_by_identity
                .insert(prototype.identity_id.clone(), prototype)
                .is_some()
        {
            return artifact;
        }
    }

    let mut merged = artifact;
    let Some(prototypes) = merged.get_mut("prototypes").and_then(Value::as_array_mut) else {
        return merged;
    };
    let mut existing = std::collections::HashSet::with_capacity(prototypes.len());
    for prototype in prototypes.iter_mut() {
        // `validate_training_artifact` above guarantees this shape.
        let identity_id = prototype
            .get("identity_id")
            .and_then(Value::as_str)
            .expect("validated artifact prototype identity_id");
        existing.insert(identity_id.to_string());
        if let Some(live) = live_by_identity.remove(identity_id) {
            *prototype = json!(live);
        }
    }
    for (_, live) in live_by_identity {
        if !existing.contains(&live.identity_id) {
            prototypes.push(json!(live));
        }
    }
    merged
}

fn load_personal_scan_context(
    database: &Database,
    model_version: Option<i64>,
) -> Result<(PersonalScanContext, Option<String>), rusqlite::Error> {
    if let Some(0) = model_version {
        return Ok((
            PersonalScanContext::Prototypes(serde_json::Value::Array(Vec::new())),
            Some("基础通用模型".to_string()),
        ));
    }
    let live_prototypes = database.personal_prototypes()?;
    if let Some(version) = model_version {
        if version > 0 {
            if let Some(artifact) = database.personal_model_artifact_by_version(version)? {
                return Ok((
                    PersonalScanContext::Model(merge_personal_model_artifact(
                        artifact,
                        &live_prototypes,
                    )),
                    Some(format!("个人模型 v{}", version)),
                ));
            }
        }
    }
    let active_artifact = database.active_personal_model_artifact()?;
    if let Some(artifact) = active_artifact {
        return Ok((
            PersonalScanContext::Model(merge_personal_model_artifact(artifact, &live_prototypes)),
            Some("个人模型 (已激活)".to_string()),
        ));
    }
    let prototypes = serde_json::to_value(live_prototypes)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    Ok((
        PersonalScanContext::Prototypes(prototypes),
        Some("基础模型 (包含学习样本)".to_string()),
    ))
}

fn run_library_scan(
    app: AppHandle,
    worker: Arc<Mutex<WorkerManager>>,
    database: Arc<Mutex<Database>>,
    control: Arc<ScanControl>,
    directory: PathBuf,
    limit: Option<usize>,
    model_version: Option<i64>,
    skip_annotated: Option<bool>,
) -> Result<ScanLibraryResult, worker_runtime::WorkerRuntimeError> {
    // The recognizer is a user preference: an empty value keeps the Worker's
    // own default (the installed model with the smallest id).
    let recognizer_model = database
        .lock()
        .ok()
        .and_then(|database| crate::app_settings::load_app_settings(&database).ok())
        .map(|settings| settings.recognition_recognizer_model)
        .unwrap_or_default();
    // Reference matching is opt-in per settings and only when a library exists.
    let reference_library = {
        let data_dir = app.state::<AppState>().data_dir.clone();
        database
            .lock()
            .ok()
            .and_then(|database| crate::app_settings::load_app_settings(&database).ok())
            .and_then(|settings| {
                crate::commands::reference::active_library_path(&settings, &data_dir)
            })
    };
    let enumerate_payload = {
        let mut manager = worker.lock().map_err(|_| {
            worker_runtime::WorkerRuntimeError::Io("Worker 状态锁已中毒".to_string())
        })?;
        let mut enum_payload = json!({
            "directory": display_path(&directory),
            "recursive": true,
            "include_hidden": false,
        });
        if let Some(l) = limit {
            enum_payload["limit"] = json!(l);
        }
        manager.request("images.enumerate", enum_payload)?
    };
    let images = enumerate_payload
        .get("images")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            worker_runtime::WorkerRuntimeError::Protocol(
                "images.enumerate payload 缺少 images 数组".to_string(),
            )
        })?;
    let total_discovered = total_discovered_from_payload(&enumerate_payload, images.len());
    let total = match limit {
        Some(l) => total_discovered.min(l),
        None => total_discovered,
    };
    let mut errors = Vec::new();
    if let Some(enumeration_errors) = enumerate_payload.get("errors").and_then(Value::as_array) {
        for item in enumeration_errors {
            let path = item
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let message = item
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("无法读取图片")
                .to_string();
            errors.push(ScanError {
                path,
                code: "ENUMERATE_IMAGE_FAILED".to_string(),
                message,
                detail: None,
            });
        }
    }

    let do_skip_annotated = skip_annotated.unwrap_or(false);
    let annotated_paths = if do_skip_annotated {
        if let Ok(db) = database.lock() {
            db.list_annotated_image_paths(&display_path(&directory))
                .unwrap_or_default()
        } else {
            std::collections::HashSet::new()
        }
    } else {
        std::collections::HashSet::new()
    };

    let (personal_context, model_name) = match database.lock() {
        Ok(database) => match load_personal_scan_context(&database, model_version) {
            Ok(context) => context,
            Err(error) => {
                errors.push(ScanError {
                    path: display_path(&directory),
                    code: "PERSONAL_MODEL_READ_FAILED".to_string(),
                    message: "读取个人分类特征失败，已跳过个人融合。".to_string(),
                    detail: Some(error.to_string()),
                });
                (
                    PersonalScanContext::Prototypes(Value::Array(Vec::new())),
                    Some("基础通用模型".to_string()),
                )
            }
        },
        Err(_) => {
            errors.push(ScanError {
                path: display_path(&directory),
                code: "DATABASE_LOCK_FAILED".to_string(),
                message: "本地数据库暂时被占用，已跳过个人融合。".to_string(),
                detail: None,
            });
            (
                PersonalScanContext::Prototypes(Value::Array(Vec::new())),
                Some("基础通用模型".to_string()),
            )
        }
    };

    let mut results = Vec::with_capacity(total);
    let mut processed = 0;
    let image_slice = match limit {
        Some(l) => &images[..images.len().min(l)],
        None => images.as_slice(),
    };
    for (index, image) in image_slice.iter().enumerate() {
        let current = index + 1;
        if control.is_cancelled() {
            break;
        }
        let pause_result = control.wait_if_paused_with(|snapshot| {
            emit_control_progress(
                &app,
                &control,
                "paused",
                snapshot.current,
                snapshot.total,
                snapshot.path,
                "扫描已暂停，等待继续或终止。",
            );
        });
        if pause_result.cancelled {
            break;
        }
        if pause_result.was_paused {
            let snapshot = control.snapshot();
            emit_control_progress(
                &app,
                &control,
                "running",
                snapshot.current,
                snapshot.total,
                snapshot.path,
                "扫描已继续。",
            );
        }
        if control.is_cancelled() {
            break;
        }
        let path = image
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string);
        let Some(path) = path else {
            processed += 1;
            errors.push(ScanError {
                path: String::new(),
                code: "INVALID_IMAGE_RECORD".to_string(),
                message: "Worker 返回的图片记录缺少 path。".to_string(),
                detail: None,
            });
            emit_scan_result(
                &app,
                &directory,
                total_discovered,
                processed,
                None,
                errors.last(),
                model_version,
                model_name.as_deref(),
                do_skip_annotated,
            );
            emit_control_progress(
                &app,
                &control,
                "error",
                current,
                total,
                "",
                "图片记录无效，已跳过。",
            );
            continue;
        };

        if do_skip_annotated && annotated_paths.contains(&path) {
            let existing_result = if let Ok(db) = database.lock() {
                db.get_annotation_as_scan_result(&path).ok().flatten()
            } else {
                None
            };
            processed += 1;
            if let Some(payload) = existing_result {
                results.push(payload);
                emit_scan_result(
                    &app,
                    &directory,
                    total_discovered,
                    processed,
                    results.last(),
                    None,
                    model_version,
                    model_name.as_deref(),
                    do_skip_annotated,
                );
                emit_control_progress(
                    &app,
                    &control,
                    "completed",
                    current,
                    total,
                    &path,
                    "已跳过已标注图片（已直接载入人工标注）。",
                );
            } else {
                emit_scan_result(
                    &app,
                    &directory,
                    total_discovered,
                    processed,
                    None,
                    None,
                    model_version,
                    model_name.as_deref(),
                    do_skip_annotated,
                );
                emit_control_progress(
                    &app,
                    &control,
                    "completed",
                    current,
                    total,
                    &path,
                    "已跳过已标注图片。",
                );
            }
            continue;
        }

        emit_control_progress(
            &app,
            &control,
            "recognizing",
            current,
            total,
            &path,
            "正在识别图片…",
        );
        let mut recognition_payload = json!({
            "path": &path,
            "top_k": 5,
        });
        if !recognizer_model.trim().is_empty() {
            recognition_payload["recognizer"] = json!(recognizer_model.trim());
        }
        if let Some(path) = reference_library.as_ref() {
            recognition_payload["reference_library_path"] = json!(path.to_string_lossy());
        }
        match &personal_context {
            PersonalScanContext::Model(artifact) => {
                recognition_payload["personal_model"] = artifact.clone();
            }
            PersonalScanContext::Prototypes(prototypes) => {
                recognition_payload["personal_prototypes"] = prototypes.clone();
            }
        }
        let recognition = {
            let mut manager = worker.lock().map_err(|_| {
                worker_runtime::WorkerRuntimeError::Io("Worker 状态锁已中毒".to_string())
            })?;
            manager.request("recognition.image", recognition_payload)
        };
        processed += 1;
        match recognition {
            Ok(payload) => {
                results.push(payload);
                emit_scan_result(
                    &app,
                    &directory,
                    total_discovered,
                    processed,
                    results.last(),
                    None,
                    model_version,
                    model_name.as_deref(),
                    do_skip_annotated,
                );
                emit_control_progress(
                    &app,
                    &control,
                    "completed",
                    current,
                    total,
                    &path,
                    "图片识别完成。",
                );
            }
            Err(error) => {
                errors.push(scan_error_from_worker(&path, error));
                emit_scan_result(
                    &app,
                    &directory,
                    total_discovered,
                    processed,
                    None,
                    errors.last(),
                    model_version,
                    model_name.as_deref(),
                    do_skip_annotated,
                );
                emit_control_progress(
                    &app,
                    &control,
                    "error",
                    current,
                    total,
                    &path,
                    "图片识别失败，已继续处理。",
                );
            }
        }
    }
    let cancelled = control.is_cancelled() && processed < total;
    emit_control_progress(
        &app,
        &control,
        if cancelled { "cancelled" } else { "complete" },
        processed,
        total,
        display_path(&directory),
        if cancelled {
            "已取消扫描，已完成的识别结果仍会保留。"
        } else {
            "扫描与识别完成。"
        },
    );
    Ok(ScanLibraryResult {
        directory: display_path(&directory),
        total_discovered,
        processed,
        cancelled,
        results,
        errors,
        model_version,
        model_name,
        skip_annotated: Some(do_skip_annotated),
        completed_at: None,
    })
}

#[tauri::command]
pub fn register_library_selection(
    app: AppHandle,
    path: String,
    request_id: Option<String>,
) -> IpcEnvelope<LibrarySelection> {
    let request_id = normalize_request_id(request_id);
    let candidate = Path::new(&path);
    if !candidate.is_absolute() || !candidate.is_dir() {
        return failure(
            "library.select",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_LIBRARY_DIRECTORY",
                "请选择一个存在的本地文件夹。",
                Some("path must be an absolute directory".to_string()),
                false,
            ),
        );
    }
    let directory = match canonicalize_for_user(candidate) {
        Ok(directory) if directory.is_dir() => directory,
        Ok(_) => {
            return failure(
                "library.select",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_LIBRARY_DIRECTORY",
                    "选择的路径不是文件夹。",
                    None,
                    false,
                ),
            )
        }
        Err(error) => {
            return failure(
                "library.select",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_LIBRARY_DIRECTORY",
                    "无法访问选择的图片文件夹。",
                    Some(error.to_string()),
                    false,
                ),
            )
        }
    };
    if let Err(detail) = allow_asset_directory(&app, &directory) {
        return failure(
            "library.select",
            request_id.clone(),
            core_error(
                &request_id,
                "ASSET_SCOPE_FAILED",
                "无法授权预览所选图片文件夹。",
                Some(detail),
                true,
            ),
        );
    }
    let display_name = display_name_for_path(&directory);
    success(
        "library.select",
        request_id,
        LibrarySelection {
            path: display_path(&directory),
            display_name,
        },
    )
}

#[tauri::command]
pub fn register_preview_file(
    app: AppHandle,
    path: String,
    request_id: Option<String>,
) -> IpcEnvelope<LibrarySelection> {
    let request_id = normalize_request_id(request_id);
    let candidate = Path::new(&path);
    if !candidate.is_absolute() {
        return failure(
            "library.preview.file",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_PREVIEW_FILE",
                "预览文件路径必须是绝对路径。",
                None,
                false,
            ),
        );
    }
    let file = match canonicalize_for_user(candidate) {
        Ok(file) if file.is_file() => file,
        Ok(_) => {
            return failure(
                "library.preview.file",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_PREVIEW_FILE",
                    "请选择一个普通图片文件。",
                    None,
                    false,
                ),
            )
        }
        Err(error) => {
            return failure(
                "library.preview.file",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_PREVIEW_FILE",
                    "无法访问选择的图片文件。",
                    Some(error.to_string()),
                    false,
                ),
            )
        }
    };
    if !is_supported_preview_image(&file) {
        return failure(
            "library.preview.file",
            request_id.clone(),
            core_error(
                &request_id,
                "UNSUPPORTED_PREVIEW_FORMAT",
                "当前仅支持 PNG、JPG、JPEG、WEBP 和 GIF 图片预览。",
                None,
                false,
            ),
        );
    }
    if let Err(detail) = allow_asset_file(&app, &file) {
        return failure(
            "library.preview.file",
            request_id.clone(),
            core_error(
                &request_id,
                "ASSET_SCOPE_FAILED",
                "无法授权预览所选图片文件。",
                Some(detail),
                true,
            ),
        );
    }
    success(
        "library.preview.file",
        request_id,
        LibrarySelection {
            path: display_path(&file),
            display_name: display_name_for_path(&file),
        },
    )
}

#[tauri::command]
pub async fn scan_library_preview(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    limit: Option<u32>,
    model_version: Option<i64>,
    skip_annotated: Option<bool>,
    request_id: Option<String>,
) -> Result<IpcEnvelope<ScanLibraryResult>, String> {
    let request_id = normalize_request_id(request_id);
    let candidate = Path::new(&path);
    if !candidate.is_absolute() || !candidate.is_dir() {
        return Ok(failure(
            "library.scan.preview",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_LIBRARY_DIRECTORY",
                "请选择一个存在的本地文件夹。",
                Some("path must be an absolute directory".to_string()),
                false,
            ),
        ));
    }
    let directory = match canonicalize_for_user(candidate) {
        Ok(directory) if directory.is_dir() => directory,
        Ok(_) => {
            return Ok(failure(
                "library.scan.preview",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_LIBRARY_DIRECTORY",
                    "选择的路径不是文件夹。",
                    None,
                    false,
                ),
            ))
        }
        Err(error) => {
            return Ok(failure(
                "library.scan.preview",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_LIBRARY_DIRECTORY",
                    "无法访问选择的图片文件夹。",
                    Some(error.to_string()),
                    false,
                ),
            ))
        }
    };
    if let Err(detail) = allow_asset_directory(&app, &directory) {
        return Ok(failure(
            "library.scan.preview",
            request_id.clone(),
            core_error(
                &request_id,
                "ASSET_SCOPE_FAILED",
                "无法授权预览所选图片文件夹。",
                Some(detail),
                true,
            ),
        ));
    }
    let limit = clamp_scan_limit(limit);
    if !state.scan_control.try_start() {
        return Ok(failure(
            "library.scan.preview",
            request_id.clone(),
            core_error(
                &request_id,
                "SCAN_ALREADY_RUNNING",
                "已有扫描任务正在运行，请等待当前任务完成。",
                None,
                true,
            ),
        ));
    }
    let control = Arc::clone(&state.scan_control);
    let worker = Arc::clone(&state.worker);
    let database = Arc::clone(&state.database);
    let scan_app = app.clone();
    let scan = tauri::async_runtime::spawn_blocking(move || {
        run_library_scan(
            scan_app,
            worker,
            database,
            control.clone(),
            directory,
            limit,
            model_version,
            skip_annotated,
        )
    })
    .await;
    state.scan_control.finish();
    match scan {
        Ok(Ok(mut payload)) => {
            let completed_at = result_store::now_rfc3339();
            payload.completed_at = Some(completed_at.clone());
            if let Err(error) = result_store::save_versioned(
                Path::new(&state.data_dir),
                ResultKind::Recognition,
                &payload.directory,
                &completed_at,
                &payload,
            ) {
                return Ok(failure(
                    "library.scan.preview",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "SCAN_RESULT_SAVE_FAILED",
                        "扫描已完成，但结果文件保存失败。",
                        Some(error.to_string()),
                        true,
                    ),
                ));
            }
            Ok(success("library.scan.preview", request_id, payload))
        }
        Ok(Err(error)) => Ok(failure(
            "library.scan.preview",
            request_id.clone(),
            core_error(
                &request_id,
                "SCAN_FAILED",
                "扫描目录失败，未应用任何文件变更。",
                Some(error.to_string()),
                true,
            ),
        )),
        Err(error) => Ok(failure(
            "library.scan.preview",
            request_id.clone(),
            core_error(
                &request_id,
                "SCAN_TASK_FAILED",
                "扫描任务异常中止，未应用任何文件变更。",
                Some(error.to_string()),
                true,
            ),
        )),
    }
}

#[tauri::command]
pub async fn get_latest_library_scan_result(
    app: AppHandle,
    path: String,
    request_id: Option<String>,
) -> IpcEnvelope<Option<ScanLibraryResult>> {
    let request_id = normalize_request_id(request_id);
    let data_dir = PathBuf::from(&app.state::<AppState>().data_dir);
    let loaded = tauri::async_runtime::spawn_blocking(move || {
        result_store::load_latest(&data_dir, ResultKind::Recognition, &path)
    })
    .await
    .unwrap_or_else(|error| Err(std::io::Error::other(error.to_string())));
    match loaded {
        Ok(payload) => success("recognition.get_results", request_id, payload),
        Err(error) => failure(
            "recognition.get_results",
            request_id.clone(),
            core_error(
                &request_id,
                "SCAN_RESULT_READ_FAILED",
                "无法读取该图库的识别结果文件。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub fn delete_latest_library_scan_result(
    state: State<'_, AppState>,
    path: String,
    request_id: Option<String>,
) -> IpcEnvelope<bool> {
    let request_id = normalize_request_id(request_id);
    match result_store::delete_latest(Path::new(&state.data_dir), ResultKind::Recognition, &path) {
        Ok(deleted) => success("recognition.get_results", request_id, deleted),
        Err(error) => failure(
            "recognition.get_results",
            request_id.clone(),
            core_error(
                &request_id,
                "SCAN_RESULT_DELETE_FAILED",
                "无法删除该图库的最新识别结果。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub fn cancel_library_scan(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<ScanCancellation> {
    let request_id = normalize_request_id(request_id);
    let running = state.scan_control.request_cancel();
    if running {
        let snapshot = state.scan_control.snapshot();
        emit_scan_progress(
            &app,
            "cancelling",
            snapshot.current,
            snapshot.total,
            snapshot.path,
            "已请求取消扫描，当前图片处理完成后停止。",
        );
    }
    success(
        "library.scan.cancel",
        request_id,
        ScanCancellation {
            requested: true,
            running,
            message: if running {
                "已请求取消扫描，当前图片处理完成后停止。".to_string()
            } else {
                "当前没有正在运行的扫描任务。".to_string()
            },
        },
    )
}

#[tauri::command]
pub fn pause_library_scan(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<ScanControlResponse> {
    let request_id = normalize_request_id(request_id);
    let response = state.scan_control.request_pause();
    if response.running {
        let snapshot = state.scan_control.snapshot();
        emit_scan_progress(
            &app,
            &response.status,
            snapshot.current,
            snapshot.total,
            snapshot.path,
            response.message.clone(),
        );
    }
    success("library.scan.pause", request_id, response)
}

#[tauri::command]
pub fn resume_library_scan(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<ScanControlResponse> {
    let request_id = normalize_request_id(request_id);
    let response = state.scan_control.request_resume();
    if response.running {
        let snapshot = state.scan_control.snapshot();
        emit_scan_progress(
            &app,
            &response.status,
            snapshot.current,
            snapshot.total,
            snapshot.path,
            response.message.clone(),
        );
    }
    success("library.scan.resume", request_id, response)
}

#[tauri::command]
pub async fn move_library_files(
    app: AppHandle,
    state: State<'_, AppState>,
    sources: Vec<String>,
    destination_directory: String,
    subfolder_by_category: Option<String>,
    request_id: Option<String>,
) -> Result<IpcEnvelope<BatchMoveResult>, String> {
    let request_id = normalize_request_id(request_id);

    let dest_base = Path::new(&destination_directory);
    if !dest_base.is_absolute() {
        return Ok(failure(
            "library.files.move",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_DESTINATION",
                "目标文件夹必须为绝对路径。",
                None,
                false,
            ),
        ));
    }

    let target_dir = match &subfolder_by_category {
        Some(sub) if !sub.trim().is_empty() => {
            let clean_sub: String = sub
                .chars()
                .filter(|c| !['\\', '/', ':', '*', '?', '"', '<', '>', '|'].contains(c))
                .collect();
            let clean_sub = clean_sub.trim();
            if clean_sub.is_empty() {
                dest_base.to_path_buf()
            } else {
                dest_base.join(clean_sub)
            }
        }
        _ => dest_base.to_path_buf(),
    };

    if let Err(err) = fs::create_dir_all(&target_dir) {
        return Ok(failure(
            "library.files.move",
            request_id.clone(),
            core_error(
                &request_id,
                "DESTINATION_CREATE_FAILED",
                "无法创建目标文件夹。",
                Some(err.to_string()),
                false,
            ),
        ));
    }

    let _ = allow_asset_directory(&app, &target_dir);

    let mut moved = Vec::new();
    let mut errors = Vec::new();

    let mut database_guard = state.database.lock().ok();

    for src_str in sources {
        let src_path = Path::new(&src_str);
        if !src_path.is_file() {
            errors.push(MoveFileErrorItem {
                source: src_str.clone(),
                error: "源文件不存在或不是普通文件。".to_string(),
            });
            continue;
        }

        let file_name = match src_path.file_name() {
            Some(name) => name,
            None => {
                errors.push(MoveFileErrorItem {
                    source: src_str.clone(),
                    error: "无法获取源文件名。".to_string(),
                });
                continue;
            }
        };

        let mut final_dest = target_dir.join(file_name);
        if final_dest.exists() && final_dest != src_path {
            let stem = src_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("file");
            let ext = src_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let mut counter = 1;
            loop {
                let candidate_name = if ext.is_empty() {
                    format!("{}_{}", stem, counter)
                } else {
                    format!("{}_{}.{}", stem, counter, ext)
                };
                let candidate_path = target_dir.join(&candidate_name);
                if !candidate_path.exists() {
                    final_dest = candidate_path;
                    break;
                }
                counter += 1;
            }
        }

        if final_dest == src_path {
            moved.push(MovedFileItem {
                source: src_str.clone(),
                destination: src_str.clone(),
            });
            continue;
        }

        let move_result = fs::rename(&src_path, &final_dest)
            .or_else(|_| fs::copy(&src_path, &final_dest).and_then(|_| fs::remove_file(&src_path)));

        match move_result {
            Ok(_) => {
                let dest_str = final_dest.to_string_lossy().into_owned();
                if let Some(ref mut db) = database_guard {
                    let _ = db.update_image_path(&src_str, &dest_str);
                }
                moved.push(MovedFileItem {
                    source: src_str,
                    destination: dest_str,
                });
            }
            Err(err) => {
                errors.push(MoveFileErrorItem {
                    source: src_str,
                    error: format!("移动文件失败: {}", err),
                });
            }
        }
    }

    let success_count = moved.len();
    let failed_count = errors.len();

    Ok(success(
        "library.files.move",
        request_id,
        BatchMoveResult {
            success_count,
            failed_count,
            moved,
            errors,
        },
    ))
}
