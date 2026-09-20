//! Reference-image library: build it from corrected samples, reuse it in scans.
//!
//! The library itself lives in `data\reference-library.json`; the Worker reads
//! it by path and caches it, so the (large) vectors never travel with every
//! recognition request.

use crate::app_settings::AppSettings;
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

pub const REFERENCE_LIBRARY_FILE: &str = "reference-library.json";
pub const BACKENDS: [&str; 2] = ["ccip", "embedding"];

pub fn library_path(data_dir: &str) -> PathBuf {
    PathBuf::from(data_dir).join(REFERENCE_LIBRARY_FILE)
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceLibraryStatus {
    pub enabled: bool,
    pub backend: String,
    pub path: String,
    pub exists: bool,
    pub identities: usize,
    pub references: usize,
    pub built_at: Option<String>,
    pub characters: Vec<ReferenceCharacterSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceCharacterSummary {
    pub identity_id: String,
    pub display_name: String,
    pub references: usize,
}

fn load_settings(app: &AppHandle) -> Result<AppSettings, String> {
    let state = app.state::<AppState>();
    let database = state
        .database
        .lock()
        .map_err(|_| "本地数据库状态锁暂时不可用。".to_string())?;
    crate::app_settings::load_app_settings(&database)
}

pub fn status_for(app: &AppHandle) -> ReferenceLibraryStatus {
    let state = app.state::<AppState>();
    let settings = load_settings(app).unwrap_or_default();
    let path = library_path(&state.data_dir);
    let mut status = ReferenceLibraryStatus {
        enabled: settings.reference_matching_enabled,
        backend: settings.reference_backend.clone(),
        path: path.to_string_lossy().into_owned(),
        exists: false,
        identities: 0,
        references: 0,
        built_at: None,
        characters: Vec::new(),
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return status;
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return status;
    };
    status.exists = true;
    if let Some(backend) = value.get("backend").and_then(Value::as_str) {
        status.backend = backend.to_string();
    }
    status.built_at = value
        .get("built_at")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(entries) = value.get("entries").and_then(Value::as_array) {
        status.identities = entries.len();
        for entry in entries {
            let references = entry
                .get("vectors")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            status.references += references;
            status.characters.push(ReferenceCharacterSummary {
                identity_id: entry
                    .get("identity_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                display_name: entry
                    .get("display_name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                references,
            });
        }
    }
    status
}

#[tauri::command]
pub fn get_reference_library_status(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<ReferenceLibraryStatus> {
    let request_id = normalize_request_id(request_id);
    success("reference.status", request_id, status_for(&app))
}

/// Build (or rebuild) the library from every character with annotations.
#[tauri::command]
pub async fn build_reference_library(
    app: AppHandle,
    backend: Option<String>,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "reference.build",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let backend = backend
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| BACKENDS.contains(&value.as_str()))
        .unwrap_or_else(|| settings.reference_backend.clone());
    if !BACKENDS.contains(&backend.as_str()) {
        return failure(
            "reference.build",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_BACKEND",
                &format!("参考库后端只能是 {}", BACKENDS.join(" / ")),
                None,
                false,
            ),
        );
    }

    let samples = {
        let state = app.state::<AppState>();
        let database = match state.database.lock() {
            Ok(database) => database,
            Err(_) => {
                return failure(
                    "reference.build",
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
        let summaries = match database.list_identity_sample_summaries() {
            Ok(rows) => rows,
            Err(error) => {
                return failure(
                    "reference.build",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "DATASET_QUERY_FAILED",
                        "无法读取已标注角色。",
                        Some(error.to_string()),
                        true,
                    ),
                )
            }
        };
        let mut samples: Vec<Value> = Vec::new();
        for summary in summaries {
            match database.list_identity_samples(&summary.identity_id) {
                Ok(rows) => {
                    for row in rows {
                        samples.push(json!({
                            "identity_id": summary.identity_id,
                            "display_name": summary.display_name,
                            "path": row.path,
                            "bbox": [row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height],
                        }));
                    }
                }
                Err(error) => {
                    return failure(
                        "reference.build",
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
            }
        }
        samples
    };
    if samples.is_empty() {
        return failure(
            "reference.build",
            request_id.clone(),
            core_error(
                &request_id,
                "REFERENCE_NO_SAMPLES",
                "还没有人工矫正过的样本，先在「角色识别」里矫正一批再建参考库。",
                None,
                false,
            ),
        );
    }

    let payload = json!({
        "backend": backend,
        "samples": samples,
        "max_per_identity": 32,
    });
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        manager
            .request("reference.build", payload)
            .map_err(|error| error.to_string())
    })
    .await;
    let payload = match result {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            return failure(
                "reference.build",
                request_id.clone(),
                core_error(&request_id, "REFERENCE_BUILD_FAILED", &message, None, true),
            )
        }
        Err(error) => {
            return failure(
                "reference.build",
                request_id.clone(),
                core_error(
                    &request_id,
                    "REFERENCE_BUILD_TASK_FAILED",
                    &format!("建库任务异常中止：{error}"),
                    None,
                    true,
                ),
            )
        }
    };
    let library = match payload.get("library") {
        Some(value) => value.clone(),
        None => {
            return failure(
                "reference.build",
                request_id.clone(),
                core_error(
                    &request_id,
                    "REFERENCE_BUILD_EMPTY",
                    "没有任何样本生成出参考向量。",
                    None,
                    false,
                ),
            )
        }
    };
    let state = app.state::<AppState>();
    let path = library_path(&state.data_dir);
    if let Err(error) = std::fs::write(
        &path,
        serde_json::to_vec_pretty(&library).unwrap_or_default(),
    ) {
        return failure(
            "reference.build",
            request_id.clone(),
            core_error(
                &request_id,
                "REFERENCE_WRITE_FAILED",
                &format!("无法写入 {}：{error}", path.display()),
                None,
                false,
            ),
        );
    }
    if let Ok(mut database) = state.database.lock() {
        let mut settings = crate::app_settings::load_app_settings(&database).unwrap_or_default();
        settings.reference_backend = backend.clone();
        settings.reference_matching_enabled = true;
        if let Err(error) = crate::app_settings::save_app_settings(&mut database, &settings) {
            return failure(
                "reference.build",
                request_id.clone(),
                core_error(
                    &request_id,
                    "SETTINGS_SAVE_FAILED",
                    &format!("参考库已生成，但保存设置失败：{error}"),
                    None,
                    false,
                ),
            );
        }
    }
    success(
        "reference.build",
        request_id,
        json!({
            "backend": backend,
            "identities": payload.get("identities").and_then(Value::as_u64).unwrap_or(0),
            "references": payload.get("references").and_then(Value::as_u64).unwrap_or(0),
            "skipped": payload.get("skipped").and_then(Value::as_u64).unwrap_or(0),
            "path": path.to_string_lossy(),
        }),
    )
}

/// Remove the library (the setting stays, so a rebuild re-enables it).
#[tauri::command]
pub fn clear_reference_library(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let state = app.state::<AppState>();
    let path = library_path(&state.data_dir);
    let removed = if path.is_file() {
        match std::fs::remove_file(&path) {
            Ok(()) => true,
            Err(error) => {
                return failure(
                    "reference.clear",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "REFERENCE_DELETE_FAILED",
                        &format!("无法删除参考库：{error}"),
                        None,
                        false,
                    ),
                )
            }
        }
    } else {
        false
    };
    success("reference.clear", request_id, json!({ "removed": removed }))
}

/// Path handed to the Worker during scans, when the library is enabled.
pub fn active_library_path(settings: &AppSettings, data_dir: &str) -> Option<PathBuf> {
    if !settings.reference_matching_enabled {
        return None;
    }
    let path = library_path(data_dir);
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

pub fn library_exists(data_dir: &str) -> bool {
    Path::new(&library_path(data_dir)).is_file()
}
