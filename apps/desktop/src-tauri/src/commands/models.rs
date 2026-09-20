//! Character recognizer inventory: list, install, remove and select.
//!
//! The catalog lives in `resources/model-catalog.json` so it can be updated
//! without touching code; installation itself is performed by the Worker
//! (it already owns the Hugging Face download path).

use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    pub path: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub repo_id: String,
    pub size_mb: f64,
    pub license: String,
    pub note: String,
    pub files: Vec<CatalogFile>,
    pub metadata: Value,
    #[serde(default)]
    pub generate: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstalledModel {
    pub id: String,
    pub kind: String,
    pub adapter: String,
    pub version: String,
    pub status: String,
    pub active: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecognizerInventory {
    pub models: Vec<InstalledModel>,
    pub active: String,
    pub catalog: Vec<CatalogEntry>,
}

fn catalog_path(app: &AppHandle) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    for root in crate::worker_runtime::discovery::resource_roots(&resource_dir) {
        let candidate = root.join("resources").join("model-catalog.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn load_catalog(app: &AppHandle) -> Vec<CatalogEntry> {
    let Some(path) = catalog_path(app) else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(entries) = serde_json::from_str::<Vec<CatalogEntry>>(&text) else {
        return Vec::new();
    };
    entries
}

fn active_model(app: &AppHandle) -> String {
    let state = app.state::<AppState>();
    state
        .database
        .lock()
        .ok()
        .and_then(|database| crate::app_settings::load_app_settings(&database).ok())
        .map(|settings| settings.recognition_recognizer_model)
        .unwrap_or_default()
}

fn collect_inventory(app: &AppHandle) -> Result<RecognizerInventory, String> {
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let models = {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        let payload = manager
            .request("model.list", json!({}))
            .map_err(|error| error.to_string())?;
        payload
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let active = active_model(app);
    let installed = models
        .into_iter()
        .filter(|item| item.get("kind").and_then(Value::as_str) == Some("character_recognizer"))
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_str)?.to_string();
            Some(InstalledModel {
                active: id == active,
                kind: item
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                adapter: item
                    .get("adapter")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                version: item
                    .get("version")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                status: item
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                error: item
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                id,
            })
        })
        .collect();
    Ok(RecognizerInventory {
        models: installed,
        active,
        catalog: load_catalog(app),
    })
}

#[tauri::command]
pub fn get_recognizer_inventory(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<RecognizerInventory> {
    let request_id = normalize_request_id(request_id);
    match collect_inventory(&app) {
        Ok(inventory) => success("model.inventory", request_id, inventory),
        Err(message) => failure(
            "model.inventory",
            request_id.clone(),
            core_error(&request_id, "MODEL_LIST_FAILED", &message, None, true),
        ),
    }
}

/// Persist the recognizer every subsequent scan should use.
#[tauri::command]
pub fn set_active_recognizer(
    app: AppHandle,
    model_id: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let trimmed = model_id.trim().to_string();
    let state = app.state::<AppState>();
    let saved = match state.database.lock() {
        Ok(mut database) => {
            let mut settings =
                crate::app_settings::load_app_settings(&database).unwrap_or_default();
            settings.recognition_recognizer_model = trimmed.clone();
            crate::app_settings::save_app_settings(&mut database, &settings)
        }
        Err(_) => Err("本地数据库状态锁暂时不可用。".to_string()),
    };
    match saved {
        Ok(()) => success("model.activate", request_id, json!({ "active": trimmed })),
        Err(message) => failure(
            "model.activate",
            request_id.clone(),
            core_error(&request_id, "SETTINGS_SAVE_FAILED", &message, None, true),
        ),
    }
}

#[tauri::command]
pub async fn install_recognizer_model(
    app: AppHandle,
    catalog_id: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let Some(entry) = load_catalog(&app)
        .into_iter()
        .find(|item| item.id == catalog_id)
    else {
        return failure(
            "model.install",
            request_id.clone(),
            core_error(
                &request_id,
                "MODEL_CATALOG_MISSING",
                &format!("模型清单里没有 {catalog_id}"),
                None,
                false,
            ),
        );
    };
    let payload = json!({
        "id": entry.id,
        "kind": entry.kind,
        "repo_id": entry.repo_id,
        "files": entry.files,
        "metadata": entry.metadata,
        "generate": entry.generate,
    });
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        manager
            .request("model.install", payload)
            .map_err(|error| error.to_string())
    })
    .await;
    match result {
        Ok(Ok(value)) => success("model.install", request_id, value),
        Ok(Err(message)) => failure(
            "model.install",
            request_id.clone(),
            core_error(&request_id, "MODEL_INSTALL_FAILED", &message, None, true),
        ),
        Err(error) => failure(
            "model.install",
            request_id.clone(),
            core_error(
                &request_id,
                "MODEL_INSTALL_TASK_FAILED",
                &format!("安装任务异常中止：{error}"),
                None,
                true,
            ),
        ),
    }
}

#[tauri::command]
pub async fn delete_recognizer_model(
    app: AppHandle,
    model_id: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    if active_model(&app) == model_id.trim() {
        return failure(
            "model.delete",
            request_id.clone(),
            core_error(
                &request_id,
                "MODEL_IN_USE",
                "该模型正在使用中，请先切换到其他识别模型再删除。",
                None,
                false,
            ),
        );
    }
    let payload = json!({ "id": model_id.trim() });
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        manager
            .request("model.delete", payload)
            .map_err(|error| error.to_string())
    })
    .await;
    match result {
        Ok(Ok(value)) => success("model.delete", request_id, value),
        Ok(Err(message)) => failure(
            "model.delete",
            request_id.clone(),
            core_error(&request_id, "MODEL_DELETE_FAILED", &message, None, true),
        ),
        Err(error) => failure(
            "model.delete",
            request_id.clone(),
            core_error(
                &request_id,
                "MODEL_DELETE_TASK_FAILED",
                &format!("删除任务异常中止：{error}"),
                None,
                true,
            ),
        ),
    }
}
