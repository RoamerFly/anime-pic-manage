//! Model inventory: everything the app ships, can download, or has cached.
//!
//! The catalog lives in `resources/model-catalog.json` so it can be updated
//! without touching code. Two delivery paths exist:
//!
//! * `models_dir` – recognizers and detectors downloaded into `models\<kind>\<id>\`
//!   by the Worker's `model.install` handler.
//! * `hf_cache` – optional models the pipeline pulls from Hugging Face on first
//!   use (the WD14 tagging model, the CCIP reference model). They stay in the
//!   shared Hugging Face cache, so the Worker owns their status and cleanup.

use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

/// Worker directory per model kind; kinds outside this map stay code-driven.
const KIND_DIRECTORIES: &[(&str, &str)] = &[
    ("character_recognizer", "recognizer"),
    ("head_detector", "detector"),
];

fn default_delivery() -> String {
    "models_dir".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    /// Optional per-file repository; falls back to the entry's `repo_id`.
    #[serde(default)]
    pub repo_id: Option<String>,
    pub path: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub kind: String,
    #[serde(default = "default_delivery")]
    pub delivery: String,
    #[serde(default)]
    pub repo_id: String,
    #[serde(default)]
    pub size_mb: f64,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub files: Vec<CatalogFile>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub generate: Option<Value>,
}

impl CatalogEntry {
    fn uses_cache(&self) -> bool {
        self.delivery == "hf_cache"
    }

    /// Every repository this entry needs, including per-file overrides.
    fn repo_ids(&self) -> Vec<String> {
        let mut repos: Vec<String> = Vec::new();
        if !self.repo_id.trim().is_empty() {
            repos.push(self.repo_id.trim().to_string());
        }
        for file in &self.files {
            if let Some(repo) = file.repo_id.as_deref() {
                let trimmed = repo.trim();
                if !trimmed.is_empty() && !repos.iter().any(|item| item == trimmed) {
                    repos.push(trimmed.to_string());
                }
            }
        }
        repos
    }
}

/// One row of 设置 → 模型配置.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInventoryEntry {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub delivery: String,
    pub repo_id: String,
    pub size_mb: f64,
    pub license: String,
    pub note: String,
    pub status: String,
    pub installed_bytes: u64,
    pub path: Option<String>,
    pub bundled: bool,
    pub downloadable: bool,
    pub active: bool,
    pub adapter: Option<String>,
    pub version: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInventory {
    pub models_dir: String,
    pub cache_dir: String,
    pub active: String,
    pub entries: Vec<ModelInventoryEntry>,
    /// Set when part of the inventory could not be collected (Worker offline).
    pub warning: Option<String>,
}

/// Status of one cache-backed catalog entry as reported by the Worker.
#[derive(Debug, Clone, Default)]
struct CacheStatus {
    status: String,
    total_bytes: u64,
    path: Option<String>,
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

/// Ask the Worker without letting a stopped Worker break the whole panel.
fn worker_request(app: &AppHandle, message_type: &str, payload: Value) -> Result<Value, String> {
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let mut manager = worker
        .lock()
        .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
    manager
        .request(message_type, payload)
        .map_err(|error| error.to_string())
}

fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0_u64;
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() {
            total = total.saturating_add(directory_size(&entry.path()));
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    total
}

fn kind_directory(kind: &str) -> Option<&'static str> {
    KIND_DIRECTORIES
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, directory)| *directory)
}

/// Hugging Face stores each repository in `<cache>\models--<owner>--<name>`.
fn cache_repo_dir(cache_dir: &str, repo_id: &str) -> Option<PathBuf> {
    if cache_dir.trim().is_empty() || repo_id.trim().is_empty() {
        return None;
    }
    let folder = format!("models--{}", repo_id.trim().replace('/', "--"));
    let candidate = Path::new(cache_dir).join(folder);
    candidate.is_dir().then_some(candidate)
}

fn parse_cache_statuses(payload: &Value) -> HashMap<String, CacheStatus> {
    let mut statuses = HashMap::new();
    let Some(entries) = payload.get("entries").and_then(Value::as_array) else {
        return statuses;
    };
    for entry in entries {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        let path = entry
            .get("files")
            .and_then(Value::as_array)
            .and_then(|files| {
                files.iter().find_map(|file| {
                    let present = file.get("present").and_then(Value::as_bool)?;
                    if !present {
                        return None;
                    }
                    file.get("path").and_then(Value::as_str).map(str::to_string)
                })
            });
        statuses.insert(
            id.to_string(),
            CacheStatus {
                status: entry
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("missing")
                    .to_string(),
                total_bytes: entry
                    .get("total_bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                path,
            },
        );
    }
    statuses
}

fn installed_entry(
    item: &Value,
    catalog: &[CatalogEntry],
    models_dir: &str,
    active: &str,
) -> Option<ModelInventoryEntry> {
    let id = item.get("id").and_then(Value::as_str)?.to_string();
    let kind = item
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let entry = catalog.iter().find(|entry| entry.id == id);
    let directory = kind_directory(&kind)
        .map(|name| Path::new(models_dir).join(name).join(&id))
        .filter(|path| path.is_dir());
    let installed_bytes = directory.as_deref().map(directory_size).unwrap_or(0);
    let error = item
        .get("error")
        .filter(|error| !error.is_null())
        .and_then(|error| {
            error
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| error.as_str().map(str::to_string))
        });
    Some(ModelInventoryEntry {
        name: entry
            .map(|entry| entry.name.clone())
            .unwrap_or_else(|| id.clone()),
        delivery: entry
            .map(|entry| entry.delivery.clone())
            .unwrap_or_else(default_delivery),
        repo_id: entry.map(|entry| entry.repo_id.clone()).unwrap_or_default(),
        size_mb: entry.map(|entry| entry.size_mb).unwrap_or(0.0),
        license: entry.map(|entry| entry.license.clone()).unwrap_or_default(),
        note: entry.map(|entry| entry.note.clone()).unwrap_or_default(),
        status: item
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("installed")
            .to_string(),
        installed_bytes,
        path: directory.map(|path| path.to_string_lossy().into_owned()),
        bundled: entry.is_none(),
        downloadable: entry.is_some(),
        active: kind == "character_recognizer" && id == active,
        adapter: item
            .get("adapter")
            .and_then(Value::as_str)
            .map(str::to_string),
        version: item
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_string),
        error,
        id,
        kind,
    })
}

fn catalog_only_entry(
    entry: &CatalogEntry,
    cache: Option<&CacheStatus>,
    models_dir: &str,
    cache_dir: &str,
    active: &str,
) -> ModelInventoryEntry {
    let (status, installed_bytes, path) = if entry.uses_cache() {
        match cache {
            Some(status) => (
                status.status.clone(),
                status.total_bytes,
                status.path.clone().or_else(|| {
                    cache_repo_dir(cache_dir, &entry.repo_id)
                        .map(|path| path.to_string_lossy().into_owned())
                }),
            ),
            None => ("missing".to_string(), 0, None),
        }
    } else {
        ("missing".to_string(), 0, None)
    };
    ModelInventoryEntry {
        id: entry.id.clone(),
        name: entry.name.clone(),
        kind: entry.kind.clone(),
        delivery: entry.delivery.clone(),
        repo_id: entry.repo_id.clone(),
        size_mb: entry.size_mb,
        license: entry.license.clone(),
        note: entry.note.clone(),
        status,
        installed_bytes,
        // A model that was never downloaded has no weights to point at, so the
        // panel stays honest and offers no "open folder" button for it.
        path: path.or_else(|| {
            kind_directory(&entry.kind)
                .map(|name| Path::new(models_dir).join(name).join(&entry.id))
                .filter(|path| path.is_dir())
                .map(|path| path.to_string_lossy().into_owned())
        }),
        bundled: false,
        downloadable: !entry.files.is_empty() || !entry.metadata.is_null(),
        active: entry.kind == "character_recognizer" && entry.id == active,
        adapter: None,
        version: entry
            .metadata
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_string),
        error: None,
    }
}

fn collect_inventory(app: &AppHandle) -> ModelInventory {
    let catalog = load_catalog(app);
    let active = active_model(app);
    let mut warning: Option<String> = None;

    let cache_entries: Vec<Value> = catalog
        .iter()
        .filter(|entry| entry.uses_cache())
        .map(|entry| {
            json!({
                "id": entry.id,
                "repo_id": entry.repo_id,
                "files": entry.files,
            })
        })
        .collect();

    // The Worker rejects an empty `entries` array, so skip the round trip when
    // the catalog happens to list no cache-backed models.
    let cache_payload = (!cache_entries.is_empty())
        .then(|| {
            worker_request(
                app,
                "model.cache.status",
                json!({ "entries": cache_entries }),
            )
        })
        .transpose();
    let (models_dir, cache_dir, cache_statuses) = match cache_payload {
        Ok(Some(payload)) => (
            payload
                .get("models_dir")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            payload
                .get("cache_dir")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            parse_cache_statuses(&payload),
        ),
        Ok(None) => (String::new(), String::new(), HashMap::new()),
        Err(message) => {
            if warning.is_none() {
                warning = Some(message);
            }
            (String::new(), String::new(), HashMap::new())
        }
    };
    let models_dir = if models_dir.is_empty() {
        crate::worker_runtime::discovery::local_models_root()
            .to_string_lossy()
            .into_owned()
    } else {
        models_dir
    };

    let installed_payload = worker_request(app, "model.list", json!({}));
    let mut entries: Vec<ModelInventoryEntry> = Vec::new();
    match installed_payload {
        Ok(payload) => {
            if let Some(items) = payload.get("models").and_then(Value::as_array) {
                for item in items {
                    if let Some(entry) = installed_entry(item, &catalog, &models_dir, &active) {
                        entries.push(entry);
                    }
                }
            }
        }
        Err(message) => warning = warning.or(Some(message)),
    }

    for entry in &catalog {
        if entries.iter().any(|item| item.id == entry.id) {
            continue;
        }
        entries.push(catalog_only_entry(
            entry,
            cache_statuses.get(&entry.id),
            &models_dir,
            &cache_dir,
            &active,
        ));
    }

    ModelInventory {
        models_dir,
        cache_dir,
        active,
        entries,
        warning,
    }
}

/// Everything 设置 → 模型配置 needs in a single round trip.
///
/// Async + `spawn_blocking` because the Worker serialises requests: while a
/// multi-hundred-megabyte download is running the inventory call must not
/// block the UI thread.
#[tauri::command]
pub async fn get_model_inventory(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<ModelInventory> {
    let request_id = normalize_request_id(request_id);
    let handle = app.clone();
    match tauri::async_runtime::spawn_blocking(move || collect_inventory(&handle)).await {
        Ok(inventory) => success("model.inventory", request_id, inventory),
        Err(error) => failure(
            "model.inventory",
            request_id.clone(),
            core_error(
                &request_id,
                "MODEL_LIST_FAILED",
                &format!("模型清单读取异常中止：{error}"),
                None,
                true,
            ),
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

fn catalog_entry(app: &AppHandle, catalog_id: &str) -> Result<CatalogEntry, String> {
    load_catalog(app)
        .into_iter()
        .find(|item| item.id == catalog_id.trim())
        .ok_or_else(|| format!("模型清单里没有 {catalog_id}"))
}

fn catalog_install_payload(entry: &CatalogEntry) -> Value {
    if entry.uses_cache() {
        json!({
            "id": entry.id,
            "repo_id": entry.repo_id,
            "files": entry.files,
        })
    } else {
        json!({
            "id": entry.id,
            "kind": entry.kind,
            "repo_id": entry.repo_id,
            "files": entry.files,
            "metadata": entry.metadata,
            "generate": entry.generate,
        })
    }
}

/// Download one catalog model into whichever location its `delivery` declares.
#[tauri::command]
pub async fn install_catalog_model(
    app: AppHandle,
    catalog_id: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let entry = match catalog_entry(&app, &catalog_id) {
        Ok(entry) => entry,
        Err(message) => {
            return failure(
                "model.install",
                request_id.clone(),
                core_error(&request_id, "MODEL_CATALOG_MISSING", &message, None, false),
            );
        }
    };
    let message_type = if entry.uses_cache() {
        "model.cache.prefetch"
    } else {
        "model.install"
    };
    let payload = catalog_install_payload(&entry);
    run_model_task(app, message_type, payload, request_id).await
}

/// Remove a downloaded catalog model; bundled models are never touched.
#[tauri::command]
pub async fn delete_catalog_model(
    app: AppHandle,
    catalog_id: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let entry = match catalog_entry(&app, &catalog_id) {
        Ok(entry) => entry,
        Err(message) => {
            return failure(
                "model.delete",
                request_id.clone(),
                core_error(&request_id, "MODEL_CATALOG_MISSING", &message, None, false),
            );
        }
    };
    if !entry.uses_cache() && active_model(&app) == entry.id {
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
    let (message_type, payload) = if entry.uses_cache() {
        (
            "model.cache.delete",
            json!({ "id": entry.id, "repo_ids": entry.repo_ids() }),
        )
    } else {
        ("model.delete", json!({ "id": entry.id }))
    };
    run_model_task(app, message_type, payload, request_id).await
}

async fn run_model_task(
    app: AppHandle,
    message_type: &'static str,
    payload: Value,
    request_id: String,
) -> IpcEnvelope<Value> {
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        manager
            .request(message_type, payload)
            .map_err(|error| error.to_string())
    })
    .await;
    // Echo back the message the Worker was asked for so the UI can tell a
    // recognizer install apart from a cache prefetch.
    let ipc_type = message_type;
    match result {
        Ok(Ok(value)) => success(ipc_type, request_id, value),
        Ok(Err(message)) => failure(
            ipc_type,
            request_id.clone(),
            core_error(&request_id, "MODEL_TASK_FAILED", &message, None, true),
        ),
        Err(error) => failure(
            ipc_type,
            request_id.clone(),
            core_error(
                &request_id,
                "MODEL_TASK_FAILED",
                &format!("模型任务异常中止：{error}"),
                None,
                true,
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(delivery: &str, files: Vec<CatalogFile>) -> CatalogEntry {
        serde_json::from_value(json!({
            "id": "sample",
            "name": "示例",
            "kind": if delivery == "hf_cache" { "tagger" } else { "character_recognizer" },
            "delivery": delivery,
            "repo_id": "owner/repo",
            "size_mb": 10,
            "license": "x",
            "note": "y",
            "files": files,
        }))
        .expect("catalog entry parses")
    }

    #[test]
    fn catalog_defaults_to_the_models_directory() {
        let entry: CatalogEntry = serde_json::from_value(json!({
            "id": "camie-initial",
            "name": "Camie",
            "kind": "character_recognizer",
            "repo_id": "deepghs/camie_tagger_onnx",
            "files": [{ "path": "initial/model.onnx", "name": "model.onnx" }],
        }))
        .expect("legacy entry still parses");

        assert_eq!(entry.delivery, "models_dir");
        assert!(!entry.uses_cache());
        assert_eq!(
            entry.repo_ids(),
            vec!["deepghs/camie_tagger_onnx".to_string()]
        );
    }

    #[test]
    fn cache_entries_collect_every_repository() {
        let entry = sample_entry(
            "hf_cache",
            vec![
                CatalogFile {
                    repo_id: None,
                    path: "a/model.onnx".to_string(),
                    name: "model.onnx".to_string(),
                },
                CatalogFile {
                    repo_id: Some("other/labels".to_string()),
                    path: "selected_tags.csv".to_string(),
                    name: "selected_tags.csv".to_string(),
                },
            ],
        );

        assert!(entry.uses_cache());
        assert_eq!(
            entry.repo_ids(),
            vec!["owner/repo".to_string(), "other/labels".to_string()]
        );
    }

    #[test]
    fn catalog_only_entries_report_missing_models() {
        let entry = sample_entry(
            "models_dir",
            vec![CatalogFile {
                repo_id: None,
                path: "initial/model.onnx".to_string(),
                name: "model.onnx".to_string(),
            }],
        );
        let inventory_entry = catalog_only_entry(&entry, None, "D:\\models", "D:\\cache", "sample");

        assert_eq!(inventory_entry.status, "missing");
        assert!(inventory_entry.downloadable);
        assert!(inventory_entry.active);
        assert_eq!(inventory_entry.installed_bytes, 0);
    }

    #[test]
    fn catalog_entries_hide_paths_until_the_weights_exist() {
        let entry = sample_entry(
            "models_dir",
            vec![CatalogFile {
                repo_id: None,
                path: "initial/model.onnx".to_string(),
                name: "model.onnx".to_string(),
            }],
        );
        let root = std::env::temp_dir().join(format!("anime-model-inv-{}", uuid::Uuid::new_v4()));
        let models_dir = root.to_string_lossy().into_owned();

        // Nothing downloaded yet: no path, so the panel offers no "open folder".
        let missing = catalog_only_entry(&entry, None, &models_dir, "D:\\cache", "");
        assert_eq!(missing.path, None);
        assert_eq!(missing.status, "missing");

        let directory = root.join("recognizer").join("sample");
        std::fs::create_dir_all(&directory).expect("create model directory");
        let present = catalog_only_entry(&entry, None, &models_dir, "D:\\cache", "");
        assert_eq!(
            present.path.as_deref(),
            Some(directory.to_string_lossy().as_ref())
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_backed_entries_use_the_reported_status() {
        let entry = sample_entry(
            "hf_cache",
            vec![CatalogFile {
                repo_id: None,
                path: "a/model.onnx".to_string(),
                name: "model.onnx".to_string(),
            }],
        );
        let status = CacheStatus {
            status: "installed".to_string(),
            total_bytes: 1234,
            path: Some("D:\\cache\\blobs\\abc".to_string()),
        };
        let inventory_entry =
            catalog_only_entry(&entry, Some(&status), "D:\\models", "D:\\cache", "");

        assert_eq!(inventory_entry.status, "installed");
        assert_eq!(inventory_entry.installed_bytes, 1234);
        assert_eq!(
            inventory_entry.path.as_deref(),
            Some("D:\\cache\\blobs\\abc")
        );
    }

    #[test]
    fn installed_entries_mark_bundled_models() {
        let catalog = vec![sample_entry(
            "models_dir",
            vec![CatalogFile {
                repo_id: None,
                path: "initial/model.onnx".to_string(),
                name: "model.onnx".to_string(),
            }],
        )];
        let item = json!({
            "id": "sample",
            "kind": "character_recognizer",
            "adapter": "animetimm_onnx",
            "version": "1",
            "status": "installed",
        });
        let entry = installed_entry(&item, &catalog, "D:\\models", "sample").expect("entry");
        assert!(!entry.bundled);
        assert!(entry.active);
        assert_eq!(entry.name, "示例");

        let bundled = installed_entry(&item, &[], "D:\\models", "sample").expect("entry");
        assert!(bundled.bundled);
        assert!(!bundled.downloadable);
    }

    #[test]
    fn cache_statuses_are_parsed_from_the_worker_payload() {
        let payload = json!({
            "models_dir": "D:\\models",
            "cache_dir": "D:\\cache",
            "entries": [
                {
                    "id": "wd14-swinv2-v3",
                    "status": "partial",
                    "total_bytes": 12,
                    "files": [
                        { "name": "model.onnx", "present": false, "size": 0, "path": null },
                        { "name": "selected_tags.csv", "present": true, "size": 12, "path": "D:\\cache\\blobs\\x" }
                    ]
                }
            ]
        });
        let statuses = parse_cache_statuses(&payload);
        let status = statuses.get("wd14-swinv2-v3").expect("status");
        assert_eq!(status.status, "partial");
        assert_eq!(status.total_bytes, 12);
        assert_eq!(status.path.as_deref(), Some("D:\\cache\\blobs\\x"));
    }
}
