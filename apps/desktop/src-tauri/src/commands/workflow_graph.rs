//! Workflow graph view: expose the raw template JSON and persist per-node edits.
//!
//! Node overrides are stored per template in
//! `data\comfy-template-overrides.json` so a graph edit survives restarts and
//! never modifies the template file itself.

use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

pub const TEMPLATE_FOLDER: &str = "comfy-workflows";
const OVERRIDES_FILE: &str = "comfy-template-overrides.json";

fn template_roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        roots.push(resource_dir.join("resources").join(TEMPLATE_FOLDER));
        roots.push(resource_dir.join(TEMPLATE_FOLDER));
    }
    let project_dir = crate::worker_runtime::discovery::local_project_dir();
    roots.push(project_dir.join("resources").join(TEMPLATE_FOLDER));
    if let Ok(current) = std::env::current_dir() {
        roots.push(current.join("resources").join(TEMPLATE_FOLDER));
        roots.push(current.join(TEMPLATE_FOLDER));
    }
    roots
}

fn read_template(app: &AppHandle, name: &str) -> Result<Value, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() || trimmed.contains(['/', '\\', ':']) {
        return Err("工作流模板名不合法。".to_string());
    }
    for root in template_roots(app) {
        let candidate = root.join(format!("{trimmed}.json"));
        if !candidate.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&candidate)
            .map_err(|error| format!("{}: {error}", candidate.display()))?;
        return serde_json::from_str(&text)
            .map_err(|error| format!("模板 {} 解析失败：{error}", candidate.display()));
    }
    Err(format!("未找到工作流模板 {trimmed}.json"))
}

fn overrides_path(app: &AppHandle) -> PathBuf {
    PathBuf::from(&app.state::<AppState>().data_dir).join(OVERRIDES_FILE)
}

fn read_overrides(app: &AppHandle) -> Map<String, Value> {
    let path = overrides_path(app);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Map::new();
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

/// Raw template plus the per-node edits currently saved for it.
#[tauri::command]
pub fn comfy_template_detail(
    app: AppHandle,
    name: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let template = match read_template(&app, &name) {
        Ok(template) => template,
        Err(message) => {
            return failure(
                "comfy.template.detail",
                request_id.clone(),
                core_error(&request_id, "COMFY_TEMPLATE_INVALID", &message, None, false),
            )
        }
    };
    let overrides = read_overrides(&app)
        .get(name.trim())
        .cloned()
        .unwrap_or_else(|| json!({}));
    success(
        "comfy.template.detail",
        request_id,
        json!({
            "name": name.trim(),
            "template": template,
            "overrides": overrides,
        }),
    )
}

#[tauri::command]
pub fn comfy_save_template_overrides(
    app: AppHandle,
    name: String,
    overrides: Value,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    if !overrides.is_object() {
        return failure(
            "comfy.template.overrides",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_TEMPLATE_INVALID",
                "覆盖值必须是对象。",
                None,
                false,
            ),
        );
    }
    let mut stored = read_overrides(&app);
    // Drop empty entries so switching nodes back to the template value really
    // removes the override instead of pinning today's value forever.
    let cleaned: Map<String, Value> = overrides
        .as_object()
        .map(|nodes| {
            nodes
                .iter()
                .filter_map(|(node_id, inputs)| {
                    let inputs = inputs.as_object()?;
                    if inputs.is_empty() {
                        return None;
                    }
                    Some((node_id.clone(), Value::Object(inputs.clone())))
                })
                .collect()
        })
        .unwrap_or_default();
    if cleaned.is_empty() {
        stored.remove(name.trim());
    } else {
        stored.insert(name.trim().to_string(), Value::Object(cleaned));
    }
    let path = overrides_path(&app);
    let payload = serde_json::to_vec_pretty(&Value::Object(stored)).unwrap_or_default();
    if let Err(error) = std::fs::write(&path, payload) {
        return failure(
            "comfy.template.overrides",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_TEMPLATE_OVERRIDES_FAILED",
                &format!("无法写入 {}：{error}", path.display()),
                None,
                false,
            ),
        );
    }
    success(
        "comfy.template.overrides",
        request_id,
        json!({ "saved": true }),
    )
}
