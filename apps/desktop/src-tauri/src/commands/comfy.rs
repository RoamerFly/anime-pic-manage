//! Tauri commands that drive the user's own ComfyUI installation.

use crate::app_settings::AppSettings;
use crate::comfy::{
    describe_device, read_version, resolve_paths, ComfyClient, ComfyError, ComfyGenerateRequest,
    ComfyGenerateResult, ComfyGeneratedImage, ComfyModelList, ComfyPaths, ComfyProgress,
    ComfyStatus, WorkflowTemplate,
};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

const SETTINGS_LOCK_ERROR: &str = "本地数据库状态锁暂时不可用。";
const POLL_INTERVAL: Duration = Duration::from_millis(700);
const DEFAULT_START_WAIT_SECS: u64 = 120;
const TEMPLATE_FOLDER: &str = "comfy-workflows";

fn load_settings(app: &AppHandle) -> Result<AppSettings, String> {
    let state = app.state::<AppState>();
    let database = state
        .database
        .lock()
        .map_err(|_| SETTINGS_LOCK_ERROR.to_string())?;
    crate::app_settings::load_app_settings(&database)
}

/// Locate the workflow template directory in dev and packaged layouts.
fn template_roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        roots.push(resource_dir.join(TEMPLATE_FOLDER));
        roots.push(resource_dir.join("resources").join(TEMPLATE_FOLDER));
    }
    let project_dir = crate::worker_runtime::discovery::local_project_dir();
    roots.push(project_dir.join("resources").join(TEMPLATE_FOLDER));
    if let Ok(current) = std::env::current_dir() {
        roots.push(current.join(TEMPLATE_FOLDER));
        roots.push(current.join("resources").join(TEMPLATE_FOLDER));
    }
    roots
}

fn load_template(app: &AppHandle, name: &str) -> Result<WorkflowTemplate, ComfyError> {
    let file = format!("{name}.json");
    for root in template_roots(app) {
        let candidate = root.join(&file);
        if candidate.is_file() {
            return WorkflowTemplate::from_file(&candidate);
        }
    }
    Err(ComfyError::Template(format!(
        "未找到工作流模板 {file}（应在 resources/{TEMPLATE_FOLDER} 下）"
    )))
}

fn list_templates(app: &AppHandle) -> Vec<String> {
    for root in template_roots(app) {
        if !root.is_dir() {
            continue;
        }
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter_map(|entry| {
                        let path = entry.path();
                        if path.extension().and_then(|value| value.to_str()) != Some("json") {
                            return None;
                        }
                        path.file_stem()
                            .and_then(|value| value.to_str())
                            .map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        if !names.is_empty() {
            return names;
        }
    }
    Vec::new()
}

fn emit_progress(
    app: &AppHandle,
    phase: &str,
    current: usize,
    total: usize,
    message: String,
    prompt_id: Option<&str>,
) {
    let _ = app.emit(
        "comfy://progress",
        ComfyProgress {
            phase: phase.to_string(),
            current,
            total,
            message,
            prompt_id: prompt_id.map(str::to_string),
        },
    );
}

async fn probe_status(
    app: &AppHandle,
    settings: &AppSettings,
    paths: Result<ComfyPaths, ComfyError>,
) -> ComfyStatus {
    let mut status = ComfyStatus {
        configured: !settings.comfy_root.trim().is_empty(),
        port: settings.comfy_port,
        ..ComfyStatus::default()
    };
    match paths {
        Ok(paths) => {
            status.installed = true;
            status.root = Some(paths.root.clone());
            status.python = Some(paths.python.clone());
            status.output_dir = Some(paths.output_dir.clone());
            status.version = read_version(Path::new(&paths.app_dir));
            match ComfyClient::new(settings.comfy_port) {
                Ok(client) => match client.system_stats().await {
                    Ok(stats) => {
                        status.running = true;
                        let (device, vram) = describe_device(&stats);
                        status.device = device;
                        status.vram_gb = vram;
                    }
                    Err(error) => status.error = Some(error.to_string()),
                },
                Err(error) => status.error = Some(error.to_string()),
            }
        }
        Err(ComfyError::NotConfigured) => {}
        Err(error) => {
            status.error = Some(error.to_string());
        }
    }
    let comfy_manager = std::sync::Arc::clone(&app.state::<AppState>().comfy);
    if let Ok(manager) = comfy_manager.lock() {
        status.owned = manager.is_owned();
    }
    status
}

#[tauri::command]
pub async fn comfy_status(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<ComfyStatus> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "comfy.status",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = resolve_paths(
        &settings.comfy_root,
        &settings.comfy_output_dir,
        settings.comfy_port,
    );
    let status = probe_status(&app, &settings, paths).await;
    success("comfy.status", request_id, status)
}

#[tauri::command]
pub async fn comfy_start(
    app: AppHandle,
    wait_secs: Option<u64>,
    request_id: Option<String>,
) -> IpcEnvelope<ComfyStatus> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "comfy.start",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = match resolve_paths(
        &settings.comfy_root,
        &settings.comfy_output_dir,
        settings.comfy_port,
    ) {
        Ok(paths) => paths,
        Err(error) => {
            return failure(
                "comfy.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_NOT_CONFIGURED",
                    &error.to_string(),
                    None,
                    false,
                ),
            )
        }
    };

    let log_dir = PathBuf::from(&app.state::<AppState>().data_dir).join("logs");
    let started = {
        let comfy = std::sync::Arc::clone(&app.state::<AppState>().comfy);
        let mut manager = match comfy.lock() {
            Ok(manager) => manager,
            Err(_) => {
                return failure(
                    "comfy.start",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "COMFY_LOCK_FAILED",
                        "ComfyUI 状态锁暂时不可用。",
                        None,
                        true,
                    ),
                )
            }
        };
        manager.start(
            &paths,
            settings.comfy_port,
            settings.comfy_low_vram,
            &log_dir,
            settings.background_priority != "normal",
        )
    };
    if let Err(error) = started {
        return failure(
            "comfy.start",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_START_FAILED",
                &error.to_string(),
                None,
                false,
            ),
        );
    }

    let client = match ComfyClient::new(settings.comfy_port) {
        Ok(client) => client,
        Err(error) => {
            return failure(
                "comfy.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_CLIENT_FAILED",
                    &error.to_string(),
                    None,
                    true,
                ),
            )
        }
    };
    let deadline =
        Instant::now() + Duration::from_secs(wait_secs.unwrap_or(DEFAULT_START_WAIT_SECS));
    while Instant::now() < deadline {
        if client.system_stats().await.is_ok() {
            emit_progress(&app, "ready", 0, 0, "ComfyUI 已就绪".to_string(), None);
            let status = probe_status(&app, &settings, Ok(paths)).await;
            return success("comfy.start", request_id, status);
        }
        emit_progress(
            &app,
            "starting",
            0,
            0,
            "正在启动 ComfyUI…".to_string(),
            None,
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    let status = probe_status(&app, &settings, Ok(paths)).await;
    failure(
        "comfy.start",
        request_id.clone(),
        core_error(
            &request_id,
            "COMFY_START_TIMEOUT",
            "ComfyUI 启动超时，请查看 logs/comfyui.log。",
            status.error.clone(),
            true,
        ),
    )
}

/// PID of the process listening on `port`, if any.
fn listener_pid(port: u16) -> Option<u32> {
    let output = std::process::Command::new("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let needle = format!(":{port}");
    for line in text.lines() {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.len() < 5 || !columns[0].eq_ignore_ascii_case("TCP") {
            continue;
        }
        if !columns[1].ends_with(&needle) || !columns[3].eq_ignore_ascii_case("LISTENING") {
            continue;
        }
        if let Ok(pid) = columns[4].parse::<u32>() {
            return Some(pid);
        }
    }
    None
}

fn process_name(pid: u32) -> Option<String> {
    let output = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next()?.trim();
    let name = first.trim_matches('"').split("\",\"").next()?.to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// End a ComfyUI instance the app did not start (manual launch, or the app was
/// restarted). Returns a message when the caller should be told why nothing
/// happened; `None` means the port is free or the process was ended.
fn stop_foreign_listener(port: u16) -> Option<String> {
    let pid = listener_pid(port)?;
    let Some(name) = process_name(pid) else {
        return Some(format!("端口 {port} 被 PID {pid} 占用，但无法确认进程名。"));
    };
    if !name.to_ascii_lowercase().contains("python") {
        return Some(format!(
            "端口 {port} 被 {name}（PID {pid}）占用，它不是 Python/ComfyUI 进程，未自动结束。"
        ));
    }
    let output = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output()
        .ok()?;
    if output.status.success() {
        None
    } else {
        Some(format!(
            "结束 {name}（PID {pid}）失败：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

#[tauri::command]
pub async fn comfy_stop(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<ComfyStatus> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "comfy.stop",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let stopped = {
        let comfy = std::sync::Arc::clone(&app.state::<AppState>().comfy);
        let mut guard = comfy.lock();
        match guard.as_mut() {
            Ok(manager) => manager.stop(),
            Err(_) => Err(ComfyError::Request("ComfyUI 状态锁暂时不可用".to_string())),
        }
    };
    if let Err(error) = stopped {
        return failure(
            "comfy.stop",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_STOP_FAILED",
                &error.to_string(),
                None,
                true,
            ),
        );
    }
    // The instance may have been started outside the app (or the app was
    // restarted): fall back to ending whatever ComfyUI process owns the port,
    // but only when it really looks like a Python process.
    if let Some(message) = stop_foreign_listener(settings.comfy_port) {
        return failure(
            "comfy.stop",
            request_id.clone(),
            core_error(&request_id, "COMFY_STOP_REFUSED", &message, None, false),
        );
    }
    let paths = resolve_paths(
        &settings.comfy_root,
        &settings.comfy_output_dir,
        settings.comfy_port,
    );
    let status = probe_status(&app, &settings, paths).await;
    success("comfy.stop", request_id, status)
}

#[tauri::command]
pub async fn comfy_models(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<ComfyModelList> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "comfy.models",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let client = match ComfyClient::new(settings.comfy_port) {
        Ok(client) => client,
        Err(error) => {
            return failure(
                "comfy.models",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_CLIENT_FAILED",
                    &error.to_string(),
                    None,
                    true,
                ),
            )
        }
    };
    let mut list = ComfyModelList {
        checkpoints: Vec::new(),
        loras: Vec::new(),
        samplers: Vec::new(),
        schedulers: Vec::new(),
    };
    match client.list_models("checkpoints").await {
        Ok(models) => list.checkpoints = models,
        Err(error) => {
            return failure(
                "comfy.models",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_NOT_RUNNING",
                    &error.to_string(),
                    None,
                    true,
                ),
            )
        }
    }
    list.loras = client.list_models("loras").await.unwrap_or_default();
    list.samplers = client
        .node_input_options("KSampler", "sampler_name")
        .await
        .unwrap_or_default();
    list.schedulers = client
        .node_input_options("KSampler", "scheduler")
        .await
        .unwrap_or_default();
    success("comfy.models", request_id, list)
}

#[tauri::command]
pub async fn comfy_generate(
    app: AppHandle,
    request: ComfyGenerateRequest,
    request_id: Option<String>,
) -> IpcEnvelope<ComfyGenerateResult> {
    let request_id = normalize_request_id(request_id);
    // A negative seed means "random"; ComfyUI's API would reject it, so resolve
    // it once here and report the value that was actually used.
    let mut request = request;
    if request.seed < 0 {
        request.seed = crate::comfy::workflow::resolved_seed(request.seed);
    }
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = match resolve_paths(
        &settings.comfy_root,
        &settings.comfy_output_dir,
        settings.comfy_port,
    ) {
        Ok(paths) => paths,
        Err(error) => {
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_NOT_CONFIGURED",
                    &error.to_string(),
                    None,
                    false,
                ),
            )
        }
    };
    let template = match load_template(&app, &request.template) {
        Ok(template) => template,
        Err(error) => {
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_TEMPLATE_INVALID",
                    &error.to_string(),
                    None,
                    false,
                ),
            )
        }
    };
    let prompt = match template.build_prompt(&request) {
        Ok(prompt) => prompt,
        Err(error) => {
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_TEMPLATE_INVALID",
                    &error.to_string(),
                    None,
                    false,
                ),
            )
        }
    };
    let client = match ComfyClient::new(settings.comfy_port) {
        Ok(client) => client,
        Err(error) => {
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_CLIENT_FAILED",
                    &error.to_string(),
                    None,
                    true,
                ),
            )
        }
    };
    if client.system_stats().await.is_err() {
        return failure(
            "comfy.generate",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_NOT_RUNNING",
                "ComfyUI 尚未运行，请先在“AI 生图”页面启动它。",
                None,
                true,
            ),
        );
    }

    let cancel_flag = {
        let state = app.state::<AppState>();
        state.comfy_cancel.store(false, Ordering::Release);
        std::sync::Arc::clone(&state.comfy_cancel)
    };

    emit_progress(&app, "queued", 0, 0, "正在提交工作流…".to_string(), None);
    let client_id = uuid::Uuid::new_v4().to_string();
    let prompt_id = match client.queue_prompt(prompt, &client_id).await {
        Ok(prompt_id) => prompt_id,
        Err(error) => {
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_QUEUE_FAILED",
                    &error.to_string(),
                    None,
                    true,
                ),
            )
        }
    };

    let started = Instant::now();
    let history = loop {
        if cancel_flag.load(Ordering::Acquire) {
            let _ = client.interrupt().await;
            emit_progress(
                &app,
                "cancelled",
                0,
                0,
                "已取消生成。".to_string(),
                Some(&prompt_id),
            );
            return failure(
                "comfy.generate",
                request_id.clone(),
                core_error(&request_id, "COMFY_CANCELLED", "生成已取消。", None, false),
            );
        }
        match client.history(&prompt_id).await {
            Ok(Some(entry)) if entry.get("outputs").is_some() => break entry,
            Ok(_) => {
                let elapsed = started.elapsed().as_secs();
                emit_progress(
                    &app,
                    "running",
                    0,
                    0,
                    format!("正在生成…（已用 {elapsed} 秒）"),
                    Some(&prompt_id),
                );
            }
            Err(error) => {
                emit_progress(&app, "error", 0, 0, error.to_string(), Some(&prompt_id));
                return failure(
                    "comfy.generate",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "COMFY_HISTORY_FAILED",
                        &error.to_string(),
                        None,
                        true,
                    ),
                );
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    };

    let images = collect_images(&history);
    if images.is_empty() {
        return failure(
            "comfy.generate",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_NO_OUTPUT",
                "工作流执行完成，但没有产生图片输出（检查模板是否包含 SaveImage 节点）。",
                None,
                false,
            ),
        );
    }

    let output_dir = PathBuf::from(&paths.output_dir);
    let total = images.len();
    let mut saved = Vec::with_capacity(total);
    for (index, image) in images.iter().enumerate() {
        emit_progress(
            &app,
            "downloading",
            index,
            total,
            format!("正在保存生成图 {}/{}…", index + 1, total),
            Some(&prompt_id),
        );
        match client.download_image(image, &output_dir, 120).await {
            Ok(path) => saved.push(ComfyGeneratedImage {
                filename: image.filename.clone(),
                subfolder: image.subfolder.clone(),
                path: path.to_string_lossy().into_owned(),
            }),
            Err(error) => {
                emit_progress(
                    &app,
                    "error",
                    index,
                    total,
                    error.to_string(),
                    Some(&prompt_id),
                );
                return failure(
                    "comfy.generate",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "COMFY_DOWNLOAD_FAILED",
                        &error.to_string(),
                        None,
                        true,
                    ),
                );
            }
        }
    }

    emit_progress(
        &app,
        "completed",
        total,
        total,
        format!("已生成 {} 张图片。", total),
        Some(&prompt_id),
    );
    success(
        "comfy.generate",
        request_id,
        ComfyGenerateResult {
            prompt_id,
            images: saved,
            output_dir: paths.output_dir,
            elapsed_ms: started.elapsed().as_millis() as u64,
            seed: request.seed,
        },
    )
}

#[tauri::command]
pub async fn comfy_cancel(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<bool> {
    let request_id = normalize_request_id(request_id);
    let state = app.state::<AppState>();
    state.comfy_cancel.store(true, Ordering::Release);
    success("comfy.cancel", request_id, true)
}

#[tauri::command]
pub fn comfy_templates(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<Vec<String>> {
    let request_id = normalize_request_id(request_id);
    success("comfy.templates", request_id, list_templates(&app))
}

#[tauri::command]
pub fn comfy_open_output(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<String> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "comfy.open_output",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = match resolve_paths(
        &settings.comfy_root,
        &settings.comfy_output_dir,
        settings.comfy_port,
    ) {
        Ok(paths) => paths,
        Err(error) => {
            return failure(
                "comfy.open_output",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_NOT_CONFIGURED",
                    &error.to_string(),
                    None,
                    false,
                ),
            )
        }
    };
    if let Err(error) = std::fs::create_dir_all(&paths.output_dir) {
        return failure(
            "comfy.open_output",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_OUTPUT_DIR_FAILED",
                &error.to_string(),
                None,
                true,
            ),
        );
    }
    if let Err(error) = crate::commands::show_item_in_folder(paths.output_dir.clone()) {
        return failure(
            "comfy.open_output",
            request_id.clone(),
            core_error(&request_id, "COMFY_OPEN_FAILED", &error, None, true),
        );
    }
    success("comfy.open_output", request_id, paths.output_dir)
}

/// Pull `images` entries out of a ComfyUI history payload.
fn collect_images(history: &Value) -> Vec<ComfyGeneratedImage> {
    let mut images = Vec::new();
    let Some(outputs) = history.get("outputs").and_then(Value::as_object) else {
        return images;
    };
    for output in outputs.values() {
        let Some(items) = output.get("images").and_then(Value::as_array) else {
            continue;
        };
        for item in items {
            let filename = item.get("filename").and_then(Value::as_str);
            let Some(filename) = filename else { continue };
            images.push(ComfyGeneratedImage {
                filename: filename.to_string(),
                subfolder: item
                    .get("subfolder")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                path: String::new(),
            });
        }
    }
    images
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collects_images_from_history_payload() {
        let history = json!({
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "anime-pic-manage_00001_.png", "subfolder": "", "type": "output"},
                        {"filename": "anime-pic-manage_00002_.png", "subfolder": "batch", "type": "output"}
                    ]
                },
                "11": {"text": ["no images here"]}
            }
        });

        let images = collect_images(&history);

        assert_eq!(images.len(), 2);
        assert_eq!(images[0].filename, "anime-pic-manage_00001_.png");
        assert_eq!(images[1].subfolder, "batch");
    }

    #[test]
    fn missing_outputs_yield_no_images() {
        assert!(collect_images(&json!({"prompt": "x"})).is_empty());
    }
}
