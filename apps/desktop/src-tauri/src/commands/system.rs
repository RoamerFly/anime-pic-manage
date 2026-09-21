use crate::database::{WORKER_COMPUTE_DEVICE_SETTING, WORKER_RUNTIME_MODE_SETTING};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::models::{RuntimeStatus, WorkerHealth, WorkerRuntimeSettings};
use crate::path_utils::display_path;
use crate::state::AppState;
use crate::worker_runtime::{
    WorkerCapabilitiesInfo, WorkerComputeDevice, WorkerManager, WorkerRuntimeMode,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};

#[derive(Debug, Clone, Serialize)]
pub struct GpuInventory {
    pub devices: Vec<String>,
    pub available: bool,
    pub detail: Option<String>,
}

/// Read the installed display adapters.
///
/// The settings page wants to distinguish "no CUDA runtime" from "no NVIDIA
/// card at all", which ONNX Runtime alone cannot report. The lookup is cached
/// in `AppState`; the registry path is preferred because a WMI query can take
/// ten seconds on cold start while `reg query` answers in milliseconds.
fn parse_driver_desc(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let value = trimmed
                .split_once("REG_SZ")
                .map(|(_, value)| value.trim())
                .filter(|value| !value.is_empty())?;
            Some(value.to_string())
        })
        .collect()
}

fn is_virtual_display(name: &str) -> bool {
    const VIRTUAL_MARKERS: [&str; 6] = [
        "virtual",
        "idddriver",
        "mirror",
        "gameviewer",
        "todesk",
        "parsec",
    ];
    let lowered = name.to_ascii_lowercase();
    VIRTUAL_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

fn query_registry_adapters() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        let mut command = std::process::Command::new("reg");
        command
            .args([
                "query",
                r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
                "/s",
                "/v",
                "DriverDesc",
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        crate::worker_runtime::process::configure_background_command(&mut command);
        if let Ok(output) = command.output() {
            if output.status.success() {
                return parse_driver_desc(&String::from_utf8_lossy(&output.stdout));
            }
        }
        Vec::new()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}

fn query_wmi_adapters() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        let script = "Get-CimInstance Win32_VideoController | \
            Select-Object -ExpandProperty Name";
        let mut command = std::process::Command::new("powershell");
        command
            .args(["-NoProfile", "-NonInteractive", "-Command", script])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        crate::worker_runtime::process::configure_background_command(&mut command);
        if let Ok(output) = command.output() {
            if output.status.success() {
                return String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect();
            }
        }
        Vec::new()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}

fn query_gpu_devices() -> Vec<String> {
    let mut devices = query_registry_adapters();
    if devices.is_empty() {
        devices = query_wmi_adapters();
    }
    let mut filtered: Vec<String> = Vec::new();
    for device in devices {
        if is_virtual_display(&device) || filtered.iter().any(|item| item == &device) {
            continue;
        }
        filtered.push(device);
    }
    filtered
}

#[tauri::command]
pub async fn get_gpu_devices(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<GpuInventory> {
    let request_id = normalize_request_id(request_id);
    let cache = Arc::clone(&app.state::<AppState>().gpu_devices);
    if let Ok(guard) = cache.lock() {
        if let Some(devices) = guard.as_ref() {
            return success(
                "system.gpu.list",
                request_id,
                GpuInventory {
                    available: !devices.is_empty(),
                    detail: None,
                    devices: devices.clone(),
                },
            );
        }
    }
    let devices = tauri::async_runtime::spawn_blocking(query_gpu_devices)
        .await
        .unwrap_or_default();
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(devices.clone());
    }
    let detail = devices
        .is_empty()
        .then(|| "未读取到独立/集成显卡信息，可继续使用 CPU 推理。".to_string());
    success(
        "system.gpu.list",
        request_id,
        GpuInventory {
            available: !devices.is_empty(),
            detail,
            devices,
        },
    )
}

/// One model session as reported by the Worker after a real CUDA self-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerModelCompute {
    pub model_id: String,
    #[serde(default)]
    pub requested_device: Option<String>,
    #[serde(default)]
    pub active_device: Option<String>,
    #[serde(default)]
    pub active_providers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerComputeError {
    pub model_id: String,
    #[serde(default)]
    pub code: Option<String>,
    pub error: String,
}

/// Result of loading the installed models with the configured execution device.
///
/// `cuda_available` only means the ONNX Runtime build advertises CUDA;
/// `cuda_session_ready` means a session was actually created with the CUDA
/// execution provider, which is what the user needs to know.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerComputeStatus {
    #[serde(default)]
    pub requested_device: String,
    #[serde(default)]
    pub cuda_available: bool,
    #[serde(default)]
    pub cuda_session_ready: bool,
    #[serde(default)]
    pub distribution: Option<String>,
    #[serde(default)]
    pub available_providers: Vec<String>,
    #[serde(default)]
    pub models: Vec<WorkerModelCompute>,
    #[serde(default)]
    pub errors: Vec<WorkerComputeError>,
}

#[tauri::command]
pub async fn check_worker_compute(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerComputeStatus> {
    let request_id = normalize_request_id(request_id);
    let worker = Arc::clone(&app.state::<AppState>().worker);
    let probed = tauri::async_runtime::spawn_blocking(move || {
        let mut manager = worker
            .lock()
            .map_err(|_| "AI Worker 状态锁暂时不可用".to_string())?;
        manager
            .request("runtime.device", json!({ "probe": true }))
            .map_err(|error| error.to_string())
    })
    .await;

    match probed {
        Ok(Ok(payload)) => match serde_json::from_value::<WorkerComputeStatus>(payload) {
            Ok(status) => success("runtime.compute.check", request_id, status),
            Err(error) => failure(
                "runtime.compute.check",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMPUTE_PROBE_INVALID",
                    "推理设备自检返回了无法解析的结果。",
                    Some(error.to_string()),
                    true,
                ),
            ),
        },
        Ok(Err(message)) => failure(
            "runtime.compute.check",
            request_id.clone(),
            core_error(
                &request_id,
                "COMPUTE_PROBE_FAILED",
                "推理设备自检失败，可能是模型尚未安装。",
                Some(message),
                true,
            ),
        ),
        Err(error) => failure(
            "runtime.compute.check",
            request_id.clone(),
            core_error(
                &request_id,
                "COMPUTE_PROBE_TASK_FAILED",
                "推理设备自检任务异常中止。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

fn now_iso() -> String {
    format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    )
}

fn worker_runtime_settings(worker: &WorkerManager) -> WorkerRuntimeSettings {
    let mode = worker.runtime_mode();
    let compute_device = worker.compute_device();
    WorkerRuntimeSettings {
        mode: mode.as_str().to_string(),
        mode_label: mode.label().to_string(),
        resolved_path: worker.runtime_path().map(|path| display_path(&path)),
        compute_device: compute_device.as_str().to_string(),
        compute_device_label: compute_device.label().to_string(),
    }
}

fn worker_health_with_runtime(
    status: String,
    message: String,
    version: Option<String>,
    model_count: Option<u64>,
    runtime: WorkerRuntimeSettings,
    capabilities: Option<WorkerCapabilitiesInfo>,
    capability_error: Option<String>,
) -> WorkerHealth {
    WorkerHealth {
        status,
        message,
        version,
        model_count,
        runtime_mode: runtime.mode,
        runtime_mode_label: runtime.mode_label,
        runtime_path: runtime.resolved_path,
        compute_device: runtime.compute_device,
        compute_device_label: runtime.compute_device_label,
        capabilities,
        capability_error,
        checked_at: now_iso(),
    }
}

fn probe_worker(worker: &Arc<Mutex<WorkerManager>>) -> WorkerHealth {
    let checked_at = now_iso();
    let mut worker = match worker.lock() {
        Ok(worker) => worker,
        Err(_) => {
            return WorkerHealth {
                status: "error".to_string(),
                message: "AI Worker 状态锁暂时不可用，请稍后重试。".to_string(),
                version: None,
                model_count: None,
                runtime_mode: WorkerRuntimeMode::default().as_str().to_string(),
                runtime_mode_label: WorkerRuntimeMode::default().label().to_string(),
                runtime_path: None,
                compute_device: WorkerComputeDevice::default().as_str().to_string(),
                compute_device_label: WorkerComputeDevice::default().label().to_string(),
                capabilities: None,
                capability_error: Some("AI Worker 状态锁暂时不可用。".to_string()),
                checked_at,
            }
        }
    };
    let runtime = worker_runtime_settings(&worker);
    match worker.health() {
        Ok(health) => match worker.capabilities() {
            Ok(capabilities) if capabilities.ready && capabilities.status == "ok" => {
                worker_health_with_runtime(
                    if health.status == "stopping" {
                        "stopped".to_string()
                    } else {
                        "healthy".to_string()
                    },
                    format!(
                        "AI Worker 已完成健康握手，发现 {} 个模型；dghs-imgutils 与 ONNX Runtime 能力就绪。",
                        health.model_count
                    ),
                    Some(health.version),
                    Some(health.model_count),
                    runtime,
                    Some(capabilities),
                    None,
                )
            }
            Ok(capabilities) => {
                let message = capabilities
                    .errors
                    .first()
                    .map(|error| error.message.clone())
                    .unwrap_or_else(|| "必需的 dghs-imgutils/ONNX Runtime 能力不可用。".to_string());
                worker_health_with_runtime(
                    "unavailable".to_string(),
                    format!("AI Worker 健康握手成功，但运行能力不可用：{message}"),
                    Some(health.version),
                    Some(health.model_count),
                    runtime,
                    Some(capabilities),
                    None,
                )
            }
            Err(error) => worker_health_with_runtime(
                "error".to_string(),
                format!("AI Worker 健康握手成功，但能力探测失败：{error}"),
                Some(health.version),
                Some(health.model_count),
                runtime,
                None,
                Some(error.to_string()),
            ),
        },
        Err(error) => worker_health_with_runtime(
            if error.is_unavailable() {
                "unavailable".to_string()
            } else {
                "error".to_string()
            },
            error.to_string(),
            None,
            None,
            runtime,
            None,
            Some(error.to_string()),
        ),
    }
}

#[tauri::command]
pub async fn get_runtime_status(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<RuntimeStatus> {
    let request_id = normalize_request_id(request_id);
    let state = app.state::<AppState>();
    let database = Arc::clone(&state.database);
    let worker = Arc::clone(&state.worker);
    let data_dir = state.data_dir.clone();

    // Health probing starts the Worker process and imports ONNX Runtime, which
    // can take seconds on a cold start. Async commands run off the main thread,
    // so file dialogs and window interactions stay responsive meanwhile.
    let probed = tauri::async_runtime::spawn_blocking(move || {
        let database = match database.lock() {
            Ok(database) => database.health().map_err(|error| error.to_string()),
            Err(_) => Err("本地数据库暂时被占用。".to_string()),
        };
        (database, probe_worker(&worker))
    })
    .await;

    match probed {
        Ok((Ok(database), worker)) => success(
            "system.health",
            request_id,
            RuntimeStatus {
                app_version: env!("CARGO_PKG_VERSION").to_string(),
                data_dir,
                database,
                worker,
            },
        ),
        Ok((Err(error), _)) => failure(
            "system.health",
            request_id.clone(),
            core_error(
                &request_id,
                "DATABASE_HEALTH_FAILED",
                "无法读取本地数据库状态。",
                Some(error),
                true,
            ),
        ),
        Err(error) => failure(
            "system.health",
            request_id.clone(),
            core_error(
                &request_id,
                "WORKER_PROBE_TASK_FAILED",
                "本地核心状态检查异常中止。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub async fn check_worker_health(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerHealth> {
    let request_id = normalize_request_id(request_id);
    let worker = Arc::clone(&app.state::<AppState>().worker);
    match tauri::async_runtime::spawn_blocking(move || probe_worker(&worker)).await {
        Ok(health) => success("worker.health", request_id, health),
        Err(error) => failure(
            "worker.health",
            request_id.clone(),
            core_error(
                &request_id,
                "WORKER_PROBE_TASK_FAILED",
                "AI Worker 状态检查异常中止。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub fn get_worker_runtime_settings(
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerRuntimeSettings> {
    let request_id = normalize_request_id(request_id);
    let worker = match state.worker.lock() {
        Ok(worker) => worker,
        Err(_) => {
            return failure(
                "settings.get",
                request_id.clone(),
                core_error(
                    &request_id,
                    "WORKER_LOCK_FAILED",
                    "AI Worker 状态锁暂时不可用。",
                    None,
                    true,
                ),
            )
        }
    };
    success("settings.get", request_id, worker_runtime_settings(&worker))
}

#[tauri::command]
pub fn set_worker_runtime_mode(
    state: State<'_, AppState>,
    mode: String,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerRuntimeSettings> {
    let request_id = normalize_request_id(request_id);
    let Some(runtime_mode) = WorkerRuntimeMode::parse(mode.trim()) else {
        return failure(
            "settings.update",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_WORKER_RUNTIME_MODE",
                "运行方式只能选择独立 EXE 或内置 ENV。",
                Some("mode must be executable or embedded_env".to_string()),
                false,
            ),
        );
    };

    let previous_mode = {
        let mut worker = match state.worker.lock() {
            Ok(worker) => worker,
            Err(_) => {
                return failure(
                    "settings.update",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "WORKER_LOCK_FAILED",
                        "AI Worker 状态锁暂时不可用。",
                        None,
                        true,
                    ),
                )
            }
        };
        let previous_mode = worker.runtime_mode();
        worker.set_runtime_mode(runtime_mode);
        previous_mode
    };

    let setting_result = match state.database.lock() {
        Ok(database) => database
            .set_setting_string(WORKER_RUNTIME_MODE_SETTING, runtime_mode.as_str())
            .map_err(|error| error.to_string()),
        Err(_) => Err("本地数据库状态锁暂时不可用".to_string()),
    };
    if let Err(error) = setting_result {
        if let Ok(mut worker) = state.worker.lock() {
            worker.set_runtime_mode(previous_mode);
        }
        return failure(
            "settings.update",
            request_id.clone(),
            core_error(
                &request_id,
                "WORKER_RUNTIME_SETTING_FAILED",
                "无法保存 Worker 运行方式，已恢复原设置。",
                Some(error.to_string()),
                true,
            ),
        );
    }

    let worker = match state.worker.lock() {
        Ok(worker) => worker,
        Err(_) => {
            return failure(
                "settings.update",
                request_id.clone(),
                core_error(
                    &request_id,
                    "WORKER_LOCK_FAILED",
                    "运行方式已保存，但无法读取当前 Worker 状态。",
                    None,
                    true,
                ),
            )
        }
    };
    success(
        "settings.update",
        request_id,
        worker_runtime_settings(&worker),
    )
}

#[tauri::command]
pub fn set_worker_compute_device(
    state: State<'_, AppState>,
    device: String,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerRuntimeSettings> {
    let request_id = normalize_request_id(request_id);
    let Some(compute_device) = WorkerComputeDevice::parse(&device) else {
        return failure(
            "settings.update",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_COMPUTE_DEVICE",
                "推理设备只能选择自动、CPU 或 NVIDIA CUDA。",
                Some("device must be auto, cpu or cuda".to_string()),
                false,
            ),
        );
    };

    let previous_device = {
        let mut worker = match state.worker.lock() {
            Ok(worker) => worker,
            Err(_) => {
                return failure(
                    "settings.update",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "WORKER_LOCK_FAILED",
                        "AI Worker 状态锁暂时不可用。",
                        None,
                        true,
                    ),
                )
            }
        };
        let previous_device = worker.compute_device();
        worker.set_compute_device(compute_device);
        previous_device
    };

    let setting_result = match state.database.lock() {
        Ok(database) => database
            .set_setting_string(WORKER_COMPUTE_DEVICE_SETTING, compute_device.as_str())
            .map_err(|error| error.to_string()),
        Err(_) => Err("本地数据库状态锁暂时不可用".to_string()),
    };
    if let Err(error) = setting_result {
        if let Ok(mut worker) = state.worker.lock() {
            worker.set_compute_device(previous_device);
        }
        return failure(
            "settings.update",
            request_id.clone(),
            core_error(
                &request_id,
                "WORKER_COMPUTE_DEVICE_SETTING_FAILED",
                "无法保存推理设备，已恢复原设置。",
                Some(error),
                true,
            ),
        );
    }

    let worker = match state.worker.lock() {
        Ok(worker) => worker,
        Err(_) => {
            return failure(
                "settings.update",
                request_id.clone(),
                core_error(
                    &request_id,
                    "WORKER_LOCK_FAILED",
                    "推理设备已保存，但无法读取当前 Worker 状态。",
                    None,
                    true,
                ),
            )
        }
    };
    success(
        "settings.update",
        request_id,
        worker_runtime_settings(&worker),
    )
}

#[tauri::command]
pub fn show_item_in_folder(path: String) -> Result<(), String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err("文件不存在。".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        let win_path = path.replace('/', "\\");
        let _ = std::process::Command::new("explorer")
            .arg(format!("/select,{}", win_path))
            .spawn()
            .map_err(|e| format!("无法打开资源管理器: {}", e))?;
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .args(["-R", &path])
            .spawn()
            .map_err(|e| format!("无法打开访达: {}", e))?;
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(parent) = p.parent() {
            let _ = std::process::Command::new("xdg-open")
                .arg(parent)
                .spawn()
                .map_err(|e| format!("无法打开文件管理器: {}", e))?;
        }
    }
    Ok(())
}

/// Open a documentation/download page in the user's browser.
///
/// Only `https` links are accepted so a compromised renderer cannot ask the
/// backend to launch arbitrary local handlers.
#[tauri::command]
pub fn open_external_url(
    app: AppHandle,
    url: String,
    request_id: Option<String>,
) -> IpcEnvelope<serde_json::Value> {
    use tauri_plugin_opener::OpenerExt;

    let request_id = normalize_request_id(request_id);
    let trimmed = url.trim();
    if !trimmed.starts_with("https://") {
        return failure(
            "system.open_url",
            request_id.clone(),
            core_error(
                &request_id,
                "URL_NOT_ALLOWED",
                "只允许打开 https 链接。",
                None,
                false,
            ),
        );
    }
    match app.opener().open_url(trimmed, None::<&str>) {
        Ok(()) => success("system.open_url", request_id, json!({ "url": trimmed })),
        Err(error) => failure(
            "system.open_url",
            request_id.clone(),
            core_error(
                &request_id,
                "URL_OPEN_FAILED",
                &format!("无法打开浏览器：{error}"),
                None,
                false,
            ),
        ),
    }
}

#[tauri::command]
pub fn delete_library_file(state: State<'_, AppState>, path: String) -> Result<bool, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err("文件不存在或已被删除。".to_string());
    }
    fs::remove_file(p).map_err(|e| format!("删除文件失败: {}", e))?;
    if let Ok(db) = state.database.lock() {
        let _ = db.delete_image_annotations_for_path(&path);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_REGISTRY_OUTPUT: &str = "\r\nHKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000\r\n    DriverDesc    REG_SZ    Intel(R) UHD Graphics\r\n\r\nHKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0001\r\n    DriverDesc    REG_SZ    NVIDIA GeForce RTX 3060 Laptop GPU\r\n\r\nEnd of search: 2 match(es) found.\r\n";

    #[test]
    fn parses_driver_desc_values_from_registry_output() {
        let devices = parse_driver_desc(SAMPLE_REGISTRY_OUTPUT);

        assert_eq!(
            devices,
            vec![
                "Intel(R) UHD Graphics".to_string(),
                "NVIDIA GeForce RTX 3060 Laptop GPU".to_string(),
            ]
        );
    }

    #[test]
    fn virtual_display_adapters_are_filtered_out() {
        assert!(is_virtual_display("Todesk Virtual Display Adapter"));
        assert!(is_virtual_display("OrayIddDriver Device"));
        assert!(is_virtual_display("GameViewer Virtual Display Adapter"));
        assert!(!is_virtual_display("NVIDIA GeForce RTX 3060 Laptop GPU"));
        assert!(!is_virtual_display("Intel(R) UHD Graphics"));
    }

    #[test]
    fn gpu_query_never_reports_virtual_adapters() {
        for device in query_gpu_devices() {
            assert!(!is_virtual_display(&device), "unexpected adapter: {device}");
        }
    }
}
