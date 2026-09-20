//! One-click LoRA training against the user's own kohya/sd-scripts install.
//!
//! Nothing is bundled or downloaded here: the commands detect an existing
//! trainer, run its training script with generated arguments and stream the
//! output back to the desktop page.

use crate::app_settings::AppSettings;
use crate::commands::dataset::default_dataset_dir;
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::kohya::{build_arguments, resolve_paths, status as kohya_status_of, TrainingRequest};
use crate::kohya_runner::{KohyaTrainingSnapshot, FINISHED_EVENT};
use crate::state::AppState;
use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
const WATCH_INTERVAL: Duration = Duration::from_millis(1500);

#[derive(Debug, Clone, Serialize)]
pub struct KohyaStatusPayload {
    #[serde(flatten)]
    pub status: crate::kohya::KohyaStatus,
    /// Arguments that would be used, so the UI can show the exact command.
    pub arguments_preview: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KohyaEnvironment {
    pub executable: String,
    pub python_version: Option<String>,
    pub torch_version: Option<String>,
    pub cuda_version: Option<String>,
    pub cuda_available: bool,
    pub device_name: Option<String>,
    pub xformers_version: Option<String>,
    pub bitsandbytes_version: Option<String>,
    pub error: Option<String>,
    pub exit_code: Option<i32>,
    pub stderr: String,
}

fn load_settings(app: &AppHandle) -> Result<AppSettings, String> {
    let state = app.state::<AppState>();
    let database = state
        .database
        .lock()
        .map_err(|_| "本地数据库状态锁暂时不可用。".to_string())?;
    crate::app_settings::load_app_settings(&database)
}

/// LoRA weights land next to the exported datasets by default.
fn default_training_output_dir(app: &AppHandle, settings: &AppSettings) -> String {
    let dataset_dir = PathBuf::from(default_dataset_dir(app, settings));
    let parent = dataset_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| dataset_dir.clone());
    // Portable packages own `<root>\output\datasets`, so the LoRA files belong
    // in the sibling `loras` folder instead of a second naming scheme.
    let folder = if dataset_dir
        .file_name()
        .map(|name| name == "datasets")
        .unwrap_or(false)
    {
        "loras"
    } else {
        "lora-models"
    };
    parent.join(folder).to_string_lossy().into_owned()
}

fn status_payload(app: &AppHandle, settings: &AppSettings) -> KohyaStatusPayload {
    let mut status = kohya_status_of(
        &settings.lora_trainer_root,
        &settings.lora_trainer_python,
        &settings.lora_trainer_base_model,
    );
    let dataset_dir = default_dataset_dir(app, settings);
    let output_dir = if settings.lora_trainer_output_dir.trim().is_empty() {
        default_training_output_dir(app, settings)
    } else {
        settings.lora_trainer_output_dir.clone()
    };
    status.train_data_dir = dataset_dir.clone();
    status.output_dir = output_dir.clone();

    let preview = if status.installed && status.base_model_exists {
        let request = TrainingRequest::default_for(
            crate::kohya::TrainerKind::Sd15,
            &PathBuf::from(&dataset_dir).join("dataset.toml"),
            Path::new(&output_dir),
            Path::new(&status.base_model),
        );
        build_arguments(&request, &PathBuf::from(&output_dir).join("logs"))
    } else {
        Vec::new()
    };

    KohyaStatusPayload {
        status,
        arguments_preview: preview,
    }
}

#[tauri::command]
pub fn kohya_status(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<KohyaStatusPayload> {
    let request_id = normalize_request_id(request_id);
    match load_settings(&app) {
        Ok(settings) => {
            let payload = status_payload(&app, &settings);
            success("kohya.status", request_id, payload)
        }
        Err(error) => failure(
            "kohya.status",
            request_id.clone(),
            core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
        ),
    }
}

#[tauri::command]
pub async fn kohya_probe_environment(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<KohyaEnvironment> {
    let request_id = normalize_request_id(request_id);
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "kohya.probe",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = match resolve_paths(&settings.lora_trainer_root, &settings.lora_trainer_python) {
        Ok(paths) => paths,
        Err(error) => {
            return failure(
                "kohya.probe",
                request_id.clone(),
                core_error(&request_id, "KOHYA_NOT_CONFIGURED", &error, None, false),
            )
        }
    };
    let python = paths.python.clone();
    let probe = tauri::async_runtime::spawn_blocking(move || run_environment_probe(&python))
        .await
        .unwrap_or_else(|error| KohyaEnvironment {
            executable: String::new(),
            python_version: None,
            torch_version: None,
            cuda_version: None,
            cuda_available: false,
            device_name: None,
            xformers_version: None,
            bitsandbytes_version: None,
            error: Some(format!("训练环境检查任务异常中止：{error}")),
            exit_code: None,
            stderr: String::new(),
        });
    success("kohya.probe", request_id, probe)
}

/// Reports torch/CUDA/optional extras without starting a training run.
const PROBE_SCRIPT: &str = r#"
import json, sys
info = {"executable": sys.executable, "python_version": sys.version.split()[0]}
for key in ("torch_version", "cuda_version", "device_name", "xformers_version",
            "bitsandbytes_version", "error"):
    info[key] = None
info["cuda_available"] = False
try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    info["error"] = f"{type(exc).__name__}: {exc}"
for module, key in (("xformers", "xformers_version"), ("bitsandbytes", "bitsandbytes_version")):
    try:
        info[key] = __import__(module).__version__
    except Exception:
        pass
print("KOHYA_ENV:" + json.dumps(info))
"#;

fn probe_failure(
    python: &Path,
    message: String,
    exit_code: Option<i32>,
    stderr: String,
) -> KohyaEnvironment {
    KohyaEnvironment {
        executable: python.to_string_lossy().into_owned(),
        python_version: None,
        torch_version: None,
        cuda_version: None,
        cuda_available: false,
        device_name: None,
        xformers_version: None,
        bitsandbytes_version: None,
        error: Some(message),
        exit_code,
        stderr,
    }
}

fn run_environment_probe(python: &Path) -> KohyaEnvironment {
    // Output goes to a temp file so a chatty interpreter can never block on a
    // full pipe while we are waiting for the timeout.
    let log_path =
        std::env::temp_dir().join(format!("apm-kohya-probe-{}.log", uuid::Uuid::new_v4()));
    let file = match fs::File::create(&log_path) {
        Ok(file) => file,
        Err(error) => {
            return probe_failure(
                python,
                format!("无法创建临时文件: {error}"),
                None,
                String::new(),
            )
        }
    };
    let errors = match file.try_clone() {
        Ok(handle) => handle,
        Err(error) => {
            return probe_failure(
                python,
                format!("无法创建临时文件: {error}"),
                None,
                String::new(),
            )
        }
    };

    let mut command = Command::new(python);
    command
        .args(["-c", PROBE_SCRIPT])
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(errors));
    crate::worker_runtime::process::configure_background_command(&mut command);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_file(&log_path);
            return probe_failure(
                python,
                format!("无法运行训练用 Python：{error}"),
                None,
                String::new(),
            );
        }
    };
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(error) => {
                let _ = fs::remove_file(&log_path);
                return probe_failure(
                    python,
                    format!("等待训练环境检查失败：{error}"),
                    None,
                    String::new(),
                );
            }
        }
    };

    let raw = fs::read_to_string(&log_path).unwrap_or_default();
    let _ = fs::remove_file(&log_path);
    let stderr: String = raw
        .lines()
        .filter(|line| !line.starts_with("KOHYA_ENV:"))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .take(2000)
        .collect();
    if status.is_none() {
        return probe_failure(
            python,
            format!(
                "训练环境检查超过 {} 秒未返回，请确认该 Python 的 torch 能正常导入。",
                PROBE_TIMEOUT.as_secs()
            ),
            None,
            stderr,
        );
    }
    let payload = raw
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix("KOHYA_ENV:"))
        .and_then(|json_line| serde_json::from_str::<Value>(json_line).ok());
    let exit_code = status.and_then(|value| value.code());
    let Some(payload) = payload else {
        return probe_failure(
            python,
            "训练环境检查没有返回结果。".to_string(),
            exit_code,
            stderr,
        );
    };
    let text = |key: &str| payload.get(key).and_then(Value::as_str).map(str::to_string);
    KohyaEnvironment {
        executable: text("executable").unwrap_or_else(|| python.to_string_lossy().into_owned()),
        python_version: text("python_version"),
        torch_version: text("torch_version"),
        cuda_version: text("cuda_version"),
        cuda_available: payload
            .get("cuda_available")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        device_name: text("device_name"),
        xformers_version: text("xformers_version"),
        bitsandbytes_version: text("bitsandbytes_version"),
        error: text("error"),
        exit_code,
        stderr,
    }
}

#[tauri::command]
pub async fn kohya_start_training(
    app: AppHandle,
    request: crate::kohya::TrainingRequest,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    if let Err(message) = request.validate() {
        return failure(
            "kohya.train",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_TRAINING_REQUEST",
                &message,
                None,
                false,
            ),
        );
    }
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "kohya.train",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = match resolve_paths(&settings.lora_trainer_root, &settings.lora_trainer_python) {
        Ok(paths) => paths,
        Err(error) => {
            return failure(
                "kohya.train",
                request_id.clone(),
                core_error(&request_id, "KOHYA_NOT_CONFIGURED", &error, None, false),
            )
        }
    };
    let script_name = request.trainer.script();
    if !paths.trainers.iter().any(|name| name == script_name) {
        return failure(
            "kohya.train",
            request_id.clone(),
            core_error(
                &request_id,
                "KOHYA_TRAINER_MISSING",
                &format!(
                    "{} 下没有找到 {}，无法按 {} 训练。",
                    paths.sd_scripts.display(),
                    script_name,
                    request.trainer.label()
                ),
                None,
                false,
            ),
        );
    }
    let script = paths.sd_scripts.join(script_name);
    // Text-encoder caching needs a dataset config without caption shuffling;
    // the adjusted copy stays next to the original so relative paths keep
    // working.
    let mut request = request;
    let mut dataset_config = request.dataset_config.trim().to_string();
    if request.cache_text_encoder_outputs {
        match std::fs::read_to_string(&dataset_config) {
            Ok(text) => {
                if let Some(rewritten) = crate::kohya::disable_caption_shuffle(&text) {
                    let original = PathBuf::from(&dataset_config);
                    let adjusted = original.with_file_name(format!(
                        "{}.cache-te.toml",
                        original
                            .file_stem()
                            .map(|stem| stem.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "dataset".to_string())
                    ));
                    if let Err(error) = std::fs::write(&adjusted, rewritten) {
                        return failure(
                            "kohya.train",
                            request_id.clone(),
                            core_error(
                                &request_id,
                                "KOHYA_CONFIG_WRITE_FAILED",
                                &format!("无法写入关闭 shuffle_caption 的训练配置：{error}"),
                                None,
                                false,
                            ),
                        );
                    }
                    dataset_config = adjusted.to_string_lossy().into_owned();
                }
            }
            Err(error) => {
                return failure(
                    "kohya.train",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "KOHYA_CONFIG_READ_FAILED",
                        &format!("无法读取训练集配置 {dataset_config}：{error}"),
                        None,
                        false,
                    ),
                )
            }
        }
    }
    request.dataset_config = dataset_config.clone();
    let output_dir = request.output_dir.trim().to_string();
    let logging_dir = PathBuf::from(&output_dir).join("logs");
    let arguments = build_arguments(&request, &logging_dir);
    let log_name = if request.output_name.trim().is_empty() {
        "lora".to_string()
    } else {
        request.output_name.trim().to_string()
    };
    let log_path = PathBuf::from(&output_dir).join(format!("{log_name}-training.log"));

    let started = match app.state::<AppState>().kohya.lock() {
        Ok(mut manager) => manager.start(
            Arc::new(app.clone()),
            &paths.python,
            &script,
            &arguments,
            log_path.clone(),
            &output_dir,
            request.output_name.trim(),
            request.base_model.trim(),
            request.dataset_config.trim(),
            settings.background_priority != "normal",
        ),
        Err(_) => Err("训练状态锁暂时不可用。".to_string()),
    };
    if let Err(message) = started {
        return failure(
            "kohya.train",
            request_id.clone(),
            core_error(&request_id, "KOHYA_START_FAILED", &message, None, false),
        );
    }

    spawn_run_watcher(Arc::clone(&app.state::<AppState>().kohya), app.clone());

    success(
        "kohya.train",
        request_id,
        json!({
            "script": script.to_string_lossy(),
            "python": paths.python.to_string_lossy(),
            "log_path": log_path.to_string_lossy(),
            "arguments": arguments,
        }),
    )
}

/// Emits one `kohya://finished` event when the watched run ends.
fn spawn_run_watcher(
    manager: Arc<std::sync::Mutex<crate::kohya_runner::KohyaManager>>,
    app: AppHandle,
) {
    tauri::async_runtime::spawn(async move {
        let mut saw_running = false;
        loop {
            tokio::time::sleep(WATCH_INTERVAL).await;
            let Ok(mut manager) = manager.lock() else {
                break;
            };
            if manager.is_running() {
                saw_running = true;
                continue;
            }
            if let Some(snapshot) = manager.poll_exit() {
                drop(manager);
                let _ = app.emit(FINISHED_EVENT, &snapshot);
                break;
            }
            if saw_running {
                break;
            }
        }
    });
}

#[tauri::command]
pub fn kohya_training_status(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<KohyaTrainingSnapshot> {
    let request_id = normalize_request_id(request_id);
    let state = app.state::<AppState>();
    let result = match state.kohya.lock() {
        Ok(mut manager) => success("kohya.status.run", request_id, manager.snapshot()),
        Err(_) => failure(
            "kohya.status.run",
            request_id.clone(),
            core_error(
                &request_id,
                "KOHYA_STATE_LOCK_FAILED",
                "训练状态锁暂时不可用。",
                None,
                true,
            ),
        ),
    };
    result
}

#[tauri::command]
pub fn kohya_stop_training(app: AppHandle, request_id: Option<String>) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let state = app.state::<AppState>();
    let stopped = match state.kohya.lock() {
        Ok(mut manager) => manager.stop(),
        Err(_) => Err("训练状态锁暂时不可用。".to_string()),
    };
    match stopped {
        Ok(stopped) => success("kohya.stop", request_id, json!({ "stopped": stopped })),
        Err(message) => failure(
            "kohya.stop",
            request_id.clone(),
            core_error(&request_id, "KOHYA_STOP_FAILED", &message, None, false),
        ),
    }
}

/// Copy a finished LoRA into the configured ComfyUI `models/loras` folder.
#[tauri::command]
pub fn kohya_install_lora(
    app: AppHandle,
    source: String,
    request_id: Option<String>,
) -> IpcEnvelope<Value> {
    let request_id = normalize_request_id(request_id);
    let source_path = PathBuf::from(source.trim());
    if !source_path.is_file() {
        return failure(
            "kohya.install",
            request_id.clone(),
            core_error(
                &request_id,
                "LORA_NOT_FOUND",
                &format!("找不到 LoRA 文件：{}", source_path.display()),
                None,
                false,
            ),
        );
    }
    let Some(file_name) = source_path.file_name().map(|name| name.to_owned()) else {
        return failure(
            "kohya.install",
            request_id.clone(),
            core_error(
                &request_id,
                "LORA_NOT_FOUND",
                "无法解析 LoRA 文件名。",
                None,
                false,
            ),
        );
    };
    let settings = match load_settings(&app) {
        Ok(settings) => settings,
        Err(error) => {
            return failure(
                "kohya.install",
                request_id.clone(),
                core_error(&request_id, "SETTINGS_READ_FAILED", &error, None, true),
            )
        }
    };
    let paths = match crate::comfy::resolve_paths(
        &settings.comfy_root,
        &settings.comfy_output_dir,
        settings.comfy_port,
    ) {
        Ok(paths) => paths,
        Err(error) => {
            return failure(
                "kohya.install",
                request_id.clone(),
                core_error(
                    &request_id,
                    "COMFY_NOT_CONFIGURED",
                    &format!("需要先配置 ComfyUI 才能安装 LoRA：{error}"),
                    None,
                    false,
                ),
            )
        }
    };
    let loras_dir = PathBuf::from(&paths.loras_dir);
    if let Err(error) = fs::create_dir_all(&loras_dir) {
        return failure(
            "kohya.install",
            request_id.clone(),
            core_error(
                &request_id,
                "COMFY_LORA_DIR_FAILED",
                &format!("无法创建 {}: {error}", loras_dir.display()),
                None,
                false,
            ),
        );
    }
    let destination = loras_dir.join(&file_name);
    let replaced = destination.is_file();
    if let Err(error) = fs::copy(&source_path, &destination) {
        return failure(
            "kohya.install",
            request_id.clone(),
            core_error(
                &request_id,
                "LORA_COPY_FAILED",
                &format!("复制 LoRA 失败：{error}"),
                None,
                false,
            ),
        );
    }
    success(
        "kohya.install",
        request_id,
        json!({
            "destination": destination.to_string_lossy(),
            "replaced": replaced,
        }),
    )
}
