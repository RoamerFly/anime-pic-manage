//! Unified application settings commands shared by the settings tabs.

use crate::app_settings::{load_app_settings, save_app_settings, AppSettings};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::state::AppState;
use tauri::State;

#[tauri::command]
pub fn get_app_settings(
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<AppSettings> {
    let request_id = normalize_request_id(request_id);
    let loaded = match state.database.lock() {
        Ok(database) => load_app_settings(&database),
        Err(_) => Err("本地数据库状态锁暂时不可用".to_string()),
    };
    match loaded {
        Ok(settings) => success("settings.get", request_id, settings),
        Err(error) => failure(
            "settings.get",
            request_id.clone(),
            core_error(
                &request_id,
                "APP_SETTINGS_READ_FAILED",
                "无法读取应用设置。",
                Some(error),
                true,
            ),
        ),
    }
}

#[tauri::command]
pub fn update_app_settings(
    state: State<'_, AppState>,
    settings: AppSettings,
    request_id: Option<String>,
) -> IpcEnvelope<AppSettings> {
    let request_id = normalize_request_id(request_id);
    let settings = match settings.validated() {
        Ok(settings) => settings,
        Err(message) => {
            return failure(
                "settings.update",
                request_id.clone(),
                core_error(&request_id, "INVALID_APP_SETTINGS", &message, None, false),
            )
        }
    };

    if let Ok(database) = state.database.lock() {
        if let Err(error) = save_app_settings(&database, &settings) {
            return failure(
                "settings.update",
                request_id.clone(),
                core_error(
                    &request_id,
                    "APP_SETTINGS_SAVE_FAILED",
                    "无法保存应用设置。",
                    Some(error),
                    true,
                ),
            );
        }
    } else {
        return failure(
            "settings.update",
            request_id.clone(),
            core_error(
                &request_id,
                "DATABASE_LOCK_FAILED",
                "本地数据库状态锁暂时不可用。",
                None,
                true,
            ),
        );
    }

    // Recognition threads take effect on the next model session load, which the
    // worker performs lazily whenever the requested threads differ.
    if let Ok(mut worker) = state.worker.lock() {
        worker.set_onnx_threads(settings.recognition_onnx_threads);
        // A changed CUDA runtime folder takes effect on the next Worker start.
        worker.set_cuda_runtime_dir(Some(settings.cuda_runtime_dir.clone()));
        worker.set_below_normal_priority(settings.background_priority != "normal");
    }

    success("settings.update", request_id, settings)
}
