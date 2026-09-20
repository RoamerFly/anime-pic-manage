use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::models::{UpdateInfo, UpdateProgress};
use crate::state::AppState;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_updater::UpdaterExt;

#[tauri::command]
pub async fn check_for_update(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<UpdateInfo> {
    let request_id = normalize_request_id(request_id);
    let current_version = env!("CARGO_PKG_VERSION").to_string();
    let updater = match app.updater() {
        Ok(updater) => updater,
        Err(error) => {
            return success(
                "updater.check",
                request_id,
                UpdateInfo {
                    status: "unconfigured".to_string(),
                    current_version,
                    version: None,
                    date: None,
                    body: None,
                    message: format!("更新服务尚未配置，暂不能检查更新：{error}"),
                },
            )
        }
    };
    match updater.check().await {
        Ok(Some(update)) => success(
            "updater.check",
            request_id,
            UpdateInfo {
                status: "available".to_string(),
                current_version,
                version: Some(update.version),
                date: update.date.map(|date| date.to_string()),
                body: update.body,
                message: "发现可用更新，安装前会再次验证版本。".to_string(),
            },
        ),
        Ok(None) => success(
            "updater.check",
            request_id,
            UpdateInfo {
                status: "up_to_date".to_string(),
                current_version,
                version: None,
                date: None,
                body: None,
                message: "当前已是最新版本。".to_string(),
            },
        ),
        Err(error) => success(
            "updater.check",
            request_id,
            UpdateInfo {
                status: "unavailable".to_string(),
                current_version,
                version: None,
                date: None,
                body: None,
                message: format!("更新服务暂不可用（开发环境可能尚未配置签名端点）：{error}"),
            },
        ),
    }
}

#[tauri::command]
pub async fn install_update(
    app: AppHandle,
    state: State<'_, AppState>,
    expected_version: String,
    request_id: Option<String>,
) -> Result<IpcEnvelope<UpdateInfo>, String> {
    let request_id = normalize_request_id(request_id);
    let current_version = env!("CARGO_PKG_VERSION").to_string();
    let updater = match app.updater() {
        Ok(updater) => updater,
        Err(error) => {
            return Ok(success(
                "updater.install",
                request_id,
                UpdateInfo {
                    status: "unavailable".to_string(),
                    current_version,
                    version: None,
                    date: None,
                    body: None,
                    message: format!("更新服务尚未配置，安装已取消：{error}"),
                },
            ))
        }
    };
    let check = match updater.check().await {
        Ok(check) => check,
        Err(error) => {
            return Ok(success(
                "updater.install",
                request_id,
                UpdateInfo {
                    status: "unavailable".to_string(),
                    current_version,
                    version: None,
                    date: None,
                    body: None,
                    message: format!("无法验证更新包，安装已取消：{error}"),
                },
            ))
        }
    };
    let Some(update) = check else {
        return Ok(success(
            "updater.install",
            request_id,
            UpdateInfo {
                status: "up_to_date".to_string(),
                current_version,
                version: None,
                date: None,
                body: None,
                message: "没有可安装的更新。".to_string(),
            },
        ));
    };
    if update.version != expected_version {
        return Ok(failure(
            "updater.install",
            request_id.clone(),
            core_error(
                &request_id,
                "UPDATE_VERSION_CHANGED",
                "更新版本已变化，请重新检查后再安装。",
                Some(format!(
                    "expected {expected_version}, got {}",
                    update.version
                )),
                true,
            ),
        ));
    }
    state
        .update_control
        .cancel_requested
        .store(false, Ordering::Release);
    let cancellation = Arc::clone(&state.update_control);
    let cancellation_wait = async move {
        loop {
            if cancellation.cancel_requested.load(Ordering::Acquire) {
                break;
            }
            cancellation.notify.notified().await;
        }
    };
    tokio::pin!(cancellation_wait);
    let progress_app = app.clone();
    let install_app = app.clone();
    let download = update.download_and_install(
        move |downloaded, total| {
            let percent = total.map(|total| (downloaded as u64).saturating_mul(100) / total.max(1));
            let _ = progress_app.emit(
                "update-progress",
                UpdateProgress {
                    phase: "downloading".to_string(),
                    downloaded: downloaded as u64,
                    total,
                    percent,
                    message: "正在下载更新…".to_string(),
                },
            );
        },
        move || {
            let _ = install_app.emit(
                "update-progress",
                UpdateProgress {
                    phase: "installing".to_string(),
                    downloaded: 0,
                    total: None,
                    percent: None,
                    message: "正在安装更新，完成后将重启应用…".to_string(),
                },
            );
        },
    );
    tokio::pin!(download);
    let result = tokio::select! {
        result = &mut download => result,
        _ = &mut cancellation_wait => {
            let _ = app.emit("update-progress", UpdateProgress { phase: "cancelled".to_string(), downloaded: 0, total: None, percent: None, message: "已取消更新下载，当前版本未改变。".to_string() });
            return Ok(success("updater.install", request_id, UpdateInfo { status: "unavailable".to_string(), current_version, version: None, date: None, body: None, message: "已取消更新下载，当前版本未改变。".to_string() }));
        }
    };
    match result {
        Ok(()) => {
            app.emit(
                "update-progress",
                UpdateProgress {
                    phase: "complete".to_string(),
                    downloaded: 0,
                    total: None,
                    percent: Some(100),
                    message: "更新完成，正在重启…".to_string(),
                },
            )
            .ok();
            app.restart();
        }
        Err(error) => Ok(failure(
            "updater.install",
            request_id.clone(),
            core_error(
                &request_id,
                "UPDATE_INSTALL_FAILED",
                "更新安装失败，当前版本保持不变。",
                Some(error.to_string()),
                true,
            ),
        )),
    }
}

#[tauri::command]
pub fn cancel_update_download(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<UpdateInfo> {
    let request_id = normalize_request_id(request_id);
    state
        .update_control
        .cancel_requested
        .store(true, Ordering::Release);
    state.update_control.notify.notify_waiters();
    let _ = app.emit(
        "update-progress",
        UpdateProgress {
            phase: "cancelled".to_string(),
            downloaded: 0,
            total: None,
            percent: None,
            message: "已请求取消更新下载。".to_string(),
        },
    );
    success(
        "updater.install",
        request_id,
        UpdateInfo {
            status: "unavailable".to_string(),
            current_version: env!("CARGO_PKG_VERSION").to_string(),
            version: None,
            date: None,
            body: None,
            message: "已取消更新下载，当前版本未改变。".to_string(),
        },
    )
}
