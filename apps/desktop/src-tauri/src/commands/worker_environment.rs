//! Install the heavyweight AI environment independently from the desktop app.
//!
//! Release application packages stay small. The matching CPU/GPU Worker
//! archive is downloaded once, verified structurally, staged on the same
//! volume, and then switched into `app\runtime` / `app\env` with rollback.

use crate::app_settings::{load_app_settings, proxy_url};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::portable::PortableLayout;
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager};

const RELEASE_ROOT: &str = "https://github.com/RoamerFly/anime-pic-manage/releases/download";
const PROGRESS_EVENT: &str = "worker-environment://install";
static INSTALLING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEnvironmentManifest {
    version: String,
    flavor: String,
    asset_name: String,
    installed_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerEnvironmentStatus {
    installed: bool,
    version: Option<String>,
    flavor: String,
    directory: String,
    asset_name: String,
    message: String,
}

struct InstallGuard;

impl Drop for InstallGuard {
    fn drop(&mut self) {
        INSTALLING.store(false, Ordering::Release);
    }
}

fn package_root(state: &AppState) -> PathBuf {
    PortableLayout::from_data_dir(Path::new(&state.data_dir)).root
}

fn build_flavor(root: &Path) -> String {
    fs::read_to_string(root.join("BUILD_FLAVOR.txt"))
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| matches!(value.as_str(), "cpu" | "gpu"))
        .unwrap_or_else(|| "cpu".to_string())
}

fn asset_name(version: &str, flavor: &str) -> String {
    format!("AnimePicManage-{version}-windows-x64-{flavor}-runtime.zip")
}

fn environment_ready(app_dir: &Path) -> bool {
    app_dir.join("runtime").join("ai-worker.exe").is_file()
        && app_dir
            .join("env")
            .join("Scripts")
            .join("python.exe")
            .is_file()
}

fn read_manifest(app_dir: &Path) -> Option<WorkerEnvironmentManifest> {
    serde_json::from_slice(&fs::read(app_dir.join("environment-runtime.json")).ok()?).ok()
}

fn status_for_root(root: &Path) -> WorkerEnvironmentStatus {
    let app_dir = root.join("app");
    let flavor = build_flavor(root);
    let version = env!("CARGO_PKG_VERSION");
    let manifest = read_manifest(&app_dir);
    let installed = environment_ready(&app_dir);
    WorkerEnvironmentStatus {
        installed,
        version: manifest.as_ref().map(|item| item.version.clone()),
        flavor: flavor.clone(),
        directory: app_dir.to_string_lossy().into_owned(),
        asset_name: asset_name(version, &flavor),
        message: if installed {
            "AI 运行环境已独立安装，应用更新会继续复用。".to_string()
        } else {
            "尚未安装 AI 运行环境；首次准备后，后续应用更新无需重复下载。".to_string()
        },
    }
}

#[tauri::command]
pub fn worker_environment_status(
    state: tauri::State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerEnvironmentStatus> {
    let request_id = normalize_request_id(request_id);
    success(
        "worker.environment.status",
        request_id,
        status_for_root(&package_root(&state)),
    )
}

fn emit(app: &AppHandle, phase: &str, current: u64, total: u64, message: &str) {
    let _ = app.emit(
        PROGRESS_EVENT,
        json!({ "phase": phase, "current": current, "total": total, "message": message }),
    );
}

fn extract_archive(archive_path: &Path, staging: &Path) -> Result<(), String> {
    let archive_file = fs::File::open(archive_path).map_err(|error| error.to_string())?;
    let mut archive = zip::ZipArchive::new(archive_file)
        .map_err(|error| format!("运行环境压缩包无效：{error}"))?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|error| error.to_string())?;
        let relative = entry
            .enclosed_name()
            .ok_or_else(|| format!("运行环境包含不安全路径：{}", entry.name()))?;
        let first = relative
            .components()
            .next()
            .map(|value| value.as_os_str().to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if !matches!(first.as_str(), "runtime" | "env") {
            return Err(format!("运行环境包含未知顶层目录：{}", relative.display()));
        }
        let destination = staging.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&destination).map_err(|error| error.to_string())?;
            continue;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut output = fs::File::create(&destination).map_err(|error| error.to_string())?;
        std::io::copy(&mut entry, &mut output).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn restore_backup(app_dir: &Path, backup: &Path) {
    for name in ["runtime", "env"] {
        let target = app_dir.join(name);
        let previous = backup.join(name);
        if target.exists() {
            let _ = fs::remove_dir_all(&target);
        }
        if previous.exists() {
            let _ = fs::rename(previous, target);
        }
    }
}

fn commit_environment(staging: &Path, app_dir: &Path) -> Result<(), String> {
    if !environment_ready(staging) {
        return Err("运行环境校验失败：缺少 Worker 或 Python。".to_string());
    }
    let backup = app_dir.join(format!(".environment-old-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&backup).map_err(|error| error.to_string())?;
    for name in ["runtime", "env"] {
        let target = app_dir.join(name);
        if target.exists() {
            if let Err(error) = fs::rename(&target, backup.join(name)) {
                restore_backup(app_dir, &backup);
                return Err(format!("备份旧{name}失败：{error}"));
            }
        }
    }
    for name in ["runtime", "env"] {
        if let Err(error) = fs::rename(staging.join(name), app_dir.join(name)) {
            restore_backup(app_dir, &backup);
            return Err(format!("启用新{name}失败：{error}"));
        }
    }
    let _ = fs::remove_dir_all(backup);
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[tauri::command]
pub async fn install_worker_environment(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<WorkerEnvironmentStatus> {
    let request_id = normalize_request_id(request_id);
    if INSTALLING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return failure(
            "worker.environment.install",
            request_id.clone(),
            core_error(
                &request_id,
                "ENVIRONMENT_BUSY",
                "AI 运行环境正在准备中。",
                None,
                true,
            ),
        );
    }
    let _guard = InstallGuard;
    let state = app.state::<AppState>();
    let root = package_root(&state);
    let app_dir = root.join("app");
    let temp_dir = root.join("temp").join("environment-download");
    let flavor = build_flavor(&root);
    let version = env!("CARGO_PKG_VERSION");
    let asset = asset_name(version, &flavor);
    let url = format!("{RELEASE_ROOT}/v{version}/{asset}");
    let proxy = state
        .database
        .lock()
        .ok()
        .and_then(|database| load_app_settings(&database).ok())
        .and_then(|settings| proxy_url(&settings.network_proxy));
    let mut builder = reqwest::Client::builder().user_agent("anime-pic-manage");
    if let Some(proxy) = proxy {
        match reqwest::Proxy::all(proxy) {
            Ok(proxy) => builder = builder.proxy(proxy),
            Err(error) => {
                return failure(
                    "worker.environment.install",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "ENVIRONMENT_PROXY_FAILED",
                        &error.to_string(),
                        None,
                        false,
                    ),
                )
            }
        }
    }
    let result: Result<(), String> = async {
        fs::create_dir_all(&temp_dir).map_err(|error| error.to_string())?;
        fs::create_dir_all(&app_dir).map_err(|error| error.to_string())?;
        if let Ok(mut worker) = state.worker.lock() {
            worker.stop();
        }
        emit(&app, "downloading", 0, 0, "正在下载独立 AI 运行环境…");
        let client = builder.build().map_err(|error| error.to_string())?;
        let checksum = client
            .get(format!("{url}.sha256"))
            .send()
            .await
            .map_err(|error| format!("下载运行环境校验文件失败：{error}"))?;
        if !checksum.status().is_success() {
            return Err(format!(
                "下载运行环境校验文件返回 HTTP {}",
                checksum.status()
            ));
        }
        let expected_sha256 = checksum
            .text()
            .await
            .map_err(|error| error.to_string())?
            .split_whitespace()
            .next()
            .map(str::to_ascii_lowercase)
            .filter(|value| value.len() == 64)
            .ok_or_else(|| "运行环境 SHA-256 校验文件无效。".to_string())?;
        let archive_path = temp_dir.join(&asset);
        let existing = fs::metadata(&archive_path)
            .map(|item| item.len())
            .unwrap_or(0);
        let mut request = client.get(&url);
        if existing > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
        }
        let mut response = request
            .send()
            .await
            .map_err(|error| format!("下载运行环境失败：{error}"))?;
        if !response.status().is_success() {
            return Err(format!("下载运行环境返回 HTTP {}", response.status()));
        }
        let resumed = existing > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        let offset = if resumed { existing } else { 0 };
        let total = offset + response.content_length().unwrap_or(0);
        let mut archive_file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&archive_path)
            .await
            .map_err(|error| error.to_string())?;
        let mut written = offset;
        while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
            tokio::io::AsyncWriteExt::write_all(&mut archive_file, &chunk)
                .await
                .map_err(|error| error.to_string())?;
            written += chunk.len() as u64;
            emit(
                &app,
                "downloading",
                written,
                total,
                "正在下载独立 AI 运行环境…",
            );
        }
        tokio::io::AsyncWriteExt::flush(&mut archive_file)
            .await
            .map_err(|error| error.to_string())?;
        drop(archive_file);
        let archive_for_hash = archive_path.clone();
        let actual_sha256 =
            tauri::async_runtime::spawn_blocking(move || sha256_file(&archive_for_hash))
                .await
                .map_err(|error| error.to_string())??;
        if actual_sha256 != expected_sha256 {
            let _ = fs::remove_file(&archive_path);
            return Err("AI 运行环境 SHA-256 校验失败，已删除损坏下载。".to_string());
        }
        emit(
            &app,
            "extracting",
            written,
            total,
            "正在校验并安装 AI 运行环境…",
        );
        let staging = app_dir.join(format!(".environment-install-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&staging).map_err(|error| error.to_string())?;
        let archive_for_task = archive_path.clone();
        let staging_for_task = staging.clone();
        let extracted = tauri::async_runtime::spawn_blocking(move || {
            extract_archive(&archive_for_task, &staging_for_task)
        })
        .await
        .map_err(|error| error.to_string())?;
        if let Err(error) = extracted {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        if let Err(error) = commit_environment(&staging, &app_dir) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        let _ = fs::remove_dir_all(&staging);
        let _ = fs::remove_file(&archive_path);
        let manifest = WorkerEnvironmentManifest {
            version: version.to_string(),
            flavor: flavor.clone(),
            asset_name: asset.clone(),
            installed_at: crate::result_store::now_rfc3339(),
        };
        fs::write(
            app_dir.join("environment-runtime.json"),
            serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        emit(&app, "completed", total, total, "AI 运行环境已就绪。");
        Ok(())
    }
    .await;

    match result {
        Ok(()) => success(
            "worker.environment.install",
            request_id,
            status_for_root(&root),
        ),
        Err(error) => failure(
            "worker.environment.install",
            request_id.clone(),
            core_error(
                &request_id,
                "ENVIRONMENT_INSTALL_FAILED",
                "无法准备 AI 运行环境。",
                Some(error),
                true,
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_asset_is_independent_from_the_application_package() {
        assert_eq!(
            asset_name("1.2.3", "gpu"),
            "AnimePicManage-1.2.3-windows-x64-gpu-runtime.zip"
        );
    }
}
