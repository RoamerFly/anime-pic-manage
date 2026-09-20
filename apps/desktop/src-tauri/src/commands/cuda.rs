//! Download the CUDA 12 / cuDNN 9 runtime into the portable package.
//!
//! The application never redistributes NVIDIA binaries: it fetches the
//! official `nvidia-*` wheels from PyPI on request, extracts the DLLs into
//! `app\cuda\` and points the inference engine at that folder. Machines that
//! already have CUDA 12 / cuDNN 9 installed keep working without any of this.

use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::portable::PortableLayout;
use crate::state::AppState;
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};

pub const CUDA_PROGRESS_EVENT: &str = "cuda://install";

/// NVIDIA publishes these wheels for exactly this purpose (runtime
/// distribution); cuFFT is a hard dependency of the ONNX Runtime CUDA provider.
const PACKAGES: [(&str, &str); 4] = [
    ("nvidia-cuda-runtime-cu12", "CUDA Runtime 12"),
    ("nvidia-cublas-cu12", "cuBLAS 12"),
    ("nvidia-cufft-cu12", "cuFFT 12"),
    ("nvidia-cudnn-cu12", "cuDNN 9"),
];

#[derive(Debug, Clone, Serialize)]
pub struct CudaRuntimeStatus {
    pub directory: String,
    pub installed: bool,
    pub packages: Vec<String>,
    pub size_mb: f64,
    pub message: String,
}

fn emit(app: &AppHandle, phase: &str, current: usize, total: usize, message: &str) {
    let _ = app.emit(
        CUDA_PROGRESS_EVENT,
        json!({
            "phase": phase,
            "current": current,
            "total": total,
            "message": message,
        }),
    );
}

fn current_exe_dir() -> Result<PathBuf, String> {
    std::env::current_exe()
        .map_err(|error| format!("无法定位程序目录：{error}"))?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "无法定位程序目录".to_string())
}

fn cuda_dir_for(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(exe_dir) = current_exe_dir() {
        return Ok(PortableLayout { root: exe_dir }.cuda_dir());
    }
    let state = app.state::<AppState>();
    Ok(PortableLayout::from_data_dir(Path::new(&state.data_dir)).cuda_dir())
}

fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.metadata().ok())
        .map(|metadata| metadata.len())
        .sum()
}

pub fn runtime_status(app: &AppHandle) -> CudaRuntimeStatus {
    let directory = cuda_dir_for(app).unwrap_or_default();
    let mut packages: Vec<String> = std::fs::read_dir(&directory)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| {
                    entry
                        .path()
                        .extension()
                        .map(|extension| extension.eq_ignore_ascii_case("dll"))
                        .unwrap_or(false)
                })
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    packages.sort();
    let size_mb = directory_size(&directory) as f64 / 1024.0 / 1024.0;
    let installed = !packages.is_empty();
    CudaRuntimeStatus {
        directory: directory.to_string_lossy().into_owned(),
        installed,
        packages,
        size_mb: (size_mb * 10.0).round() / 10.0,
        message: if installed {
            "已下载 CUDA 运行时，推理引擎会优先使用它。".to_string()
        } else {
            "尚未下载 CUDA 运行时；GPU 版目前会回落到 CPU。".to_string()
        },
    }
}

#[tauri::command]
pub fn cuda_runtime_status(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<CudaRuntimeStatus> {
    let request_id = normalize_request_id(request_id);
    success("cuda.status", request_id, runtime_status(&app))
}

/// Resolve the newest Windows wheel of a PyPI package.
async fn resolve_wheel(
    client: &reqwest::Client,
    package: &str,
) -> Result<(String, String), String> {
    let url = format!("https://pypi.org/pypi/{package}/json");
    let payload: serde_json::Value = client
        .get(&url)
        .send()
        .await
        .map_err(|error| format!("无法查询 {package}：{error}"))?
        .json()
        .await
        .map_err(|error| format!("{package} 的版本信息无法解析：{error}"))?;
    let version = payload
        .get("info")
        .and_then(|info| info.get("version"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("{package} 没有可用版本"))?
        .to_string();
    let files = payload
        .get("releases")
        .and_then(|releases| releases.get(&version))
        .and_then(|value| value.as_array())
        .ok_or_else(|| format!("{package} {version} 没有发布文件"))?;
    let wheel = files
        .iter()
        .find(|file| {
            file.get("filename")
                .and_then(|value| value.as_str())
                .map(|name| name.ends_with("win_amd64.whl"))
                .unwrap_or(false)
        })
        .and_then(|file| file.get("url").and_then(|value| value.as_str()))
        .ok_or_else(|| format!("{package} {version} 没有 Windows wheel"))?;
    Ok((version, wheel.to_string()))
}

/// True for the DLLs inside an `nvidia-*` wheel that ONNX Runtime loads.
fn is_runtime_dll(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    lowered.ends_with(".dll") && lowered.contains("/bin/")
}

async fn download_wheel(client: &reqwest::Client, url: &str, target: &Path) -> Result<u64, String> {
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("下载失败：{error}"))?;
    if !response.status().is_success() {
        return Err(format!("下载返回 HTTP {}", response.status()));
    }
    let mut file = tokio::fs::File::create(target)
        .await
        .map_err(|error| format!("{}: {error}", target.display()))?;
    let mut written: u64 = 0;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("下载中断：{error}"))?
    {
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|error| format!("写入失败：{error}"))?;
        written += chunk.len() as u64;
    }
    tokio::io::AsyncWriteExt::flush(&mut file)
        .await
        .map_err(|error| format!("写入失败：{error}"))?;
    Ok(written)
}

/// Extract every `nvidia/**/bin/*.dll` into `target`, flattened by file name.
fn extract_runtime_dlls(wheel: &Path, target: &Path) -> Result<Vec<String>, String> {
    let file =
        std::fs::File::open(wheel).map_err(|error| format!("{}: {error}", wheel.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| format!("{} 不是有效的 wheel：{error}", wheel.display()))?;
    let mut extracted = Vec::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("读取 wheel 条目失败：{error}"))?;
        let name = entry.name().replace('\\', "/");
        if !is_runtime_dll(&name) {
            continue;
        }
        let Some(file_name) = name.rsplit('/').next() else {
            continue;
        };
        let destination = target.join(file_name);
        let mut output = std::fs::File::create(&destination)
            .map_err(|error| format!("{}: {error}", destination.display()))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|error| format!("解压 {file_name} 失败：{error}"))?;
        output
            .flush()
            .map_err(|error| format!("写入 {file_name} 失败：{error}"))?;
        extracted.push(file_name.to_string());
    }
    Ok(extracted)
}

#[tauri::command]
pub async fn install_cuda_runtime(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<CudaRuntimeStatus> {
    let request_id = normalize_request_id(request_id);
    let target = match cuda_dir_for(&app) {
        Ok(target) => target,
        Err(message) => {
            return failure(
                "cuda.install",
                request_id.clone(),
                core_error(&request_id, "CUDA_DIR_FAILED", &message, None, false),
            )
        }
    };
    if let Err(error) = std::fs::create_dir_all(&target) {
        return failure(
            "cuda.install",
            request_id.clone(),
            core_error(
                &request_id,
                "CUDA_DIR_FAILED",
                &format!("无法创建 {}：{error}", target.display()),
                None,
                false,
            ),
        );
    }
    let staging = target.join("_download");
    if let Err(error) = std::fs::create_dir_all(&staging) {
        return failure(
            "cuda.install",
            request_id.clone(),
            core_error(
                &request_id,
                "CUDA_DIR_FAILED",
                &format!("无法创建临时目录：{error}"),
                None,
                false,
            ),
        );
    }

    let client = match reqwest::Client::builder()
        .user_agent("anime-pic-manage")
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return failure(
                "cuda.install",
                request_id.clone(),
                core_error(
                    &request_id,
                    "HTTP_CLIENT_FAILED",
                    &error.to_string(),
                    None,
                    false,
                ),
            )
        }
    };

    emit(
        &app,
        "resolving",
        0,
        PACKAGES.len(),
        "正在查询 NVIDIA 运行时版本…",
    );
    let mut installed_packages: Vec<String> = Vec::new();
    for (index, (package, label)) in PACKAGES.iter().enumerate() {
        emit(
            &app,
            "downloading",
            index,
            PACKAGES.len(),
            &format!("正在下载 {label}…"),
        );
        let (version, url) = match resolve_wheel(&client, package).await {
            Ok(value) => value,
            Err(message) => {
                return failure(
                    "cuda.install",
                    request_id.clone(),
                    core_error(&request_id, "CUDA_RESOLVE_FAILED", &message, None, true),
                )
            }
        };
        let wheel = staging.join(format!("{package}-{version}.whl"));
        if let Err(message) = download_wheel(&client, &url, &wheel).await {
            return failure(
                "cuda.install",
                request_id.clone(),
                core_error(&request_id, "CUDA_DOWNLOAD_FAILED", &message, None, true),
            );
        }
        emit(
            &app,
            "extracting",
            index,
            PACKAGES.len(),
            &format!("正在解压 {label}…"),
        );
        let extract_target = target.clone();
        let wheel_path = wheel.clone();
        let extracted = tauri::async_runtime::spawn_blocking(move || {
            extract_runtime_dlls(&wheel_path, &extract_target)
        })
        .await;
        match extracted {
            Ok(Ok(files)) => {
                installed_packages.push(format!("{label} {version}（{} 个 DLL）", files.len()));
                let _ = std::fs::remove_file(&wheel);
            }
            Ok(Err(message)) => {
                return failure(
                    "cuda.install",
                    request_id.clone(),
                    core_error(&request_id, "CUDA_EXTRACT_FAILED", &message, None, true),
                )
            }
            Err(error) => {
                return failure(
                    "cuda.install",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "CUDA_EXTRACT_FAILED",
                        &format!("解压任务异常中止：{error}"),
                        None,
                        true,
                    ),
                )
            }
        }
    }
    let _ = std::fs::remove_dir(&staging);

    let manifest = json!({
        "created_at": crate::result_store::now_rfc3339(),
        "directory": target.to_string_lossy(),
        "packages": installed_packages,
    });
    if let Err(error) = std::fs::write(
        target.join("cuda-runtime.json"),
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    ) {
        return failure(
            "cuda.install",
            request_id.clone(),
            core_error(
                &request_id,
                "CUDA_MANIFEST_FAILED",
                &format!("无法写入清单：{error}"),
                None,
                false,
            ),
        );
    }

    // Point the inference engine at the freshly downloaded runtime.
    let directory = target.to_string_lossy().into_owned();
    {
        let state = app.state::<AppState>();
        if let Ok(database) = state.database.lock() {
            let mut settings =
                crate::app_settings::load_app_settings(&database).unwrap_or_default();
            settings.cuda_runtime_dir = directory.clone();
            if let Err(error) = crate::app_settings::save_app_settings(&database, &settings) {
                return failure(
                    "cuda.install",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "SETTINGS_SAVE_FAILED",
                        &format!("运行时已下载，但保存设置失败：{error}"),
                        None,
                        false,
                    ),
                );
            }
        }
        if let Ok(mut worker) = state.worker.lock() {
            worker.set_cuda_runtime_dir(Some(directory.clone()));
        };
    }

    emit(
        &app,
        "completed",
        PACKAGES.len(),
        PACKAGES.len(),
        "CUDA 运行时已就绪。",
    );
    success("cuda.install", request_id, runtime_status(&app))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_runtime_dlls_are_extracted() {
        assert!(is_runtime_dll("nvidia/cudnn/bin/cudnn64_9.dll"));
        assert!(is_runtime_dll("nvidia/cublas/bin/cublasLt64_12.dll"));
        assert!(!is_runtime_dll("nvidia/cudnn/bin/cudnn.lib"));
        assert!(!is_runtime_dll(
            "nvidia_cudnn_cu12-9.0.dist-info/LICENSE.txt"
        ));
        assert!(!is_runtime_dll("nvidia/cudnn/include/cudnn.h"));
    }

    #[test]
    fn extracts_and_flattens_wheel_payload() {
        let root = std::env::temp_dir().join(format!("cuda-wheel-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let wheel = root.join("fake.whl");
        {
            let file = std::fs::File::create(&wheel).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            let options: zip::write::SimpleFileOptions = Default::default();
            writer
                .start_file("nvidia/cudnn/bin/cudnn64_9.dll", options)
                .unwrap();
            writer.write_all(b"# dll").unwrap();
            writer
                .start_file("nvidia/cudnn/bin/cudnn_ops64_9.dll", options)
                .unwrap();
            writer.write_all(b"# dll").unwrap();
            writer
                .start_file("nvidia_cudnn_cu12-9.dist-info/METADATA", options)
                .unwrap();
            writer.write_all(b"metadata").unwrap();
            writer.finish().unwrap();
        }
        let target = root.join("cuda");
        std::fs::create_dir_all(&target).unwrap();

        let mut names = extract_runtime_dlls(&wheel, &target).expect("extraction works");
        names.sort();

        assert_eq!(names, vec!["cudnn64_9.dll", "cudnn_ops64_9.dll"]);
        assert!(target.join("cudnn64_9.dll").is_file());
        assert!(!target.join("METADATA").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
