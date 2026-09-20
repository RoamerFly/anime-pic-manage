//! Locate, start and stop the user's own ComfyUI installation.
//!
//! The application never bundles ComfyUI (GPL-3.0): it only drives an existing
//! install, usually a portable package such as `ComfyUI-aki-v3` that contains
//! `ComfyUI/` and `python/` next to each other.

use super::types::{ComfyError, ComfyPaths, ComfyStatus};
use crate::worker_runtime::process::configure_background_command_with;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub struct ComfyManager {
    child: Option<Child>,
    /// True when we started the process, so we may stop it again.
    owned: bool,
}

impl ComfyManager {
    pub fn new() -> Self {
        Self {
            child: None,
            owned: false,
        }
    }

    pub fn is_owned(&self) -> bool {
        self.owned
    }

    /// True while the process we started is still alive.
    pub fn owns_running_process(&mut self) -> bool {
        if !self.owned {
            return false;
        }
        match self.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                _ => {
                    self.clear();
                    false
                }
            },
            None => false,
        }
    }

    pub fn start(
        &mut self,
        paths: &ComfyPaths,
        port: u16,
        low_vram: bool,
        log_dir: &Path,
        below_normal_priority: bool,
    ) -> Result<PathBuf, ComfyError> {
        if self.owns_running_process() {
            return Err(ComfyError::Response("ComfyUI 已由本应用启动".to_string()));
        }
        fs::create_dir_all(log_dir)
            .map_err(|error| ComfyError::Io(format!("{}: {error}", log_dir.display())))?;
        let log_path = log_dir.join("comfyui.log");
        let stdout = fs::File::create(&log_path)
            .map_err(|error| ComfyError::Io(format!("{}: {error}", log_path.display())))?;
        let stderr = stdout
            .try_clone()
            .map_err(|error| ComfyError::Io(error.to_string()))?;

        let mut command = Command::new(&paths.python);
        command
            .arg(&paths.main_script)
            .arg("--listen")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--preview-method")
            .arg("none")
            .current_dir(&paths.app_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        if low_vram {
            command.arg("--lowvram");
        }
        configure_background_command_with(&mut command, below_normal_priority);

        let child = command
            .spawn()
            .map_err(|error| ComfyError::Io(format!("启动 ComfyUI 失败: {error}")))?;
        self.child = Some(child);
        self.owned = true;
        Ok(log_path)
    }

    /// Stop the process only when this application started it.
    pub fn stop(&mut self) -> Result<bool, ComfyError> {
        if !self.owned {
            return Ok(false);
        }
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.clear();
        Ok(true)
    }

    fn clear(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.owned = false;
    }
}

impl Drop for ComfyManager {
    fn drop(&mut self) {
        // Only reap what we started; a user-launched ComfyUI keeps running.
        self.clear();
    }
}

/// Resolve the ComfyUI layout from the configured root.
pub fn resolve_paths(
    root: &str,
    output_override: &str,
    _port: u16,
) -> Result<ComfyPaths, ComfyError> {
    if root.trim().is_empty() {
        return Err(ComfyError::NotConfigured);
    }
    let root_path = PathBuf::from(root.trim());
    let (root_dir, app_dir, python) = if root_path.join("ComfyUI").join("main.py").is_file() {
        let python = root_path.join("python").join("python.exe");
        (root_path.clone(), root_path.join("ComfyUI"), python)
    } else if root_path.join("main.py").is_file() {
        let parent = root_path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| ComfyError::InvalidRoot(root.to_string()))?;
        let app_dir = root_path.clone();
        let python = {
            let candidate = parent.join("python").join("python.exe");
            if candidate.is_file() {
                candidate
            } else {
                parent.join("venv").join("Scripts").join("python.exe")
            }
        };
        (parent, app_dir, python)
    } else {
        return Err(ComfyError::InvalidRoot(format!(
            "{} 下没有找到 ComfyUI\\main.py",
            root_path.display()
        )));
    };
    if !python.is_file() {
        return Err(ComfyError::InvalidRoot(format!(
            "{} 下没有找到 python.exe",
            python.display()
        )));
    }
    let models_dir = app_dir.join("models");
    let output_dir = if output_override.trim().is_empty() {
        root_dir.join("Images").join("generated")
    } else {
        PathBuf::from(output_override.trim())
    };
    Ok(ComfyPaths {
        root: root_dir.to_string_lossy().into_owned(),
        app_dir: app_dir.to_string_lossy().into_owned(),
        python: python.to_string_lossy().into_owned(),
        main_script: app_dir.join("main.py").to_string_lossy().into_owned(),
        models_dir: models_dir.to_string_lossy().into_owned(),
        loras_dir: models_dir.join("loras").to_string_lossy().into_owned(),
        output_dir: output_dir.to_string_lossy().into_owned(),
    })
}

/// Read `comfyui_version.py` without importing it.
pub fn read_version(app_dir: &Path) -> Option<String> {
    let raw = fs::read_to_string(app_dir.join("comfyui_version.py")).ok()?;
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("__version__") {
            let value = rest.trim_start_matches(['=', ' ']).trim();
            return Some(value.trim_matches(['"', '\'']).to_string());
        }
    }
    None
}

/// Extract the first device name and its VRAM from `/system_stats`.
pub fn describe_device(system_stats: &Value) -> (Option<String>, Option<f64>) {
    let Some(device) = system_stats
        .get("devices")
        .and_then(Value::as_array)
        .and_then(|devices| devices.first())
    else {
        return (None, None);
    };
    let name = device
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_string);
    let vram_gb = device
        .get("vram_total")
        .and_then(Value::as_f64)
        .map(|bytes| (bytes / 1024.0 / 1024.0 / 1024.0 * 10.0).round() / 10.0);
    (name, vram_gb)
}

pub fn empty_status(port: u16) -> ComfyStatus {
    ComfyStatus {
        port,
        ..ComfyStatus::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fake_install(root: &Path) {
        fs::create_dir_all(root.join("ComfyUI")).unwrap();
        fs::create_dir_all(root.join("python")).unwrap();
        fs::write(root.join("ComfyUI").join("main.py"), b"# comfy").unwrap();
        fs::write(root.join("python").join("python.exe"), b"# python").unwrap();
        fs::write(
            root.join("ComfyUI").join("comfyui_version.py"),
            b"# generated\n__version__ = \"0.9.2\"\n",
        )
        .unwrap();
    }

    #[test]
    fn resolves_a_portable_package_layout() {
        let root = std::env::temp_dir().join(format!("comfy-test-{}", uuid::Uuid::new_v4()));
        fake_install(&root);

        let paths = resolve_paths(&root.to_string_lossy(), "", 8188).expect("paths resolve");

        assert!(paths.app_dir.ends_with("ComfyUI"));
        assert!(paths.python.ends_with("python.exe"));
        assert!(paths.models_dir.ends_with("models"));
        assert!(paths.loras_dir.ends_with("loras"));
        assert!(paths.output_dir.ends_with("generated"));
        assert_eq!(
            read_version(Path::new(&paths.app_dir)).as_deref(),
            Some("0.9.2")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_when_pointed_directly_at_the_app_directory() {
        let root = std::env::temp_dir().join(format!("comfy-test-{}", uuid::Uuid::new_v4()));
        fake_install(&root);

        let paths = resolve_paths(&root.join("ComfyUI").to_string_lossy(), "", 8188)
            .expect("paths resolve");

        assert!(paths.app_dir.ends_with("ComfyUI"));
        assert!(paths.python.ends_with("python.exe"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_an_unconfigured_or_invalid_root() {
        assert!(matches!(
            resolve_paths("", "", 8188),
            Err(ComfyError::NotConfigured)
        ));
        assert!(matches!(
            resolve_paths("D:/definitely/missing", "", 8188),
            Err(ComfyError::InvalidRoot(_))
        ));
    }

    #[test]
    fn reads_device_and_vram_from_system_stats() {
        let stats = json!({
            "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3060 Laptop GPU", "vram_total": 6442450944u64}]
        });

        let (name, vram) = describe_device(&stats);

        assert_eq!(
            name.as_deref(),
            Some("cuda:0 NVIDIA GeForce RTX 3060 Laptop GPU")
        );
        assert_eq!(vram, Some(6.0));
    }
}
