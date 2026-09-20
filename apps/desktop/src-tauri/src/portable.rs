//! Portable-package layout.
//!
//! A released build keeps only three entries in the root (`启动.bat`,
//! `anime-pic-manage.exe`, `使用说明.txt`) and moves everything a user should
//! not touch into `app\`:
//!
//! ```text
//! AnimePicManage\
//! ├─ 启动.bat            启动器（自检 + 启动）
//! ├─ anime-pic-manage.exe
//! ├─ 使用说明.txt
//! ├─ app\                程序本体：runtime\ env\ cuda\
//! ├─ models\             识别模型
//! ├─ resources\          角色集合与工作流模板
//! ├─ data\               数据库、扫描结果、日志
//! ├─ output\             产出：generated\ loras\ datasets\
//! └─ temp\
//! ```
//!
//! Older flat packages (worker directly in `runtime\`) keep working: the
//! worker discovery search list simply also looks inside `app\`.

use std::path::{Path, PathBuf};

/// Root of the portable package that contains the given executable.
pub fn portable_root(exe_dir: &Path) -> PathBuf {
    exe_dir.to_path_buf()
}

pub fn portable_app_dir(exe_dir: &Path) -> PathBuf {
    portable_root(exe_dir).join("app")
}

pub fn portable_data_dir(exe_dir: &Path) -> PathBuf {
    portable_root(exe_dir).join("data")
}

pub fn portable_temp_dir(exe_dir: &Path) -> PathBuf {
    portable_root(exe_dir).join("temp")
}

pub fn portable_output_dir(exe_dir: &Path) -> PathBuf {
    portable_root(exe_dir).join("output")
}

/// Where a downloaded CUDA 12 / cuDNN 9 runtime lives.
pub fn portable_cuda_dir(exe_dir: &Path) -> PathBuf {
    portable_app_dir(exe_dir).join("cuda")
}

/// Layout derived from an application data directory.
///
/// The data directory is always `<root>\data`, so the portable root is its
/// parent. Development checkouts use `apps\desktop\src-tauri\...\data`, which
/// simply resolves to their own parent and stays harmless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableLayout {
    pub root: PathBuf,
}

impl PortableLayout {
    pub fn from_data_dir(data_dir: &Path) -> Self {
        Self {
            root: data_dir
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| data_dir.to_path_buf()),
        }
    }

    pub fn output_dir(&self) -> PathBuf {
        self.root.join("output")
    }

    /// Exported kohya training sets.
    pub fn dataset_dir(&self) -> PathBuf {
        self.output_dir().join("datasets")
    }

    /// Trained LoRA files.
    pub fn lora_dir(&self) -> PathBuf {
        self.output_dir().join("loras")
    }

    /// Generated images when ComfyUI has no output directory configured.
    pub fn generated_dir(&self) -> PathBuf {
        self.output_dir().join("generated")
    }

    pub fn cuda_dir(&self) -> PathBuf {
        self.root.join("app").join("cuda")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_the_portable_layout_from_the_data_directory() {
        let layout = PortableLayout::from_data_dir(Path::new(r"D:\Apps\AnimePicManage\data"));

        assert_eq!(layout.root, PathBuf::from(r"D:\Apps\AnimePicManage"));
        assert!(layout.dataset_dir().ends_with(r"output\datasets"));
        assert!(layout.lora_dir().ends_with(r"output\loras"));
        assert!(layout.generated_dir().ends_with(r"output\generated"));
        assert!(layout.cuda_dir().ends_with(r"app\cuda"));
    }

    #[test]
    fn a_bare_directory_stays_its_own_root() {
        let layout = PortableLayout::from_data_dir(Path::new("data"));

        assert_eq!(layout.root, PathBuf::from(""));
        assert!(layout.output_dir().ends_with("output"));
    }
}
