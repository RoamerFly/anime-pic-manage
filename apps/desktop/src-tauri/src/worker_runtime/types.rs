use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const WORKER_SCHEMA_VERSION: &str = "1.0";
pub const REQUIRED_CAPABILITY_FEATURES: [&str; 3] = [
    "imgutils.preprocess.pillow",
    "imgutils.generic.yolo",
    "imgutils.data",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerHealthInfo {
    pub status: String,
    pub version: String,
    pub model_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerCapabilityError {
    pub code: String,
    pub message: String,
    pub detail: Option<String>,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerDependencyCapability {
    pub distribution: String,
    pub version: Option<String>,
    pub status: String,
    pub error: Option<WorkerCapabilityError>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerFeatureCapability {
    pub module: String,
    pub status: String,
    pub error: Option<WorkerCapabilityError>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerCapabilitiesInfo {
    pub status: String,
    pub ready: bool,
    pub dghs_imgutils: WorkerDependencyCapability,
    pub onnxruntime: WorkerDependencyCapability,
    pub compute: Option<WorkerComputeCapability>,
    pub features: BTreeMap<String, WorkerFeatureCapability>,
    pub errors: Vec<WorkerCapabilityError>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerComputeCapability {
    pub distribution: Option<String>,
    pub available_providers: Vec<String>,
    pub cuda_available: bool,
    /// True only when the CUDA provider is advertised *and* the CUDA/cuDNN
    /// runtime libraries on this machine can actually be loaded.
    pub cuda_usable: bool,
    pub cuda_runtime: Option<WorkerCudaRuntime>,
    pub error: Option<WorkerCapabilityError>,
}

/// Which CUDA/cuDNN libraries are missing, so the settings page can say why a
/// CUDA-capable build still runs on CPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerCudaRuntime {
    pub status: String,
    pub missing: Vec<String>,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum WorkerRuntimeError {
    #[error("未找到本地 AI Worker（开发环境需要 apps/ai-worker/run_worker.py，发布环境需要 ai-worker 可执行文件）")]
    NotFound,
    #[error(
        "已找到 Worker 源码，但本机未安装 uv；请安装 uv 后重试，或提供已打包的 Worker 可执行文件"
    )]
    UvMissing,
    #[error("启动 AI Worker 失败: {0}")]
    Spawn(String),
    #[error("AI Worker 通信失败: {0}")]
    Io(String),
    #[error("AI Worker 健康响应无效: {0}")]
    Protocol(String),
    #[error("AI Worker 返回错误 [{code}]: {message}")]
    WorkerResponse { code: String, message: String },
}

impl WorkerRuntimeError {
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::NotFound | Self::UvMissing)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WorkerRuntimeMode {
    Executable,
    #[default]
    EmbeddedEnv,
}

impl WorkerRuntimeMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "executable" | "exe" => Some(Self::Executable),
            "embedded_env" | "env" => Some(Self::EmbeddedEnv),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Executable => "executable",
            Self::EmbeddedEnv => "embedded_env",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Executable => "独立 EXE",
            Self::EmbeddedEnv => "内置 ENV",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerComputeDevice {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl WorkerComputeDevice {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" | "gpu" => Some(Self::Cuda),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "自动（优先显卡）",
            Self::Cpu => "仅 CPU",
            Self::Cuda => "NVIDIA CUDA",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerLaunchSpec {
    Executable {
        executable: PathBuf,
        working_dir: PathBuf,
        models_root: PathBuf,
    },
    PythonScript {
        interpreter: PathBuf,
        project_dir: PathBuf,
        script: PathBuf,
        models_root: PathBuf,
    },
    UvScript {
        project_dir: PathBuf,
        script: PathBuf,
        models_root: PathBuf,
    },
}

impl WorkerLaunchSpec {
    pub fn working_dir(&self) -> &Path {
        match self {
            Self::Executable { working_dir, .. } => working_dir,
            Self::PythonScript { project_dir, .. } => project_dir,
            Self::UvScript { project_dir, .. } => project_dir,
        }
    }

    pub fn models_root(&self) -> &Path {
        match self {
            Self::Executable { models_root, .. }
            | Self::PythonScript { models_root, .. }
            | Self::UvScript { models_root, .. } => models_root,
        }
    }
}
