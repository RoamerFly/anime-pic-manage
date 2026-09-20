//! Integration with a user-installed ComfyUI (kept external on purpose).
//!
//! ComfyUI is GPL-3.0 while this project ships under MIT, so the application
//! never bundles or copies it: it only detects an existing installation,
//! optionally starts it, and drives it over its local HTTP API.

pub mod client;
pub mod manager;
pub mod types;
pub mod workflow;

#[allow(unused_imports)]
pub use client::ComfyClient;
#[allow(unused_imports)]
pub use manager::{describe_device, empty_status, read_version, resolve_paths, ComfyManager};
#[allow(unused_imports)]
pub use types::{
    ComfyError, ComfyGenerateRequest, ComfyGenerateResult, ComfyGeneratedImage, ComfyLoraSelection,
    ComfyModelList, ComfyPaths, ComfyProgress, ComfyStatus, DEFAULT_COMFY_PORT, MIN_COMFY_PORT,
};
#[allow(unused_imports)]
pub use workflow::{WorkflowTemplate, TEMPLATE_VERSION};
