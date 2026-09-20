pub mod discovery;
pub mod manager;
pub mod process;
pub mod protocol;
pub mod types;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use discovery::{resolve_worker_candidate, resolve_worker_candidate_for_mode};
pub use manager::WorkerManager;
#[allow(unused_imports)]
pub use protocol::{parse_health_response, parse_worker_response};
#[allow(unused_imports)]
pub use types::{
    WorkerCapabilitiesInfo, WorkerCapabilityError, WorkerComputeCapability, WorkerComputeDevice,
    WorkerCudaRuntime, WorkerDependencyCapability, WorkerFeatureCapability, WorkerHealthInfo,
    WorkerLaunchSpec, WorkerRuntimeError, WorkerRuntimeMode, REQUIRED_CAPABILITY_FEATURES,
    WORKER_SCHEMA_VERSION,
};
