use crate::worker_runtime::types::{
    WorkerCapabilitiesInfo, WorkerCapabilityError, WorkerComputeCapability, WorkerCudaRuntime,
    WorkerDependencyCapability, WorkerFeatureCapability, WorkerHealthInfo, WorkerRuntimeError,
    REQUIRED_CAPABILITY_FEATURES, WORKER_SCHEMA_VERSION,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
struct WorkerResponseEnvelope {
    schema_version: String,
    request_id: String,
    task_id: String,
    message_type: String,
    payload: Value,
    error: Option<WorkerErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct WorkerErrorPayload {
    code: String,
    message: String,
    #[allow(dead_code)]
    detail: Option<String>,
    #[allow(dead_code)]
    retryable: bool,
}

#[derive(Debug, Deserialize)]
struct HealthPayload {
    status: String,
    worker_version: String,
    schema_version: String,
    model_count: u64,
}

#[derive(Debug, Deserialize)]
struct CapabilityErrorPayload {
    code: String,
    message: String,
    detail: Option<String>,
    retryable: bool,
}

#[derive(Debug, Deserialize)]
struct DependencyCapabilityPayload {
    distribution: String,
    version: Option<String>,
    status: String,
    error: Option<CapabilityErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct FeatureCapabilityPayload {
    module: String,
    status: String,
    error: Option<CapabilityErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct CapabilitiesPayload {
    status: String,
    ready: bool,
    dghs_imgutils: DependencyCapabilityPayload,
    onnxruntime: DependencyCapabilityPayload,
    #[serde(default)]
    compute: Option<ComputeCapabilityPayload>,
    features: BTreeMap<String, FeatureCapabilityPayload>,
    errors: Vec<CapabilityErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct ComputeCapabilityPayload {
    distribution: Option<String>,
    #[serde(default)]
    available_providers: Vec<String>,
    #[serde(default)]
    cuda_available: bool,
    #[serde(default)]
    cuda_usable: bool,
    #[serde(default)]
    cuda_runtime: Option<CudaRuntimePayload>,
    error: Option<CapabilityErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct CudaRuntimePayload {
    status: String,
    #[serde(default)]
    missing: Vec<String>,
    #[serde(default)]
    message: String,
}

#[allow(dead_code)]
pub fn parse_health_response(
    line: &str,
    request_id: &str,
    task_id: &str,
) -> Result<WorkerHealthInfo, WorkerRuntimeError> {
    let payload = parse_worker_response(line, request_id, task_id, "health.result")?;
    parse_health_payload(payload)
}

/// Validate a generic Worker response envelope and return its payload.
pub fn parse_worker_response(
    line: &str,
    request_id: &str,
    task_id: &str,
    expected_message_type: &str,
) -> Result<Value, WorkerRuntimeError> {
    let raw: Value = serde_json::from_str(line.trim())
        .map_err(|error| WorkerRuntimeError::Protocol(format!("JSON 解析失败: {error}")))?;
    let object = raw
        .as_object()
        .ok_or_else(|| WorkerRuntimeError::Protocol("响应根节点必须是 JSON 对象".to_string()))?;
    for field in [
        "schema_version",
        "request_id",
        "task_id",
        "message_type",
        "payload",
        "error",
    ] {
        if !object.contains_key(field) {
            return Err(WorkerRuntimeError::Protocol(format!(
                "响应缺少字段: {field}"
            )));
        }
    }
    let response: WorkerResponseEnvelope = serde_json::from_value(raw)
        .map_err(|error| WorkerRuntimeError::Protocol(format!("响应字段类型无效: {error}")))?;
    if response.schema_version != WORKER_SCHEMA_VERSION {
        return Err(WorkerRuntimeError::Protocol(format!(
            "schema_version 不匹配: {}",
            response.schema_version
        )));
    }
    if response.request_id != request_id {
        return Err(WorkerRuntimeError::Protocol(format!(
            "request_id 不匹配: {}",
            response.request_id
        )));
    }
    if response.task_id != task_id {
        return Err(WorkerRuntimeError::Protocol(format!(
            "task_id 不匹配: {}",
            response.task_id
        )));
    }
    if response.message_type != expected_message_type {
        return Err(WorkerRuntimeError::Protocol(format!(
            "message_type 不匹配: {}",
            response.message_type
        )));
    }
    if let Some(error) = response.error {
        return Err(WorkerRuntimeError::WorkerResponse {
            code: error.code,
            message: error.message,
        });
    }
    Ok(response.payload)
}

pub fn parse_health_payload(payload: Value) -> Result<WorkerHealthInfo, WorkerRuntimeError> {
    let payload: HealthPayload = serde_json::from_value(payload)
        .map_err(|error| WorkerRuntimeError::Protocol(format!("health payload 无效: {error}")))?;
    if payload.schema_version != WORKER_SCHEMA_VERSION {
        return Err(WorkerRuntimeError::Protocol(format!(
            "health payload schema_version 不匹配: {}",
            payload.schema_version
        )));
    }
    if payload.status != "ok" && payload.status != "stopping" {
        return Err(WorkerRuntimeError::Protocol(format!(
            "health status 无效: {}",
            payload.status
        )));
    }
    Ok(WorkerHealthInfo {
        status: payload.status,
        version: payload.worker_version,
        model_count: payload.model_count,
    })
}

fn parse_capability_error(error: CapabilityErrorPayload) -> WorkerCapabilityError {
    WorkerCapabilityError {
        code: error.code,
        message: error.message,
        detail: error.detail,
        retryable: error.retryable,
    }
}

pub fn validate_capability_status(status: &str, field: &str) -> Result<(), WorkerRuntimeError> {
    if status != "ready" && status != "unavailable" {
        return Err(WorkerRuntimeError::Protocol(format!(
            "runtime.capabilities {field} status 无效: {status}"
        )));
    }
    Ok(())
}

pub fn parse_capabilities_payload(
    payload: Value,
) -> Result<WorkerCapabilitiesInfo, WorkerRuntimeError> {
    let payload: CapabilitiesPayload = serde_json::from_value(payload).map_err(|error| {
        WorkerRuntimeError::Protocol(format!("runtime.capabilities payload 无效: {error}"))
    })?;
    if payload.status != "ok" && payload.status != "unavailable" {
        return Err(WorkerRuntimeError::Protocol(format!(
            "runtime.capabilities status 无效: {}",
            payload.status
        )));
    }
    validate_capability_status(&payload.dghs_imgutils.status, "dghs_imgutils")?;
    validate_capability_status(&payload.onnxruntime.status, "onnxruntime")?;
    for feature_name in REQUIRED_CAPABILITY_FEATURES {
        let Some(feature) = payload.features.get(feature_name) else {
            return Err(WorkerRuntimeError::Protocol(format!(
                "runtime.capabilities 缺少必需 feature: {feature_name}"
            )));
        };
        validate_capability_status(&feature.status, feature_name)?;
    }
    let dghs_imgutils = WorkerDependencyCapability {
        distribution: payload.dghs_imgutils.distribution,
        version: payload.dghs_imgutils.version,
        status: payload.dghs_imgutils.status,
        error: payload.dghs_imgutils.error.map(parse_capability_error),
    };
    let onnxruntime = WorkerDependencyCapability {
        distribution: payload.onnxruntime.distribution,
        version: payload.onnxruntime.version,
        status: payload.onnxruntime.status,
        error: payload.onnxruntime.error.map(parse_capability_error),
    };
    let compute = payload.compute.map(|compute| WorkerComputeCapability {
        distribution: compute.distribution,
        available_providers: compute.available_providers,
        cuda_available: compute.cuda_available,
        cuda_usable: compute.cuda_usable,
        cuda_runtime: compute.cuda_runtime.map(|runtime| WorkerCudaRuntime {
            status: runtime.status,
            missing: runtime.missing,
            message: runtime.message,
        }),
        error: compute.error.map(parse_capability_error),
    });
    let features = payload
        .features
        .into_iter()
        .map(|(name, feature)| {
            (
                name,
                WorkerFeatureCapability {
                    module: feature.module,
                    status: feature.status,
                    error: feature.error.map(parse_capability_error),
                },
            )
        })
        .collect();
    let errors = payload
        .errors
        .into_iter()
        .map(parse_capability_error)
        .collect();
    Ok(WorkerCapabilitiesInfo {
        status: payload.status,
        ready: payload.ready,
        dghs_imgutils,
        onnxruntime,
        compute,
        features,
        errors,
    })
}
