use serde::Serialize;
use uuid::Uuid;

pub const IPC_VERSION: &str = "1.0";

#[derive(Debug, Clone, Serialize)]
pub struct IpcError {
    pub code: String,
    pub message: String,
    pub detail: Option<String>,
    pub retryable: bool,
    pub request_id: String,
    pub task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IpcEnvelope<T> {
    pub version: String,
    pub request_id: String,
    pub task_id: Option<String>,
    pub message_type: String,
    pub payload: Option<T>,
    pub error: Option<IpcError>,
}

pub fn normalize_request_id(value: Option<String>) -> String {
    value
        .filter(|id| !id.trim().is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

pub fn success<T>(message_type: &str, request_id: String, payload: T) -> IpcEnvelope<T> {
    IpcEnvelope {
        version: IPC_VERSION.to_string(),
        request_id,
        task_id: None,
        message_type: message_type.to_string(),
        payload: Some(payload),
        error: None,
    }
}

pub fn failure<T>(message_type: &str, request_id: String, error: IpcError) -> IpcEnvelope<T> {
    IpcEnvelope {
        version: IPC_VERSION.to_string(),
        request_id,
        task_id: None,
        message_type: message_type.to_string(),
        payload: None,
        error: Some(error),
    }
}

pub fn core_error(
    request_id: &str,
    code: &str,
    message: &str,
    detail: impl Into<Option<String>>,
    retryable: bool,
) -> IpcError {
    IpcError {
        code: code.to_string(),
        message: message.to_string(),
        detail: detail.into(),
        retryable,
        request_id: request_id.to_string(),
        task_id: None,
    }
}
