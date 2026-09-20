use crate::worker_runtime::discovery::{
    local_models_root, local_project_dir, resolve_worker_candidate_for_mode,
};
use crate::worker_runtime::process::{
    configure_background_command_with, should_clear_process, uv_available,
};
use crate::worker_runtime::protocol::{
    parse_capabilities_payload, parse_health_payload, parse_worker_response,
};
use crate::worker_runtime::types::{
    WorkerCapabilitiesInfo, WorkerComputeDevice, WorkerHealthInfo, WorkerLaunchSpec,
    WorkerRuntimeError, WorkerRuntimeMode, WORKER_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use uuid::Uuid;

pub struct WorkerManager {
    resource_dir: Option<PathBuf>,
    runtime_mode: WorkerRuntimeMode,
    compute_device: WorkerComputeDevice,
    onnx_threads: u32,
    /// Optional folder holding CUDA 12 / cuDNN 9 DLLs for GPU inference.
    cuda_runtime_dir: Option<String>,
    /// Application data folder. The Worker pins its Hugging Face cache inside
    /// it so every downloaded model lives with the package.
    data_dir: Option<String>,
    /// Lower the Worker's CPU priority so long scans stay polite.
    below_normal_priority: bool,
    child: Option<Child>,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: Option<BufReader<ChildStdout>>,
}

impl WorkerManager {
    pub fn new(resource_dir: Option<PathBuf>) -> Self {
        Self::with_settings(
            resource_dir,
            WorkerRuntimeMode::default(),
            WorkerComputeDevice::default(),
            crate::app_settings::AppSettings::default().recognition_onnx_threads,
        )
    }

    pub fn with_runtime_mode(
        resource_dir: Option<PathBuf>,
        runtime_mode: WorkerRuntimeMode,
    ) -> Self {
        Self::with_settings(
            resource_dir,
            runtime_mode,
            WorkerComputeDevice::default(),
            crate::app_settings::AppSettings::default().recognition_onnx_threads,
        )
    }

    pub fn with_settings(
        resource_dir: Option<PathBuf>,
        runtime_mode: WorkerRuntimeMode,
        compute_device: WorkerComputeDevice,
        onnx_threads: u32,
    ) -> Self {
        Self {
            resource_dir,
            runtime_mode,
            compute_device,
            onnx_threads,
            cuda_runtime_dir: None,
            data_dir: None,
            below_normal_priority: true,
            child: None,
            stdin: None,
            stdout: None,
        }
    }

    pub fn runtime_mode(&self) -> WorkerRuntimeMode {
        self.runtime_mode
    }

    pub fn set_runtime_mode(&mut self, runtime_mode: WorkerRuntimeMode) {
        if self.runtime_mode != runtime_mode {
            self.clear_process();
            self.runtime_mode = runtime_mode;
        }
    }

    pub fn compute_device(&self) -> WorkerComputeDevice {
        self.compute_device
    }

    /// Store the inference device used by every subsequent recognition request.
    ///
    /// The long-lived Worker keeps model sessions cached, so the device travels
    /// with each recognition payload and the Worker reloads a session only when
    /// the requested device actually changes.
    pub fn set_compute_device(&mut self, compute_device: WorkerComputeDevice) {
        self.compute_device = compute_device;
    }

    pub fn set_onnx_threads(&mut self, onnx_threads: u32) {
        self.onnx_threads = onnx_threads;
    }

    /// Point the Worker at a folder with CUDA 12 / cuDNN 9 DLLs.
    ///
    /// The value is read when the Worker process starts, so an already running
    /// process is restarted to pick up a change.
    pub fn set_cuda_runtime_dir(&mut self, cuda_runtime_dir: Option<String>) {
        let normalized = cuda_runtime_dir
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        if self.cuda_runtime_dir != normalized {
            self.clear_process();
            self.cuda_runtime_dir = normalized;
        }
    }

    /// `false` lets long jobs use every cycle they can get.
    pub fn set_below_normal_priority(&mut self, below_normal: bool) {
        if self.below_normal_priority != below_normal {
            self.clear_process();
            self.below_normal_priority = below_normal;
        }
    }

    /// Point the Worker at the application data folder.
    ///
    /// The Worker keeps its Hugging Face cache under `<data_dir>\hf-cache` so
    /// the tagging model, the CCIP model and any reference embeddings stay with
    /// the package instead of leaking into the user profile.
    pub fn set_data_dir(&mut self, data_dir: Option<String>) {
        let normalized = data_dir
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        if self.data_dir != normalized {
            self.clear_process();
            self.data_dir = normalized;
        }
    }

    /// `None` when the data folder is unknown; the Worker then falls back to
    /// the machine-wide Hugging Face cache.
    pub fn hf_cache_dir(&self) -> Option<PathBuf> {
        let data_dir = self.data_dir.as_deref()?;
        Some(PathBuf::from(data_dir).join("hf-cache"))
    }

    #[cfg(test)]
    pub fn onnx_threads(&self) -> u32 {
        self.onnx_threads
    }

    pub fn runtime_path(&self) -> Option<PathBuf> {
        let spec = resolve_worker_candidate_for_mode(
            self.resource_dir.as_deref(),
            &local_project_dir(),
            &local_models_root(),
            self.runtime_mode,
        )
        .ok()?;
        match spec {
            WorkerLaunchSpec::Executable { executable, .. }
            | WorkerLaunchSpec::PythonScript {
                interpreter: executable,
                ..
            } => Some(executable),
            WorkerLaunchSpec::UvScript { .. } => Some(PathBuf::from("uv")),
        }
    }

    /// Send a request to the long-lived Worker and return its payload.
    ///
    /// The request/response envelope is deliberately validated at this
    /// boundary for every message type.  This keeps higher-level commands
    /// from accidentally consuming a response belonging to another task.
    pub fn request(
        &mut self,
        message_type: &str,
        payload: Value,
    ) -> Result<Value, WorkerRuntimeError> {
        let payload = self.with_compute_device(message_type, payload);
        if let Err(error) = self.ensure_started() {
            self.clear_process();
            return Err(error);
        }
        let request_id = Uuid::new_v4().to_string();
        let task_id = Uuid::new_v4().to_string();
        let request = json!({
            "schema_version": WORKER_SCHEMA_VERSION,
            "request_id": request_id,
            "task_id": task_id,
            "message_type": message_type,
            "payload": payload,
            "error": Value::Null,
        });
        let result = self.round_trip(&request.to_string(), &request_id, &task_id, message_type);
        if result.as_ref().is_err_and(should_clear_process) {
            self.clear_process();
        }
        result
    }

    /// Attach the configured compute device to every model-inference payload.
    pub(crate) fn with_compute_device(&self, message_type: &str, mut payload: Value) -> Value {
        let needs_device = matches!(
            message_type,
            "recognition.image" | "image.recognize" | "recognition.embedding" | "embedding.extract"
        );
        if !needs_device {
            return payload;
        }
        if let Some(object) = payload.as_object_mut() {
            object
                .entry("device")
                .or_insert_with(|| Value::String(self.compute_device.as_str().to_string()));
            object
                .entry("onnx_threads")
                .or_insert_with(|| Value::from(self.onnx_threads));
        }
        payload
    }

    pub fn health(&mut self) -> Result<WorkerHealthInfo, WorkerRuntimeError> {
        let result = self
            .request("health", json!({}))
            .and_then(parse_health_payload);
        if result.as_ref().is_err_and(should_clear_process) {
            self.clear_process();
        }
        result
    }

    pub fn capabilities(&mut self) -> Result<WorkerCapabilitiesInfo, WorkerRuntimeError> {
        self.request("runtime.capabilities", json!({}))
            .and_then(parse_capabilities_payload)
    }

    fn ensure_started(&mut self) -> Result<(), WorkerRuntimeError> {
        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(None) => return Ok(()),
                Ok(Some(_)) => self.clear_process(),
                Err(error) => {
                    self.clear_process();
                    return Err(WorkerRuntimeError::Io(error.to_string()));
                }
            }
        }

        let spec = resolve_worker_candidate_for_mode(
            self.resource_dir.as_deref(),
            &local_project_dir(),
            &local_models_root(),
            self.runtime_mode,
        )?;
        let mut command = match &spec {
            WorkerLaunchSpec::Executable { executable, .. } => Command::new(executable),
            WorkerLaunchSpec::PythonScript {
                interpreter,
                script,
                ..
            } => {
                let mut command = Command::new(interpreter);
                command.arg(script);
                command
            }
            WorkerLaunchSpec::UvScript {
                project_dir,
                script,
                ..
            } => {
                if !uv_available() {
                    return Err(WorkerRuntimeError::UvMissing);
                }
                let mut command = Command::new("uv");
                command
                    .arg("run")
                    .arg("--project")
                    .arg(project_dir)
                    .arg("--extra")
                    .arg("model")
                    .arg("--extra")
                    .arg("detector")
                    .arg("python")
                    .arg(script);
                command
            }
        };
        command
            .current_dir(spec.working_dir())
            .env("PYTHONUNBUFFERED", "1")
            .env("ANIME_PIC_MODELS", spec.models_root())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(cuda_runtime_dir) = self.cuda_runtime_dir.as_deref() {
            command.env("ANIME_PIC_CUDA_RUNTIME_DIR", cuda_runtime_dir);
        }
        if let Some(hf_cache_dir) = self.hf_cache_dir() {
            command.env("ANIME_PIC_HF_CACHE", hf_cache_dir);
        }
        configure_background_command_with(&mut command, self.below_normal_priority);
        let mut child = command
            .spawn()
            .map_err(|error| WorkerRuntimeError::Spawn(error.to_string()))?;
        let stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(WorkerRuntimeError::Spawn("Worker stdin 未打开".to_string()));
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(WorkerRuntimeError::Spawn(
                    "Worker stdout 未打开".to_string(),
                ));
            }
        };
        self.child = Some(child);
        self.stdin = Some(BufWriter::new(stdin));
        self.stdout = Some(BufReader::new(stdout));
        Ok(())
    }

    fn round_trip(
        &mut self,
        request: &str,
        request_id: &str,
        task_id: &str,
        message_type: &str,
    ) -> Result<Value, WorkerRuntimeError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| WorkerRuntimeError::Io("Worker stdin 不可用".to_string()))?;
        stdin
            .write_all(request.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .map_err(|error| WorkerRuntimeError::Io(error.to_string()))?;
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| WorkerRuntimeError::Io("Worker stdout 不可用".to_string()))?;
        let mut line = String::new();
        let count = stdout
            .read_line(&mut line)
            .map_err(|error| WorkerRuntimeError::Io(error.to_string()))?;
        if count == 0 {
            return Err(WorkerRuntimeError::Io(
                "Worker 已退出且没有返回响应".to_string(),
            ));
        }
        parse_worker_response(
            &line,
            request_id,
            task_id,
            &format!("{message_type}.result"),
        )
    }

    fn clear_process(&mut self) {
        self.stdin.take();
        self.stdout.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) {
        if self.child.is_some() {
            let request_id = Uuid::new_v4().to_string();
            let task_id = Uuid::new_v4().to_string();
            let request = json!({
                "schema_version": WORKER_SCHEMA_VERSION,
                "request_id": request_id,
                "task_id": task_id,
                "message_type": "shutdown",
                "payload": {},
                "error": Value::Null,
            });
            if let (Some(stdin), Some(stdout)) = (self.stdin.as_mut(), self.stdout.as_mut()) {
                let _ = stdin
                    .write_all(request.to_string().as_bytes())
                    .and_then(|_| stdin.write_all(b"\n"))
                    .and_then(|_| stdin.flush());
                let mut response = String::new();
                let _ = stdout.read_line(&mut response);
            }
        }
        self.clear_process();
    }
}
