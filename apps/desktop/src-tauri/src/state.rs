use crate::comfy::ComfyManager;
use crate::database::Database;
use crate::kohya_runner::KohyaManager;
use crate::worker_runtime::WorkerManager;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use tokio::sync::Notify;

pub struct AppState {
    pub database: Arc<Mutex<Database>>,
    pub data_dir: String,
    pub worker: Arc<Mutex<WorkerManager>>,
    pub comfy: Arc<Mutex<ComfyManager>>,
    pub comfy_cancel: Arc<AtomicBool>,
    pub gpu_devices: Arc<Mutex<Option<Vec<String>>>>,
    pub update_control: Arc<UpdateControl>,
    pub scan_control: Arc<ScanControl>,
    pub similarity_scan_control: Arc<ScanControl>,
    pub training_control: Arc<TrainingControl>,
    /// External kohya/sd-scripts LoRA training process (never bundled).
    pub kohya: Arc<Mutex<KohyaManager>>,
}

impl AppState {
    pub fn new(database: Database, data_dir: String, worker: WorkerManager) -> Self {
        Self {
            database: Arc::new(Mutex::new(database)),
            data_dir,
            worker: Arc::new(Mutex::new(worker)),
            comfy: Arc::new(Mutex::new(ComfyManager::new())),
            comfy_cancel: Arc::new(AtomicBool::new(false)),
            gpu_devices: Arc::new(Mutex::new(None)),
            update_control: Arc::new(UpdateControl::new()),
            scan_control: Arc::new(ScanControl::new()),
            similarity_scan_control: Arc::new(ScanControl::new()),
            training_control: Arc::new(TrainingControl::new()),
            kohya: Arc::new(Mutex::new(KohyaManager::new())),
        }
    }
}

pub struct UpdateControl {
    pub cancel_requested: AtomicBool,
    pub notify: Notify,
}

impl UpdateControl {
    pub fn new() -> Self {
        Self {
            cancel_requested: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct ScanControlState {
    pub running: bool,
    pub paused: bool,
    pub cancel_requested: bool,
    pub current: usize,
    pub total: usize,
    pub path: String,
    pub message: String,
    pub phase: String,
    pub directory: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanControlResponse {
    pub status: String,
    pub running: bool,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ScanProgressSnapshot {
    pub current: usize,
    pub total: usize,
    pub path: String,
    pub message: String,
    pub phase: String,
    pub directory: String,
    pub running: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct PauseWaitResult {
    pub was_paused: bool,
    pub cancelled: bool,
}

pub struct ScanControl {
    pub state: Mutex<ScanControlState>,
    pub wake: Condvar,
}

pub struct TrainingControl {
    pub running: AtomicBool,
}

pub struct TrainingGuard<'a> {
    pub control: &'a TrainingControl,
}

impl TrainingControl {
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
        }
    }

    pub fn try_start(&self) -> Option<TrainingGuard<'_>> {
        self.running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| TrainingGuard { control: self })
    }

    #[cfg(test)]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

impl Drop for TrainingGuard<'_> {
    fn drop(&mut self) {
        self.control.running.store(false, Ordering::Release);
    }
}

impl ScanControl {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ScanControlState::default()),
            wake: Condvar::new(),
        }
    }

    pub fn try_start(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.running {
            return false;
        }
        state.running = true;
        state.paused = false;
        state.cancel_requested = false;
        state.current = 0;
        state.total = 0;
        state.path.clear();
        state.message.clear();
        state.phase.clear();
        state.directory.clear();
        true
    }

    pub fn finish(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.running = false;
            state.paused = false;
            state.cancel_requested = false;
            state.current = 0;
            state.total = 0;
            state.path.clear();
            state.message.clear();
            state.phase.clear();
            state.directory.clear();
        }
        self.wake.notify_all();
    }

    pub fn request_cancel(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let running = state.running;
        if running {
            state.cancel_requested = true;
        }
        drop(state);
        self.wake.notify_all();
        running
    }

    pub fn is_cancelled(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.cancel_requested)
            .unwrap_or(true)
    }

    /// Record which directory the running scan belongs to.
    ///
    /// The desktop re-reads this after a page remount so a scan that is still
    /// running can be re-attached without reloading a previous result file.
    pub fn set_directory(&self, directory: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.directory = directory.to_string();
        }
    }

    pub fn update_progress(&self, current: usize, total: usize, path: &str, message: &str) {
        self.update_phase_progress(current, total, path, message, "");
    }

    pub fn update_phase_progress(
        &self,
        current: usize,
        total: usize,
        path: &str,
        message: &str,
        phase: &str,
    ) {
        if let Ok(mut state) = self.state.lock() {
            state.current = current;
            state.total = total;
            state.path = path.to_string();
            state.message = message.to_string();
            if !phase.is_empty() {
                state.phase = phase.to_string();
            }
        }
    }

    pub fn snapshot(&self) -> ScanProgressSnapshot {
        self.state
            .lock()
            .map(|state| ScanProgressSnapshot {
                current: state.current,
                total: state.total,
                path: state.path.clone(),
                message: state.message.clone(),
                phase: state.phase.clone(),
                directory: state.directory.clone(),
                running: state.running,
            })
            .unwrap_or(ScanProgressSnapshot {
                current: 0,
                total: 0,
                path: String::new(),
                message: String::new(),
                phase: String::new(),
                directory: String::new(),
                running: false,
            })
    }

    pub fn request_pause(&self) -> ScanControlResponse {
        let Ok(mut state) = self.state.lock() else {
            return ScanControlResponse {
                status: "error".to_string(),
                running: false,
                message: "无法读取扫描控制状态。".to_string(),
            };
        };
        let response = if !state.running {
            ScanControlResponse {
                status: "idle".to_string(),
                running: false,
                message: "当前没有正在运行的扫描任务。".to_string(),
            }
        } else if state.cancel_requested {
            ScanControlResponse {
                status: "cancelling".to_string(),
                running: true,
                message: "扫描正在取消，暂不能暂停。".to_string(),
            }
        } else if state.paused {
            ScanControlResponse {
                status: "paused".to_string(),
                running: true,
                message: "扫描已暂停。".to_string(),
            }
        } else {
            state.paused = true;
            ScanControlResponse {
                status: "pausing".to_string(),
                running: true,
                message: "已请求暂停，当前图片处理完成后暂停。".to_string(),
            }
        };
        drop(state);
        self.wake.notify_all();
        response
    }

    pub fn request_resume(&self) -> ScanControlResponse {
        let Ok(mut state) = self.state.lock() else {
            return ScanControlResponse {
                status: "error".to_string(),
                running: false,
                message: "无法读取扫描控制状态。".to_string(),
            };
        };
        let response = if !state.running {
            ScanControlResponse {
                status: "idle".to_string(),
                running: false,
                message: "当前没有暂停中的扫描任务。".to_string(),
            }
        } else if state.cancel_requested {
            ScanControlResponse {
                status: "cancelling".to_string(),
                running: true,
                message: "扫描正在取消，不能继续。".to_string(),
            }
        } else if state.paused {
            state.paused = false;
            ScanControlResponse {
                status: "resumed".to_string(),
                running: true,
                message: "扫描已继续。".to_string(),
            }
        } else {
            ScanControlResponse {
                status: "running".to_string(),
                running: true,
                message: "扫描正在运行。".to_string(),
            }
        };
        drop(state);
        self.wake.notify_all();
        response
    }

    pub fn wait_if_paused_with<F>(&self, on_paused: F) -> PauseWaitResult
    where
        F: FnOnce(ScanProgressSnapshot),
    {
        let Ok(mut state) = self.state.lock() else {
            return PauseWaitResult {
                was_paused: false,
                cancelled: true,
            };
        };
        let was_paused = state.paused && state.running && !state.cancel_requested;
        if was_paused {
            let snapshot = ScanProgressSnapshot {
                current: state.current,
                total: state.total,
                path: state.path.clone(),
                message: state.message.clone(),
                phase: state.phase.clone(),
                directory: state.directory.clone(),
                running: state.running,
            };
            drop(state);
            on_paused(snapshot);
            state = match self.state.lock() {
                Ok(next) => next,
                Err(_) => {
                    return PauseWaitResult {
                        was_paused,
                        cancelled: true,
                    }
                }
            };
        }
        while state.running && state.paused && !state.cancel_requested {
            state = match self.wake.wait(state) {
                Ok(next) => next,
                Err(_) => {
                    return PauseWaitResult {
                        was_paused,
                        cancelled: true,
                    }
                }
            };
        }
        PauseWaitResult {
            was_paused,
            cancelled: state.cancel_requested,
        }
    }
}
