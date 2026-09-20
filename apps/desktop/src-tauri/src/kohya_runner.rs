//! Process plumbing for an external kohya/sd-scripts training run.
//!
//! Training takes tens of minutes, so the child process is owned by a manager
//! that keeps a rolling log tail and parsed progress. That lets the desktop
//! page re-attach after a remount instead of losing the run.

use crate::kohya::{list_lora_files, parse_progress};
use crate::worker_runtime::process::configure_background_command_with;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

pub const MAX_LOG_LINES: usize = 400;
pub const PROGRESS_EVENT: &str = "kohya://progress";
pub const FINISHED_EVENT: &str = "kohya://finished";
/// Minimum gap between streamed progress snapshots.
const PROGRESS_THROTTLE: Duration = Duration::from_millis(400);

#[derive(Debug, Clone, Default, Serialize)]
pub struct KohyaTrainingSnapshot {
    pub running: bool,
    pub finished: bool,
    pub stopping: bool,
    pub exit_code: Option<i32>,
    pub output_dir: String,
    pub output_name: String,
    pub base_model: String,
    pub dataset_config: String,
    pub log_path: String,
    pub started_at: u64,
    pub elapsed_secs: u64,
    pub epoch: Option<u32>,
    pub step: Option<u32>,
    pub total_steps: Option<u32>,
    pub loss: Option<f64>,
    pub message: String,
    pub artifacts: Vec<String>,
    pub log_tail: Vec<String>,
}

struct RunState {
    snapshot: KohyaTrainingSnapshot,
    tail: VecDeque<String>,
    started: Option<Instant>,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            snapshot: KohyaTrainingSnapshot::default(),
            tail: VecDeque::new(),
            started: None,
        }
    }
}

impl RunState {
    fn export(&self) -> KohyaTrainingSnapshot {
        let mut snapshot = self.snapshot.clone();
        snapshot.log_tail = self.tail.iter().cloned().collect();
        snapshot
    }
}

pub struct KohyaManager {
    child: Option<Child>,
    state: Arc<Mutex<RunState>>,
    /// Application cache folder handed to the trainer as its Hugging Face home,
    /// so anything sd-scripts pulls down stays inside the package.
    hf_cache_dir: Option<PathBuf>,
}

/// Sink for streamed progress, so the process plumbing can be tested without a
/// running Tauri app.
pub trait EventSink: Send + Sync + 'static {
    fn emit_progress(&self, snapshot: &KohyaTrainingSnapshot);
}

impl EventSink for AppHandle {
    fn emit_progress(&self, snapshot: &KohyaTrainingSnapshot) {
        let _ = self.emit(PROGRESS_EVENT, snapshot);
    }
}

/// Bookkeeping written into the snapshot when a run starts.
#[derive(Debug, Clone)]
pub struct RunMetadata {
    pub output_dir: String,
    pub output_name: String,
    pub base_model: String,
    pub dataset_config: String,
    pub log_path: PathBuf,
}

impl Default for KohyaManager {
    fn default() -> Self {
        Self::new()
    }
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or(0)
}

fn terminate_tree(child: &mut Child) {
    let pid = child.id();
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output();
    }
    #[cfg(not(windows))]
    {
        let _ = pid;
        let _ = child.kill();
    }
    let _ = child.wait();
}

impl KohyaManager {
    pub fn new() -> Self {
        Self {
            child: None,
            state: Arc::new(Mutex::new(RunState::default())),
            hf_cache_dir: None,
        }
    }

    /// Keep the trainer's Hugging Face downloads inside the application folder.
    pub fn set_hf_cache_dir(&mut self, dir: Option<PathBuf>) {
        self.hf_cache_dir = dir;
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, RunState>, String> {
        self.state
            .lock()
            .map_err(|_| "训练状态锁暂时不可用。".to_string())
    }

    pub fn snapshot(&mut self) -> KohyaTrainingSnapshot {
        self.poll_exit();
        match self.lock() {
            Ok(mut state) => {
                if let Some(started) = state.started {
                    state.snapshot.elapsed_secs = started.elapsed().as_secs();
                }
                // Re-scan so intermediate `--save_every_n_epochs` checkpoints
                // appear while a long run is still going.
                if !state.snapshot.output_dir.is_empty() {
                    state.snapshot.artifacts =
                        list_lora_files(Path::new(&state.snapshot.output_dir));
                }
                state.export()
            }
            Err(_) => KohyaTrainingSnapshot::default(),
        }
    }

    /// True while the child process is still alive.
    pub fn is_running(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.snapshot.running)
            .unwrap_or(false)
    }

    /// Detect a finished child. Returns the final snapshot exactly once.
    pub fn poll_exit(&mut self) -> Option<KohyaTrainingSnapshot> {
        let child = self.child.as_mut()?;
        let Ok(Some(status)) = child.try_wait() else {
            return None;
        };
        let code = status.code();
        self.child = None;
        let mut state = self.lock().ok()?;
        state.snapshot.running = false;
        state.snapshot.finished = true;
        state.snapshot.stopping = false;
        state.snapshot.exit_code = code;
        state.snapshot.artifacts = list_lora_files(Path::new(&state.snapshot.output_dir));
        state.snapshot.message = match code {
            Some(0) => "训练完成。".to_string(),
            Some(value) => format!("训练进程退出（退出码 {value}），请查看日志。"),
            None => "训练进程已结束，请查看日志。".to_string(),
        };
        state.started = None;
        Some(state.export())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        sink: Arc<dyn EventSink>,
        python: &Path,
        script: &Path,
        arguments: &[String],
        log_path: PathBuf,
        output_dir: &str,
        output_name: &str,
        base_model: &str,
        dataset_config: &str,
        below_normal_priority: bool,
    ) -> Result<(), String> {
        let mut command = Command::new(python);
        command
            .arg(script)
            .args(arguments)
            .current_dir(script.parent().unwrap_or(Path::new(".")))
            // The desktop tails this output live, so stdout must not be
            // block-buffered; the log would otherwise arrive in 8 KB chunks.
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SDXL LoRA training on 6-8 GB cards fragments badly without this;
        // an explicit user setting still wins.
        if std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none() {
            command.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
        }
        self.start_process(
            sink,
            command,
            RunMetadata {
                output_dir: output_dir.to_string(),
                output_name: output_name.to_string(),
                base_model: base_model.to_string(),
                dataset_config: dataset_config.to_string(),
                log_path,
            },
            below_normal_priority,
        )
    }

    /// Spawn an already prepared command and stream its output.
    pub fn start_process(
        &mut self,
        sink: Arc<dyn EventSink>,
        mut command: Command,
        meta: RunMetadata,
        below_normal_priority: bool,
    ) -> Result<(), String> {
        if self.child.is_some() || self.poll_exit().is_some() {
            return Err("已经有一个训练任务在运行，请先停止它。".to_string());
        }
        if let Some(parent) = meta.log_path.parent() {
            fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        fs::create_dir_all(&meta.output_dir)
            .map_err(|error| format!("无法创建训练输出目录 {}: {error}", meta.output_dir))?;
        let log_file = fs::File::create(&meta.log_path)
            .map_err(|error| format!("{}: {error}", meta.log_path.display()))?;
        let stderr_file = log_file
            .try_clone()
            .map_err(|error| format!("无法写入训练日志: {error}"))?;

        // sd-scripts fetches tokenizers and base models through
        // ``huggingface_hub``; pointing it at the application cache keeps the
        // download next to the rest of the software instead of in the user
        // profile.
        if let Some(cache_dir) = self.hf_cache_dir.as_ref() {
            command
                .env("HF_HOME", cache_dir)
                .env("HF_HUB_CACHE", cache_dir.join("hub"));
        }
        configure_background_command_with(&mut command, below_normal_priority);

        let mut child = command
            .spawn()
            .map_err(|error| format!("启动训练进程失败: {error}"))?;
        let pid = child.id();

        {
            let mut state = self.lock()?;
            state.snapshot = KohyaTrainingSnapshot {
                running: true,
                message: format!("训练进程已启动（PID {pid}）"),
                output_dir: meta.output_dir.clone(),
                output_name: meta.output_name.clone(),
                base_model: meta.base_model.clone(),
                dataset_config: meta.dataset_config.clone(),
                log_path: meta.log_path.to_string_lossy().into_owned(),
                started_at: now_seconds(),
                ..KohyaTrainingSnapshot::default()
            };
            state.tail.clear();
            state.started = Some(Instant::now());
        }

        let stdout = child.stdout.take();
        if let Some(pipe) = stdout {
            let shared = Arc::clone(&self.state);
            let sink = Arc::clone(&sink);
            std::thread::spawn(move || {
                stream_lines(BufReader::new(pipe), log_file, shared, sink);
            });
        }
        if let Some(pipe) = child.stderr.take() {
            let shared = Arc::clone(&self.state);
            std::thread::spawn(move || {
                stream_lines(BufReader::new(pipe), stderr_file, shared, sink);
            });
        }

        self.child = Some(child);
        Ok(())
    }

    pub fn stop(&mut self) -> Result<bool, String> {
        let shared = Arc::clone(&self.state);
        let Some(child) = self.child.as_mut() else {
            return Ok(false);
        };
        if let Ok(mut state) = shared.lock() {
            state.snapshot.stopping = true;
            state.snapshot.message = "正在停止训练…".to_string();
        }
        terminate_tree(child);
        self.child = None;
        if let Ok(mut state) = shared.lock() {
            state.snapshot.running = false;
            state.snapshot.finished = true;
            state.snapshot.stopping = false;
            state.snapshot.artifacts = list_lora_files(Path::new(&state.snapshot.output_dir));
            state.snapshot.message = "训练已停止。".to_string();
            state.started = None;
        }
        Ok(true)
    }
}

impl Drop for KohyaManager {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            terminate_tree(child);
        }
    }
}

fn stream_lines<R: BufRead>(
    reader: R,
    mut log_file: fs::File,
    shared: Arc<Mutex<RunState>>,
    sink: Arc<dyn EventSink>,
) {
    let mut last_emit: Option<Instant> = None;
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let _ = writeln!(log_file, "{line}");
        let _ = log_file.flush();
        let progress = parse_progress(&line);
        let Ok(mut state) = shared.lock() else { break };
        state.tail.push_back(line.clone());
        while state.tail.len() > MAX_LOG_LINES {
            state.tail.pop_front();
        }
        if progress.epoch.is_some() {
            state.snapshot.epoch = progress.epoch;
        }
        if progress.step.is_some() {
            state.snapshot.step = progress.step;
        }
        if progress.total_steps.is_some() {
            state.snapshot.total_steps = progress.total_steps;
        }
        if progress.loss.is_some() {
            state.snapshot.loss = progress.loss;
        }
        if !line.trim().is_empty() {
            state.snapshot.message = line.trim().to_string();
        }
        // Emit the first line immediately, then throttle so a fast trainer
        // cannot flood the webview.
        let due = last_emit.is_none_or(|last: Instant| last.elapsed() >= PROGRESS_THROTTLE);
        let snapshot = if due { Some(state.export()) } else { None };
        drop(state);
        if let Some(snapshot) = snapshot {
            last_emit = Some(Instant::now());
            sink.emit_progress(&snapshot);
        }
    }
    // Always deliver the tail once the stream ends, even for very short runs.
    if let Ok(state) = shared.lock() {
        sink.emit_progress(&state.export());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct RecordingSink {
        snapshots: StdMutex<Vec<KohyaTrainingSnapshot>>,
    }

    impl EventSink for RecordingSink {
        fn emit_progress(&self, snapshot: &KohyaTrainingSnapshot) {
            if let Ok(mut snapshots) = self.snapshots.lock() {
                snapshots.push(snapshot.clone());
            }
        }
    }

    impl RecordingSink {
        fn last(&self) -> Option<KohyaTrainingSnapshot> {
            self.snapshots.lock().ok()?.last().cloned()
        }
    }

    fn scratch(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("kohya-run-{label}-{}", uuid::Uuid::new_v4()))
    }

    /// A stand-in trainer that prints tqdm-style lines and exits cleanly.
    fn fake_trainer_command(dir: &Path) -> Command {
        if cfg!(windows) {
            fs::write(
                dir.join("train.cmd"),
                "@echo off\r\necho epoch 1/2\r\necho steps: 5/10 loss=0.2500\r\necho steps: 10/10 loss=0.1250\r\necho training finished\r\n",
            )
            .unwrap();
            let mut command = Command::new("cmd");
            command.args(["/C", "train.cmd"]).current_dir(dir);
            command
        } else {
            fs::write(
                dir.join("train.sh"),
                "echo 'epoch 1/2'\necho 'steps: 5/10 loss=0.2500'\necho 'steps: 10/10 loss=0.1250'\necho 'training finished'\n",
            )
            .unwrap();
            let mut command = Command::new("sh");
            command.arg("train.sh").current_dir(dir);
            command
        }
    }

    /// A stand-in trainer that reports the environment it was started with.
    fn environment_probe_script(dir: &Path) -> (PathBuf, Vec<String>) {
        if cfg!(windows) {
            fs::write(
                dir.join("env.cmd"),
                "@echo off\r\necho unbuffered=%PYTHONUNBUFFERED%\r\necho alloc=%PYTORCH_CUDA_ALLOC_CONF%\r\necho hf_home=%HF_HOME%\r\necho hub=%HF_HUB_CACHE%\r\n",
            )
            .unwrap();
            (
                PathBuf::from("cmd"),
                vec!["/C".to_string(), "env.cmd".to_string()],
            )
        } else {
            fs::write(
                dir.join("env.sh"),
                "echo unbuffered=$PYTHONUNBUFFERED\necho alloc=$PYTORCH_CUDA_ALLOC_CONF\necho hf_home=$HF_HOME\necho hub=$HF_HUB_CACHE\n",
            )
            .unwrap();
            (PathBuf::from("sh"), vec!["env.sh".to_string()])
        }
    }

    #[test]
    fn training_processes_run_unbuffered_with_expandable_segments() {
        let dir = scratch("env");
        fs::create_dir_all(&dir).unwrap();
        let (interpreter, arguments) = environment_probe_script(&dir);
        let script = dir.join(if cfg!(windows) { "env.cmd" } else { "env.sh" });
        let log_path = dir.join("env.log");

        // Uses the real `start` path so the variables under test are exactly
        // the ones production sets.
        let mut manager = KohyaManager::new();
        manager.set_hf_cache_dir(Some(dir.join("hf-cache")));
        manager
            .start(
                Arc::new(RecordingSink::default()),
                &interpreter,
                &script,
                &arguments,
                log_path.clone(),
                &dir.to_string_lossy(),
                "probe",
                "base",
                "dataset.toml",
                true,
            )
            .expect("probe run starts");
        let deadline = Instant::now() + Duration::from_secs(20);
        while manager.poll_exit().is_none() {
            assert!(Instant::now() < deadline, "the probe never finished");
            std::thread::sleep(Duration::from_millis(30));
        }
        let logged = fs::read_to_string(&log_path).unwrap();
        assert!(logged.contains("unbuffered=1"), "log was {logged}");
        assert!(logged.contains("expandable_segments"), "log was {logged}");
        // Training downloads stay inside the application folder.
        assert!(
            logged.contains("hf-cache") && logged.contains("hub"),
            "log was {logged}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn idle_manager_reports_no_run() {
        let mut manager = KohyaManager::new();
        let snapshot = manager.snapshot();

        assert!(!snapshot.running);
        assert!(!snapshot.finished);
        assert!(snapshot.log_tail.is_empty());
        assert!(manager.stop().is_ok());
    }

    #[test]
    fn stopping_without_a_run_is_a_no_op() {
        let mut manager = KohyaManager::new();

        assert!(!manager.stop().expect("stop should not fail"));
    }

    #[test]
    fn streams_a_finished_run_and_collects_its_artifacts() {
        let dir = scratch("stream");
        fs::create_dir_all(&dir).unwrap();
        let log_path = dir.join("train.log");
        let mut command = fake_trainer_command(&dir);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut manager = KohyaManager::new();
        let sink = Arc::new(RecordingSink::default());
        manager
            .start_process(
                sink.clone(),
                command,
                RunMetadata {
                    output_dir: dir.to_string_lossy().into_owned(),
                    output_name: "character_alpha".to_string(),
                    base_model: "base.safetensors".to_string(),
                    dataset_config: "dataset.toml".to_string(),
                    log_path: log_path.clone(),
                },
                true,
            )
            .expect("run starts");
        assert!(manager.is_running());

        let deadline = Instant::now() + Duration::from_secs(20);
        let final_snapshot = loop {
            if let Some(snapshot) = manager.poll_exit() {
                break snapshot;
            }
            assert!(Instant::now() < deadline, "the fake trainer never exited");
            std::thread::sleep(Duration::from_millis(50));
        };

        // A LoRA dropped into the output folder shows up in the snapshot.
        fs::write(dir.join("character_alpha.safetensors"), b"# lora").unwrap();
        assert_eq!(final_snapshot.exit_code, Some(0));
        assert!(final_snapshot.finished);
        assert!(!final_snapshot.running);
        assert_eq!(final_snapshot.output_name, "character_alpha");

        let logged = fs::read_to_string(&log_path).unwrap();
        assert!(logged.contains("training finished"));
        assert!(logged.contains("loss=0.1250"));

        let streamed = sink.last().expect("progress was streamed");
        assert_eq!(streamed.step, Some(10));
        assert_eq!(streamed.total_steps, Some(10));

        let after = manager.snapshot();
        assert!(after
            .artifacts
            .iter()
            .any(|path| path.ends_with(".safetensors")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_to_start_a_second_run() {
        let dir = scratch("double");
        fs::create_dir_all(&dir).unwrap();
        let mut first = fake_trainer_command(&dir);
        first.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut second = fake_trainer_command(&dir);
        second.stdout(Stdio::piped()).stderr(Stdio::piped());
        let meta = RunMetadata {
            output_dir: dir.to_string_lossy().into_owned(),
            output_name: "character_alpha".to_string(),
            base_model: "base.safetensors".to_string(),
            dataset_config: "dataset.toml".to_string(),
            log_path: dir.join("train.log"),
        };

        let mut manager = KohyaManager::new();
        manager
            .start_process(
                Arc::new(RecordingSink::default()),
                first,
                meta.clone(),
                true,
            )
            .expect("first run starts");

        let error = manager
            .start_process(Arc::new(RecordingSink::default()), second, meta, true)
            .expect_err("second run is refused");

        assert!(error.contains("已经有一个训练任务在运行"));
        manager.stop().expect("stop succeeds");
        fs::remove_dir_all(dir).unwrap();
    }
}
