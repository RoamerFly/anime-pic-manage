//! Detect and drive the user's own kohya_ss / sd-scripts LoRA trainer.
//!
//! The application never bundles a trainer: kohya_ss and sd-scripts are
//! separate projects with their own licences, so this module only looks for an
//! existing installation and builds the argument list for its training script.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SD15_TRAINER: &str = "train_network.py";
pub const SDXL_TRAINER: &str = "sdxl_train_network.py";

pub const MIN_NETWORK_DIM: u32 = 1;
pub const MAX_NETWORK_DIM: u32 = 512;
pub const MIN_LEARNING_RATE: f64 = 0.000_001;
pub const MAX_LEARNING_RATE: f64 = 0.1;
pub const MIN_EPOCHS: u32 = 1;
pub const MAX_EPOCHS: u32 = 1000;
pub const MIN_BATCH_SIZE: u32 = 1;
pub const MAX_BATCH_SIZE: u32 = 16;
pub const MIN_ACCUMULATION: u32 = 1;
pub const MAX_ACCUMULATION: u32 = 64;
pub const MAX_DATA_LOADER_WORKERS: u32 = 8;

/// Where the Python interpreter came from, so the UI can explain the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonSource {
    Settings,
    VirtualEnv,
    Embedded,
    Path,
}

impl PythonSource {
    pub fn label(self) -> &'static str {
        match self {
            PythonSource::Settings => "设置中指定",
            PythonSource::VirtualEnv => "训练器自带虚拟环境",
            PythonSource::Embedded => "便携版内置 Python",
            PythonSource::Path => "系统 PATH",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KohyaPaths {
    pub root: PathBuf,
    pub sd_scripts: PathBuf,
    pub python: PathBuf,
    pub python_source: PythonSource,
    pub python_exists: bool,
    pub trainers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KohyaStatus {
    pub configured: bool,
    pub installed: bool,
    pub root: String,
    pub sd_scripts_dir: String,
    pub python: String,
    pub python_source: String,
    pub trainers: Vec<String>,
    pub base_model: String,
    pub base_model_exists: bool,
    pub train_data_dir: String,
    pub output_dir: String,
    pub error: Option<String>,
}

impl KohyaStatus {
    pub fn unconfigured(base_model: &str) -> Self {
        Self {
            configured: false,
            installed: false,
            root: String::new(),
            sd_scripts_dir: String::new(),
            python: String::new(),
            python_source: PythonSource::Path.label().to_string(),
            trainers: Vec::new(),
            base_model: base_model.to_string(),
            base_model_exists: !base_model.trim().is_empty() && Path::new(base_model).is_file(),
            train_data_dir: String::new(),
            output_dir: String::new(),
            error: Some(
                "还没有配置 LoRA 训练器目录。请指定 kohya_ss 或 sd-scripts 的安装目录。"
                    .to_string(),
            ),
        }
    }
}

fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|path| path.is_file()).cloned()
}

/// Resolve the trainer layout from the configured root and Python override.
pub fn resolve_paths(root: &str, python_override: &str) -> Result<KohyaPaths, String> {
    let root = root.trim();
    if root.is_empty() {
        return Err("还没有配置 LoRA 训练器目录。".to_string());
    }
    let root_path = PathBuf::from(root);
    if !root_path.is_dir() {
        return Err(format!("训练器目录不存在：{}", root_path.display()));
    }

    // kohya_ss keeps the scripts in `sd-scripts/`, while a direct sd-scripts
    // clone has them at the root.
    let mut sd_scripts = PathBuf::new();
    let mut trainers: Vec<String> = Vec::new();
    for candidate in [root_path.join("sd-scripts"), root_path.clone()] {
        let found: Vec<String> = [SD15_TRAINER, SDXL_TRAINER]
            .iter()
            .filter(|name| candidate.join(name).is_file())
            .map(|name| (*name).to_string())
            .collect();
        if !found.is_empty() {
            sd_scripts = candidate;
            trainers = found;
            break;
        }
    }
    if trainers.is_empty() {
        return Err(format!(
            "{} 下没有找到 {SD15_TRAINER} 或 {SDXL_TRAINER}，请确认这是 kohya_ss 或 sd-scripts 目录。",
            root_path.display()
        ));
    }

    let override_path = python_override.trim();
    let (python, python_source) = if !override_path.is_empty() {
        (PathBuf::from(override_path), PythonSource::Settings)
    } else if let Some(path) = first_existing(&[
        root_path.join("venv").join("Scripts").join("python.exe"),
        root_path.join(".venv").join("Scripts").join("python.exe"),
        sd_scripts.join("venv").join("Scripts").join("python.exe"),
        sd_scripts.join(".venv").join("Scripts").join("python.exe"),
    ]) {
        (path, PythonSource::VirtualEnv)
    } else if let Some(path) = first_existing(&[
        root_path.join("python").join("python.exe"),
        root_path
            .parent()
            .map(|parent| parent.join("python_embeded").join("python.exe"))
            .unwrap_or_default(),
    ]) {
        (path, PythonSource::Embedded)
    } else {
        (PathBuf::from("python"), PythonSource::Path)
    };
    let python_exists = python.is_file() || python_source == PythonSource::Path;

    Ok(KohyaPaths {
        root: root_path,
        sd_scripts,
        python,
        python_source,
        python_exists,
        trainers,
    })
}

pub fn status(root: &str, python_override: &str, base_model: &str) -> KohyaStatus {
    let base_model = base_model.trim();
    let base_model_exists = !base_model.is_empty() && Path::new(base_model).is_file();
    match resolve_paths(root, python_override) {
        Ok(paths) => KohyaStatus {
            configured: true,
            installed: paths.python_exists,
            root: paths.root.to_string_lossy().into_owned(),
            sd_scripts_dir: paths.sd_scripts.to_string_lossy().into_owned(),
            python: paths.python.to_string_lossy().into_owned(),
            python_source: paths.python_source.label().to_string(),
            trainers: paths.trainers,
            base_model: base_model.to_string(),
            base_model_exists,
            train_data_dir: String::new(),
            output_dir: String::new(),
            error: if paths.python_exists {
                None
            } else {
                Some(format!(
                    "找不到可用的 Python：{}。请在设置里指定训练用的 python.exe。",
                    paths.python.display()
                ))
            },
        },
        Err(error) => KohyaStatus {
            error: Some(error),
            ..KohyaStatus::unconfigured(base_model)
        },
    }
}

/// Which training script to run for a base model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainerKind {
    Sd15,
    Sdxl,
}

impl TrainerKind {
    pub fn script(self) -> &'static str {
        match self {
            TrainerKind::Sd15 => SD15_TRAINER,
            TrainerKind::Sdxl => SDXL_TRAINER,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TrainerKind::Sd15 => "SD1.5 / SD2.x",
            TrainerKind::Sdxl => "SDXL",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "sd15" | "sd1.5" | "sd" => Ok(TrainerKind::Sd15),
            "sdxl" => Ok(TrainerKind::Sdxl),
            other => Err(format!("未知的训练脚本类型：{other}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub dataset_config: String,
    pub output_dir: String,
    pub output_name: String,
    pub base_model: String,
    pub trainer: TrainerKind,
    pub network_dim: u32,
    pub network_alpha: u32,
    pub learning_rate: f64,
    pub max_train_epochs: u32,
    pub train_batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub optimizer: String,
    pub mixed_precision: String,
    pub save_every_n_epochs: u32,
    pub seed: u64,
    pub max_data_loader_n_workers: u32,
    #[serde(default)]
    pub gradient_checkpointing: bool,
    #[serde(default)]
    pub cache_latents: bool,
    #[serde(default)]
    pub network_train_unet_only: bool,
    /// Cache the text-encoder outputs and free both CLIPs (~1.4 GB on SDXL).
    /// kohya requires UNet-only training and a dataset config without
    /// `shuffle_caption` for this.
    #[serde(default)]
    pub cache_text_encoder_outputs: bool,
    #[serde(default)]
    pub flip_aug: bool,
    #[serde(default)]
    pub noise_offset: f64,
}

impl TrainingRequest {
    /// A conservative starting point for a 6 GB card: AdamW8bit, gradient
    /// checkpointing and cached latents keep SD1.5 training inside VRAM.
    pub fn default_for(
        trainer: TrainerKind,
        dataset_config: &Path,
        output_dir: &Path,
        base_model: &Path,
    ) -> Self {
        Self {
            dataset_config: dataset_config.to_string_lossy().into_owned(),
            output_dir: output_dir.to_string_lossy().into_owned(),
            output_name: "lora".to_string(),
            base_model: base_model.to_string_lossy().into_owned(),
            trainer,
            network_dim: 32,
            network_alpha: 16,
            learning_rate: 0.0001,
            max_train_epochs: 10,
            train_batch_size: 1,
            gradient_accumulation_steps: 1,
            optimizer: "AdamW8bit".to_string(),
            mixed_precision: "fp16".to_string(),
            save_every_n_epochs: 1,
            seed: 42,
            max_data_loader_n_workers: 2,
            gradient_checkpointing: true,
            cache_latents: true,
            network_train_unet_only: true,
            cache_text_encoder_outputs: false,
            flip_aug: false,
            noise_offset: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.dataset_config.trim().is_empty() {
            return Err("缺少训练集配置（dataset.toml）。".to_string());
        }
        if !Path::new(self.dataset_config.trim()).is_file() {
            return Err(format!(
                "找不到训练集配置：{}（请先导出训练集）",
                self.dataset_config
            ));
        }
        if self.output_dir.trim().is_empty() {
            return Err("缺少训练输出目录。".to_string());
        }
        let output_name = self.output_name.trim();
        if output_name.is_empty() {
            return Err("请填写 LoRA 名称。".to_string());
        }
        if output_name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) {
            return Err("LoRA 名称不能包含 / \\ : * ? \" < > | 等字符。".to_string());
        }
        if self.base_model.trim().is_empty() {
            return Err("请选择训练用的底模（.safetensors）。".to_string());
        }
        if !Path::new(self.base_model.trim()).is_file() {
            return Err(format!("找不到底模文件：{}", self.base_model));
        }
        if !(MIN_NETWORK_DIM..=MAX_NETWORK_DIM).contains(&self.network_dim) {
            return Err(format!(
                "network_dim 必须在 {MIN_NETWORK_DIM} 到 {MAX_NETWORK_DIM} 之间。"
            ));
        }
        if !(MIN_NETWORK_DIM..=MAX_NETWORK_DIM).contains(&self.network_alpha) {
            return Err(format!(
                "network_alpha 必须在 {MIN_NETWORK_DIM} 到 {MAX_NETWORK_DIM} 之间。"
            ));
        }
        if !self.learning_rate.is_finite()
            || !(MIN_LEARNING_RATE..=MAX_LEARNING_RATE).contains(&self.learning_rate)
        {
            return Err(format!(
                "学习率必须在 {MIN_LEARNING_RATE} 到 {MAX_LEARNING_RATE} 之间。"
            ));
        }
        if !(MIN_EPOCHS..=MAX_EPOCHS).contains(&self.max_train_epochs) {
            return Err(format!(
                "训练轮数必须在 {MIN_EPOCHS} 到 {MAX_EPOCHS} 之间。"
            ));
        }
        if !(MIN_BATCH_SIZE..=MAX_BATCH_SIZE).contains(&self.train_batch_size) {
            return Err(format!(
                "batch size 必须在 {MIN_BATCH_SIZE} 到 {MAX_BATCH_SIZE} 之间。"
            ));
        }
        if !(MIN_ACCUMULATION..=MAX_ACCUMULATION).contains(&self.gradient_accumulation_steps) {
            return Err(format!(
                "梯度累积必须在 {MIN_ACCUMULATION} 到 {MAX_ACCUMULATION} 之间。"
            ));
        }
        if self.max_data_loader_n_workers > MAX_DATA_LOADER_WORKERS {
            return Err(format!(
                "数据加载进程数不能超过 {MAX_DATA_LOADER_WORKERS}（Windows 下过大会占满内存）。"
            ));
        }
        if self.cache_text_encoder_outputs && !self.network_train_unet_only {
            return Err(
                "缓存文本编码器输出时只能训练 UNet（请同时勾选“仅训练 UNet”）。".to_string(),
            );
        }
        if !self.noise_offset.is_finite() || !(0.0..=1.0).contains(&self.noise_offset) {
            return Err("noise_offset 必须在 0 到 1 之间。".to_string());
        }
        if !matches!(self.mixed_precision.as_str(), "no" | "fp16" | "bf16") {
            return Err("mixed_precision 只能是 no / fp16 / bf16。".to_string());
        }
        Ok(())
    }
}

/// Build the sd-scripts argument list. Kept pure so it can be unit tested.
pub fn build_arguments(request: &TrainingRequest, logging_dir: &Path) -> Vec<String> {
    let mut args = vec![
        "--pretrained_model_name_or_path".to_string(),
        request.base_model.trim().to_string(),
        "--dataset_config".to_string(),
        request.dataset_config.trim().to_string(),
        "--output_dir".to_string(),
        request.output_dir.trim().to_string(),
        "--output_name".to_string(),
        request.output_name.trim().to_string(),
        "--logging_dir".to_string(),
        logging_dir.to_string_lossy().into_owned(),
        "--network_module".to_string(),
        "networks.lora".to_string(),
        "--network_dim".to_string(),
        request.network_dim.to_string(),
        "--network_alpha".to_string(),
        request.network_alpha.to_string(),
        "--learning_rate".to_string(),
        format!("{:.8}", request.learning_rate),
        "--max_train_epochs".to_string(),
        request.max_train_epochs.to_string(),
        "--train_batch_size".to_string(),
        request.train_batch_size.to_string(),
        "--gradient_accumulation_steps".to_string(),
        request.gradient_accumulation_steps.to_string(),
        "--optimizer_type".to_string(),
        request.optimizer.trim().to_string(),
        "--mixed_precision".to_string(),
        request.mixed_precision.clone(),
        "--save_precision".to_string(),
        "fp16".to_string(),
        "--save_every_n_epochs".to_string(),
        request.save_every_n_epochs.to_string(),
        "--seed".to_string(),
        request.seed.to_string(),
        "--max_data_loader_n_workers".to_string(),
        request.max_data_loader_n_workers.to_string(),
        "--noise_offset".to_string(),
        format!("{:.4}", request.noise_offset),
    ];
    if request.gradient_checkpointing {
        args.push("--gradient_checkpointing".to_string());
    }
    if request.cache_latents {
        args.push("--cache_latents".to_string());
    }
    if request.network_train_unet_only {
        args.push("--network_train_unet_only".to_string());
    }
    if request.cache_text_encoder_outputs {
        args.push("--cache_text_encoder_outputs".to_string());
    }
    if request.flip_aug {
        args.push("--flip_aug".to_string());
    }
    args
}

/// sd-scripts refuses to cache text-encoder outputs while captions are
/// shuffled, so a copy of the dataset config with shuffling turned off is
/// written next to the original and used for this run.
///
/// Returns `None` when the config already has shuffling disabled.
pub fn disable_caption_shuffle(config: &str) -> Option<String> {
    let mut changed = false;
    let mut lines: Vec<String> = Vec::new();
    for line in config.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("shuffle_caption") {
            if let Some((_, value)) = rest.split_once('=') {
                if value.trim().eq_ignore_ascii_case("true") {
                    lines.push("shuffle_caption = false".to_string());
                    changed = true;
                    continue;
                }
            }
        }
        lines.push(line.to_string());
    }
    if !changed {
        return None;
    }
    let mut text = lines.join("\n");
    if config.ends_with('\n') {
        text.push('\n');
    }
    Some(text)
}

/// Parse one line of sd-scripts/tqdm output for progress information.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KohyaProgress {
    pub epoch: Option<u32>,
    pub step: Option<u32>,
    pub total_steps: Option<u32>,
    pub loss: Option<f64>,
}

pub fn parse_progress(line: &str) -> KohyaProgress {
    let mut progress = KohyaProgress::default();
    if let Some(rest) = line.split("epoch ").nth(1) {
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(value) = digits.parse::<u32>() {
            progress.epoch = Some(value);
        }
    }
    if let Some(rest) = line.split("loss=").nth(1) {
        let digits: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if let Ok(value) = digits.parse::<f64>() {
            progress.loss = Some(value);
        }
    }
    // tqdm prints `steps:  12%|#2| 180/1500 [00:20<02:30, 1.2it/s]`.
    if line.contains("steps:") {
        for chunk in line.split_whitespace() {
            let truncated = chunk.trim_end_matches(|c| matches!(c, '%' | ']' | '['));
            if let Some((left, right)) = truncated.split_once('/') {
                let left: String = left.chars().filter(char::is_ascii_digit).collect();
                let right: String = right.chars().filter(char::is_ascii_digit).collect();
                if let (Ok(step), Ok(total)) = (left.parse::<u32>(), right.parse::<u32>()) {
                    if total > 0 {
                        progress.step = Some(step);
                        progress.total_steps = Some(total);
                        break;
                    }
                }
            }
        }
    }
    progress
}

/// List the LoRA files a training run produced.
pub fn list_lora_files(output_dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(output_dir) else {
        return Vec::new();
    };
    let mut files: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .map(|ext| {
                    ext.eq_ignore_ascii_case("safetensors") || ext.eq_ignore_ascii_case("pt")
                })
                .unwrap_or(false)
        })
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_install(root: &Path) {
        std::fs::create_dir_all(root.join("sd-scripts")).unwrap();
        std::fs::create_dir_all(root.join("venv").join("Scripts")).unwrap();
        std::fs::write(root.join("sd-scripts").join(SD15_TRAINER), b"# trainer").unwrap();
        std::fs::write(root.join("sd-scripts").join(SDXL_TRAINER), b"# trainer").unwrap();
        std::fs::write(
            root.join("venv").join("Scripts").join("python.exe"),
            b"# python",
        )
        .unwrap();
    }

    fn scratch(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("kohya-{label}-{}", uuid::Uuid::new_v4()))
    }

    fn request(root: &Path) -> TrainingRequest {
        TrainingRequest {
            dataset_config: root.join("dataset.toml").to_string_lossy().into_owned(),
            output_dir: root.join("out").to_string_lossy().into_owned(),
            output_name: "character_alpha".to_string(),
            base_model: root.join("base.safetensors").to_string_lossy().into_owned(),
            trainer: TrainerKind::Sd15,
            network_dim: 32,
            network_alpha: 16,
            learning_rate: 0.0001,
            max_train_epochs: 10,
            train_batch_size: 1,
            gradient_accumulation_steps: 1,
            optimizer: "AdamW8bit".to_string(),
            mixed_precision: "fp16".to_string(),
            save_every_n_epochs: 1,
            seed: 42,
            max_data_loader_n_workers: 2,
            gradient_checkpointing: true,
            cache_latents: true,
            network_train_unet_only: true,
            cache_text_encoder_outputs: false,
            flip_aug: false,
            noise_offset: 0.0,
        }
    }

    #[test]
    fn resolves_a_kohya_ss_layout_with_a_bundled_venv() {
        let root = scratch("layout");
        fake_install(&root);

        let paths = resolve_paths(&root.to_string_lossy(), "").expect("paths resolve");

        assert!(paths.sd_scripts.ends_with("sd-scripts"));
        assert!(paths.python.ends_with("python.exe"));
        assert_eq!(paths.python_source, PythonSource::VirtualEnv);
        assert_eq!(paths.trainers.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_a_bare_sd_scripts_clone_and_prefers_an_explicit_python() {
        let root = scratch("bare");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(SD15_TRAINER), b"# trainer").unwrap();
        let python = root.join("custom-python.exe");
        std::fs::write(&python, b"# python").unwrap();

        let paths = resolve_paths(&root.to_string_lossy(), &python.to_string_lossy())
            .expect("paths resolve");

        assert_eq!(paths.sd_scripts, root);
        assert_eq!(paths.python_source, PythonSource::Settings);
        assert_eq!(paths.trainers, vec![SD15_TRAINER.to_string()]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_an_unconfigured_or_unknown_root() {
        assert!(resolve_paths("", "").is_err());
        let empty = scratch("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(resolve_paths(&empty.to_string_lossy(), "")
            .unwrap_err()
            .contains("train_network.py"));
        std::fs::remove_dir_all(empty).unwrap();
    }

    #[test]
    fn unconfigured_status_explains_what_is_missing() {
        let status = status("", "", "");

        assert!(!status.configured);
        assert!(!status.installed);
        assert!(status.error.unwrap().contains("还没有配置"));
    }

    #[test]
    fn builds_a_sd_scripts_argument_list() {
        let root = scratch("args");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("dataset.toml"), b"[general]").unwrap();
        std::fs::write(root.join("base.safetensors"), b"# weights").unwrap();
        let request = request(&root);

        request.validate().expect("request is valid");
        let args = build_arguments(&request, &root.join("logs"));

        assert!(args.contains(&"--dataset_config".to_string()));
        assert!(args.contains(&"--gradient_checkpointing".to_string()));
        assert!(args.contains(&"--cache_latents".to_string()));
        assert!(args.contains(&"--network_train_unet_only".to_string()));
        assert!(!args.contains(&"--flip_aug".to_string()));
        let dim = args.iter().position(|arg| arg == "--network_dim").unwrap();
        assert_eq!(args[dim + 1], "32");
        let lr = args
            .iter()
            .position(|arg| arg == "--learning_rate")
            .unwrap();
        assert_eq!(args[lr + 1], "0.00010000");
        // The dataset config owns resolution/batch layout, so the CLI must not
        // repeat them or sd-scripts refuses to start.
        assert!(!args.iter().any(|arg| arg == "--resolution"));
        assert!(!args.iter().any(|arg| arg == "--train_data_dir"));
        // Batch size and caption extension come from the CLI / dataset config
        // respectively; duplicating either makes sd-scripts abort.
        assert!(args.iter().any(|arg| arg == "--train_batch_size"));
        assert!(!args.iter().any(|arg| arg == "--caption_extension"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn caption_shuffle_is_disabled_for_text_encoder_caching() {
        let config = "[general]\nshuffle_caption = true\nkeep_tokens = 3\n\n[[datasets]]\nresolution = 1024\n";

        let rewritten = disable_caption_shuffle(config).expect("shuffling is turned off");

        assert!(rewritten.contains("shuffle_caption = false"));
        assert!(rewritten.contains("keep_tokens = 3"));
        assert!(rewritten.ends_with('\n'));
        // Nothing to do when shuffling is already disabled.
        assert!(disable_caption_shuffle(&rewritten).is_none());
        assert!(disable_caption_shuffle("[general]\nkeep_tokens = 1\n").is_none());
    }

    #[test]
    fn text_encoder_caching_requires_unet_only_training() {
        let root = scratch("te-cache");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("dataset.toml"), b"[general]").unwrap();
        std::fs::write(root.join("base.safetensors"), b"# weights").unwrap();
        let request = TrainingRequest {
            cache_text_encoder_outputs: true,
            network_train_unet_only: false,
            ..request(&root)
        };

        let error = request.validate().expect_err("combination is rejected");

        assert!(error.contains("仅训练 UNet"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn caching_flag_is_forwarded_to_sd_scripts() {
        let root = scratch("te-args");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("dataset.toml"), b"[general]").unwrap();
        std::fs::write(root.join("base.safetensors"), b"# weights").unwrap();
        let request = TrainingRequest {
            cache_text_encoder_outputs: true,
            ..request(&root)
        };

        request.validate().expect("request is valid");
        let args = build_arguments(&request, &root.join("logs"));

        assert!(args.iter().any(|arg| arg == "--cache_text_encoder_outputs"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_invalid_training_values() {
        let root = scratch("invalid");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("dataset.toml"), b"[general]").unwrap();
        std::fs::write(root.join("base.safetensors"), b"# weights").unwrap();
        let request = request(&root);
        request.validate().expect("baseline is valid");

        let bad_epochs = TrainingRequest {
            max_train_epochs: 0,
            ..request.clone()
        };
        assert!(bad_epochs.validate().is_err());

        let bad_name = TrainingRequest {
            output_name: "../escape".to_string(),
            ..request.clone()
        };
        assert!(bad_name.validate().is_err());

        let bad_lr = TrainingRequest {
            learning_rate: 5.0,
            ..request.clone()
        };
        assert!(bad_lr.validate().is_err());

        let missing_base = TrainingRequest {
            base_model: root.join("nope.safetensors").to_string_lossy().into_owned(),
            ..request.clone()
        };
        assert!(missing_base.validate().unwrap_err().contains("底模"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_tqdm_step_and_loss_lines() {
        let progress =
            parse_progress("steps:  12%|#2| 180/1500 [00:20<02:30, 1.2it/s, loss=0.0842]");

        assert_eq!(progress.step, Some(180));
        assert_eq!(progress.total_steps, Some(1500));
        assert_eq!(progress.loss, Some(0.0842));
        assert_eq!(parse_progress("epoch 3/10").epoch, Some(3));
        assert_eq!(parse_progress("loading model").step, None);
    }

    #[test]
    fn lists_only_lora_artifacts() {
        let root = scratch("artifacts");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("a.safetensors"), b"# lora").unwrap();
        std::fs::write(root.join("notes.txt"), b"log").unwrap();
        std::fs::write(root.join("sub").join("b.safetensors"), b"# nested").unwrap();

        let files = list_lora_files(&root);

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("a.safetensors"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
