import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  AlertCircle,
  CircleCheck,
  FolderOpen,
  GraduationCap,
  Loader2,
  Play,
  RefreshCw,
  Square,
} from "lucide-react";
import type {
  KohyaInstallResult,
  KohyaStatus,
  KohyaTrainerKind,
  KohyaTrainingRequest,
  KohyaTrainingStatus,
} from "@anime-pic-manage/shared-types";
import { invokeCore, isTauriRuntime } from "../lib/tauri";

type PresetId = "sd15_6gb" | "sdxl_6gb" | "custom";

const PRESETS: Array<{
  id: PresetId;
  label: string;
  hint: string;
  trainer: KohyaTrainerKind;
  networkDim: number;
  networkAlpha: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
  optimizer: string;
  precision: string;
  cacheTextEncoderOutputs: boolean;
}> = [
  {
    id: "sd15_6gb",
    label: "SD1.5 · 6GB 显存",
    hint: "512 分辨率起步，AdamW8bit + 梯度检查点，最稳",
    trainer: "sd15",
    networkDim: 32,
    networkAlpha: 16,
    learningRate: 0.0001,
    epochs: 10,
    batchSize: 1,
    optimizer: "AdamW8bit",
    precision: "fp16",
    // SD1.5 fits 6 GB with the text encoder still trainable.
    cacheTextEncoderOutputs: false,
  },
  {
    id: "sdxl_6gb",
    label: "SDXL · 6GB 显存",
    hint: "分辨率交给训练集配置（建议 768），更省显存",
    trainer: "sdxl",
    networkDim: 16,
    networkAlpha: 8,
    learningRate: 0.0001,
    epochs: 8,
    batchSize: 1,
    optimizer: "AdamW8bit",
    precision: "fp16",
    // 6 GB SDXL only fits when both CLIPs are dropped after caching.
    cacheTextEncoderOutputs: true,
  },
  {
    id: "custom",
    label: "自定义",
    hint: "完全手动指定下面的参数",
    trainer: "sd15",
    networkDim: 32,
    networkAlpha: 16,
    learningRate: 0.0001,
    epochs: 10,
    batchSize: 1,
    optimizer: "AdamW8bit",
    precision: "fp16",
    cacheTextEncoderOutputs: false,
  },
];

const OPTIMIZERS = [
  "AdamW8bit",
  "AdamW",
  "Lion",
  "Prodigy",
  "AdaFactor",
  "SGDNesterov8bit",
];

function basename(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

/**
 * Drives the user's own kohya/sd-scripts install. The application never
 * bundles or downloads the trainer; it only builds the argument list and
 * streams the run's output back here.
 */
export function TrainingPanel({
  datasetToml,
  outputName,
  onNotice,
  onError,
}: {
  datasetToml: string;
  outputName: string;
  onNotice: (message: string | null) => void;
  onError: (message: string | null) => void;
}) {
  const [status, setStatus] = useState<KohyaStatus | null>(null);
  const [run, setRun] = useState<KohyaTrainingStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [preset, setPreset] = useState<PresetId>("sd15_6gb");
  const [trainer, setTrainer] = useState<KohyaTrainerKind>("sd15");
  const [config, setConfig] = useState(datasetToml);
  const [name, setName] = useState(outputName);
  const [outputDir, setOutputDir] = useState("");
  const [networkDim, setNetworkDim] = useState(32);
  const [networkAlpha, setNetworkAlpha] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(1);
  const [optimizer, setOptimizer] = useState("AdamW8bit");
  const [precision, setPrecision] = useState("fp16");
  const [dataLoaderWorkers, setDataLoaderWorkers] = useState(2);
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);
  const [cacheLatents, setCacheLatents] = useState(true);
  const [unetOnly, setUnetOnly] = useState(true);
  const [cacheTextEncoderOutputs, setCacheTextEncoderOutputs] = useState(false);
  const [installed, setInstalled] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const refresh = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const response = await invokeCore<KohyaStatus>("kohya_status", "kohya.status");
    if (response.payload) {
      setStatus(response.payload);
      setOutputDir((current) => current || response.payload!.output_dir);
    }
  }, []);

  const refreshRun = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const response = await invokeCore<KohyaTrainingStatus>(
      "kohya_training_status",
      "kohya.status.run",
    );
    if (response.payload) setRun(response.payload);
  }, []);

  useEffect(() => {
    void refresh();
    void refreshRun();
  }, [refresh, refreshRun]);

  // The run also lives in the Rust manager, so re-attach after a remount.
  useEffect(() => {
    if (!isTauriRuntime()) return;
    let active = true;
    const unlisteners: Array<() => void> = [];
    void listen<KohyaTrainingStatus>("kohya://progress", (event) => {
      if (active) setRun(event.payload);
    }).then((stop) => unlisteners.push(stop));
    void listen<KohyaTrainingStatus>("kohya://finished", (event) => {
      if (!active) return;
      setRun(event.payload);
      onNotice(event.payload.message || "训练进程已结束。");
    }).then((stop) => unlisteners.push(stop));
    return () => {
      active = false;
      unlisteners.forEach((stop) => stop());
    };
  }, [onNotice]);

  useEffect(() => {
    if (datasetToml) setConfig(datasetToml);
  }, [datasetToml]);

  useEffect(() => {
    if (outputName) setName(outputName);
  }, [outputName]);

  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [run?.log_tail.length]);

  const applyPreset = (id: PresetId) => {
    setPreset(id);
    const picked = PRESETS.find((item) => item.id === id);
    if (!picked) return;
    setTrainer(picked.trainer);
    setNetworkDim(picked.networkDim);
    setNetworkAlpha(picked.networkAlpha);
    setLearningRate(picked.learningRate);
    setEpochs(picked.epochs);
    setBatchSize(picked.batchSize);
    setOptimizer(picked.optimizer);
    setPrecision(picked.precision);
    setCacheTextEncoderOutputs(picked.cacheTextEncoderOutputs);
    if (picked.cacheTextEncoderOutputs) setUnetOnly(true);
  };

  const start = async () => {
    if (busy) return;
    onError(null);
    setBusy("start");
    try {
      const request: KohyaTrainingRequest = {
        dataset_config: config.trim(),
        output_dir: outputDir.trim(),
        output_name: name.trim(),
        base_model: status?.base_model?.trim() ?? "",
        trainer,
        network_dim: networkDim,
        network_alpha: networkAlpha,
        learning_rate: learningRate,
        max_train_epochs: epochs,
        train_batch_size: batchSize,
        gradient_accumulation_steps: 1,
        optimizer,
        mixed_precision: precision,
        save_every_n_epochs: 1,
        seed: 42,
        max_data_loader_n_workers: dataLoaderWorkers,
        gradient_checkpointing: gradientCheckpointing,
        cache_latents: cacheLatents,
        network_train_unet_only: unetOnly,
        cache_text_encoder_outputs: cacheTextEncoderOutputs,
        flip_aug: false,
        noise_offset: 0,
      };
      const response = await invokeCore<{ log_path: string }>(
        "kohya_start_training",
        "kohya.train",
        { request },
      );
      if (response.error || !response.payload) {
        onError(response.error?.message ?? "启动训练失败。");
        return;
      }
      setInstalled(null);
      onNotice("训练已启动，日志会实时显示在下方。");
      await refreshRun();
    } finally {
      setBusy(null);
    }
  };

  const stop = async () => {
    if (busy) return;
    setBusy("stop");
    try {
      const response = await invokeCore<{ stopped: boolean }>(
        "kohya_stop_training",
        "kohya.stop",
      );
      if (response.error) {
        onError(response.error.message);
        return;
      }
      onNotice("已请求停止训练。");
      await refreshRun();
    } finally {
      setBusy(null);
    }
  };

  const install = async (source: string) => {
    setBusy(`install:${source}`);
    try {
      const response = await invokeCore<KohyaInstallResult>("kohya_install_lora", "kohya.install", {
        source,
      });
      if (response.error || !response.payload) {
        onError(response.error?.message ?? "安装 LoRA 失败。");
        return;
      }
      setInstalled(response.payload.destination);
      onNotice(
        `${response.payload.replaced ? "已覆盖" : "已安装"} LoRA：${response.payload.destination}`,
      );
    } finally {
      setBusy(null);
    }
  };

  const running = run?.running ?? false;
  const percent =
    run?.total_steps && run.step ? Math.min(100, (run.step / run.total_steps) * 100) : null;
  const statusText = !status
    ? "尚未检测"
    : !status.configured
      ? status.error ?? "未配置训练器"
      : !status.installed
        ? status.error ?? "找不到可用的 Python"
        : status.base_model_exists
          ? `已就绪 · ${status.python_source} · ${basename(status.python)}`
          : "已就绪，但还没在设置里选定训练底模";

  return (
    <section className="panel-card generation-dataset">
      <div className="generation-results-heading">
        <strong>LoRA 训练（本机 kohya）</strong>
        <small>
          调用你自己安装的 kohya_ss / sd-scripts，把上一步导出的训练集训练成 LoRA；本应用不打包、不修改训练器。
        </small>
      </div>

      <div className="banner-callout banner-info">
        <AlertCircle size={16} />
        <span>{statusText}</span>
      </div>

      <div className="generation-grid">
        <label className="generation-field">
          <span>训练集配置（dataset.toml）</span>
          <input
            type="text"
            value={config}
            onChange={(event) => setConfig(event.target.value)}
            placeholder="先在上一栏导出训练集"
          />
        </label>
        <label className="generation-field">
          <span>LoRA 名称</span>
          <input
            type="text"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="例如 character_alpha"
          />
        </label>
        <label className="generation-field">
          <span>输出目录</span>
          <input
            type="text"
            value={outputDir}
            onChange={(event) => setOutputDir(event.target.value)}
          />
        </label>
        <label className="generation-field">
          <span>底模（在设置 → 生图配置里选择）</span>
          <input
            type="text"
            value={status?.base_model ?? ""}
            readOnly
            placeholder="尚未选择底模"
          />
        </label>
      </div>

      <div className="generation-grid">
        <label className="generation-field">
          <span>预设</span>
          <select value={preset} onChange={(event) => applyPreset(event.target.value as PresetId)}>
            {PRESETS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}｜{item.hint}
              </option>
            ))}
          </select>
        </label>
        <label className="generation-field">
          <span>训练脚本</span>
          <select
            value={trainer}
            onChange={(event) => setTrainer(event.target.value as KohyaTrainerKind)}
          >
            <option value="sd15">train_network.py（SD1.5 / SD2.x）</option>
            <option value="sdxl">sdxl_train_network.py（SDXL）</option>
          </select>
        </label>
        <label className="generation-field">
          <span>network_dim / alpha</span>
          <div className="generation-inline">
            <input
              type="number"
              min={1}
              max={512}
              value={networkDim}
              onChange={(event) => setNetworkDim(Number(event.target.value))}
            />
            <input
              type="number"
              min={1}
              max={512}
              value={networkAlpha}
              onChange={(event) => setNetworkAlpha(Number(event.target.value))}
            />
          </div>
        </label>
        <label className="generation-field">
          <span>学习率 / 轮数 / batch</span>
          <div className="generation-inline">
            <input
              type="number"
              step={0.00001}
              min={0.000001}
              max={0.1}
              value={learningRate}
              onChange={(event) => setLearningRate(Number(event.target.value))}
            />
            <input
              type="number"
              min={1}
              max={1000}
              value={epochs}
              onChange={(event) => setEpochs(Number(event.target.value))}
            />
            <input
              type="number"
              min={1}
              max={16}
              value={batchSize}
              onChange={(event) => setBatchSize(Number(event.target.value))}
            />
          </div>
        </label>
      </div>

      <div className="generation-grid">
        <label className="generation-field">
          <span>优化器</span>
          <select value={optimizer} onChange={(event) => setOptimizer(event.target.value)}>
            {OPTIMIZERS.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label className="generation-field">
          <span>精度</span>
          <select value={precision} onChange={(event) => setPrecision(event.target.value)}>
            <option value="fp16">fp16（NVIDIA 推荐）</option>
            <option value="bf16">bf16</option>
            <option value="no">no（全精度，最慢）</option>
          </select>
        </label>
        <label className="generation-field">
          <span>数据加载进程</span>
          <input
            type="number"
            min={0}
            max={8}
            value={dataLoaderWorkers}
            onChange={(event) => setDataLoaderWorkers(Number(event.target.value))}
          />
        </label>
        <div className="generation-field">
          <span>显存与一致性</span>
          <div className="generation-inline">
            <label className="switch-control">
              <input
                type="checkbox"
                checked={gradientCheckpointing}
                onChange={(event) => setGradientCheckpointing(event.target.checked)}
              />
              <span>梯度检查点</span>
            </label>
            <label className="switch-control">
              <input
                type="checkbox"
                checked={cacheLatents}
                onChange={(event) => setCacheLatents(event.target.checked)}
              />
              <span>缓存 latents</span>
            </label>
            <label className="switch-control">
              <input
                type="checkbox"
                checked={unetOnly}
                onChange={(event) => {
                  setUnetOnly(event.target.checked);
                  if (!event.target.checked) setCacheTextEncoderOutputs(false);
                }}
              />
              <span>仅训练 UNet</span>
            </label>
            <label className="switch-control">
              <input
                type="checkbox"
                checked={cacheTextEncoderOutputs}
                onChange={(event) => {
                  setCacheTextEncoderOutputs(event.target.checked);
                  if (event.target.checked) setUnetOnly(true);
                }}
              />
              <span>缓存文本编码器输出（小显存）</span>
            </label>
          </div>
        </div>
      </div>

      {cacheTextEncoderOutputs && (
        <small className="generation-progress-text">
          文本编码器输出缓存后会释放两个 CLIP（SDXL 约 1.4GB），但本次不再训练文本编码器；
          因为 sd-scripts 要求关掉 caption 洗牌，应用会自动用同目录的
          <code>dataset.cache-te.toml</code> 训练。
        </small>
      )}

      <div className="generation-actions">
        <button
          className="primary-button"
          onClick={() => void start()}
          disabled={busy !== null || running || !status?.installed}
        >
          {busy === "start" ? <Loader2 size={16} className="spin" /> : <Play size={16} />}
          开始训练
        </button>
        <button className="ghost-button" onClick={() => void stop()} disabled={busy !== null || !running}>
          {busy === "stop" ? <Loader2 size={16} className="spin" /> : <Square size={15} />}
          停止训练
        </button>
        <button className="ghost-button" onClick={() => void refresh()} disabled={busy !== null}>
          <RefreshCw size={15} /> 重新检测
        </button>
        {run && !run.running && run.finished && (
          <span className="generation-progress-text">{run.message}</span>
        )}
        {running && (
          <span className="generation-progress-text">
            {percent !== null ? `进度 ${percent.toFixed(0)}%` : "训练中"} · 已用{" "}
            {formatDuration(run?.elapsed_secs ?? 0)}
            {run?.loss !== null && run?.loss !== undefined ? ` · loss ${run.loss.toFixed(4)}` : ""}
          </span>
        )}
      </div>

      {percent !== null && running && (
        <div className="training-progress">
          <div className="training-progress-bar" style={{ width: `${percent}%` }} />
        </div>
      )}

      {status && status.installed && status.arguments_preview.length > 0 && (
        <details className="training-command">
          <summary>查看将要执行的命令（默认参数预览）</summary>
          <code>
            {basename(status.python)} {status.trainers[0]} {status.arguments_preview.join(" ")}
          </code>
        </details>
      )}

      {run && run.log_tail.length > 0 && (
        <pre className="training-log" ref={logRef}>
          {run.log_tail.join("\n")}
        </pre>
      )}

      {run && run.artifacts.length > 0 && (
        <div className="generation-caption-preview">
          <strong>训练产物</strong>
          {run.artifacts.map((artifact) => (
            <div key={artifact} className="generation-caption-row">
              <code>{basename(artifact)}</code>
              <button
                className="ghost-button"
                onClick={() => void install(artifact)}
                disabled={busy !== null}
              >
                <GraduationCap size={14} /> 安装到 ComfyUI
              </button>
              <button
                className="ghost-button"
                onClick={() =>
                  void invokeCore<void>("show_item_in_folder", "system.explorer.show", {
                    path: artifact,
                  })
                }
              >
                <FolderOpen size={14} /> 定位
              </button>
            </div>
          ))}
          {installed && (
            <small>
              <CircleCheck size={12} /> 已安装到 <code>{installed}</code>
            </small>
          )}
        </div>
      )}

      {run?.log_path && (
        <small className="generation-progress-text">
          完整日志：<code>{run.log_path}</code>
        </small>
      )}
    </section>
  );
}
