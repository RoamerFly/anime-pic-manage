import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  Activity,
  AlertCircle,
  ChevronRight,
  CircleCheck,
  CircleHelp,
  Cpu,
  Download,
  FolderCog,
  Github,
  ImagePlus,
  Info,
  LoaderCircle,
  Package,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import type {
  AppSettings,
  ComfyModelList,
  ComfyStatus,
  CudaInstallProgress,
  CudaRuntimeStatus,
  GpuInventory,
  KohyaEnvironment,
  KohyaStatus,
  ModelInventory,
  ModelInventoryEntry,
  ReferenceLibraryStatus,
  RuntimeStatus,
  SettingsTab,
  UpdateInfo,
  UpdateProgress,
  WorkerCapabilities,
  WorkerComputeDevice,
  WorkerComputeStatus,
  WorkerRuntimeMode,
  WorkerRuntimeSettings,
} from "@anime-pic-manage/shared-types";
import { invokeCore, isTauriRuntime } from "../lib/tauri";
import { open } from "@tauri-apps/plugin-dialog";
import {
  applyUiFontSize,
  cacheUiFontSize,
  readCachedUiFontSize,
  stepUiFontSize,
  UI_FONT_BASE_PX,
  UI_FONT_MAX_PX,
  UI_FONT_MIN_PX,
  UI_FONT_STEP_PX,
} from "../lib/ui-font";
import {
  registerSettingsSave,
  updateSettingsSaveState,
} from "../lib/settings-bridge";
import {
  describeCudaAvailability,
  describeCudaRuntime,
  describeMeasuredCompute,
  describeProviders,
  nvidiaGpuNames,
} from "../lib/gpu-label";

const DEFAULT_APP_SETTINGS: AppSettings = {
  recognition_onnx_threads: 4,
  recognition_skip_annotated: true,
  similarity_workers: 8,
  similarity_threshold: 0.85,
  similarity_include_subfolders: true,
  similarity_archive_dir: "_duplicates",
  ui_font_size: readCachedUiFontSize(),
  comfy_root: "",
  comfy_port: 8188,
  comfy_low_vram: true,
  comfy_auto_start: false,
  comfy_output_dir: "",
  lora_trainer_root: "",
  lora_trainer_python: "",
  lora_trainer_base_model: "",
  lora_trainer_output_dir: "",
  cuda_runtime_dir: "",
  recognition_recognizer_model: "",
  background_priority: "below_normal",
  reference_matching_enabled: false,
  reference_backend: "ccip",
};

const SETTINGS_TABS: Array<{
  id: SettingsTab;
  label: string;
  hint: string;
  icon: typeof Activity;
}> = [
  {
    id: "environment",
    label: "基础配置",
    hint: "运行方式、推理设备与依赖状态",
    icon: FolderCog,
  },
  {
    id: "recognition",
    label: "识别配置",
    hint: "推理并行度与识别扫描默认值",
    icon: Cpu,
  },
  {
    id: "similarity",
    label: "相似度配置",
    hint: "并行度、默认阈值与归档行为",
    icon: SlidersHorizontal,
  },
  {
    id: "models",
    label: "模型配置",
    hint: "自带与可下载模型的清单、状态与存储位置",
    icon: Package,
  },
  {
    id: "comfy",
    label: "生图配置",
    hint: "ComfyUI 路径、端口与输出目录",
    icon: ImagePlus,
  },
];


export const developerUrl = "https://github.com/RoamerFly";
export const repositoryUrl = "https://github.com/RoamerFly/anime-pic-manage";
export const issuesUrl = `${repositoryUrl}/issues`;

export function RuntimeDependencyCard({
  label,
  dependency,
}: {
  label: string;
  dependency: WorkerCapabilities["dghs_imgutils"];
}) {
  const ready = dependency.status === "ready";
  return (
    <div className={`runtime-dependency ${ready ? "ready" : "unavailable"}`}>
      <div>
        <strong>{label}</strong>
        <small>
          {dependency.version
            ? `版本 ${dependency.version}`
            : dependency.error?.message ?? "版本不可用"}
        </small>
      </div>
      <span>{ready ? "已就绪" : "不可用"}</span>
    </div>
  );
}

export function WorkerRuntimeSettingsPanel({
  status,
  gpus,
  compute,
  mode,
  device,
  busy = false,
  onChangeMode,
  onChangeDevice,
  cudaRuntimeDir,
  onChangeCudaRuntimeDir,
  onCudaInstalled,
  backgroundPriority,
  onChangeBackgroundPriority,
}: {
  status: RuntimeStatus | null;
  gpus?: GpuInventory | null;
  compute?: WorkerComputeStatus | null;
  mode: WorkerRuntimeMode;
  device: WorkerComputeDevice;
  busy?: boolean;
  onChangeMode: (mode: WorkerRuntimeMode) => void;
  onChangeDevice: (device: WorkerComputeDevice) => void;
  cudaRuntimeDir: string;
  onChangeCudaRuntimeDir: (value: string) => void;
  /** Re-probe the engine after the runtime was downloaded (it restarts). */
  onCudaInstalled?: () => void;
  backgroundPriority: AppSettings["background_priority"];
  onChangeBackgroundPriority: (value: AppSettings["background_priority"]) => void;
}) {
  const worker = status?.worker;
  const resolvedPath = worker?.runtime_path ?? null;
  const healthMessage = worker?.message ?? "等待健康检查";
  const capabilities = worker?.capabilities ?? null;
  const capabilityError = worker?.capability_error ?? null;
  const deviceLabel = worker?.compute_device_label ?? "自动（优先显卡）";
  const cudaAvailable = capabilities?.compute?.cuda_available ?? false;
  const availableProviders = describeProviders(
    capabilities?.compute?.available_providers ?? [],
  );
  const measured = describeMeasuredCompute(compute ?? null, gpus ?? null);
  const nvidiaCards = nvidiaGpuNames(gpus?.devices);
  const [cudaRuntime, setCudaRuntime] = useState<CudaRuntimeStatus | null>(null);
  const [cudaProgress, setCudaProgress] = useState<CudaInstallProgress | null>(null);
  const [installing, setInstalling] = useState(false);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void (async () => {
      const response = await invokeCore<CudaRuntimeStatus>(
        "cuda_runtime_status",
        "cuda.status",
      );
      setCudaRuntime(response.payload ?? null);
    })();
    let active = true;
    let stop: (() => void) | undefined;
    void listen<CudaInstallProgress>("cuda://install", (event) => {
      if (active) setCudaProgress(event.payload);
    }).then((unlisten) => {
      if (active) stop = unlisten;
      else unlisten();
    });
    return () => {
      active = false;
      stop?.();
    };
  }, []);

  const installCudaRuntime = async () => {
    setInstalling(true);
    setCudaProgress({
      phase: "resolving",
      current: 0,
      total: 0,
      message: "正在准备下载…",
    });
    try {
      const response = await invokeCore<CudaRuntimeStatus>(
        "install_cuda_runtime",
        "cuda.install",
      );
      setCudaRuntime(response.payload ?? null);
      if (response.error) {
        setCudaProgress({
          phase: "error",
          current: 0,
          total: 0,
          message: response.error.message,
        });
      } else {
        onCudaInstalled?.();
      }
    } finally {
      setInstalling(false);
    }
  };

  const pickCudaRuntimeDir = async () => {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择包含 CUDA 12 / cuDNN 9 DLL 的目录",
    });
    if (typeof selected === "string" && selected.trim()) {
      onChangeCudaRuntimeDir(selected.trim());
    }
  };

  return (
    <section className="settings-section runtime-settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <WandSparkles size={19} />
        </div>
        <div>
          <h3>推理运行环境</h3>
          <p>
            选择本机推理引擎；切换会停止旧进程，并在下次检查时使用新方式。
          </p>
        </div>
      </div>
      <div className="runtime-mode-controls">
        <label htmlFor="worker-runtime-mode">
          <strong>运行方式</strong>
          <small>
            内置推理引擎适合日常识别；兼容环境用于缺少内置引擎时的备用运行。
          </small>
        </label>
        <div className="runtime-mode-actions">
          <select
            id="worker-runtime-mode"
            value={mode}
            onChange={(event) =>
              onChangeMode(event.target.value as WorkerRuntimeMode)
            }
            disabled={busy}
          >
            <option value="executable">内置推理引擎</option>
            <option value="embedded_env">兼容环境</option>
          </select>
        </div>
      </div>
      <div className="runtime-mode-controls">
        <label htmlFor="worker-compute-device">
          <strong>推理设备</strong>
          <small>
            {describeCudaAvailability(gpus ?? null, capabilities?.compute)}
            {measured ? ` 实测结果：${measured.summary}。` : ""}
          </small>
        </label>
        <div className="runtime-mode-actions">
          <select
            id="worker-compute-device"
            value={device}
            onChange={(event) =>
              onChangeDevice(event.target.value as WorkerComputeDevice)
            }
            disabled={busy}
          >
            <option value="auto">自动（优先显卡）</option>
            <option value="cpu">仅 CPU</option>
            <option value="cuda">NVIDIA CUDA</option>
          </select>
        </div>
      </div>
      <div className="settings-row runtime-path-row">
        <div>
          <strong>当前解析路径</strong>
          <small>{healthMessage}</small>
        </div>
        <code title={resolvedPath ?? undefined}>
          {resolvedPath ?? "当前模式未找到可用 Worker"}
        </code>
      </div>
      <div className="settings-row runtime-path-row">
        <div>
          <strong>当前推理设备</strong>
          <small>
            {measured
              ? `实测：${measured.summary} · ${measured.detail}`
              : availableProviders.length > 0
                ? `可用执行提供器：${availableProviders.join("、")}${
                    nvidiaCards.length > 0
                      ? ` · 显卡：${nvidiaCards.join("、")}`
                      : gpus?.devices?.length
                        ? ` · 显卡：${gpus.devices.join("、")}`
                        : ""
                  }`
                : "尚未完成运行能力检查"}
          </small>
        </div>
        <code title={deviceLabel}>{deviceLabel}</code>
      </div>
      <div className="settings-field settings-field-stacked">
        <div>
          <strong>CUDA 运行时目录（可选）</strong>
          <small>
            {describeCudaRuntime(capabilities?.compute)}
            {cudaRuntimeDir.trim()
              ? " 改完保存后会自动重启推理引擎。"
              : ""}
          </small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={cudaRuntimeDir}
            placeholder="留空使用系统 PATH"
            onChange={(event) => onChangeCudaRuntimeDir(event.target.value)}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickCudaRuntimeDir()}
          >
            选择目录
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={() =>
              void invokeCore<void>("open_external_url", "system.open_url", {
                url: "https://developer.nvidia.com/cuda-toolkit-archive",
              })
            }
          >
            CUDA 12 下载
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={() =>
              void invokeCore<void>("open_external_url", "system.open_url", {
                url: "https://developer.nvidia.com/cudnn-downloads",
              })
            }
          >
            cuDNN 9 下载
          </button>
        </div>
        <div className="generation-inline">
          <button
            type="button"
            className="primary-button"
            disabled={installing || !isTauriRuntime()}
            onClick={() => void installCudaRuntime()}
          >
            {installing ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Sparkles size={15} />
            )}
            一键下载 CUDA 运行时（约 1.4GB）
          </button>
          <small>
            {cudaProgress?.message ??
              (cudaRuntime?.installed
                ? `已下载 ${cudaRuntime.packages.length} 个 DLL · ${cudaRuntime.size_mb} MB · 位于 ${cudaRuntime.directory}`
                : "从 PyPI 下载 NVIDIA 官方运行库到 app\\cuda，无需在系统里安装 CUDA，也不会重分发这些文件。")}
          </small>
        </div>
      </div>
      <div className="settings-field settings-field-stacked">
        <div>
          <strong>后台任务优先级</strong>
          <small>
            推理引擎、ComfyUI 与 LoRA 训练都以子进程运行。选「低调」时它们会让出 CPU
            给桌面和其他程序，机器不会卡；选「普通」时跑得更快，但长时间扫描可能让界面发涩。
            调整后会自动重启推理引擎。
          </small>
        </div>
        <div className="generation-inline">
          <select
            value={backgroundPriority}
            onChange={(event) =>
              onChangeBackgroundPriority(
                event.target.value as AppSettings["background_priority"],
              )
            }
          >
            <option value="below_normal">低调（推荐，不抢 CPU）</option>
            <option value="normal">普通（跑满 CPU，更快）</option>
          </select>
          <small>线程数仍由「识别配置」与「相似度配置」分别控制。</small>
        </div>
      </div>
      <div className="runtime-capabilities" aria-live="polite">
        <div className="runtime-capabilities-heading">
          <strong>运行能力</strong>
          <small>实际导入 dghs-imgutils 与 ONNX Runtime 模块，不加载模型。</small>
        </div>
        {capabilities ? (
          <>
            <div className="runtime-dependency-grid">
              <RuntimeDependencyCard
                label="dghs-imgutils"
                dependency={capabilities.dghs_imgutils}
              />
              <RuntimeDependencyCard
                label="ONNX Runtime"
                dependency={capabilities.onnxruntime}
              />
            </div>
            <div className="runtime-feature-list">
              {Object.entries(capabilities.features).map(([name, feature]) => (
                <span
                  className={
                    feature.status === "ready"
                      ? "runtime-feature ready"
                      : "runtime-feature unavailable"
                  }
                  key={name}
                >
                  <i />
                  {name}
                  {feature.status === "ready"
                    ? "已就绪"
                    : feature.error?.message ?? "不可用"}
                </span>
              ))}
            </div>
            {!capabilities.ready && (
              <div className="runtime-feedback error">
                <AlertCircle size={14} />
                <span>
                  {capabilities.errors[0]?.message ??
                    "必需运行能力不可用，无法开始识别。"}
                </span>
              </div>
            )}
          </>
        ) : (
          <div className="runtime-feedback error">
            <AlertCircle size={14} />
            <span>{capabilityError ?? "尚未完成运行能力检查。"}</span>
          </div>
        )}
      </div>
    </section>
  );
}

const WORKER_STATUS_LABELS: Record<string, string> = {
  healthy: "已就绪",
  starting: "启动中",
  stopped: "已停止",
  unavailable: "不可用",
  error: "异常",
};

function formatCheckedAt(value?: string | null): string | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds > 0) {
    return new Date(seconds * 1000).toLocaleString();
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

export function EnvironmentCheckSection({
  status,
  gpus,
  compute,
  onProbeCompute,
  onRefresh,
}: {
  status: RuntimeStatus | null;
  gpus?: GpuInventory | null;
  compute?: WorkerComputeStatus | null;
  onProbeCompute?: () => Promise<void> | void;
  onRefresh?: () => Promise<void> | void;
}) {
  const [current, setCurrent] = useState<RuntimeStatus | null>(status);
  const [checking, setChecking] = useState(false);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setCurrent(status);
  }, [status]);

  const runCheck = async () => {
    if (checking) return;
    setChecking(true);
    setError(null);
    // The CUDA self-test loads the installed models, so it reports what the
    // sessions really use instead of what the runtime merely advertises.
    const probePromise = onProbeCompute?.();
    const response = await invokeCore<RuntimeStatus>(
      "get_runtime_status",
      "system.health",
    );
    if (response.error || !response.payload) {
      setError(response.error?.message ?? "环境状态检查失败。");
    } else {
      setCurrent(response.payload);
      setCheckedAt(new Date().toLocaleString());
      await onRefresh?.();
    }
    try {
      await probePromise;
    } catch {
      setError((current) => current ?? "推理设备自检未能执行。");
    }
    setChecking(false);
  };

  const worker = current?.worker;
  const capabilities = worker?.capabilities ?? null;
  const database = current?.database;
  const databaseReady = database?.status === "ready";
  const workerReady = worker?.status === "healthy";
  const capabilitiesReady = capabilities?.ready === true;
  const cudaAvailable = capabilities?.compute?.cuda_available ?? false;
  const providers = describeProviders(
    capabilities?.compute?.available_providers ?? [],
  );
  const lastChecked = checkedAt ?? formatCheckedAt(worker?.checked_at);
  const measured = describeMeasuredCompute(compute ?? null, gpus ?? null);
  const nvidiaCards = nvidiaGpuNames(gpus?.devices);

  const rows = [
    {
      key: "database",
      label: "本地数据库",
      detail: databaseReady
        ? `schema v${database?.schema_version} · ${database?.table_count} 张表`
        : database?.message ?? "尚未读取数据库状态",
      ok: databaseReady,
      when: "数据库健康检查失败",
    },
    {
      key: "worker",
      label: "AI Worker",
      detail: worker?.message ?? "尚未完成健康握手",
      ok: workerReady,
      when: "Worker 未就绪",
    },
    {
      key: "dependencies",
      label: "运行依赖",
      detail: capabilitiesReady
        ? "dghs-imgutils 与 ONNX Runtime 已就绪"
        : capabilities?.errors[0]?.message ?? "尚未完成运行能力检查",
      ok: capabilitiesReady,
      when: "运行依赖不可用",
    },
    {
      key: "providers",
      label: "推理设备",
      detail: measured
        ? `${measured.summary}${
            nvidiaCards.length > 0 ? `（${nvidiaCards.join("、")}）` : ""
          } · ${worker?.compute_device_label ?? "自动"}`
        : providers.length
          ? `${providers.join("、")} · ${worker?.compute_device_label ?? "自动"}${
              nvidiaCards.length > 0
                ? ` · ${nvidiaCards.join("、")}`
                : cudaAvailable
                  ? ""
                  : " · 未检测到 CUDA"
            }`
          : "尚未检测到执行提供器",
      ok: measured ? measured.ok : providers.length > 0,
      when: measured ? "未使用 CUDA" : "执行提供器未检测",
    },
    {
      key: "compute-probe",
      label: "CUDA 实测",
      detail:
        measured?.detail ??
        "尚未实测；点击“开始检查”会用已安装模型实际加载一次，确认是否真的跑在 CUDA 上。",
      ok: measured?.ok ?? false,
      when: "未实测",
    },
    {
      key: "cuda-runtime",
      label: "CUDA 运行时",
      detail: describeCudaRuntime(capabilities?.compute),
      // Only a problem when the build offers CUDA but the runtime is missing.
      ok:
        cudaAvailable
          ? capabilities?.compute?.cuda_runtime?.status !== "missing"
          : true,
      when: "缺少 CUDA / cuDNN 动态库",
    },
    {
      key: "models",
      label: "已安装模型",
      detail: `${worker?.model_count ?? 0} 个模型可用于识别`,
      ok: (worker?.model_count ?? 0) > 0,
      when: "未发现已安装模型",
    },
  ];
  const blocking = rows.filter((row) => !row.ok);

  return (
    <section className="settings-section environment-check-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <Activity size={19} />
        </div>
        <div>
          <h3>环境状态检查</h3>
          <p>一次性确认数据库、Worker、运行依赖、推理设备与模型是否可用。</p>
        </div>
      </div>
      <div className="environment-check-actions">
        <div>
          <strong>
            {blocking.length === 0
              ? "全部检查项通过，可以开始识别"
              : `${blocking.length} 项需要处理`}
          </strong>
          <small>{lastChecked ? `最近检查：${lastChecked}` : "尚未执行检查"}</small>
        </div>
        <button
          className="primary-button"
          onClick={() => void runCheck()}
          disabled={checking || !isTauriRuntime()}
        >
          {checking ? (
            <LoaderCircle size={15} className="spin" />
          ) : (
            <Activity size={15} />
          )}
          开始检查
        </button>
      </div>
      {error && (
        <div className="runtime-feedback error">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}
      <div className="environment-check-list">
        {rows.map((row) => (
          <div
            className={`environment-check-item ${row.ok ? "ready" : "unavailable"}`}
            key={row.key}
          >
            <i />
            <strong>{row.label}</strong>
            <small title={row.detail}>{row.detail}</small>
            <span>{row.ok ? "正常" : row.when}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function AboutUpdateModal({
  show,
  currentVersion,
  onClose,
}: {
  show: boolean;
  currentVersion: string;
  onClose: () => void;
}) {
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [progress, setProgress] = useState<UpdateProgress | null>(null);
  const [checking, setChecking] = useState(false);
  const [installing, setInstalling] = useState(false);

  useEffect(() => {
    if (!show) return;
    let active = true;
    let unlisten: (() => void) | undefined;
    void listen<UpdateProgress>("update-progress", (event) => {
      if (active) setProgress(event.payload);
    })
      .then((stop) => {
        unlisten = stop;
      })
      .catch(() => undefined);
    return () => {
      active = false;
      unlisten?.();
    };
  }, [show]);

  if (!show) return null;

  const checkUpdate = async () => {
    setChecking(true);
    const response = await invokeCore<UpdateInfo>(
      "check_for_update",
      "updater.check",
    );
    setUpdate(
      response.payload ??
        (response.error
          ? {
              status: "unavailable",
              current_version: currentVersion,
              message: response.error.message,
            }
          : null),
    );
    setChecking(false);
  };

  const installUpdate = async () => {
    if (!update?.version) return;
    setInstalling(true);
    const response = await invokeCore<UpdateInfo>(
      "install_update",
      "updater.install",
      { expectedVersion: update.version },
    );
    if (response.payload) setUpdate(response.payload);
    setInstalling(false);
  };

  const cancelUpdate = async () => {
    await invokeCore<UpdateInfo>("cancel_update_download", "updater.install");
    setInstalling(false);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="about-modal-card"
        role="dialog"
        aria-modal="true"
        aria-label="关于与更新"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div className="about-modal-title">
            <Info size={19} />
            <h3>关于与更新</h3>
          </div>
          <button className="icon-close" onClick={onClose} aria-label="关闭">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <div className="about-version-row">
            <div>
              <strong>Anime Pic Manage</strong>
              <small>本地优先的动漫图片识别与整理工具</small>
            </div>
            <span className="soft-badge blue">v{currentVersion}</span>
          </div>
          <div className="about-links">
            <a href={developerUrl} target="_blank" rel="noreferrer">
              <Github size={17} />
              <span>
                <strong>开发者</strong>
                <small>RoamerFly</small>
              </span>
              <ChevronRight size={16} />
            </a>
            <a href={repositoryUrl} target="_blank" rel="noreferrer">
              <Sparkles size={17} />
              <span>
                <strong>项目仓库</strong>
                <small>查看源代码与版本</small>
              </span>
              <ChevronRight size={16} />
            </a>
            <a href={issuesUrl} target="_blank" rel="noreferrer">
              <CircleHelp size={17} />
              <span>
                <strong>问题反馈</strong>
                <small>报告问题或提出建议</small>
              </span>
              <ChevronRight size={16} />
            </a>
          </div>
          <div className="update-box">
            <div className="update-status">
              <span
                className={`update-indicator ${
                  update?.status === "available" ? "available" : ""
                }`}
              />
              {update?.message ?? "尚未检查更新"}
            </div>
            <div className="update-actions">
              <button
                className="ghost-button"
                disabled={checking || installing}
                onClick={() => void checkUpdate()}
              >
                {checking ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <RefreshCw size={16} />
                )}
                检查更新
              </button>
              {update?.status === "available" && (
                <button
                  className="primary-button"
                  disabled={installing}
                  onClick={() => void installUpdate()}
                >
                  {installing ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Sparkles size={16} />
                  )}
                  安装并重启
                </button>
              )}
              {installing && (
                <button
                  className="ghost-button"
                  onClick={() => void cancelUpdate()}
                >
                  取消下载
                </button>
              )}
            </div>
            {progress && (
              <div className="progress-wrap">
                <div className="progress-label">
                  <span>{progress.message}</span>
                  <span>
                    {progress.percent !== undefined ? `${progress.percent}%` : ""}
                  </span>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-bar"
                    style={{
                      width: `${Math.min(100, progress.percent ?? 0)}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
        <div className="modal-footer">
          <button className="ghost-button" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}

function AppearanceSettingsPanel({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  const fontSize = settings.ui_font_size;
  return (
    <section className="settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <SlidersHorizontal size={19} />
        </div>
        <div>
          <h3>界面显示</h3>
          <p>调整界面文字大小，改动会立即预览，保存后下次启动仍然生效。</p>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>界面字号</strong>
          <small>
            以 {UI_FONT_BASE_PX}px 为基准整体缩放界面文字，可拖动滑块无级调整，
            或用 − / + 按 {UI_FONT_STEP_PX}px 步进（{UI_FONT_MIN_PX}–
            {UI_FONT_MAX_PX}px）。
          </small>
        </div>
        <div className="font-size-control">
          <button
            type="button"
            className="ghost-button font-step-button"
            onClick={() => onChange({ ui_font_size: stepUiFontSize(fontSize, -1) })}
            disabled={fontSize <= UI_FONT_MIN_PX}
            title={`减小 ${UI_FONT_STEP_PX}px`}
          >
            −
          </button>
          <input
            type="range"
            min={UI_FONT_MIN_PX}
            max={UI_FONT_MAX_PX}
            step={UI_FONT_STEP_PX}
            value={fontSize}
            onChange={(event) =>
              onChange({ ui_font_size: Number(event.target.value) })
            }
            aria-label="界面字号"
          />
          <button
            type="button"
            className="ghost-button font-step-button"
            onClick={() => onChange({ ui_font_size: stepUiFontSize(fontSize, 1) })}
            disabled={fontSize >= UI_FONT_MAX_PX}
            title={`增大 ${UI_FONT_STEP_PX}px`}
          >
            ＋
          </button>
          <code>{fontSize.toFixed(1)} px</code>
          <button
            type="button"
            className="ghost-button font-reset-button"
            onClick={() => onChange({ ui_font_size: UI_FONT_BASE_PX })}
            disabled={fontSize === UI_FONT_BASE_PX}
          >
            恢复默认
          </button>
        </div>
      </div>

      <div className="settings-field-note">
        字号影响整个应用的文字（含识别检查、相似计算与设置页），不会改变图片与文件本身。
      </div>

    </section>
  );
}

function ComfySettingsPanel({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  const [status, setStatus] = useState<ComfyStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [models, setModels] = useState<ComfyModelList | null>(null);

  const check = useCallback(async () => {
    setChecking(true);
    const response = await invokeCore<ComfyStatus>("comfy_status", "comfy.status");
    if (response.payload) {
      setStatus(response.payload);
      if (response.payload.running) {
        const list = await invokeCore<ComfyModelList>("comfy_models", "comfy.models");
        setModels(list.payload ?? null);
      }
    }
    setChecking(false);
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const pickDirectory = async (key: "comfy_root" | "comfy_output_dir") => {
    const selected = await open({
      directory: true,
      multiple: false,
      title: key === "comfy_root" ? "选择 ComfyUI 安装目录（含 ComfyUI 与 python 的文件夹）" : "选择生成图输出目录",
    });
    if (typeof selected === "string" && selected.trim()) {
      onChange({ [key]: selected.trim() } as Partial<AppSettings>);
    }
  };

  const statusText = !status
    ? "尚未检测"
    : !status.configured
      ? "未配置目录"
      : !status.installed
        ? status.error ?? "目录无效"
        : status.running
          ? `运行中${status.version ? ` · v${status.version}` : ""}${
              status.device ? ` · ${status.device}` : ""
            }${status.vram_gb ? ` · ${status.vram_gb} GB 显存` : ""}`
          : `已停止${status.version ? ` · v${status.version}` : ""}`;

  return (
    <section className="settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <ImagePlus size={19} />
        </div>
        <div>
          <h3>AI 生图（ComfyUI）</h3>
          <p>
            使用你自己安装的 ComfyUI（不会随本应用分发）。填好目录后即可在「AI 生图」页面启动与出图。
          </p>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>ComfyUI 目录</strong>
          <small>
            指向包含 <code>ComfyUI</code> 与 <code>python</code> 两层的目录，例如
            <code>E:\freetime\AI_Draw\ComfyUI-aki\ComfyUI-aki-v3</code>；也接受直接指向 <code>ComfyUI</code> 本体。
          </small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={settings.comfy_root}
            placeholder="E:\\path\\to\\ComfyUI-aki-v3"
            onChange={(event) => onChange({ comfy_root: event.target.value })}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickDirectory("comfy_root")}
          >
            选择目录
          </button>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>生成图输出目录</strong>
          <small>
            留空则使用 <code>&lt;ComfyUI 根目录&gt;\Images\generated-manage</code>
            （带 <code>-manage</code> 后缀，便于和你在 ComfyUI 网页端自己出的图区分）。
          </small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={settings.comfy_output_dir}
            placeholder="留空使用默认目录"
            onChange={(event) =>
              onChange({ comfy_output_dir: event.target.value })
            }
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickDirectory("comfy_output_dir")}
          >
            选择目录
          </button>
        </div>
      </div>

      <div className="settings-field">
        <div>
          <strong>端口</strong>
          <small>与 ComfyUI 启动参数一致，默认 8188。</small>
        </div>
        <input
          type="number"
          min={1024}
          max={65535}
          className="settings-text-input"
          value={settings.comfy_port}
          onChange={(event) =>
            onChange({ comfy_port: Number(event.target.value) })
          }
        />
      </div>

      <div className="settings-field">
        <div>
          <strong>低显存模式（--lowvram）</strong>
          <small>显存 ≤ 8GB 建议开启；6GB 显卡跑 SDXL 基本必须开。</small>
        </div>
        <label className="switch-control">
          <input
            type="checkbox"
            checked={settings.comfy_low_vram}
            onChange={(event) =>
              onChange({ comfy_low_vram: event.target.checked })
            }
          />
          <span>{settings.comfy_low_vram ? "已启用" : "已关闭"}</span>
        </label>
      </div>

      <div className="settings-field">
        <div>
          <strong>随应用自动启动</strong>
          <small>打开「AI 生图」页面时，如果 ComfyUI 未运行则自动启动它。</small>
        </div>
        <label className="switch-control">
          <input
            type="checkbox"
            checked={settings.comfy_auto_start}
            onChange={(event) =>
              onChange({ comfy_auto_start: event.target.checked })
            }
          />
          <span>{settings.comfy_auto_start ? "已启用" : "已关闭"}</span>
        </label>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>当前状态</strong>
          <small>{statusText}</small>
        </div>
        <div className="generation-inline">
          <button
            type="button"
            className="ghost-button"
            onClick={() => void check()}
            disabled={checking}
          >
            {checking ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Activity size={15} />
            )}
            检测
          </button>
          {models && (
            <span className="muted">
              发现 {models.checkpoints.length} 个 checkpoint、{models.loras.length} 个 LoRA
            </span>
          )}
        </div>
      </div>

      <div className="settings-field-note">
        ComfyUI 采用 GPL-3.0 许可，本应用只调用你本机已安装的实例，不打包、不修改它的代码；生图与训练所需的模型与 LoRA 许可证由各自上游决定。
      </div>
    </section>
  );
}

function LoraTrainingSettingsPanel({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  const [status, setStatus] = useState<KohyaStatus | null>(null);
  const [environment, setEnvironment] = useState<KohyaEnvironment | null>(null);
  const [checking, setChecking] = useState(false);
  const [probing, setProbing] = useState(false);

  const check = useCallback(async () => {
    setChecking(true);
    const response = await invokeCore<KohyaStatus>("kohya_status", "kohya.status");
    setStatus(response.payload ?? null);
    setChecking(false);
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const probe = async () => {
    setProbing(true);
    const response = await invokeCore<KohyaEnvironment>(
      "kohya_probe_environment",
      "kohya.probe",
    );
    setEnvironment(response.payload ?? null);
    setProbing(false);
  };

  const pickPath = async (key: keyof AppSettings, mode: "directory" | "file") => {
    const selected = await open({
      directory: mode === "directory",
      multiple: false,
      title:
        mode === "directory"
          ? key === "lora_trainer_root"
            ? "选择 kohya_ss 或 sd-scripts 目录"
            : "选择 LoRA 输出目录"
          : key === "lora_trainer_python"
            ? "选择训练用的 python.exe"
            : "选择训练底模（.safetensors）",
    });
    if (typeof selected === "string" && selected.trim()) {
      onChange({ [key]: selected.trim() } as Partial<AppSettings>);
    }
  };

  const statusText = !status
    ? "尚未检测"
    : !status.configured
      ? status.error ?? "未配置"
      : !status.installed
        ? status.error ?? "找不到可用的 Python"
        : `已就绪 · ${status.python_source} · ${status.trainers.join(" / ")}`;

  const environmentText = !environment
    ? null
    : environment.error
      ? `环境异常：${environment.error}`
      : `Python ${environment.python_version ?? "?"} · torch ${environment.torch_version ?? "未安装"}${
          environment.cuda_available
            ? ` · CUDA ${environment.cuda_version ?? "?"} ${environment.device_name ?? ""}`
            : " · 仅 CPU"
        }${environment.bitsandbytes_version ? ` · bitsandbytes ${environment.bitsandbytes_version}` : ""}`;

  return (
    <section className="settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <SlidersHorizontal size={19} />
        </div>
        <div>
          <h3>LoRA 训练（kohya / sd-scripts）</h3>
          <p>
            训练同样只驱动你本机已安装的训练器，本应用不打包、不下载 kohya_ss 或 sd-scripts。
            填好下面几项后，就能在「AI 生图」页把导出的训练集一键训练成 LoRA。
          </p>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>训练器目录</strong>
          <small>
            kohya_ss 根目录（含 <code>sd-scripts</code>）或直接指向 <code>sd-scripts</code>，
            里面需要有 <code>train_network.py</code> 或 <code>sdxl_train_network.py</code>。
          </small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={settings.lora_trainer_root}
            placeholder="E:\\path\\to\\kohya_ss"
            onChange={(event) => onChange({ lora_trainer_root: event.target.value })}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickPath("lora_trainer_root", "directory")}
          >
            选择目录
          </button>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>训练用 Python</strong>
          <small>留空会自动使用训练器目录下的 <code>venv</code>，再不行才回退到系统 PATH。</small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={settings.lora_trainer_python}
            placeholder="留空自动探测"
            onChange={(event) => onChange({ lora_trainer_python: event.target.value })}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickPath("lora_trainer_python", "file")}
          >
            选择文件
          </button>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>训练底模</strong>
          <small>
            必须是 <code>.safetensors</code>，且与要训练的 LoRA 家族一致（SD1.5 的 LoRA 不能配 SDXL 底模）。
          </small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={settings.lora_trainer_base_model}
            placeholder="E:\\models\\anime.safetensors"
            onChange={(event) => onChange({ lora_trainer_base_model: event.target.value })}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickPath("lora_trainer_base_model", "file")}
          >
            选择文件
          </button>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>LoRA 输出目录</strong>
          <small>留空则输出到训练集同级的 <code>lora-models</code>。</small>
        </div>
        <div className="generation-inline">
          <input
            type="text"
            className="settings-text-input settings-text-input-wide"
            value={settings.lora_trainer_output_dir}
            placeholder="留空使用默认目录"
            onChange={(event) => onChange({ lora_trainer_output_dir: event.target.value })}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => void pickPath("lora_trainer_output_dir", "directory")}
          >
            选择目录
          </button>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>训练器状态</strong>
          <small>{statusText}</small>
          {environmentText && <small>{environmentText}</small>}
        </div>
        <div className="generation-inline">
          <button
            type="button"
            className="ghost-button"
            onClick={() => void check()}
            disabled={checking}
          >
            {checking ? <LoaderCircle size={15} className="spin" /> : <Activity size={15} />}
            检测训练器
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={() => void probe()}
            disabled={probing || !status?.installed}
          >
            {probing ? <LoaderCircle size={15} className="spin" /> : <Cpu size={15} />}
            检查 torch / CUDA
          </button>
        </div>
      </div>

      <div className="settings-field-note">
        kohya_ss 与 sd-scripts 是独立项目（各有自己的许可证），本应用只调用它们、不打包也不修改；
        训练产生的 LoRA 版权与使用限制由训练者与底模许可证共同决定。
      </div>
    </section>
  );
}

/**
 * Character recognizer inventory: pick the active model and install extra
 * ones from the shipped catalog (the Worker performs the download).
 */
function ReferenceLibrarySection({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  const [status, setStatus] = useState<ReferenceLibraryStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const response = await invokeCore<ReferenceLibraryStatus>(
      "get_reference_library_status",
      "reference.status",
    );
    setStatus(response.payload ?? null);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const build = async () => {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const response = await invokeCore<{
        identities: number;
        references: number;
        skipped: number;
      }>("build_reference_library", "reference.build", {
        backend: settings.reference_backend,
      });
      if (response.error) {
        setError(response.error.message);
      } else if (response.payload) {
        setMessage(
          `参考库已生成：${response.payload.identities} 个角色 · ${response.payload.references} 条参考` +
            (response.payload.skipped > 0 ? ` · ${response.payload.skipped} 条跳过` : ""),
        );
        onChange({ reference_matching_enabled: true });
      }
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const response = await invokeCore<{ removed: boolean }>(
        "clear_reference_library",
        "reference.clear",
      );
      if (response.error) setError(response.error.message);
      else setMessage("参考库已删除。");
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="settings-field settings-field-stacked">
        <div>
          <strong>参考图识别（模型里没有的角色）</strong>
          <small>
            识别模型的标签表决定它能认出谁；像一些冷门角色不在表里，无论怎么扫描都是「未识别」。
            用人工矫正过的样本建一个参考库后，扫描会对**非高置信**的人物做相似度匹配，
            命中就给出「参考匹配 xx%」。CCIP 后端判别力最好（实测 AUC 1.000），需要额外下载模型；
            embedding 后端复用当前识别模型、无需下载。
          </small>
        </div>
        <div className="generation-inline">
          <label className="switch-control">
            <input
              type="checkbox"
              checked={settings.reference_matching_enabled}
              onChange={(event) =>
                onChange({ reference_matching_enabled: event.target.checked })
              }
            />
            <span>{settings.reference_matching_enabled ? "已启用" : "已关闭"}</span>
          </label>
          <select
            value={settings.reference_backend}
            onChange={(event) =>
              onChange({
                reference_backend: event.target.value as AppSettings["reference_backend"],
              })
            }
          >
            <option value="ccip">CCIP（推荐，需下载模型约 143MB）</option>
            <option value="embedding">当前识别模型的 embedding（无需下载）</option>
          </select>
          <button
            type="button"
            className="primary-button"
            onClick={() => void build()}
            disabled={busy || !isTauriRuntime()}
          >
            {busy ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
            用矫正结果建立参考库
          </button>
          {status?.exists && (
            <button
              type="button"
              className="ghost-button"
              onClick={() => void clear()}
              disabled={busy}
            >
              删除参考库
            </button>
          )}
        </div>
        <small>
          {status?.exists
            ? `当前参考库：${status.identities} 个角色 · ${status.references} 条参考 · 后端 ${status.backend}` +
              (status.built_at ? ` · 建于 ${status.built_at}` : "")
            : "尚未建立参考库。"}
          {status?.characters?.length
            ? ` 角色：${status.characters
                .slice(0, 6)
                .map((item) => `${item.display_name}(${item.references})`)
                .join("、")}`
            : ""}
        </small>
      </div>
      {(message || error) && (
        <div className="settings-status-banner" aria-live="polite">
          {message && (
            <span className="runtime-feedback success">
              <CircleCheck size={13} />
              {message}
            </span>
          )}
          {error && (
            <span className="runtime-feedback error">
              <AlertCircle size={13} />
              {error}
            </span>
          )}
        </div>
      )}
    </>
  );
}

/** Model kinds, in the order 模型配置 lists them. */
const MODEL_GROUPS: Array<{ id: string; label: string; hint: string }> = [
  {
    id: "character_recognizer",
    label: "角色识别模型",
    hint: "扫描时给每张图打角色标签；同一时间只有一个处于“使用中”。",
  },
  {
    id: "head_detector",
    label: "头部检测模型",
    hint: "先框出人物头部再裁剪，随安装包一起分发。",
  },
  {
    id: "tagger",
    label: "训练集打标模型",
    hint: "导出 LoRA 训练集时把裁剪图转成 Danbooru 标签，首次导出会自动下载。",
  },
  {
    id: "reference_backend",
    label: "参考图匹配模型",
    hint: "“参考图识别”选 CCIP 后端时使用，纯 embedding 后端不需要。",
  },
];

const MODEL_STATUS_META: Record<string, { label: string; tone: string }> = {
  installed: { label: "已就绪", tone: "ready" },
  partial: { label: "不完整", tone: "warn" },
  missing: { label: "未下载", tone: "missing" },
  unavailable: { label: "不可用", tone: "missing" },
};

function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  // Keep one decimal for small models, where the difference between 2 MB and
  // 45 MB still tells the user whether a download is trivial.
  if (value >= 100 * 1024 ** 2) return `${Math.round(value / 1024 ** 2)} MB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${value} B`;
}

function modelSizeLabel(entry: ModelInventoryEntry): string {
  if (entry.installed_bytes > 0) return formatBytes(entry.installed_bytes);
  if (entry.size_mb > 0) return `约 ${formatBytes(entry.size_mb * 1024 ** 2)}`;
  return "体积未知";
}

/**
 * 模型配置: every model the app ships, can download, or has cached, in one
 * list. The Worker owns the downloads; this panel only reflects its inventory.
 */
function ModelSettingsPanel({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  const [inventory, setInventory] = useState<ModelInventory | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const response = await invokeCore<ModelInventory>(
      "get_model_inventory",
      "model.inventory",
    );
    if (response.payload) {
      setInventory(response.payload);
      setError(response.payload.warning ?? null);
      // Adopt the persisted choice on first load so the markers match what the
      // next scan will really use.
      if (!settings.recognition_recognizer_model && response.payload.active) {
        onChange({ recognition_recognizer_model: response.payload.active });
      }
    } else if (response.error) {
      setError(response.error.message);
    }
  }, [onChange, settings.recognition_recognizer_model]);

  useEffect(() => {
    void refresh();
    // The inventory only needs to load once per tab visit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Download a catalog model into the models directory or the HF cache. */
  const install = async (entry: ModelInventoryEntry) => {
    setBusy(entry.id);
    setMessage(null);
    setError(null);
    try {
      const response = await invokeCore<unknown>(
        "install_catalog_model",
        "model.install",
        { catalogId: entry.id },
      );
      if (response.error) {
        setError(response.error.message);
      } else {
        setMessage(`${entry.name} 已下载完成，可直接离线使用。`);
      }
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  const remove = async (entry: ModelInventoryEntry) => {
    const target = entry.delivery === "hf_cache" ? "缓存中的模型文件" : "模型目录";
    if (
      !globalThis.confirm(
        `确定删除「${entry.name}」吗？\n将从${target}移除，之后需要重新下载才能使用。`,
      )
    ) {
      return;
    }
    setBusy(entry.id);
    setMessage(null);
    setError(null);
    try {
      const response = await invokeCore<unknown>(
        "delete_catalog_model",
        "model.delete",
        { catalogId: entry.id },
      );
      if (response.error) setError(response.error.message);
      else setMessage(`已删除「${entry.name}」。`);
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  /** Persist the recognizer every later scan should use. */
  const activate = async (entry: ModelInventoryEntry) => {
    setBusy(entry.id);
    setMessage(null);
    setError(null);
    try {
      const response = await invokeCore<{ active: string }>(
        "set_active_recognizer",
        "model.activate",
        { modelId: entry.id },
      );
      if (response.error) {
        setError(response.error.message);
      } else {
        onChange({ recognition_recognizer_model: entry.id });
        setMessage(`已把「${entry.name}」设为当前识别模型，下一次扫描生效。`);
      }
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  const openPath = (path: string | null | undefined) => {
    if (!path) return;
    void invokeCore<void>("show_item_in_folder", "system.explorer.show", { path });
  };

  const entries = inventory?.entries ?? [];
  const ready = entries.filter((entry) => entry.status === "installed");
  const usedBytes = ready.reduce((total, entry) => total + entry.installed_bytes, 0);
  const groups = MODEL_GROUPS.map((group) => ({
    ...group,
    entries: entries.filter((entry) => entry.kind === group.id),
  })).filter((group) => group.entries.length > 0);
  const otherEntries = entries.filter(
    (entry) => !MODEL_GROUPS.some((group) => group.id === entry.kind),
  );

  return (
    <section className="settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <Package size={19} />
        </div>
        <div>
          <h3>模型配置</h3>
          <p>
            这里汇总随安装包自带、可从 Hugging Face 下载以及已经缓存到本机的全部模型。
          </p>
        </div>
      </div>

      <div className="settings-field settings-field-stacked">
        <div>
          <strong>存储位置</strong>
          <small>
            角色识别模型与检测模型下载到模型目录；打标、参考匹配模型走 Hugging Face
            缓存。“删除”只清理这两个目录里的模型文件。
          </small>
        </div>
        <div className="model-path-list">
          <div className="model-path-row">
            <span className="model-path-label">模型目录</span>
            <code title={inventory?.models_dir}>
              {inventory?.models_dir || "等待 AI Worker 就绪…"}
            </code>
            <button
              type="button"
              className="ghost-button"
              onClick={() => openPath(inventory?.models_dir)}
              disabled={!inventory?.models_dir}
            >
              打开
            </button>
          </div>
          <div className="model-path-row">
            <span className="model-path-label">缓存目录</span>
            <code title={inventory?.cache_dir}>
              {inventory?.cache_dir || "等待 AI Worker 就绪…"}
            </code>
            <button
              type="button"
              className="ghost-button"
              onClick={() => openPath(inventory?.cache_dir)}
              disabled={!inventory?.cache_dir}
            >
              打开
            </button>
          </div>
        </div>
      </div>

      <div className="model-toolbar">
        <span>
          {inventory
            ? `${ready.length} / ${entries.length} 个模型已就绪${
                usedBytes > 0 ? ` · 占用 ${formatBytes(usedBytes)}` : ""
              }`
            : "正在读取模型清单…"}
        </span>
        <button
          type="button"
          className="ghost-button"
          onClick={() => void refresh()}
          disabled={busy !== null}
        >
          {busy !== null ? (
            <LoaderCircle size={15} className="spin" />
          ) : (
            <RefreshCw size={15} />
          )}
          刷新
        </button>
      </div>

      {groups.map((group) => (
        <div className="model-group" key={group.id}>
          <div className="model-group-heading">
            <strong>{group.label}</strong>
            <small>{group.hint}</small>
          </div>
          {group.entries.map((entry) => (
            <ModelCard
              key={entry.id}
              entry={entry}
              busy={busy}
              onInstall={() => void install(entry)}
              onRemove={() => void remove(entry)}
              onActivate={() => void activate(entry)}
              onOpenPath={() => openPath(entry.path)}
            />
          ))}
        </div>
      ))}

      {otherEntries.length > 0 && (
        <div className="model-group">
          <div className="model-group-heading">
            <strong>其他模型</strong>
            <small>清单里未归类的模型。</small>
          </div>
          {otherEntries.map((entry) => (
            <ModelCard
              key={entry.id}
              entry={entry}
              busy={busy}
              onInstall={() => void install(entry)}
              onRemove={() => void remove(entry)}
              onActivate={() => void activate(entry)}
              onOpenPath={() => openPath(entry.path)}
            />
          ))}
        </div>
      )}

      <div className="settings-field-note">
        “自动融合模型”会把当前激活的个人模型与最新的人工矫正样本合并使用；个人模型
        由“再训练”生成，与这里的基础识别模型互相配合，不需要在这里切换。
      </div>

      {(message || error) && (
        <div className="settings-status-banner" aria-live="polite">
          {message && (
            <span className="runtime-feedback success">
              <CircleCheck size={13} />
              {message}
            </span>
          )}
          {error && (
            <span className="runtime-feedback error">
              <AlertCircle size={13} />
              {error}
            </span>
          )}
        </div>
      )}
    </section>
  );
}

/** One row of the model list: identity, status badge and its actions. */
export function ModelCard({
  entry,
  busy,
  onInstall,
  onRemove,
  onActivate,
  onOpenPath,
}: {
  entry: ModelInventoryEntry;
  /** Id of the entry whose download/delete is running, if any. */
  busy: string | null;
  onInstall: () => void;
  onRemove: () => void;
  onActivate: () => void;
  onOpenPath: () => void;
}) {
  const status = MODEL_STATUS_META[entry.status] ?? {
    label: entry.status,
    tone: "missing",
  };
  const installed = entry.status === "installed";
  // Only the row being worked on spins; every other row just locks its buttons
  // so two downloads cannot race against each other.
  const rowBusy = busy === entry.id;
  const locked = busy !== null;

  return (
    <div className={`model-card${entry.active ? " active" : ""}`}>
      <div className="model-card-body">
        <div className="model-card-title">
          <strong>{entry.name}</strong>
          <span className={`model-badge ${status.tone}`}>{status.label}</span>
          {entry.active && <span className="model-badge active">使用中</span>}
          {entry.bundled && <span className="model-badge">随包提供</span>}
        </div>
        <small className="model-card-meta">
          <code>{entry.id}</code>
          {entry.repo_id ? ` · ${entry.repo_id}` : " · 随安装包分发"}
          {` · ${modelSizeLabel(entry)}`}
          {entry.version ? ` · v${entry.version}` : ""}
        </small>
        {entry.note && <small className="model-card-note">{entry.note}</small>}
        {entry.error && <small className="model-card-note error">{entry.error}</small>}
        {entry.path && (
          <small className="model-card-path" title={entry.path}>
            {entry.path}
          </small>
        )}
      </div>
      <div className="model-card-actions">
        {entry.kind === "character_recognizer" && installed && !entry.active && (
          <button
            type="button"
            className="ghost-button"
            onClick={onActivate}
            disabled={locked}
          >
            <CircleCheck size={15} /> 设为当前
          </button>
        )}
        {entry.downloadable && !installed && (
          <button
            type="button"
            className="primary-button"
            onClick={onInstall}
            disabled={locked}
          >
            {rowBusy ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
            下载{entry.size_mb > 0 ? ` ${formatBytes(entry.size_mb * 1024 ** 2)}` : ""}
          </button>
        )}
        {entry.path && (
          <button
            type="button"
            className="ghost-button"
            onClick={onOpenPath}
            disabled={locked}
          >
            打开目录
          </button>
        )}
        {!entry.bundled && entry.downloadable && entry.status !== "missing" && (
          <button
            type="button"
            className="ghost-button danger"
            onClick={onRemove}
            disabled={locked || entry.active}
            title={entry.active ? "正在使用中，请先切换到其他识别模型" : undefined}
          >
            <Trash2 size={15} /> 删除
          </button>
        )}
      </div>
    </div>
  );
}

function RecognitionSettingsPanel({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  return (
    <section className="settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <Cpu size={19} />
        </div>
        <div>
          <h3>识别配置</h3>
          <p>控制角色识别推理使用多少 CPU 线程，以及扫描时的默认行为。</p>
        </div>
      </div>

      <ReferenceLibrarySection settings={settings} onChange={onChange} />

      <div className="settings-field">
        <div>
          <strong>ONNX 推理线程数</strong>
          <small>
            单次推理内部的 CPU 并行度，数值越大越快，但会占用更多核心；显卡执行时该值不影响 GPU 运算。
          </small>
        </div>
        <select
          value={settings.recognition_onnx_threads}
          onChange={(event) =>
            onChange({ recognition_onnx_threads: Number(event.target.value) })
          }
        >
          {Array.from({ length: 16 }, (_, index) => index + 1).map((value) => (
            <option value={value} key={value}>
              {value} 线程{value === 4 ? "（默认）" : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="settings-field">
        <div>
          <strong>默认跳过已标注图片</strong>
          <small>识别扫描时默认只处理未人工标注的图片，避免覆盖此前的标注与再训练样本。</small>
        </div>
        <label className="switch-control">
          <input
            type="checkbox"
            checked={settings.recognition_skip_annotated}
            onChange={(event) =>
              onChange({ recognition_skip_annotated: event.target.checked })
            }
          />
          <span>{settings.recognition_skip_annotated ? "已启用" : "已关闭"}</span>
        </label>
      </div>

      <div className="settings-field-note">
        识别任务由单个 AI Worker 顺序执行：线程数控制的是每次模型推理的并行度，而不是同时处理多张图片。显卡与 CUDA 运行时请在“基础配置”中设置。
      </div>
    </section>
  );
}

function SimilaritySettingsPanel({
  settings,
  onChange,
}: {
  settings: AppSettings;
  onChange: (patch: Partial<AppSettings>) => void;
}) {
  return (
    <section className="settings-section">
      <div className="settings-heading">
        <div className="settings-heading-icon">
          <SlidersHorizontal size={19} />
        </div>
        <div>
          <h3>相似度配置</h3>
          <p>相似度扫描使用本地感知哈希与颜色直方图算法，并行度直接决定特征提取速度。</p>
        </div>
      </div>

      <div className="settings-field">
        <div>
          <strong>特征提取并行度</strong>
          <small>
            同时解码与计算多少张图片的特征。数值越高越快，但会占用更多 CPU 与磁盘 IO。
          </small>
        </div>
        <select
          value={settings.similarity_workers}
          onChange={(event) =>
            onChange({ similarity_workers: Number(event.target.value) })
          }
        >
          {Array.from({ length: 32 }, (_, index) => index + 1).map((value) => (
            <option value={value} key={value}>
              {value}
              {value === 8 ? "（默认）" : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="settings-field">
        <div>
          <strong>默认匹配灵敏度</strong>
          <small>
            新扫描的初始阈值，仍可在相似计算页面临时调整。推荐 85% 识别差分与重压缩。
          </small>
        </div>
        <div className="settings-slider">
          <input
            type="range"
            min="0.70"
            max="0.98"
            step="0.01"
            value={settings.similarity_threshold}
            onChange={(event) =>
              onChange({ similarity_threshold: Number(event.target.value) })
            }
          />
          <code>{Math.round(settings.similarity_threshold * 100)}%</code>
        </div>
      </div>

      <div className="settings-field">
        <div>
          <strong>默认包含子目录</strong>
          <small>新扫描默认递归遍历所选目录下的所有子文件夹。</small>
        </div>
        <label className="switch-control">
          <input
            type="checkbox"
            checked={settings.similarity_include_subfolders}
            onChange={(event) =>
              onChange({ similarity_include_subfolders: event.target.checked })
            }
          />
          <span>{settings.similarity_include_subfolders ? "已启用" : "已关闭"}</span>
        </label>
      </div>

      <div className="settings-field">
        <div>
          <strong>默认归档文件夹名</strong>
          <small>
            执行“隔离归档”时在该图片所在目录下创建的子文件夹名称，不能包含路径分隔符。
          </small>
        </div>
        <input
          type="text"
          className="settings-text-input"
          value={settings.similarity_archive_dir}
          maxLength={64}
          onChange={(event) =>
            onChange({ similarity_archive_dir: event.target.value })
          }
        />
      </div>

      <div className="settings-field-note">
        相似度扫描为纯 CPU 路径，设置并行度后可重新扫描直接生效；结果文件与人工修改仍会被保留。
      </div>

    </section>
  );
}

export function SettingsPage({
  status,
  onRefresh,
}: {
  status: RuntimeStatus | null;
  onRefresh?: () => Promise<void> | void;
}) {
  const [tab, setTab] = useState<SettingsTab>("environment");
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS);
  const [savedSettings, setSavedSettings] = useState<AppSettings>(
    DEFAULT_APP_SETTINGS,
  );
  const [runtimeMode, setRuntimeMode] = useState<WorkerRuntimeMode>(
    status?.worker.runtime_mode ?? "embedded_env",
  );
  const [runtimeDevice, setRuntimeDevice] = useState<WorkerComputeDevice>(
    status?.worker.compute_device ?? "auto",
  );
  const [gpus, setGpus] = useState<GpuInventory | null>(null);
  const [compute, setCompute] = useState<WorkerComputeStatus | null>(null);
  const [savingSettings, setSavingSettings] = useState(false);
  const [settingsNotice, setSettingsNotice] = useState<string | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const savedMode = status?.worker.runtime_mode ?? "embedded_env";
  const savedDevice = status?.worker.compute_device ?? "auto";
  const dirty =
    JSON.stringify(settings) !== JSON.stringify(savedSettings) ||
    runtimeMode !== savedMode ||
    runtimeDevice !== savedDevice;

  useEffect(() => {
    let active = true;
    void invokeCore<GpuInventory>("get_gpu_devices", "system.gpu.list")
      .then((response) => {
        if (active && response.payload) setGpus(response.payload);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  const probeCompute = useCallback(async () => {
    const response = await invokeCore<WorkerComputeStatus>(
      "check_worker_compute",
      "runtime.compute.check",
    );
    if (response.payload) setCompute(response.payload);
  }, []);

  // Verify CUDA automatically only when the runtime advertises it; on a CPU
  // build the self-test would just load several hundred megabytes of models.
  useEffect(() => {
    if (compute) return;
    if (status?.worker.capabilities?.compute?.cuda_available !== true) return;
    void probeCompute().catch(() => undefined);
  }, [compute, status, probeCompute]);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await invokeCore<AppSettings>(
          "get_app_settings",
          "settings.get",
        );
        if (!active) return;
        if (response.payload) {
          setSettings(response.payload);
          // Keep the dirty baseline in sync, otherwise the page looks
          // "unsaved" forever and a mismatched baseline hides real edits.
          setSavedSettings(response.payload);
          setSettingsError(null);
        } else if (response.error) {
          setSettingsError(
            `无法读取应用设置，已使用默认值：${response.error.message}`,
          );
        }
      } catch (error) {
        if (!active) return;
        setSettingsError(
          `无法读取应用设置，已使用默认值：${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }
    };
    void load();
    return () => {
      active = false;
    };
  }, []);

  const patchSettings = (patch: Partial<AppSettings>) => {
    setSettings((current) => ({ ...current, ...patch }));
    setSettingsNotice(null);
  };

  // Live preview: the slider and the 0.5px stepper apply immediately, while the
  // topbar save button persists the value for the next launch.
  useEffect(() => {
    applyUiFontSize(settings.ui_font_size ?? UI_FONT_BASE_PX);
  }, [settings.ui_font_size]);

  /** Persist every pending change: app settings, runtime mode and device. */
  const saveSettings = useCallback(async () => {
    if (savingSettings) return;
    setSavingSettings(true);
    setSettingsNotice(null);
    setSettingsError(null);
    const errors: string[] = [];
    const notices: string[] = [];

    const settingsResponse = await invokeCore<AppSettings>(
      "update_app_settings",
      "settings.update",
      { settings },
    );
    if (settingsResponse.error || !settingsResponse.payload) {
      errors.push(settingsResponse.error?.message ?? "无法保存应用设置。");
    } else {
      setSettings(settingsResponse.payload);
      setSavedSettings(settingsResponse.payload);
      cacheUiFontSize(settingsResponse.payload.ui_font_size);
    }

    if (runtimeMode !== savedMode) {
      const modeResponse = await invokeCore<WorkerRuntimeSettings>(
        "set_worker_runtime_mode",
        "settings.update",
        { mode: runtimeMode },
      );
      if (modeResponse.error || !modeResponse.payload) {
        errors.push(modeResponse.error?.message ?? "无法保存 Worker 运行方式。");
      } else if (status?.worker.runtime_mode !== modeResponse.payload.mode) {
        notices.push(
          `运行方式已写入 ${modeResponse.payload.mode_label}，将在 Worker 重启后生效。`,
        );
      }
    }

    if (runtimeDevice !== savedDevice) {
      const deviceResponse = await invokeCore<WorkerRuntimeSettings>(
        "set_worker_compute_device",
        "settings.update",
        { device: runtimeDevice },
      );
      if (deviceResponse.error || !deviceResponse.payload) {
        errors.push(deviceResponse.error?.message ?? "无法保存推理设备。");
      }
    }

    await onRefresh?.();
    if (errors.length > 0) {
      setSettingsError(errors.join(" "));
    } else {
      setSettingsNotice(["设置已保存。", ...notices].join(" "));
    }
    setSavingSettings(false);
  }, [
    onRefresh,
    runtimeDevice,
    runtimeMode,
    savedDevice,
    savedMode,
    savingSettings,
    settings,
    status?.worker.runtime_mode,
  ]);

  // The topbar button owns the save action, so the page registers a handler
  // and publishes whether there is anything to save.
  useEffect(() => {
    registerSettingsSave(saveSettings);
    return () => registerSettingsSave(null);
  }, [saveSettings]);

  useEffect(() => {
    updateSettingsSaveState({ saving: savingSettings, dirty });
  }, [dirty, savingSettings]);

  return (
    <div className="page-stack settings-page">
      <div className="settings-tabs" role="tablist" aria-label="设置分类">
        {SETTINGS_TABS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              role="tab"
              aria-selected={tab === item.id}
              className={`settings-tab${tab === item.id ? " active" : ""}`}
              onClick={() => setTab(item.id)}
            >
              <Icon size={16} />
              <span>
                <strong>{item.label}</strong>
                <small>{item.hint}</small>
              </span>
            </button>
          );
        })}
      </div>

      {tab === "environment" && (
        <div className="settings-stack">
          <AppearanceSettingsPanel
            settings={settings}
            onChange={patchSettings}
          />
          <WorkerRuntimeSettingsPanel
            status={status}
            gpus={gpus}
            compute={compute}
            mode={runtimeMode}
            device={runtimeDevice}
            busy={savingSettings}
            onChangeMode={setRuntimeMode}
            onChangeDevice={setRuntimeDevice}
            cudaRuntimeDir={settings.cuda_runtime_dir}
            onChangeCudaRuntimeDir={(value) =>
              patchSettings({ cuda_runtime_dir: value })
            }
            onCudaInstalled={() => {
              void probeCompute().catch(() => undefined);
              void onRefresh?.();
            }}
            backgroundPriority={settings.background_priority}
            onChangeBackgroundPriority={(value) =>
              patchSettings({ background_priority: value })
            }
          />
          <EnvironmentCheckSection
            status={status}
            gpus={gpus}
            compute={compute}
            onProbeCompute={probeCompute}
            onRefresh={onRefresh}
          />
        </div>
      )}

      {tab === "recognition" && (
        <RecognitionSettingsPanel
          settings={settings}
          onChange={patchSettings}
        />
      )}

      {tab === "similarity" && (
        <SimilaritySettingsPanel
          settings={settings}
          onChange={patchSettings}
        />
      )}

      {tab === "models" && (
        <ModelSettingsPanel settings={settings} onChange={patchSettings} />
      )}

      {tab === "comfy" && (
        <>
          <ComfySettingsPanel settings={settings} onChange={patchSettings} />
          <LoraTrainingSettingsPanel settings={settings} onChange={patchSettings} />
        </>
      )}

      {(settingsNotice || settingsError) && (
        <div className="settings-status-banner" aria-live="polite">
          {settingsNotice && (
            <span className="runtime-feedback success">
              <CircleCheck size={13} />
              {settingsNotice}
            </span>
          )}
          {settingsError && (
            <span className="runtime-feedback error">
              <AlertCircle size={13} />
              {settingsError}
            </span>
          )}
        </div>
      )}

    </div>
  );
}
