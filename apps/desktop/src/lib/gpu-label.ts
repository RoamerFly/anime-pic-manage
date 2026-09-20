import type {
  GpuInventory,
  WorkerComputeCapability,
  WorkerComputeStatus,
} from "@anime-pic-manage/shared-types";

/** ONNX Runtime reports several providers users cannot act on; keep it readable. */
const PROVIDER_LABELS: Record<string, string> = {
  CUDAExecutionProvider: "NVIDIA CUDA",
  TensorrtExecutionProvider: "TensorRT",
  ROCMExecutionProvider: "AMD ROCm",
  MIGraphXExecutionProvider: "AMD MIGraphX",
  DmlExecutionProvider: "DirectML",
  OpenVINOExecutionProvider: "OpenVINO",
  CPUExecutionProvider: "CPU",
};

const NVIDIA_MARKERS = ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"];

export function describeProviders(providers: string[]): string[] {
  return providers
    .filter((provider) => provider in PROVIDER_LABELS)
    .map((provider) => PROVIDER_LABELS[provider]);
}

/**
 * Report the CUDA/cuDNN runtime state. A CUDA build without the matching
 * runtime silently runs on CPU, which looks like "GPU 没生效也没报错".
 */
export function describeCudaRuntime(
  compute: WorkerComputeCapability | null | undefined,
): string {
  const runtime = compute?.cuda_runtime;
  if (!compute?.cuda_available) {
    return "当前推理引擎不是 CUDA 版；识别使用 CPU。";
  }
  if (!runtime) {
    return "CUDA 版推理引擎已就绪（未做动态库探测）。";
  }
  if (runtime.status === "ready") {
    return "CUDA 12 与 cuDNN 9 运行时均已加载，可用于 GPU 推理。";
  }
  if (runtime.status === "unsupported") {
    return runtime.message;
  }
  const searched = runtime.configured_dir
    ? `（已尝试指定目录 ${runtime.configured_dir}）`
    : "";
  return `${runtime.message}${searched}`;
}

export function nvidiaGpuNames(devices?: string[] | null): string[] {
  return (devices ?? []).filter((device) => {
    const lowered = device.toLowerCase();
    return NVIDIA_MARKERS.some((marker) => lowered.includes(marker));
  });
}

/**
 * Hint under the inference-device selector: is CUDA usable, and on which card?
 *
 * `cuda_available` only means the wheel was built with CUDA. When the local
 * CUDA/cuDNN runtime is missing, ONNX Runtime silently falls back to CPU, so
 * the missing libraries are reported by name instead.
 */
export function describeCudaAvailability(
  gpus: GpuInventory | null,
  compute: WorkerComputeCapability | null | undefined,
): string {
  const cudaAvailable = compute?.cuda_available ?? false;
  const runtime = compute?.cuda_runtime ?? null;
  const nvidia = nvidiaGpuNames(gpus?.devices);
  const other = (gpus?.devices ?? []).filter(
    (device) => !nvidia.includes(device),
  );
  if (cudaAvailable && runtime?.status === "missing" && runtime.missing.length > 0) {
    return `${nvidia.length > 0 ? `检测到 ${nvidia.join("、")}，` : ""}但缺少 ${runtime.missing.join(
      "、",
    )}，CUDA 实际不可用，识别会回落到 CPU。`;
  }
  if (cudaAvailable) {
    const suffix =
      compute?.cuda_usable === true ? "，CUDA 运行时已就绪" : "";
    return nvidia.length > 0
      ? `已检测到 NVIDIA CUDA，可用于：${nvidia.join("、")}${suffix}`
      : `已检测到 NVIDIA CUDA 执行提供器（未能读取显卡型号）${suffix}`;
  }
  if (nvidia.length > 0) {
    return `检测到 NVIDIA 显卡 ${nvidia.join(
      "、",
    )}，但当前运行时未提供 CUDA，识别将使用 CPU。`;
  }
  if (other.length > 0) {
    return `检测到显卡 ${other.join(
      "、",
    )}，未检测到 NVIDIA CUDA 执行提供器，识别将使用 CPU。`;
  }
  return "未检测到 CUDA 执行提供器；当前运行时只提供 CPU 推理。";
}

export interface MeasuredCompute {
  ok: boolean;
  summary: string;
  detail: string;
}

/**
 * Summarize the Worker's real CUDA self-test.
 *
 * When CUDA really took effect the GPU model is included, so the settings page
 * answers "which card is inference actually running on?" rather than only
 * listing execution providers.
 */
export function describeMeasuredCompute(
  compute: WorkerComputeStatus | null,
  gpus: GpuInventory | null,
): MeasuredCompute | null {
  if (!compute) return null;
  const models = compute.models ?? [];
  const onCuda = models.filter((model) => model.active_device === "cuda");
  const failures = compute.errors ?? [];
  const gpuLabel = nvidiaGpuNames(gpus?.devices).join("、");

  if (onCuda.length > 0) {
    return {
      ok: true,
      summary: "CUDA 实测通过",
      detail: `${onCuda.map((model) => model.model_id).join("、")} 实际使用 CUDAExecutionProvider${
        gpuLabel ? `（显卡：${gpuLabel}）` : ""
      }`,
    };
  }
  if (models.length === 0) {
    return {
      ok: false,
      summary: "未能实测",
      detail: failures[0]?.error ?? "没有可加载的模型，无法验证实际执行设备",
    };
  }
  const activeProviders = describeProviders(
    models.flatMap((model) => model.active_providers),
  );
  return {
    ok: false,
    summary: "实测运行在 CPU",
    detail:
      failures.length > 0
        ? failures.map((item) => `${item.model_id}：${item.error}`).join("；")
        : `${models.map((model) => model.model_id).join("、")} 实际使用 ${
            activeProviders.join("、") || "CPU"
          }${gpuLabel ? `（已检测到 ${gpuLabel}，但会话未使用 CUDA）` : ""}`,
  };
}
