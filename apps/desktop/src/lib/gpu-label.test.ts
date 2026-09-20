import { describe, expect, it } from "vitest";
import type {
  GpuInventory,
  WorkerComputeCapability,
  WorkerComputeStatus,
} from "@anime-pic-manage/shared-types";
import {
  describeCudaAvailability,
  describeCudaRuntime,
  describeMeasuredCompute,
  describeProviders,
  nvidiaGpuNames,
} from "./gpu-label";

const gpus: GpuInventory = {
  available: true,
  devices: [
    "Intel(R) UHD Graphics",
    "NVIDIA GeForce RTX 3060 Laptop GPU",
    "Todesk Virtual Display Adapter",
  ],
};

describe("gpu labels", () => {
  it("finds the NVIDIA adapter and ignores integrated graphics", () => {
    expect(nvidiaGpuNames(gpus.devices)).toEqual([
      "NVIDIA GeForce RTX 3060 Laptop GPU",
    ]);
  });

  it("names the card when CUDA is available", () => {
    expect(
      describeCudaAvailability(gpus, {
        available_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        cuda_available: true,
        cuda_usable: true,
      }),
    ).toContain("RTX 3060");
  });

  it("explains that a present card is unused without CUDA", () => {
    const message = describeCudaAvailability(gpus, {
      available_providers: ["CPUExecutionProvider"],
      cuda_available: false,
    });

    expect(message).toContain("RTX 3060");
    expect(message).toContain("未提供 CUDA");
  });

  it("blames the missing runtime when CUDA is advertised but unusable", () => {
    const message = describeCudaAvailability(gpus, {
      available_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
      cuda_available: true,
      cuda_usable: false,
      cuda_runtime: {
        status: "missing",
        missing: ["cuDNN 9"],
        message: "缺少 cuDNN 9",
      },
    });

    expect(message).toContain("RTX 3060");
    expect(message).toContain("cuDNN 9");
    expect(message).toContain("回落到 CPU");
  });

  it("only lists actionable execution providers", () => {
    expect(
      describeProviders([
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
      ]),
    ).toEqual(["TensorRT", "NVIDIA CUDA", "CPU"]);
  });
});

describe("cuda runtime summary", () => {
  const base: WorkerComputeCapability = {
    available_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    cuda_available: true,
  };

  it("reports a fully loaded runtime", () => {
    const message = describeCudaRuntime({
      ...base,
      cuda_usable: true,
      cuda_runtime: { status: "ready", missing: [], message: "" },
    });

    expect(message).toContain("cuDNN 9");
    expect(message).toContain("GPU 推理");
  });

  it("surfaces the missing libraries verbatim", () => {
    const message = describeCudaRuntime({
      ...base,
      cuda_usable: false,
      cuda_runtime: {
        status: "missing",
        missing: ["cuDNN 9", "cuBLAS 12"],
        message: "缺少 cuDNN 9、cuBLAS 12：ONNX Runtime 会自动回落到 CPU。",
      },
    });

    expect(message).toContain("缺少 cuDNN 9、cuBLAS 12");
  });

  it("says so when the runtime is CPU-only", () => {
    const message = describeCudaRuntime({
      available_providers: ["CPUExecutionProvider"],
      cuda_available: false,
    });

    expect(message).toContain("CPU");
  });
});

describe("measured compute summary", () => {
  const onCuda: WorkerComputeStatus = {
    requested_device: "auto",
    cuda_available: true,
    cuda_session_ready: true,
    distribution: "onnxruntime-gpu",
    available_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    models: [
      {
        model_id: "anime-head-v2.0-s",
        requested_device: "auto",
        active_device: "cuda",
        active_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
      },
    ],
    errors: [],
  };

  it("reports the GPU model when the session really used CUDA", () => {
    const measured = describeMeasuredCompute(onCuda, gpus);

    expect(measured?.ok).toBe(true);
    expect(measured?.detail).toContain("NVIDIA GeForce RTX 3060 Laptop GPU");
    expect(measured?.detail).toContain("CUDAExecutionProvider");
  });

  it("reports CPU sessions together with the unused card", () => {
    const onCpu: WorkerComputeStatus = {
      ...onCuda,
      cuda_available: false,
      cuda_session_ready: false,
      models: [
        {
          model_id: "anime-head-v2.0-s",
          requested_device: "auto",
          active_device: "cpu",
          active_providers: ["CPUExecutionProvider"],
        },
      ],
    };

    const measured = describeMeasuredCompute(onCpu, gpus);

    expect(measured?.ok).toBe(false);
    expect(measured?.summary).toBe("实测运行在 CPU");
    expect(measured?.detail).toContain("RTX 3060");
  });

  it("returns nothing before a self-test ran", () => {
    expect(describeMeasuredCompute(null, gpus)).toBeNull();
  });
});
