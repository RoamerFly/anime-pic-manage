import { describe, expect, it } from "vitest";
import { IPC_SCHEMA_VERSION, createRequest, isIpcEnvelope } from "./index.js";

describe("IPC schema", () => {
  it("creates a versioned request with a request id", () => {
    const request = createRequest("system.health", {});
    expect(request.version).toBe(IPC_SCHEMA_VERSION);
    expect(request.request_id).toMatch(/^req-|^[0-9a-f-]{36}$/i);
    expect(request.message_type).toBe("system.health");
  });

  it("rejects an unversioned or incomplete message", () => {
    expect(isIpcEnvelope({ version: "0.1", message_type: "system.health" })).toBe(false);
    expect(isIpcEnvelope({ version: IPC_SCHEMA_VERSION, message_type: "system.health" })).toBe(false);
  });

  it("keeps annotation editing messages in the versioned contract", () => {
    const request = createRequest("annotation.save", {
      path: "C:/library/image.png",
      image_size: [1920, 1080],
      base_revision: 2,
      annotations: [{
        label_name: "Example",
        bbox: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 },
        source: "manual",
        manually_adjusted: true,
      }],
    });
    expect(request.message_type).toBe("annotation.save");
    expect(isIpcEnvelope(request)).toBe(true);
  });

  it("keeps personal-model actions and status fields explicit", () => {
    const status = {
      status: "ready",
      message: "样本已满足训练条件。",
      readiness: {
        verified_sample_count: 6,
        verified_class_count: 2,
        eligible_class_count: 2,
        ready: true,
        reasons: [],
      },
      settings: { min_samples_per_class: 3 },
      active_version: 1,
      versions: [{
        version: 1,
        status: "active" as const,
        algorithm: "prototype-centroid",
        sample_count: 6,
        class_count: 2,
        metrics: { accuracy: 0.92, macro_f1: 0.88 },
        warnings: [],
        created_at: "2026-09-03T12:00:00Z",
      }],
    };
    const request = createRequest("personal.training.status", status);
    const activate = createRequest("personal.training.activate", { version: status.active_version });
    expect(request.message_type).toBe("personal.training.status");
    expect(activate.message_type).toBe("personal.training.activate");
    expect(isIpcEnvelope(request)).toBe(true);
    expect(isIpcEnvelope(activate)).toBe(true);
  });

  it("keeps Worker runtime selection and resolved path in the shared contract", () => {
    const settings = createRequest("settings.update", {
      mode: "embedded_env",
      mode_label: "内置 ENV",
      resolved_path: "D:/dist_windows/env/Scripts/python.exe",
    });
    expect(settings.message_type).toBe("settings.update");
    expect(isIpcEnvelope(settings)).toBe(true);
  });

  it("keeps runtime capability versions and required feature statuses explicit", () => {
    const capabilities = {
      status: "ok" as const,
      ready: true,
      dghs_imgutils: { distribution: "dghs-imgutils", version: "0.19.0", status: "ready" as const, error: null },
      onnxruntime: { distribution: "onnxruntime", version: "1.20.1", status: "ready" as const, error: null },
      features: {
        "imgutils.preprocess.pillow": { module: "imgutils.preprocess.pillow", status: "ready" as const, error: null },
        "imgutils.generic.yolo": { module: "imgutils.generic.yolo", status: "ready" as const, error: null },
        "imgutils.data": { module: "imgutils.data", status: "ready" as const, error: null },
      },
      errors: [],
    };
    const response = createRequest("system.health", { worker: { capabilities } });
    expect(response.payload).toEqual({ worker: { capabilities } });
    expect(capabilities.dghs_imgutils.version).toBe("0.19.0");
    expect(Object.values(capabilities.features).every((feature) => feature.status === "ready")).toBe(true);
  });

  it("handles similarity scan and decision contracts", () => {
    const scanReq = createRequest("similarity.scan.start", {
      path: "D:/images",
      threshold: 0.85,
    });
    expect(scanReq.message_type).toBe("similarity.scan.start");
    expect(isIpcEnvelope(scanReq)).toBe(true);

    const decisionReq = createRequest("similarity.decision.apply", {
      archive_directory: "D:/images/_duplicates",
      decisions: [{ path: "D:/images/dup.png", action: "archive" }],
    });
    expect(decisionReq.message_type).toBe("similarity.decision.apply");
    expect(isIpcEnvelope(decisionReq)).toBe(true);
  });
});
