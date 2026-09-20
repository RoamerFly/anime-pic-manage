import { describe, expect, it } from "vitest";
import {
  isTauriRuntime,
  normalizeIpcArgs,
  normalizePreviewPath,
  requestEnvelope,
  settleWithin,
} from "./tauri";

describe("desktop IPC helpers", () => {
  it("uses the shared versioned request envelope", () => {
    const request = requestEnvelope("worker.health", {});
    expect(request.version).toBe("1.0");
    expect(request.message_type).toBe("worker.health");
    expect(request.request_id).toBeTruthy();
  });

  it("detects browser mode without pretending the core is healthy", () => {
    expect(isTauriRuntime()).toBe(false);
  });

  it("normalizes snake_case arguments to camelCase for Tauri v2 commands", () => {
    const args = {
      auto_activate: true,
      overwrite_version: 3,
      image_size: [1920, 1080],
      base_revision: 5,
      requestId: "custom",
    };
    const normalized = normalizeIpcArgs(args, "req-123");
    expect(normalized).toEqual({
      requestId: "custom",
      autoActivate: true,
      overwriteVersion: 3,
      imageSize: [1920, 1080],
      baseRevision: 5,
    });
  });

  it("stops waiting when a core request does not settle", async () => {
    const never = new Promise<string>(() => undefined);
    await expect(settleWithin(never, 1)).resolves.toBeNull();
  });

  it("normalizes Windows verbatim paths for image preview URLs", () => {
    expect(normalizePreviewPath("\\\\?\\G:\\Pictures\\sample.png")).toBe(
      "G:\\Pictures\\sample.png",
    );
    expect(normalizePreviewPath("\\\\?\\UNC\\server\\share\\sample.png")).toBe(
      "\\\\server\\share\\sample.png",
    );
    expect(normalizePreviewPath("D:\\Pictures\\sample.png")).toBe(
      "D:\\Pictures\\sample.png",
    );
  });
});
