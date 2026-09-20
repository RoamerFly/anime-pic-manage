import { describe, expect, it } from "vitest";
import { isScanActive, scanStateFromPhase } from "./scan-state";

describe("scan progress state mapping", () => {
  it("keeps per-image errors in a running task", () => {
    expect(scanStateFromPhase("error")).toBe("running");
    expect(scanStateFromPhase("recognizing")).toBe("running");
  });

  it("represents pause, cancellation, and completion phases", () => {
    expect(scanStateFromPhase("pausing")).toBe("pausing");
    expect(scanStateFromPhase("paused")).toBe("paused");
    expect(scanStateFromPhase("cancelling")).toBe("cancelling");
    expect(scanStateFromPhase("cancelled")).toBe("cancelled");
    expect(scanStateFromPhase("complete")).toBe("completed");
  });

  it("only marks controllable task states active", () => {
    expect(isScanActive("running")).toBe(true);
    expect(isScanActive("paused")).toBe(true);
    expect(isScanActive("completed")).toBe(false);
    expect(isScanActive("failed")).toBe(false);
  });
});
