import { describe, expect, it } from "vitest";
import type { LibraryScanPayload } from "@anime-pic-manage/shared-types";
import { parseStoredScan, scanForDirectory, scanSummary, type StoredScan } from "./scan-session";

const payload: LibraryScanPayload = {
  directory: "D:/Pictures",
  total_discovered: 3,
  processed: 2,
  cancelled: false,
  results: [
    { path: "D:/Pictures/one.png", image_size: [100, 100], people: [{ person_index: 0, box: { x1: 1, y1: 2, x2: 40, y2: 50, confidence: 0.9 }, crops: {}, status: "high_confidence", margin: 0.4, consistency: 0.8, candidates: [] }] },
  ],
  errors: [{ path: "D:/Pictures/two.png", message: "read failed" }],
};

describe("scan session persistence", () => {
  it("parses a stored payload and rejects malformed values", () => {
    const stored = parseStoredScan({ payload, completedAt: "2026-09-03T08:00:00.000Z" });
    expect(stored?.payload.directory).toBe("D:/Pictures");
    expect(parseStoredScan({ payload, completedAt: 42 })).toBeNull();
    expect(parseStoredScan(null)).toBeNull();
  });

  it("only restores results for the selected library directory", () => {
    const stored: StoredScan = { payload, completedAt: "2026-09-03T08:00:00.000Z" };
    expect(scanForDirectory(stored, "D:/Pictures")).toBe(payload);
    expect(scanForDirectory(stored, "D:/Other")).toBeNull();
    expect(scanForDirectory(stored, undefined)).toBeNull();
  });

  it("summarizes real scan counts without inventing completed work", () => {
    expect(scanSummary(payload, "2026-09-03T08:00:00.000Z")).toMatchObject({
      totalDiscovered: 3,
      processed: 2,
      resultCount: 1,
      personCount: 1,
      errorCount: 1,
      cancelled: false,
    });
  });

  it("persists and deletes scan results correctly", async () => {
    const { persistScan, readStoredScanForDirectory, deleteStoredScan } = await import("./scan-session");
    persistScan(payload);
    const restored = readStoredScanForDirectory("D:/Pictures");
    expect(restored?.directory).toBe("D:/Pictures");
    expect(restored?.results.length).toBe(1);

    const deleted = deleteStoredScan("D:/Pictures");
    expect(deleted).toBe(true);
    expect(readStoredScanForDirectory("D:/Pictures")).toBeNull();
  });
});
