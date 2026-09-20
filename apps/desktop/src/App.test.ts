import { describe, expect, it } from "vitest";
import type {
  LibraryScanPayload,
  LibraryScanResultEvent,
} from "@anime-pic-manage/shared-types";
import { mergeIncrementalScanResult } from "./App";

const firstResult = {
  path: "D:/Pictures/one.png",
  image_size: [100, 100] as [number, number],
  people: [],
};

describe("incremental library scan results", () => {
  it("appends each completed image while preserving earlier results", () => {
    const previous: LibraryScanPayload = {
      directory: "D:\\Pictures",
      total_discovered: 3,
      processed: 1,
      cancelled: false,
      results: [firstResult],
      errors: [],
    };
    const event: LibraryScanResultEvent = {
      directory: "d:/pictures/",
      total_discovered: 3,
      processed: 2,
      result: {
        path: "D:/Pictures/two.png",
        image_size: [200, 120],
        people: [],
      },
      model_version: 4,
      model_name: "个人模型 v4",
      skip_annotated: true,
    };

    const merged = mergeIncrementalScanResult(previous, event);

    expect(merged.results.map((result) => result.path)).toEqual([
      "D:/Pictures/one.png",
      "D:/Pictures/two.png",
    ]);
    expect(merged.processed).toBe(2);
    expect(merged.model_version).toBe(4);
  });

  it("replaces repeated paths and resets stale results for another directory", () => {
    const initial = mergeIncrementalScanResult(null, {
      directory: "D:/Pictures",
      total_discovered: 1,
      processed: 1,
      result: firstResult,
    });
    const replaced = mergeIncrementalScanResult(initial, {
      directory: "D:/Pictures",
      total_discovered: 1,
      processed: 1,
      result: { ...firstResult, image_size: [300, 300] },
    });
    const anotherDirectory = mergeIncrementalScanResult(replaced, {
      directory: "D:/Other",
      total_discovered: 1,
      processed: 1,
      error: { path: "D:/Other/broken.png", message: "read failed" },
    });

    expect(replaced.results).toHaveLength(1);
    expect(replaced.results[0]?.image_size).toEqual([300, 300]);
    expect(anotherDirectory.results).toEqual([]);
    expect(anotherDirectory.errors).toEqual([
      { path: "D:/Other/broken.png", message: "read failed" },
    ]);
  });
});
