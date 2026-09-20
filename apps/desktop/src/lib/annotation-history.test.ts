import { describe, expect, it } from "vitest";
import { appendAnnotationHistory } from "./annotation-history";

describe("annotation history", () => {
  it("appends a snapshot without mutating an unfinished history", () => {
    const history = [["initial"], ["edited"]];
    const next = ["saved"];

    const result = appendAnnotationHistory(history, 1, next, 50);

    expect(result).toEqual({
      snapshots: [["initial"], ["edited"], ["saved"]],
      historyIndex: 2,
    });
    expect(history).toEqual([["initial"], ["edited"]]);
    expect(next).toEqual(["saved"]);
  });

  it("drops the oldest snapshots when the 50-entry cap is exceeded", () => {
    const history = Array.from({ length: 50 }, (_, index) => [index]);

    const result = appendAnnotationHistory(history, 49, [50], 50);

    expect(result.snapshots).toHaveLength(50);
    expect(result.snapshots[0]).toEqual([1]);
    expect(result.snapshots.at(-1)).toEqual([50]);
    expect(result.historyIndex).toBe(49);
  });

  it("truncates the redo branch before appending a new edit", () => {
    const history = [["initial"], ["first edit"], ["redo candidate"]];

    const result = appendAnnotationHistory(history, 1, ["new branch"], 50);

    expect(result).toEqual({
      snapshots: [["initial"], ["first edit"], ["new branch"]],
      historyIndex: 2,
    });
  });
});
