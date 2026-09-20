import { describe, expect, it } from "vitest";
import type { SimilarityGroup, SimilarityGroupItem } from "@anime-pic-manage/shared-types";
import { getParentDirectory } from "../components/SimilarityGroupCard";

describe("Similarity logic and decision calculation", () => {
  const sampleGroup: SimilarityGroup = {
    group_id: "group-001",
    group_type: "exact_duplicate",
    average_similarity: 0.98,
    items: [
      {
        path: "D:/images/sample_high.png",
        file_size: 2048000,
        dimensions: [1920, 1080],
        format: "png",
        clarity_score: 180.5,
        is_recommended: true,
        recommend_reason: "最高画质",
        decision: "keep",
      },
      {
        path: "D:/images/sample_low.jpg",
        file_size: 512000,
        dimensions: [1280, 720],
        format: "jpg",
        clarity_score: 95.0,
        is_recommended: false,
        recommend_reason: "低清版本",
        decision: "archive",
      },
    ],
  };

  it("identifies recommended high quality item in group", () => {
    const highItem = sampleGroup.items.find((it) => it.is_recommended);
    expect(highItem).toBeDefined();
    expect(highItem?.format).toBe("png");
    expect(highItem?.dimensions[0]).toBe(1920);
  });

  it("calculates potential space saved for archive or delete decisions", () => {
    const groups: SimilarityGroup[] = [sampleGroup];
    let freedBytes = 0;
    let archiveCount = 0;
    let keepCount = 0;

    for (const g of groups) {
      for (const item of g.items) {
        if (item.decision === "archive") {
          archiveCount++;
          freedBytes += item.file_size;
        } else if (item.decision === "keep") {
          keepCount++;
        }
      }
    }

    expect(archiveCount).toBe(1);
    expect(keepCount).toBe(1);
    expect(freedBytes).toBe(512000);
  });

  it("supports updating individual item decisions", () => {
    const updatedItems = sampleGroup.items.map((it) =>
      it.path === "D:/images/sample_low.jpg" ? { ...it, decision: "delete" as const } : it,
    );
    expect(updatedItems[1].decision).toBe("delete");
  });

  it("shows the parent directory for items from Windows and POSIX paths", () => {
    expect(getParentDirectory("D:\\wallpapers\\set-a\\image.png")).toBe(
      "D:\\wallpapers\\set-a",
    );
    expect(getParentDirectory("/mnt/wallpapers/set-b/image.png")).toBe(
      "/mnt/wallpapers/set-b",
    );
    expect(getParentDirectory("C:\\image.png")).toBe("C:\\");
    expect(getParentDirectory("/image.png")).toBe("/");
  });
});
