import { describe, expect, it } from "vitest";
import { buildFolderTree, resultsForFolder } from "./ScanFolderTree";
import type { LibraryScanResult } from "@anime-pic-manage/shared-types";

describe("buildFolderTree", () => {
  it("organizes flat scan results into a hierarchical directory tree", () => {
    const mockResults: LibraryScanResult[] = [
      {
        path: "D:/Pictures/Anime/Season1/001.jpg",
        image_size: [1920, 1080],
        people: [
          {
            person_index: 0,
            status: "high_confidence",
            box: { x1: 100, y1: 100, x2: 300, y2: 300, confidence: 0.95 },
            crops: {},
            margin: 0.2,
            consistency: 0.9,
            candidates: [],
          },
        ],
      },
      {
        path: "D:/Pictures/Anime/Season1/002.jpg",
        image_size: [1920, 1080],
        people: [
          {
            person_index: 0,
            status: "needs_review",
            box: { x1: 200, y1: 200, x2: 400, y2: 400, confidence: 0.8 },
            crops: {},
            margin: 0.1,
            consistency: 0.5,
            candidates: [],
          },
        ],
      },
      {
        path: "D:/Pictures/Anime/Season2/003.jpg",
        image_size: [1080, 1920],
        people: [],
      },
    ];

    const tree = buildFolderTree(mockResults, "D:/Pictures/Anime");

    expect(tree.id).toBe("root");
    expect(tree.name).toBe("Anime");
    expect(tree.totalFileCount).toBe(3);
    expect(tree.totalPeopleCount).toBe(2);
    expect(tree.hasReview).toBe(true);
    expect(tree.subFolders.length).toBe(2);

    const s1 = tree.subFolders.find((f) => f.name === "Season1");
    expect(s1).toBeDefined();
    expect(s1?.totalFileCount).toBe(2);
    expect(s1?.totalPeopleCount).toBe(2);
    expect(s1?.hasReview).toBe(true);
    expect(s1?.files.length).toBe(2);

    const s2 = tree.subFolders.find((f) => f.name === "Season2");
    expect(s2).toBeDefined();
    expect(s2?.totalFileCount).toBe(1);
    expect(s2?.totalPeopleCount).toBe(0);
  });

  it("filters a selected parent folder recursively and sorts by directory path", () => {
    const results = [
      { path: "D:/Pictures/Anime/B/002.jpg", image_size: [1, 1], people: [] },
      { path: "D:/Pictures/Anime/A/010.jpg", image_size: [1, 1], people: [] },
      { path: "D:/Pictures/Anime/A/002.jpg", image_size: [1, 1], people: [] },
      { path: "D:/Pictures/Other/skip.jpg", image_size: [1, 1], people: [] },
    ] satisfies LibraryScanResult[];

    expect(resultsForFolder(results, "D:/Pictures", "Anime").map((item) => item.path)).toEqual([
      "D:/Pictures/Anime/A/002.jpg",
      "D:/Pictures/Anime/A/010.jpg",
      "D:/Pictures/Anime/B/002.jpg",
    ]);
  });
});
