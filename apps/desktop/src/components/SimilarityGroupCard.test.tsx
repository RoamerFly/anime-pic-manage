import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { SimilarityGroup } from "@anime-pic-manage/shared-types";
import { SimilarityGroupCard } from "./SimilarityGroupCard";

const group: SimilarityGroup = {
  group_id: "group-001",
  group_type: "exact_duplicate",
  average_similarity: 0.99,
  items: [
    {
      path: "D:/Pictures/Anime/motion.gif",
      file_size: 2048,
      dimensions: [512, 512],
      format: "gif",
      clarity_score: 42.5,
      is_recommended: true,
      recommend_reason: "最高画质",
      decision: "keep",
      frame_count: 12,
      is_animated: true,
      sampled_frames: [0, 6, 11],
    },
    {
      path: "D:/Pictures/Anime/still.png",
      file_size: 4096,
      dimensions: [512, 512],
      format: "png",
      clarity_score: 50.1,
      is_recommended: false,
      recommend_reason: "低清版本",
      decision: "archive",
      frame_count: 1,
      is_animated: false,
      sampled_frames: [0],
    },
  ],
};

function render() {
  return renderToStaticMarkup(
    <SimilarityGroupCard
      group={group}
      groupIdx={0}
      onApplyRecommended={() => undefined}
      onKeepAll={() => undefined}
      onArchiveAll={() => undefined}
      onDeleteAll={() => undefined}
      onItemDecisionChange={() => undefined}
      onPreviewImage={() => undefined}
      onContextMenu={() => undefined}
    />,
  );
}

describe("SimilarityGroupCard animation badge", () => {
  it("marks animated items and states which frames were compared", () => {
    const markup = render();

    expect(markup).toContain("animated-badge");
    expect(markup).toContain("动图 12 帧 · 比对首/中/末");
    expect(markup).toContain("已比较第 1/7/12 帧");
  });

  it("only badges the animated file", () => {
    expect(render().match(/animated-badge/g)).toHaveLength(1);
  });
});
