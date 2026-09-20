import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { LibraryScanResult } from "@anime-pic-manage/shared-types";
import { Filmstrip } from "./Filmstrip";

const results: LibraryScanResult[] = [
  {
    path: "D:/Pictures/Anime/001.png",
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
    path: "D:/Pictures/Anime/002.png",
    image_size: [1280, 720],
    people: [],
  },
];

function render() {
  return renderToStaticMarkup(
    <Filmstrip
      results={results}
      selectedPath={results[0].path}
      onSelectResult={() => undefined}
    />,
  );
}

describe("Filmstrip cards", () => {
  it("adds an expand button to every card for the large preview", () => {
    const markup = render();

    expect(markup.match(/filmstrip-expand-btn/g)).toHaveLength(2);
    expect(markup).toContain("大图查看 001.png");
    expect(markup).toContain("大图查看 002.png");
  });

  it("keeps the card clickable and keyboard reachable", () => {
    const markup = render();

    expect(markup.match(/role="button"/g)).toHaveLength(2);
    expect(markup.match(/tabindex="0"/g)).toHaveLength(2);
  });

  it("does not render the lightbox before a card is expanded", () => {
    expect(render()).not.toContain("filmstrip-lightbox");
  });
});
