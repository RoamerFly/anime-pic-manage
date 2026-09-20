import { describe, expect, it } from "vitest";
import {
  clampViewport,
  fitViewport,
  graphNodesOf,
  layoutWorkflow,
  linkOf,
  panViewport,
  zoomAtPoint,
} from "./WorkflowGraph";

const template = {
  prompt: {
    "3": {
      class_type: "KSampler",
      inputs: {
        seed: 1,
        steps: 20,
        positive: ["6", 0],
        negative: ["7", 0],
        model: ["4", 0],
      },
    },
    "4": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "a.safetensors" } },
    "6": { class_type: "CLIPTextEncode", inputs: { text: "a positive prompt", clip: ["4", 1] } },
    "7": { class_type: "CLIPTextEncode", inputs: { text: "a negative prompt", clip: ["4", 1] } },
  },
};

describe("workflow graph layout", () => {
  it("connects KSampler positive/negative to their CLIP nodes", () => {
    const layout = layoutWorkflow(template);
    const sampler = layout.nodes.find((node) => node.id === "3")!;
    const positive = layout.nodes.find((node) => node.id === "6")!;
    const negative = layout.nodes.find((node) => node.id === "7")!;

    const edge = (input: string) => layout.edges.find((item) => item.input === input)!;
    expect(edge("positive").from).toBe("6");
    expect(edge("negative").from).toBe("7");
    expect(edge("model").from).toBe("4");
    // The CLIP encoders sit to the left of the sampler they feed.
    expect(positive.x).toBeLessThan(sampler.x);
    expect(negative.x).toBeLessThan(sampler.x);
    // Edges start at the source's right edge and end on the target's left edge.
    expect(edge("positive").x1).toBe(positive.x + positive.width);
    expect(edge("positive").x2).toBe(sampler.x);
  });

  it("lands each edge on the row it actually feeds", () => {
    const layout = layoutWorkflow(template);
    const sampler = layout.nodes.find((node) => node.id === "3")!;
    const positiveRow = sampler.rows.find((row) => row.name === "positive")!;
    const modelRow = sampler.rows.find((row) => row.name === "model")!;
    const positiveEdge = layout.edges.find((edge) => edge.input === "positive")!;
    const modelEdge = layout.edges.find((edge) => edge.input === "model")!;

    expect(positiveEdge.y2).toBeCloseTo(sampler.y + positiveRow.offsetY, 5);
    expect(modelEdge.y2).toBeCloseTo(sampler.y + modelRow.offsetY, 5);
    expect(positiveEdge.y2).not.toBeCloseTo(modelEdge.y2, 1);
  });

  it("never overlaps two nodes in the same column", () => {
    const layout = layoutWorkflow(template);
    const byColumn = new Map<number, typeof layout.nodes>();
    for (const node of layout.nodes) {
      byColumn.set(node.depth, [...(byColumn.get(node.depth) ?? []), node]);
    }
    for (const nodes of byColumn.values()) {
      const sorted = [...nodes].sort((left, right) => left.y - right.y);
      for (let index = 1; index < sorted.length; index += 1) {
        const previous = sorted[index - 1];
        expect(sorted[index].y).toBeGreaterThanOrEqual(previous.y + previous.height);
      }
    }
  });

  it("accepts numeric node ids and rejects malformed links", () => {
    expect(linkOf([6, 1])).toEqual({ nodeId: "6", outputIndex: 1 });
    expect(linkOf(["6", 0])).toEqual({ nodeId: "6", outputIndex: 0 });
    expect(linkOf("a string input")).toBeNull();
    expect(linkOf(null)).toBeNull();
    expect(linkOf([])).toBeNull();

    const numeric = {
      prompt: {
        3: { class_type: "KSampler", inputs: { positive: [6, 0] } },
        6: { class_type: "CLIPTextEncode", inputs: { text: "x" } },
      },
    };
    const layout = layoutWorkflow(numeric);
    expect(layout.edges).toHaveLength(1);
    expect(layout.edges[0]).toMatchObject({ from: "6", to: "3", input: "positive" });
  });

  it("ignores links that point at nodes outside the graph", () => {
    const broken = {
      prompt: {
        "3": { class_type: "KSampler", inputs: { positive: ["999", 0] } },
      },
    };

    expect(layoutWorkflow(broken).edges).toHaveLength(0);
  });

  it("reads the node map from an envelope or a raw graph", () => {
    expect(Object.keys(graphNodesOf(template))).toEqual(["3", "4", "6", "7"]);
    expect(Object.keys(graphNodesOf(template.prompt))).toHaveLength(4);
  });
});

describe("workflow viewport", () => {
  it("pans freely on both axes", () => {
    const start = { x: 10, y: 20, zoom: 1 };

    expect(panViewport(start, 30, -15)).toEqual({ x: 40, y: 5, zoom: 1 });
    // Dragging right and down must not be clamped to a single axis.
    expect(panViewport(start, -30, 45)).toEqual({ x: -20, y: 65, zoom: 1 });
  });

  it("keeps the point under the cursor fixed while zooming", () => {
    const start = { x: 0, y: 0, zoom: 1 };

    const zoomed = zoomAtPoint(start, 2, 100, 50);

    // content point (100,50) stays under the cursor after scaling 1 -> 2
    expect(zoomed.zoom).toBe(2);
    expect(zoomed.x + 100 * 2).toBeCloseTo(100, 5);
    expect(zoomed.y + 50 * 2).toBeCloseTo(50, 5);
  });

  it("allows overscroll but keeps part of the graph reachable", () => {
    const layout = { width: 1000, height: 600 };
    const canvas = { width: 400, height: 300 };

    const dragged = clampViewport({ x: 9999, y: 9999, zoom: 1 }, layout, canvas);
    expect(dragged.x).toBeLessThanOrEqual(120);
    expect(dragged.y).toBeLessThanOrEqual(120);

    const farLeft = clampViewport({ x: -9999, y: -9999, zoom: 1 }, layout, canvas);
    expect(farLeft.x).toBeGreaterThanOrEqual(canvas.width - layout.width - 120);
    expect(farLeft.y).toBeGreaterThanOrEqual(canvas.height - layout.height - 120);
  });

  it("fits and centres the graph for fullscreen", () => {
    const layout = { width: 1600, height: 900 };
    const canvas = { width: 800, height: 600 };

    const fitted = fitViewport(layout, canvas);

    expect(fitted.zoom).toBeLessThan(1);
    expect(layout.width * fitted.zoom).toBeLessThanOrEqual(canvas.width);
    expect(layout.height * fitted.zoom).toBeLessThanOrEqual(canvas.height);
    // Centred within the canvas.
    expect(fitted.x).toBeGreaterThan(0);
    expect(fitted.y).toBeGreaterThan(0);
  });

  it("does not zoom past 100% for a small graph", () => {
    const fitted = fitViewport({ width: 200, height: 120 }, { width: 1200, height: 800 });

    expect(fitted.zoom).toBe(1);
  });
});
