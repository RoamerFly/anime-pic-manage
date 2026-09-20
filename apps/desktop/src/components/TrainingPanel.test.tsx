import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { TrainingPanel } from "./TrainingPanel";

function render() {
  return renderToStaticMarkup(
    <TrainingPanel
      datasetToml="E:/lora-datasets/dataset.toml"
      outputName="character_alpha"
      onNotice={() => {}}
      onError={() => {}}
    />,
  );
}

describe("TrainingPanel", () => {
  it("explains that the trainer is external and shows the dataset path", () => {
    const html = render();

    expect(html).toContain("LoRA 训练（本机 kohya）");
    expect(html).toContain("E:/lora-datasets/dataset.toml");
    expect(html).toContain("character_alpha");
    expect(html).toContain("不打包");
  });

  it("disables starting until a trainer is detected", () => {
    const html = render();

    // No status has been loaded in a static browser render, so the start
    // button must stay disabled instead of launching something unexpected.
    expect(html).toContain("开始训练");
    expect(html).toMatch(/<button[^>]*disabled[^>]*>[^<]*<svg/);
  });
});
