import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type {
  LegacyModelCache,
  ModelInventoryEntry,
} from "@anime-pic-manage/shared-types";
import { ModelAdoptBanner, ModelCard } from "./SettingsPage";

function entry(overrides: Partial<ModelInventoryEntry> = {}): ModelInventoryEntry {
  return {
    id: "camie-initial",
    name: "Camie tagger",
    kind: "character_recognizer",
    delivery: "models_dir",
    repo_id: "deepghs/camie_tagger_onnx",
    size_mb: 824,
    license: "",
    note: "7 万级标签",
    status: "missing",
    installed_bytes: 0,
    path: null,
    bundled: false,
    downloadable: true,
    active: false,
    ...overrides,
  };
}

function render(model: ModelInventoryEntry) {
  return renderToStaticMarkup(
    <ModelCard
      entry={model}
      busy={null}
      onInstall={() => {}}
      onRemove={() => {}}
      onActivate={() => {}}
      onOpenPath={() => {}}
    />,
  );
}

describe("模型配置的模型卡片", () => {
  it("offers a download with its expected size while the model is missing", () => {
    const html = render(entry());

    expect(html).toContain("未下载");
    expect(html).toContain("下载 824 MB");
    expect(html).toContain("deepghs/camie_tagger_onnx");
    // Nothing is on disk yet, so there is nothing to delete or open.
    expect(html).not.toContain("删除");
    expect(html).not.toContain("打开目录");
  });

  it("marks the active recognizer and keeps it out of the delete path", () => {
    const html = render(
      entry({
        status: "installed",
        installed_bytes: 864_000_000,
        path: "D:/models/recognizer/camie-initial",
        active: true,
      }),
    );

    expect(html).toContain("已就绪");
    expect(html).toContain("使用中");
    expect(html).toContain("824 MB");
    expect(html).toContain("打开目录");
    // The active model may not be deleted, and it cannot be re-activated.
    expect(html).toMatch(/<button[^>]*disabled[^>]*>(?:(?!<\/button>).)*删除/s);
    expect(html).not.toContain("设为当前");
  });

  it("lets another installed recognizer take over", () => {
    const html = render(
      entry({ status: "installed", bundled: true, downloadable: false }),
    );

    expect(html).toContain("设为当前");
    expect(html).toContain("随包提供");
    // Bundled models ship with the app, so they can never be deleted.
    expect(html).not.toContain("删除");
  });

  it("reports a partially cached model and offers the missing download", () => {
    const html = render(
      entry({
        id: "wd14-swinv2-v3",
        name: "WD14 打标模型（SwinV2 v3）",
        kind: "tagger",
        delivery: "hf_cache",
        size_mb: 446,
        status: "partial",
        installed_bytes: 2_048_000,
      }),
    );

    expect(html).toContain("不完整");
    expect(html).toContain("2.0 MB");
    expect(html).toContain("下载 446 MB");
    expect(html).toContain("删除");
  });
});

describe("模型配置的旧缓存迁入提示", () => {
  const legacy: LegacyModelCache = {
    path: "E:/Cache/huggingface_cache/hub",
    repos: ["deepghs/ccip_onnx", "SmilingWolf/wd-swinv2-tagger-v3"],
    bytes: 618_000_000,
  };

  it("names the other cache, its size and what copying costs", () => {
    const html = renderToStaticMarkup(
      <ModelAdoptBanner legacy={legacy} busy={false} onAdopt={() => {}} />,
    );

    expect(html).toContain("E:/Cache/huggingface_cache/hub");
    expect(html).toContain("2 个模型");
    expect(html).toContain("589 MB");
    expect(html).toContain("复制到软件目录");
    expect(html).toContain("原位置的文件不会被删除");
    expect(html).not.toMatch(/<button[^>]*disabled/);
  });

  it("locks the button while the copy runs", () => {
    const html = renderToStaticMarkup(
      <ModelAdoptBanner legacy={legacy} busy onAdopt={() => {}} />,
    );

    expect(html).toMatch(/<button[^>]*disabled/);
  });
});
