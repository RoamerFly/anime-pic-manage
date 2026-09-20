import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertCircle,
  CircleCheck,
  FolderOpen,
  ImagePlus,
  Loader2,
  Play,
  Power,
  RefreshCw,
  Square,
  X,
} from "lucide-react";
import type {
  ComfyGenerateRequest,
  ComfyGenerateResult,
  ComfyModelList,
  ComfyProgress,
  ComfyStatus,
  IdentitySampleSummary,
  LoraCropMode,
  LoraExportCandidates,
  LoraExportResult,
  LoraQualityPreset,
} from "@anime-pic-manage/shared-types";
import { invokeCore, isTauriRuntime, normalizePreviewPath } from "../lib/tauri";
import { TrainingPanel } from "../components/TrainingPanel";
import { WorkflowGraph, type NodeOverrides } from "../components/WorkflowGraph";

/** Prompt presets matched to the checkpoint families used by the library. */
const QUALITY_PRESETS: Array<{
  id: string;
  label: string;
  hint: string;
  positive: string;
  negative: string;
  size: number;
  steps: number;
  cfg: number;
}> = [
  {
    id: "illustrious",
    label: "Illustrious / NoobAI",
    hint: "SDXL 动漫底模",
    positive: "masterpiece, best quality, amazing quality, very aesthetic",
    negative: "worst quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts",
    size: 1024,
    steps: 28,
    cfg: 6.0,
  },
  {
    id: "pony",
    label: "Pony / 动漫 Pony",
    hint: "需要 score_* 质量标签",
    positive: "score_9, score_8_up, score_7_up, source_anime",
    negative: "score_1, score_2, score_3, worst quality, low quality, bad anatomy",
    size: 1024,
    steps: 28,
    cfg: 7.0,
  },
  {
    id: "sd15",
    label: "SD1.5 动漫",
    hint: "primeMix / AnythingV5，6GB 显存最快",
    positive: "masterpiece, best quality, ultra-detailed, illustration",
    negative: "worst quality, low quality, lowres, bad anatomy, extra digits",
    size: 832,
    steps: 24,
    cfg: 7.0,
  },
];

const SIZE_PRESETS = [768, 832, 1024, 1216];

export function GenerationPage() {
  const [status, setStatus] = useState<ComfyStatus | null>(null);
  const [models, setModels] = useState<ComfyModelList | null>(null);
  const [templates, setTemplates] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // Routine and error feedback appears as a toast at the top of the page and
  // fades away on its own, so it never pushes the controls down.
  const [toast, setToast] = useState<{ text: string; kind: "info" | "error" } | null>(null);
  const [progress, setProgress] = useState<ComfyProgress | null>(null);
  const [result, setResult] = useState<ComfyGenerateResult | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const generatingRef = useRef(false);

  const [candidates, setCandidates] = useState<IdentitySampleSummary[]>([]);
  const [datasetDir, setDatasetDir] = useState("");
  const [exportIdentity, setExportIdentity] = useState("");
  const [exportTrigger, setExportTrigger] = useState("");
  const [cropMode, setCropMode] = useState<LoraCropMode>("person");
  const [qualityPreset, setQualityPreset] = useState<LoraQualityPreset>("pony");
  const [exportResolution, setExportResolution] = useState(1024);
  const [maxImages, setMaxImages] = useState(40);
  const [repeats, setRepeats] = useState(10);
  const [keepTokens, setKeepTokens] = useState(4);
  const [exportResult, setExportResult] = useState<LoraExportResult | null>(null);
  const [trainDatasetToml, setTrainDatasetToml] = useState("");
  const [trainOutputName, setTrainOutputName] = useState("");

  const [template, setTemplate] = useState("txt2img");
  const [templateDetail, setTemplateDetail] = useState<unknown>(null);
  const [nodeOverrides, setNodeOverrides] = useState<NodeOverrides>({});
  const [checkpoint, setCheckpoint] = useState("");
  const [lora, setLora] = useState("");
  const [loraStrength, setLoraStrength] = useState(0.8);
  const [positive, setPositive] = useState(
    "masterpiece, best quality, 1girl, solo",
  );
  const [negative, setNegative] = useState(
    QUALITY_PRESETS[0].negative,
  );
  const [steps, setSteps] = useState(28);
  const [cfg, setCfg] = useState(6);
  const [sampler, setSampler] = useState("euler_ancestral");
  const [scheduler, setScheduler] = useState("normal");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [batch, setBatch] = useState(1);
  const [seed, setSeed] = useState(-1);

  const refreshStatus = useCallback(async () => {
    const response = await invokeCore<ComfyStatus>("comfy_status", "comfy.status");
    if (response.payload) {
      setStatus(response.payload);
      setError(response.payload.error ?? null);
    } else if (response.error) {
      setError(response.error.message);
    }
    return response.payload ?? null;
  }, []);

  const refreshModels = useCallback(async () => {
    const response = await invokeCore<ComfyModelList>("comfy_models", "comfy.models");
    if (response.payload) {
      setModels(response.payload);
      setCheckpoint((current) => current || response.payload!.checkpoints[0] || "");
      setSampler(
        (current) =>
          current || response.payload!.samplers[0] || "euler_ancestral",
      );
      setScheduler(
        (current) => current || response.payload!.schedulers[0] || "normal",
      );
    }
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void refreshStatus();
    void invokeCore<string[]>("comfy_templates", "comfy.templates").then(
      (response) => {
        if (response.payload?.length) {
          setTemplates(response.payload);
          setTemplate(response.payload[0]);
        }
      },
    );
  }, [refreshStatus]);

  const refreshCandidates = useCallback(async () => {
    const response = await invokeCore<LoraExportCandidates>(
      "get_lora_export_candidates",
      "dataset.candidates",
    );
    if (!response.payload) return;
    setCandidates(response.payload.identities);
    setDatasetDir((current) => current || response.payload!.default_output_dir);
    const first = response.payload.identities[0];
    if (first) {
      setExportIdentity((current) => current || first.identity_id);
      setExportTrigger(
        (current) => current || first.label_name.toLowerCase().replace(/ /g, "_"),
      );
    }
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void refreshCandidates();
  }, [refreshCandidates]);

  useEffect(() => {
    if (status?.running) void refreshModels();
  }, [refreshModels, status?.running]);

  // Load the selected template's raw graph plus its saved node overrides.
  useEffect(() => {
    if (!isTauriRuntime() || !template) return;
    let active = true;
    void (async () => {
      const response = await invokeCore<{
        template: unknown;
        overrides: NodeOverrides;
      }>("comfy_template_detail", "comfy.template.detail", { name: template });
      if (!active || !response.payload) return;
      setTemplateDetail(response.payload.template);
      setNodeOverrides(response.payload.overrides ?? {});
      // Mirror the workflow into the quick fields so the panel shows the
      // template's own values instead of silently overriding them.
      const detail = response.payload.template as {
        prompt?: Record<string, unknown>;
        bindings?: Record<string, unknown>;
      } | null;
      const graph = detail?.prompt ?? (detail as Record<string, unknown> | null);
      const bindings = detail?.bindings ?? {};
      const bound = (key: string): unknown => {
        const path = bindings[key];
        if (!Array.isArray(path)) return undefined;
        let node: unknown = graph?.[String(path[0])];
        for (const step of path.slice(1)) {
          node = (node as Record<string, unknown> | undefined)?.[String(step)];
        }
        return node;
      };
      const text = bound("positive");
      if (typeof text === "string") setPositive(text);
      const negativeText = bound("negative");
      if (typeof negativeText === "string") setNegative(negativeText);
      const stepsValue = bound("steps");
      if (typeof stepsValue === "number" && stepsValue > 0) setSteps(stepsValue);
      const cfgValue = bound("cfg");
      if (typeof cfgValue === "number" && cfgValue > 0) setCfg(cfgValue);
      const samplerValue = bound("sampler");
      if (typeof samplerValue === "string" && samplerValue) setSampler(samplerValue);
      const schedulerValue = bound("scheduler");
      if (typeof schedulerValue === "string" && schedulerValue) setScheduler(schedulerValue);
      const widthValue = bound("width");
      if (typeof widthValue === "number" && widthValue > 0) setWidth(widthValue);
      const heightValue = bound("height");
      if (typeof heightValue === "number" && heightValue > 0) setHeight(heightValue);
      const batchValue = bound("batch");
      if (typeof batchValue === "number" && batchValue > 0) setBatch(batchValue);
    })();
    return () => {
      active = false;
    };
  }, [template]);

  const saveOverrides = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (error) {
      setToast({ text: error, kind: "error" });
      return;
    }
    if (notice) setToast({ text: notice, kind: "info" });
  }, [error, notice]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), toast.kind === "error" ? 8000 : 4000);
    return () => clearTimeout(timer);
  }, [toast]);
  const updateNodeOverride = useCallback(
    (nodeId: string, input: string, value: unknown) => {
      setNodeOverrides((current) => {
        const next: NodeOverrides = { ...current };
        const node = { ...(next[nodeId] ?? {}) };
        if (value === undefined) delete node[input];
        else node[input] = value;
        if (Object.keys(node).length === 0) delete next[nodeId];
        else next[nodeId] = node;
        // Persist with a small debounce so typing a prompt is one write.
        if (saveOverrides.current) clearTimeout(saveOverrides.current);
        saveOverrides.current = setTimeout(() => {
          void invokeCore("comfy_save_template_overrides", "comfy.template.overrides", {
            name: template,
            overrides: next,
          });
        }, 600);
        return next;
      });
    },
    [template],
  );

  const pickDatasetDir = async () => {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择 LoRA 训练集输出目录",
    });
    if (typeof selected === "string" && selected.trim()) {
      setDatasetDir(selected.trim());
    }
  };

  // Progress arrives while a workflow runs in ComfyUI.
  useEffect(() => {
    let active = true;
    let unlisten: (() => void) | undefined;
    void listen<ComfyProgress>("comfy://progress", (event) => {
      if (!active) return;
      setProgress(event.payload);
      if (event.payload.phase === "error") setError(event.payload.message);
    }).then((stop) => {
      if (!active) {
        stop();
        return;
      }
      unlisten = stop;
    });
    return () => {
      active = false;
      unlisten?.();
    };
  }, []);

  const runAction = async (name: string, action: () => Promise<void>) => {
    if (busy) return;
    setBusy(name);
    setError(null);
    try {
      await action();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  };

  const startComfy = () =>
    runAction("start", async () => {
      setNotice("正在启动 ComfyUI，首次启动需要十几秒…");
      const response = await invokeCore<ComfyStatus>(
        "comfy_start",
        "comfy.start",
        { waitSecs: 180 },
      );
      if (response.error) {
        setError(response.error.message);
        setNotice(null);
        return;
      }
      setStatus(response.payload ?? null);
      setNotice("ComfyUI 已就绪。");
    });

  const stopComfy = () =>
    runAction("stop", async () => {
      const response = await invokeCore<ComfyStatus>("comfy_stop", "comfy.stop");
      setStatus(response.payload ?? null);
      setNotice("已停止本应用启动的 ComfyUI。");
    });

  const generate = () =>
    runAction("generate", async () => {
      if (!checkpoint) {
        setError("请先选择一个 checkpoint。");
        return;
      }
      setGenerating(true);
      generatingRef.current = true;
      setResult(null);
      setProgress({ phase: "queued", current: 0, total: 0, message: "正在提交工作流…" });
      const request: ComfyGenerateRequest = {
        template,
        checkpoint,
        lora: lora ? { name: lora, strength: loraStrength } : null,
        positive,
        negative,
        steps,
        cfg,
        sampler,
        scheduler,
        width,
        height,
        batch,
        seed,
        filename_prefix: "anime-pic-manage",
        node_overrides:
          Object.keys(nodeOverrides).length > 0 ? nodeOverrides : undefined,
      };
      const response = await invokeCore<ComfyGenerateResult>(
        "comfy_generate",
        "comfy.generate",
        { request },
      );
      setGenerating(false);
      generatingRef.current = false;
      if (response.error || !response.payload) {
        setError(response.error?.message ?? "生成失败。");
        return;
      }
      setResult(response.payload);
      setSeed(response.payload.seed);
      setNotice(
        `已生成 ${response.payload.images.length} 张图片，耗时 ${(
          response.payload.elapsed_ms / 1000
        ).toFixed(1)} 秒。`,
      );
    });

  const cancel = async () => {
    await invokeCore<boolean>("comfy_cancel", "comfy.cancel");
    setNotice("已请求取消生成。");
  };

  const runExport = () =>
    runAction("export", async () => {
      if (!exportIdentity) {
        setError("请先选择一个有标注样本的角色。");
        return;
      }
      setExportResult(null);
      setNotice("正在导出训练集并调用 WD14 打标，首次运行需要下载打标模型…");
      const response = await invokeCore<LoraExportResult>(
        "export_lora_dataset",
        "dataset.export",
        {
          options: {
            identity_id: exportIdentity,
            trigger: exportTrigger,
            output_dir: datasetDir,
            crop_mode: cropMode,
            resolution: exportResolution,
            max_images: maxImages,
            quality_preset: qualityPreset,
            repeats,
            keep_tokens: keepTokens,
            shuffle_caption: true,
            drop_character_tags: true,
          },
        },
      );
      if (response.error || !response.payload) {
        setNotice(null);
        setError(response.error?.message ?? "训练集导出失败。");
        return;
      }
      setExportResult(response.payload);
      setTrainDatasetToml(response.payload.dataset_toml);
      setTrainOutputName(
        response.payload.trigger ||
          response.payload.train_dir.split(/[\\/]/).filter(Boolean).pop() ||
          "lora",
      );
      setNotice(
        `训练集已导出：保留 ${response.payload.kept} 张，跳过 ${response.payload.skipped} 张。`,
      );
    });

  const applyPreset = (preset: (typeof QUALITY_PRESETS)[number]) => {
    setPositive((current) =>
      current.includes(preset.positive.split(",")[0])
        ? current
        : `${preset.positive}, ${current}`.trim(),
    );
    setNegative(preset.negative);
    setWidth(preset.size);
    setHeight(preset.size);
    setSteps(preset.steps);
    setCfg(preset.cfg);
  };

  const statusLabel = useMemo(() => {
    if (!status) return "未检测";
    if (!status.configured) return "未配置";
    if (!status.installed) return "目录无效";
    if (status.running) return "运行中";
    return "已停止";
  }, [status]);

  return (
    <div className="page-stack generation-page">
      <section className="library-toolbar panel-card generation-toolbar">
        <div className="library-identity">
          <span className="toolbar-icon">
            <ImagePlus size={18} />
          </span>
          <div>
            <span className="eyebrow">COMFYUI</span>
            <strong>AI 生图</strong>
            <small>
              {status?.root ?? "在“设置 → 生图配置”中填写你的 ComfyUI 目录"}
            </small>
          </div>
        </div>
        <div className="generation-status">
          <span
            className={`status-badge ${status?.running ? "high" : "unknown"}`}
          >
            <span />
            {statusLabel}
            {status?.version ? ` · v${status.version}` : ""}
            {status?.device ? ` · ${status.device}` : ""}
            {status?.vram_gb ? ` · ${status.vram_gb} GB` : ""}
          </span>
          <button
            className="ghost-button"
            onClick={() =>
              void runAction("status", async () => {
                await refreshStatus();
              })
            }
            disabled={busy !== null}
          >
            {busy === "status" ? (
              <Loader2 size={15} className="spin" />
            ) : (
              <RefreshCw size={15} />
            )}
            检测
          </button>
          {status?.running ? (
            <button
              className="ghost-button"
              onClick={() => void stopComfy()}
              disabled={busy !== null || !status?.owned}
              title={status?.owned ? "停止本应用启动的 ComfyUI" : "该实例由你手动启动，应用不会关闭它"}
            >
              {busy === "stop" ? (
                <Loader2 size={15} className="spin" />
              ) : (
                <Power size={15} />
              )}
              停止
            </button>
          ) : (
            <button
              className="primary-button"
              onClick={() => void startComfy()}
              disabled={busy !== null || !status?.installed}
            >
              {busy === "start" ? (
                <Loader2 size={15} className="spin" />
              ) : (
                <Play size={15} />
              )}
              启动 ComfyUI
            </button>
          )}
          <button
            className="ghost-button"
            onClick={() =>
              void invokeCore<string>("comfy_open_output", "comfy.open_output")
            }
          >
            <FolderOpen size={15} /> 输出目录
          </button>
        </div>
      </section>

      {toast && (
        <div
          className={`app-toast ${toast.kind}`}
          role="status"
          onClick={() => setToast(null)}
        >
          {toast.kind === "error" ? <AlertCircle size={15} /> : <CircleCheck size={15} />}
          <span>{toast.text}</span>
        </div>
      )}

      <section className="panel-card generation-controls">
        <div className="generation-grid">
          <label className="generation-field">
            <span>工作流模板</span>
            <select value={template} onChange={(e) => setTemplate(e.target.value)}>
              {(templates.length ? templates : ["txt2img"]).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="generation-field">
            <span>Checkpoint</span>
            <select
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
            >
              <option value="">请选择</option>
              {(models?.checkpoints ?? []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="generation-field">
            <span>角色 LoRA</span>
            <select value={lora} onChange={(e) => setLora(e.target.value)}>
              <option value="">不使用</option>
              {(models?.loras ?? []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="generation-field">
            <span>LoRA 强度 {loraStrength.toFixed(2)}</span>
            <input
              type="range"
              min="0"
              max="1.5"
              step="0.05"
              value={loraStrength}
              onChange={(e) => setLoraStrength(Number(e.target.value))}
            />
          </label>
        </div>

        <div className="generation-grid">
          <label className="generation-field">
            <span>尺寸</span>
            <div className="generation-inline">
              <select
                value={width === height ? width : 0}
                onChange={(e) => {
                  const size = Number(e.target.value);
                  if (size > 0) {
                    setWidth(size);
                    setHeight(size);
                  }
                }}
              >
                <option value={0}>自定义</option>
                {SIZE_PRESETS.map((size) => (
                  <option key={size} value={size}>
                    {size} × {size}
                  </option>
                ))}
              </select>
              <input
                type="number"
                min={256}
                max={2048}
                step={64}
                value={width}
                onChange={(e) => setWidth(Number(e.target.value))}
              />
              <input
                type="number"
                min={256}
                max={2048}
                step={64}
                value={height}
                onChange={(e) => setHeight(Number(e.target.value))}
              />
            </div>
          </label>
          <label className="generation-field">
            <span>步数 / CFG</span>
            <div className="generation-inline">
              <input
                type="number"
                min={1}
                max={150}
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value))}
              />
              <input
                type="number"
                min={1}
                max={30}
                step={0.5}
                value={cfg}
                onChange={(e) => setCfg(Number(e.target.value))}
              />
            </div>
          </label>
          <label className="generation-field">
            <span>采样器 / 调度器</span>
            <div className="generation-inline">
              <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
                {(models?.samplers?.length
                  ? models.samplers
                  : [sampler]
                ).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
              <select
                value={scheduler}
                onChange={(e) => setScheduler(e.target.value)}
              >
                {(models?.schedulers?.length
                  ? models.schedulers
                  : [scheduler]
                ).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </div>
          </label>
          <label className="generation-field">
            <span>种子 / 批量</span>
            <div className="generation-inline">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
              <button
                className="ghost-button compact"
                onClick={() => setSeed(-1)}
                title="随机：由应用生成一个有效种子（ComfyUI 接口不接受 -1）"
              >
                随机
              </button>
              <input
                type="number"
                min={1}
                max={8}
                value={batch}
                onChange={(e) => setBatch(Number(e.target.value))}
              />
            </div>
          </label>
        </div>

        <div className="generation-actions">
          <button
            className="primary-button"
            onClick={() => void generate()}
            disabled={generating || busy !== null || !status?.running}
          >
            {generating ? (
              <Loader2 size={16} className="spin" />
            ) : (
              <Play size={16} />
            )}
            开始生成
          </button>
          {generating && (
            <button className="danger-button" onClick={() => void cancel()}>
              <Square size={15} /> 取消
            </button>
          )}
          {progress && (
            <span className="generation-progress-text">{progress.message}</span>
          )}
        </div>
      </section>

      {result && result.images.length > 0 && (
        <section className="panel-card generation-results">
          <div className="generation-results-heading">
            <strong>本次生成（{result.images.length} 张）</strong>
            <small title={result.output_dir}>{result.output_dir}</small>
          </div>
          <div className="generation-results-grid">
            {result.images.map((image) => {
              const src = isTauriRuntime()
                ? convertFileSrc(normalizePreviewPath(image.path))
                : image.path;
              return (
                <button
                  key={image.path}
                  className="generation-result-card"
                  onClick={() => setPreview(image.path)}
                  title={image.filename}
                >
                  <img src={src} alt={image.filename} loading="lazy" />
                  <span>{image.filename}</span>
                </button>
              );
            })}
          </div>
        </section>
      )}

      <section className="panel-card generation-workflow">
        <div className="generation-results-heading">
          <strong>工作流（{template}）</strong>
          <small>
            按数据流展示所选模板的节点与连线；可以直接改节点里的文本、数字与开关，
            改动作为本机覆盖值保存，生成时优先于上面的快捷参数。
          </small>
        </div>
        <WorkflowGraph
          template={templateDetail}
          overrides={nodeOverrides}
          onChange={updateNodeOverride}
        />
      </section>

      <section className="panel-card generation-dataset">
        <div className="generation-results-heading">
          <strong>训练集导出（角色 LoRA）</strong>
          <small>
            从人工矫正后的标注里导出该角色的训练数据：按人物框裁剪 → 去重 → WD14
            打标 → 生成 kohya 配置
          </small>
        </div>

        {candidates.length === 0 ? (
          <div className="banner-callout banner-info">
            <AlertCircle size={16} />
            <span>
              还没有已标注的角色样本。请先在「角色识别」里完成识别与人工矫正，再回来导出。
            </span>
          </div>
        ) : (
          <>
            <div className="generation-grid">
              <label className="generation-field">
                <span>角色（按标注样本数排序）</span>
                <select
                  value={exportIdentity}
                  onChange={(event) => {
                    setExportIdentity(event.target.value);
                    const picked = candidates.find(
                      (item) => item.identity_id === event.target.value,
                    );
                    if (picked) {
                      setExportTrigger(picked.label_name.toLowerCase().replace(/ /g, "_"));
                    }
                  }}
                >
                  {candidates.map((item) => (
                    <option key={item.identity_id} value={item.identity_id}>
                      {item.display_name}（{item.image_count} 张，矫正 {item.manual_count}）
                    </option>
                  ))}
                </select>
              </label>
              <label className="generation-field">
                <span>触发词</span>
                <input
                  type="text"
                  value={exportTrigger}
                  onChange={(event) => setExportTrigger(event.target.value)}
                  placeholder="例如 character_alpha"
                />
              </label>
              <label className="generation-field">
                <span>裁剪方式</span>
                <select
                  value={cropMode}
                  onChange={(event) => setCropMode(event.target.value as LoraCropMode)}
                >
                  <option value="person">按人物框（推荐，自动去背景杂物）</option>
                  <option value="head">头部特写</option>
                  <option value="bust">半身</option>
                  <option value="large">人物框 + 少量环境</option>
                  <option value="none">整图不裁剪</option>
                </select>
              </label>
              <label className="generation-field">
                <span>质量标签前缀（按基座选择）</span>
                <select
                  value={qualityPreset}
                  onChange={(event) =>
                    setQualityPreset(event.target.value as LoraQualityPreset)
                  }
                >
                  <option value="pony">Pony：score_9,score_8_up,score_7_up</option>
                  <option value="illustrious">Illustrious：masterpiece,best quality…</option>
                  <option value="sd15">SD1.5：masterpiece,best quality</option>
                  <option value="none">不加前缀</option>
                </select>
              </label>
            </div>

            <div className="generation-grid">
              <label className="generation-field">
                <span>训练分辨率</span>
                <select
                  value={exportResolution}
                  onChange={(event) => setExportResolution(Number(event.target.value))}
                >
                  <option value={512}>512（SD1.5，6GB 显存最稳）</option>
                  <option value={768}>768（SDXL 省显存）</option>
                  <option value={1024}>1024（SDXL 标准）</option>
                </select>
              </label>
              <label className="generation-field">
                <span>最多导出张数 / repeats</span>
                <div className="generation-inline">
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={maxImages}
                    onChange={(event) => setMaxImages(Number(event.target.value))}
                  />
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={repeats}
                    onChange={(event) => setRepeats(Number(event.target.value))}
                  />
                </div>
              </label>
              <label className="generation-field">
                <span>keep_tokens（保护前缀+触发词）</span>
                <input
                  type="number"
                  min={0}
                  max={20}
                  value={keepTokens}
                  onChange={(event) => setKeepTokens(Number(event.target.value))}
                />
              </label>
              <label className="generation-field">
                <span>输出目录</span>
                <div className="generation-inline">
                  <input
                    type="text"
                    value={datasetDir}
                    onChange={(event) => setDatasetDir(event.target.value)}
                  />
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => void pickDatasetDir()}
                  >
                    <FolderOpen size={15} /> 选择
                  </button>
                </div>
              </label>
            </div>

            <div className="generation-actions">
              <button
                className="primary-button"
                onClick={() => void runExport()}
                disabled={busy !== null}
              >
                {busy === "export" ? (
                  <Loader2 size={16} className="spin" />
                ) : (
                  <ImagePlus size={16} />
                )}
                导出训练集
              </button>
              {exportResult && (
                <button
                  className="ghost-button"
                  onClick={() =>
                    void invokeCore<void>("show_item_in_folder", "system.explorer.show", {
                      path: exportResult.train_dir,
                    })
                  }
                >
                  <FolderOpen size={15} /> 打开训练集目录
                </button>
              )}
              {exportResult && (
                <span className="generation-progress-text">
                  {exportResult.kept} 张已导出 · {exportResult.skipped} 张被跳过 · 触发词{" "}
                  <code>{exportResult.trigger}</code>
                </span>
              )}
            </div>

            {exportResult && (
              <div className="generation-caption-preview">
                <strong>caption 抽样（可直接检查打标质量）</strong>
                {exportResult.samples.map((sample) => (
                  <div key={sample.image} className="generation-caption-row">
                    <code>{sample.image.split(/[\\/]/).pop()}</code>
                    <span>{sample.caption}</span>
                  </div>
                ))}
                {exportResult.skip_reasons.length > 0 && (
                  <small>
                    跳过示例：
                    {exportResult.skip_reasons
                      .slice(0, 3)
                      .map((item) => `${item.path.split(/[\\/]/).pop()}（${item.reason}）`)
                      .join("；")}
                  </small>
                )}
              </div>
            )}
          </>
        )}
      </section>

      <TrainingPanel
        datasetToml={trainDatasetToml}
        outputName={trainOutputName}
        onNotice={setNotice}
        onError={setError}
      />

      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)}>
          <div
            className="filmstrip-lightbox"
            role="dialog"
            aria-modal="true"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="filmstrip-lightbox-header">
              <strong>{preview.split(/[\\/]/).pop()}</strong>
              <small>{preview}</small>
              <button
                className="icon-close"
                onClick={() => setPreview(null)}
                aria-label="关闭"
              >
                <X size={18} />
              </button>
            </div>
            <div className="filmstrip-lightbox-body">
              <img
                src={
                  isTauriRuntime()
                    ? convertFileSrc(normalizePreviewPath(preview))
                    : preview
                }
                alt="生成结果"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
