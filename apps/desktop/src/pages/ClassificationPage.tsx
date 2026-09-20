import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { convertFileSrc } from "@tauri-apps/api/core";
import {
  AlertCircle,
  BrainCircuit,
  Clock3,
  FileImage,
  FolderOpen,
  Info,
  Play,
  X,
} from "lucide-react";
import type {
  BatchMoveResult,
  CharacterIdentity,
  CharacterIdentityListPayload,
  ImageAnnotation,
  ImageAnnotationsPayload,
  LibraryScanPayload,
  LibraryScanPerson,
  LibraryScanResult,
  LibrarySelection,
  NormalizedBoundingBox,
  PersonalTrainingStatus,
  WorkerStatus,
} from "@anime-pic-manage/shared-types";
import { invokeCore, isTauriRuntime } from "../lib/tauri";
import { appendAnnotationHistory } from "../lib/annotation-history";
import { readStoredScanForDirectory } from "../lib/scan-session";
import { isScanActive } from "../lib/scan-state";
import {
  isUsablePersonalModelVersion,
  readPersonalTrainingStatus,
  type PersonalTrainingView,
} from "../lib/personal-model";
import { Filmstrip } from "../components/Filmstrip";
import {
  UnifiedInspector,
  type EditableAnnotation,
  type InspectorPerson,
  type PersonStatus,
  type ResultFilter,
} from "../components/UnifiedInspector";
import { AnnotationCanvas } from "../components/AnnotationCanvas";
import {
  ScanFolderTree,
  resultsForFolder,
  type FolderTreeNode,
} from "../components/ScanFolderTree";
import { ImageContextMenu } from "../components/ContextMenu";
import type { DemoPerson, PreviewFile, ScanController } from "../types";

const MAX_ANNOTATION_HISTORY = 50;

const demoPeople: DemoPerson[] = [
  {
    id: "person-01",
    title: "人物 01",
    bbox: { left: 16, top: 14, width: 35, height: 62 },
    confidence: 0.94,
    status: "high",
    source: { person_index: 0 },
    candidates: [
      { label: "character_tag_alpha", score: 0.94 },
      { label: "character_tag_beta", score: 0.03 },
      { label: "character_tag_gamma", score: 0.01 },
      { label: "character_tag_delta", score: 0.01 },
      { label: "character_tag_epsilon", score: 0.01 },
    ],
  },
  {
    id: "person-02",
    title: "人物 02",
    bbox: { left: 58, top: 22, width: 28, height: 54 },
    confidence: 0.78,
    status: "review",
    source: { person_index: 1 },
    margin: 0.06,
    consistency: 0.82,
    reason: "候选间距较小",
    candidates: [
      { label: "character_tag_beta", score: 0.78 },
      { label: "character_tag_gamma", score: 0.72 },
      { label: "character_tag_alpha", score: 0.08 },
    ],
  },
];

export function clamp01(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

export function isValidAnnotationLabel(value: string): boolean {
  const normalized = value.trim();
  return (
    normalized.length > 0 &&
    !["未命名分类", "未命名", "新建分类"].includes(normalized)
  );
}

export function normalizePersonStatus(status: string): PersonStatus {
  if (status === "high" || status === "high_confidence") return "high";
  if (status === "review" || status === "needs_review") return "review";
  // A reference-image match is a suggestion the user still confirms, so it
  // belongs in the review bucket rather than "unrecognised".
  if (status === "reference_match") return "review";
  return "unknown";
}

function absoluteFolderPath(root: string, relativePath: string): string {
  if (!relativePath) return root;
  const separator = root.includes("\\") ? "\\" : "/";
  return `${root.replace(/[\\/]+$/, "")}${separator}${relativePath.replace(/[\\/]/g, separator)}`;
}

export function ClassificationPage({
  library,
  worker,
  onChooseLibrary,
  initialFilter = "all",
  autoStart: _autoStart = false,
  onScanCompleted: _onScanCompleted,
  controller,
  onOpenPersonalModel,
}: {
  library: LibrarySelection | null;
  worker: WorkerStatus;
  onChooseLibrary: () => void;
  initialFilter?: ResultFilter;
  autoStart?: boolean;
  onScanCompleted?: (payload: LibraryScanPayload) => void;
  controller: ScanController;
  onOpenPersonalModel?: () => void;
}) {
  const [filter, setFilter] = useState<ResultFilter>(initialFilter);
  const [activePersonId, setActivePersonId] = useState<string | null>(null);
  const [selectedResultPath, setSelectedResultPath] = useState<string | null>(
    null,
  );
  const [previewFile, setPreviewFile] = useState<PreviewFile | null>(null);
  const initialStoredScan = useMemo(
    () => readStoredScanForDirectory(library?.path),
    [library?.path],
  );
  const [scan, setScan] = useState<LibraryScanPayload | null>(initialStoredScan);
  const [notice, setNotice] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [editorMode, setEditorMode] = useState(false);
  const [annotations, setAnnotations] = useState<EditableAnnotation[]>([]);
  const [annotationLoading, setAnnotationLoading] = useState(false);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  const [annotationRevision, setAnnotationRevision] = useState(0);
  const [identities, setIdentities] = useState<CharacterIdentity[]>([]);
  const [activeAnnotationId, setActiveAnnotationId] = useState<string | null>(
    null,
  );
  const [labelDraft, setLabelDraft] = useState("");
  const [history, setHistory] = useState<EditableAnnotation[][]>([[]]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [personalStatus, setPersonalStatus] =
    useState<PersonalTrainingView | null>(null);
  const [quickTraining, setQuickTraining] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const initialAnnotationsRef = useRef<EditableAnnotation[]>([]);
  const trainingTriggerRef = useRef<Set<string>>(new Set());

  const handleQuickTrain = useCallback(async () => {
    if (quickTraining) return;
    if (!personalStatus?.ready) {
      setScanError(
        personalStatus?.readiness_message ||
          "已验证样本尚不足以训练个人模型，至少需要2个角色达标（各至少3张已保存标注）。",
      );
      return;
    }
    setQuickTraining(true);
    setNotice("正在汇聚已保存的全部标注样本进行再训练...");
    setScanError(null);
    const response = await invokeCore<PersonalTrainingStatus>(
      "train_personal_model",
      "personal.training.start",
      { autoActivate: true, auto_activate: true },
    );
    if (response.error || !response.payload) {
      setScanError(
        response.error?.message ?? "个人模型训练失败，请检查推理引擎日志。",
      );
    } else {
      const next = readPersonalTrainingStatus(response.payload);
      if (next) {
        setPersonalStatus(next);
        if (next.active_version != null) {
          controller.setSelectedModelVersion(next.active_version);
        }
        setNotice(
          `个人模型 personal-v${next.active_version ?? ""} 训练完成并已自动设为当前扫描模型！可在上方下拉框切换。`,
        );
        setDirectoryDialogMode("scan");
      }
    }
    setQuickTraining(false);
  }, [quickTraining, personalStatus, controller]);

  const [directoryDialogMode, setDirectoryDialogMode] = useState<
    "filter" | "scan" | null
  >(null);
  const [selectedFolderPath, setSelectedFolderPath] = useState("");
  const [dialogFolder, setDialogFolder] = useState<FolderTreeNode | null>(null);

  const scanResults = scan?.results ?? [];
  const scanRootPath = scan?.directory ?? library?.path;
  const visibleScanResults = useMemo(
    () => resultsForFolder(scanResults, scanRootPath, selectedFolderPath),
    [scanResults, scanRootPath, selectedFolderPath],
  );
  const selectedResult =
    visibleScanResults.find((result) => result.path === selectedResultPath) ??
    visibleScanResults[0];
  const isDemo = !scan && !previewFile;
  const progress = controller.progress;
  const scanState = controller.state;
  const scanBusy = isScanActive(scanState);

  const selectedIndex = visibleScanResults.findIndex(
    (r) => r.path === (selectedResultPath ?? selectedResult?.path),
  );
  const hasPrevImage = selectedIndex > 0;
  const hasNextImage =
    selectedIndex >= 0 && selectedIndex < visibleScanResults.length - 1;

  const handlePrevImage = useCallback(() => {
    const idx = visibleScanResults.findIndex(
      (r) => r.path === (selectedResultPath ?? selectedResult?.path),
    );
    if (idx > 0) {
      if (
        dirty &&
        !globalThis.confirm(
          "当前图片有未保存的标注调整，切换将放弃未保存内容，是否继续？",
        )
      ) {
        return;
      }
      setSelectedResultPath(visibleScanResults[idx - 1].path);
      setActivePersonId(null);
    }
  }, [visibleScanResults, selectedResultPath, selectedResult?.path, dirty]);

  const handleNextImage = useCallback(() => {
    const idx = visibleScanResults.findIndex(
      (r) => r.path === (selectedResultPath ?? selectedResult?.path),
    );
    if (idx >= 0 && idx < visibleScanResults.length - 1) {
      if (
        dirty &&
        !globalThis.confirm(
          "当前图片有未保存的标注调整，切换将放弃未保存内容，是否继续？",
        )
      ) {
        return;
      }
      setSelectedResultPath(visibleScanResults[idx + 1].path);
      setActivePersonId(null);
    }
  }, [visibleScanResults, selectedResultPath, selectedResult?.path, dirty]);

  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    path: string;
  } | null>(null);

  const handleShowInFolder = useCallback(async (path: string) => {
    const res = await invokeCore<void>(
      "show_item_in_folder",
      "system.explorer.show",
      { path },
    );
    if (res.error) {
      setNotice("无法打开所在文件夹: " + res.error.message);
    }
  }, []);

  const handleMoveSingleFile = useCallback(
    async (path: string) => {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "选择图片移动目标文件夹",
      });
      if (typeof selected === "string" && selected.trim()) {
        const dest = selected.trim();
        const res = await invokeCore<BatchMoveResult>(
          "move_library_files",
          "library.files.move",
          {
            sources: [path],
            destination_directory: dest,
          },
        );
        if (res.payload && res.payload.success_count > 0) {
          controller.updateScanResults(res.payload.moved);
          setNotice(`已成功将文件移动至: ${dest}`);
        } else {
          setScanError(res.error?.message ?? "移动文件失败。");
        }
      }
    },
    [controller],
  );

  const handleDeleteSingleFile = useCallback(
    async (path: string) => {
      const fileName = path.split(/[\\/]/).pop() || path;
      const confirmed = globalThis.confirm(
        `确定要从磁盘永久删除以下图片文件吗？\n\n${fileName}\n(${path})\n\n此操作不可撤销。`,
      );
      if (!confirmed) return;
      const res = await invokeCore<boolean>(
        "delete_library_file",
        "library.file.delete",
        { path },
      );
      if (res.error) {
        setScanError(res.error.message ?? "删除文件失败。");
        return;
      }
      const currentResults = scan?.results ?? [];
      const nextResults = currentResults.filter((r) => r.path !== path);
      const nextScan: LibraryScanPayload = {
        ...(scan ?? {
          directory: library?.path ?? "",
          total_discovered: nextResults.length,
          processed: nextResults.length,
          results: nextResults,
          errors: [],
          model_name: "基础通用模型",
          cancelled: false,
        }),
        results: nextResults,
        total_discovered: nextResults.length,
        processed: nextResults.length,
      };
      setScan(nextScan);
      controller.updateScanResults([]);
      if (selectedResultPath === path || selectedResult?.path === path) {
        const nextItem =
          nextResults[
            Math.min(selectedIndex, Math.max(0, nextResults.length - 1))
          ];
        setSelectedResultPath(nextItem?.path ?? null);
      }
      setNotice(`已删除图片文件: ${fileName}`);
    },
    [
      controller,
      library?.path,
      scan,
      selectedIndex,
      selectedResult?.path,
      selectedResultPath,
    ],
  );

  // 全局键盘左右键快速切图
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        handlePrevImage();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        handleNextImage();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handlePrevImage, handleNextImage]);

  // 加载个人模型状态用于模型选择器和训练入口
  useEffect(() => {
    if (!isTauriRuntime()) return;
    void invokeCore<PersonalTrainingStatus>(
      "get_personal_training_status",
      "personal.training.status",
    ).then((res) => {
      if (res.payload) {
        setPersonalStatus(readPersonalTrainingStatus(res.payload));
      }
    });
  }, []);

  useEffect(() => {
    const restored = readStoredScanForDirectory(library?.path);
    setScan(restored);
    setSelectedFolderPath("");
    setSelectedResultPath(restored?.results[0]?.path ?? null);
    setActivePersonId(null);
    setEditorMode(false);
    setAnnotations([]);
    setActiveAnnotationId(null);
    setDirty(false);
    initialAnnotationsRef.current = [];
  }, [library?.path]);

  useEffect(() => {
    const latest = controller.latestScan;
    if (!latest) return;
    setScan(latest);
    setSelectedResultPath((current) =>
      current && latest.results.some((result) => result.path === current)
        ? current
        : latest.results[0]?.path ?? null,
    );
  }, [controller.latestScan]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void invokeCore<CharacterIdentityListPayload | CharacterIdentity[]>(
      "list_character_identities",
      "identity.list",
    ).then((response) => {
      if (response.error || !response.payload) return;
      const value: unknown = response.payload;
      const rows = Array.isArray(value)
        ? value
        : (value as CharacterIdentityListPayload).identities;
      if (!Array.isArray(rows)) return;
      setIdentities(
        rows.flatMap((item) => {
          if (!item || typeof item !== "object") return [];
          const raw = item as Record<string, unknown>;
          const id = typeof raw.id === "string" ? raw.id : "";
          const name =
            typeof raw.display_name === "string"
              ? raw.display_name
              : typeof raw.name === "string"
                ? raw.name
                : "";
          return id && name
            ? [
                {
                  id,
                  display_name: name,
                  aliases: Array.isArray(raw.aliases)
                    ? raw.aliases.filter(
                        (alias): alias is string => typeof alias === "string",
                      )
                    : undefined,
                  sample_count:
                    typeof raw.sample_count === "number"
                      ? raw.sample_count
                      : undefined,
                },
              ]
            : [];
        }),
      );
    });
  }, []);

  const toUiPerson = useCallback(
    (result: LibraryScanResult, person: LibraryScanPerson): InspectorPerson => {
      const top1 = person.top1;
      return {
        id: `${result.path}#${person.person_index}`,
        title:
          top1?.display_name ||
          top1?.character_tag ||
          `人物 ${String(person.person_index + 1).padStart(2, "0")}`,
        confidence: top1?.confidence ?? 0,
        status: normalizePersonStatus(person.status),
        personIndex: person.person_index,
        candidates: person.candidates.map((candidate) => ({
          label: candidate.display_name || candidate.character_tag,
          score: candidate.confidence,
          source: candidate.source,
          baseScore: candidate.base_score,
          personalScore: candidate.personal_score,
          sampleCount: candidate.sample_count,
          reason: candidate.reason,
        })),
        margin: person.margin,
        consistency: person.consistency,
        reason: person.reason,
      };
    },
    [],
  );

  const realPeople = useMemo(
    () =>
      selectedResult
        ? selectedResult.people.map((p) => toUiPerson(selectedResult, p))
        : [],
    [selectedResult, toUiPerson],
  );

  const modelAnnotations = useCallback(
    (result: LibraryScanResult): EditableAnnotation[] => {
      const [imageWidth, imageHeight] = result.image_size;
      const width = Math.max(imageWidth, 1);
      const height = Math.max(imageHeight, 1);
      return result.people.map((person, index) => {
        const ui = toUiPerson(result, person);
        return {
          clientId: `model-${result.path}-${index}`,
          personId: ui.id,
          label_name: ui.title,
          identity_id: null,
          bbox: {
            x: clamp01(person.box.x1 / width),
            y: clamp01(person.box.y1 / height),
            width: clamp01((person.box.x2 - person.box.x1) / width),
            height: clamp01((person.box.y2 - person.box.y1) / height),
          },
          source: "model",
          manually_adjusted: false,
        };
      });
    },
    [toUiPerson],
  );

  const clone = useCallback(
    (items: EditableAnnotation[]) =>
      items.map((item) => ({ ...item, bbox: { ...item.bbox } })),
    [],
  );
  const sameAnnotations = useCallback(
    (a: EditableAnnotation[], b: EditableAnnotation[]) =>
      JSON.stringify(
        a.map(({ clientId: _c, personId: _p, ...item }) => item),
      ) ===
      JSON.stringify(
        b.map(({ clientId: _c, personId: _p, ...item }) => item),
      ),
    [],
  );

  const resetEditor = useCallback(
    (items: EditableAnnotation[], revision: number) => {
      const next = clone(items);
      setAnnotations(next);
      setHistory([clone(next)]);
      setHistoryIndex(0);
      setDirty(false);
      setAnnotationRevision(revision);
      initialAnnotationsRef.current = clone(next);
      setActiveAnnotationId(next[0]?.clientId ?? null);
      setLabelDraft(next[0]?.label_name ?? "");
      setActivePersonId(next[0]?.personId ?? null);
    },
    [clone],
  );

  useEffect(() => {
    if (!selectedResult) {
      resetEditor([], 0);
      setEditorMode(false);
      setAnnotationLoading(false);
      return;
    }
    const fallback = modelAnnotations(selectedResult);
    let active = true;
    resetEditor(fallback, 0);
    setAnnotationLoading(true);
    setAnnotationError(null);
    setEditorMode(false);

    void invokeCore<ImageAnnotationsPayload>(
      "list_image_annotations",
      "annotation.list",
      { path: selectedResult.path },
    )
      .then((response) => {
        if (!active) return;
        if (
          response.error ||
          !response.payload ||
          !Array.isArray(response.payload.annotations) ||
          response.payload.image_size[0] <= 0 ||
          response.payload.image_size[1] <= 0
        ) {
          if (
            response.error &&
            response.error.code !== "NOT_FOUND" &&
            response.error.code !== "ANNOTATION_NOT_FOUND"
          )
            setAnnotationError(response.error.message);
          return;
        }
        const serverItems: EditableAnnotation[] =
          response.payload.annotations.map((item, index) => ({
            ...item,
            clientId: item.id ?? `saved-${selectedResult.path}-${index}`,
            personId: fallback[index]?.personId,
          }));
        resetEditor(
          serverItems,
          Number.isFinite(response.payload.revision)
            ? response.payload.revision
            : 0,
        );
      })
      .finally(() => {
        if (active) setAnnotationLoading(false);
      });
    return () => {
      active = false;
    };
  }, [modelAnnotations, resetEditor, selectedResult]);

  const activeAnnotation = annotations.find(
    (item) => item.clientId === activeAnnotationId,
  );
  const annotationForPerson = useCallback(
    (person: { id: string }) =>
      annotations.find((item) => item.personId === person.id),
    [annotations],
  );

  const peopleForCurrentImage: InspectorPerson[] = useMemo(() => {
    if (isDemo) {
      return demoPeople.map((p) => ({
        id: p.id,
        title: p.title,
        confidence: p.confidence,
        status: p.status,
        candidates: p.candidates,
        personIndex: p.source.person_index,
        margin: p.margin,
        consistency: p.consistency,
        reason: p.reason,
      }));
    }
    if (!selectedResult) return [];
    return realPeople.map((person) => {
      const annotation = annotationForPerson(person);
      return annotation
        ? { ...person, title: annotation.label_name || person.title }
        : person;
    });
  }, [annotationForPerson, isDemo, realPeople, selectedResult]);

  const displaySource = selectedResult
    ? convertFileSrc(selectedResult.path)
    : previewFile?.source;

  const commitAnnotations = useCallback(
    (nextItems: EditableAnnotation[]) => {
      const next = clone(nextItems);
      if (sameAnnotations(next, annotations)) return;
      const updatedHistory = appendAnnotationHistory(
        history,
        historyIndex,
        next,
        MAX_ANNOTATION_HISTORY,
      );
      setAnnotations(next);
      setHistory(updatedHistory.snapshots);
      setHistoryIndex(updatedHistory.historyIndex);
      setDirty(true);
    },
    [annotations, clone, history, historyIndex, sameAnnotations],
  );

  const updateAnnotationBox = (id: string, box: NormalizedBoundingBox) => {
    const next = annotations.map((item) =>
      item.clientId === id
        ? { ...item, bbox: box, source: "manual" as const, manually_adjusted: true }
        : item,
    );
    commitAnnotations(next);
  };

  const handleAddAnnotation = (box: NormalizedBoundingBox) => {
    const newId = `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const newAnnotation: EditableAnnotation = {
      clientId: newId,
      label_name: "新角色",
      identity_id: null,
      bbox: box,
      source: "manual",
      manually_adjusted: true,
    };
    const next = [...annotations, newAnnotation];
    commitAnnotations(next);
    setActiveAnnotationId(newId);
    setLabelDraft("新角色");
    setEditorMode(true);
  };

  const removeAnnotation = useCallback(
    (id: string) => {
      const next = annotations.filter((item) => item.clientId !== id);
      commitAnnotations(next);
      if (activeAnnotationId === id) {
        setActiveAnnotationId(next[0]?.clientId ?? null);
        setLabelDraft(next[0]?.label_name ?? "");
      }
    },
    [activeAnnotationId, annotations, commitAnnotations],
  );

  const commitLabel = (explicitLabel?: string) => {
    if (!activeAnnotation) return;
    const label = (explicitLabel !== undefined ? explicitLabel : labelDraft).trim();
    if (!label) return;
    const identity = identities.find(
      (item) => item.display_name === label || item.aliases?.includes(label),
    );
    const next = annotations.map((item) =>
      item.clientId === activeAnnotation.clientId
        ? {
            ...item,
            label_name: label,
            identity_id: identity?.id ?? null,
            source: "manual" as const,
            manually_adjusted: true,
          }
        : item,
    );
    commitAnnotations(next);
    setLabelDraft(label);
  };

  const assignCandidate = (personId: string, label: string) => {
    const target = annotations.find((item) => item.personId === personId);
    if (!target) {
      setAnnotationError("未找到候选角色对应的识别框，请先选择人物框后重试。");
      return;
    }
    const identity = identities.find(
      (item) => item.display_name === label || item.aliases?.includes(label),
    );
    const next = annotations.map((item) =>
      item.clientId === target.clientId
        ? {
            ...item,
            label_name: label,
            identity_id: identity?.id ?? null,
            source: "manual" as const,
            manually_adjusted: true,
          }
        : item,
    );
    setActivePersonId(personId);
    setActiveAnnotationId(target.clientId);
    setLabelDraft(label);
    setEditorMode(true);
    commitAnnotations(next);
    setAnnotationError(null);
    setNotice(`已将当前人物指派为“${label}”，保存后会加入训练样本。`);
  };

  const undo = useCallback(() => {
    if (historyIndex <= 0) return;
    const next = clone(history[historyIndex - 1]);
    setHistoryIndex(historyIndex - 1);
    setAnnotations(next);
    setDirty(!sameAnnotations(next, initialAnnotationsRef.current));
    setActiveAnnotationId(next[0]?.clientId ?? null);
    setLabelDraft(next[0]?.label_name ?? "");
  }, [clone, history, historyIndex, sameAnnotations]);

  const redo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;
    const next = clone(history[historyIndex + 1]);
    setHistoryIndex(historyIndex + 1);
    setAnnotations(next);
    setDirty(!sameAnnotations(next, initialAnnotationsRef.current));
    setActiveAnnotationId(next[0]?.clientId ?? null);
    setLabelDraft(next[0]?.label_name ?? "");
  }, [clone, history, historyIndex, sameAnnotations]);

  const saveAnnotations = async () => {
    if (!selectedResult || previewFile || !editorMode || !dirty) return;
    const pendingLabel = labelDraft.trim();
    const effectiveAnnotations =
      activeAnnotation && pendingLabel && pendingLabel !== activeAnnotation.label_name
        ? annotations.map((item) =>
            item.clientId === activeAnnotation.clientId
              ? {
                  ...item,
                  label_name: pendingLabel,
                  identity_id:
                    identities.find(
                      (id) =>
                        id.display_name === pendingLabel ||
                        id.aliases?.includes(pendingLabel),
                    )?.id ?? null,
                  source: "manual" as const,
                  manually_adjusted: true,
                }
              : item,
          )
        : annotations;

    const unnamed = effectiveAnnotations.find(
      (item) => !isValidAnnotationLabel(item.label_name),
    );
    if (unnamed) {
      setActiveAnnotationId(unnamed.clientId);
      setLabelDraft(unnamed.label_name);
      setAnnotationError(
        "保存前请为每个新增识别框填写有效分类名称，不能使用“未命名分类”作为训练标签。",
      );
      return;
    }

    const payloadAnnotations: ImageAnnotation[] = effectiveAnnotations.map(
      ({ clientId: _c, personId: _p, ...item }) => item,
    );
    const response = await invokeCore<ImageAnnotationsPayload>(
      "save_image_annotations",
      "annotation.save",
      {
        path: selectedResult.path,
        imageSize: selectedResult.image_size,
        image_size: selectedResult.image_size,
        baseRevision: annotationRevision,
        base_revision: annotationRevision,
        annotations: payloadAnnotations,
      },
    );

    if (response.error || !response.payload) {
      setAnnotationError(response.error?.message ?? "人工标注保存失败。");
      return;
    }

    const saved = response.payload.annotations?.map((item, index) => ({
      ...item,
      clientId: item.id ?? `saved-${selectedResult.path}-${index}`,
      personId: effectiveAnnotations[index]?.personId,
    }));
    const next = saved ?? effectiveAnnotations;
    const savedRevision = Number.isFinite(response.payload.revision)
      ? response.payload.revision
      : annotationRevision + 1;
    resetEditor(next, savedRevision);
    setNotice(`已保存当前图片的 ${next.length} 个标注调整结果。`);
    setAnnotationError(null);

    // 刷新个人模型状态与训练进度
    void invokeCore<PersonalTrainingStatus>(
      "get_personal_training_status",
      "personal.training.status",
    ).then((res) => {
      if (res.payload) {
        setPersonalStatus(readPersonalTrainingStatus(res.payload));
      }
    });

    // 推荐训练时触发
    if (response.payload.training_recommended === true) {
      const triggerKey = `${selectedResult.path}:${savedRevision}`;
      if (!trainingTriggerRef.current.has(triggerKey)) {
        trainingTriggerRef.current.add(triggerKey);
        setNotice(
          `已保存 ${next.length} 个标注，新样本已满足个人模型训练条件！可直接在右侧点击「训练新模型」。`,
        );
      }
    }
  };

  const confirmDiscard = useCallback(
    () =>
      !dirty ||
      globalThis.confirm("当前图片有未保存的人工标注，是否放弃修改并继续？"),
    [dirty],
  );

  const selectResult = (result: LibraryScanResult) => {
    if (!confirmDiscard() || result.path === selectedResult?.path) return;
    setSelectedResultPath(result.path);
    setPreviewFile(null);
    setActivePersonId(null);
    setEditorMode(false);
  };

  const choosePreview = async () => {
    if (!confirmDiscard()) return;
    if (!isTauriRuntime()) {
      inputRef.current?.click();
      return;
    }
    const selected = await open({
      directory: false,
      multiple: false,
      title: "选择一张图片预览",
      filters: [
        {
          name: "图片",
          extensions: ["png", "jpg", "jpeg", "webp", "gif"],
        },
      ],
    });
    if (typeof selected !== "string") return;
    const response = await invokeCore<LibrarySelection>(
      "register_preview_file",
      "library.preview.file",
      { path: selected },
    );
    if (response.error || !response.payload) {
      setScanError(response.error?.message ?? "无法授权预览所选图片。");
      return;
    }
    const preview = response.payload;
    setScanError(null);
    setScan(null);
    setSelectedResultPath(null);
    setPreviewFile({
      name: preview.display_name,
      path: preview.path,
      source: convertFileSrc(preview.path),
    });
    setActivePersonId(null);
    setEditorMode(false);
  };

  const handleBrowserPreview = (event: ChangeEvent<HTMLInputElement>) => {
    if (!confirmDiscard()) {
      event.target.value = "";
      return;
    }
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setScan(null);
      setSelectedResultPath(null);
      setPreviewFile({
        name: file.name,
        path: "浏览器临时预览",
        source: typeof reader.result === "string" ? reader.result : undefined,
      });
      setActivePersonId(null);
      setEditorMode(false);
    };
    reader.readAsDataURL(file);
    event.target.value = "";
  };

  const handleSelectCandidateIndex = (idx: number) => {
    const active =
      peopleForCurrentImage.find((p) => p.id === activePersonId) ||
      peopleForCurrentImage[0];
    if (active && active.candidates[idx]) {
      const label = active.candidates[idx].label;
      setLabelDraft(label);
      commitLabel(label);
    }
  };

  const startDirectoryScan = useCallback(
    async (directory: string) => {
      if (!directory || !confirmDiscard()) return;
      setDirectoryDialogMode(null);
      setDialogFolder(null);
      setSelectedFolderPath("");
      await controller.start(undefined, undefined, directory);
    },
    [confirmDiscard, controller],
  );

  const chooseDirectoryToScan = useCallback(async () => {
    if (!isTauriRuntime()) return;
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择要使用当前模型识别的目录",
    });
    if (typeof selected === "string" && selected.trim()) {
      await startDirectoryScan(selected.trim());
    }
  }, [startDirectoryScan]);

  const applyDirectoryDialog = () => {
    if (!directoryDialogMode) return;
    const relativePath = dialogFolder?.relativePath ?? "";
    if (directoryDialogMode === "filter") {
      const nextResults = resultsForFolder(scanResults, scanRootPath, relativePath);
      setSelectedFolderPath(relativePath);
      setSelectedResultPath(nextResults[0]?.path ?? null);
      setDirectoryDialogMode(null);
      setDialogFolder(null);
      return;
    }
    if (scanRootPath) {
      void startDirectoryScan(absoluteFolderPath(scanRootPath, relativePath));
    }
  };

  return (
    <div className="page-stack classification-page">
      {/* 顶部工具栏：图库选择 + 识别模型选择下拉框 + 扫描控制 */}
      <section className="library-toolbar">
        <div className="library-identity">
          <span className="toolbar-icon">
            <FolderOpen size={17} />
          </span>
          <div>
            <span className="eyebrow">CURRENT LIBRARY</span>
            <strong>{library?.display_name ?? "尚未选择图库"}</strong>
            <small>
              {library?.path ?? "选择目录后，真实扫描任务会显示在这里"}
            </small>
          </div>
        </div>

        <div className="toolbar-actions">
          <div
            className="model-selector-container"
            title="选择本次识别扫描所使用的模型"
          >
            <BrainCircuit size={14} />
            <select
              className="model-select-dropdown"
              value={
                controller.selectedModelVersion === null
                  ? "default"
                  : controller.selectedModelVersion
              }
              onChange={(e) => {
                const val = e.target.value;
                controller.setSelectedModelVersion(
                  val === "default" ? null : Number(val),
                );
              }}
              disabled={scanBusy}
            >
              <option value="default">自动融合模型 (默认推荐)</option>
              <option value="0">基础通用模型 (不含个人学习)</option>
              {personalStatus?.versions
                .filter((v) => isUsablePersonalModelVersion(v))
                .map((v) => (
                  <option key={v.version} value={v.version}>
                    个人模型 v{v.version} (
                    {v.status === "active"
                      ? "当前激活 · "
                      : v.status === "archived"
                        ? "已归档 · "
                        : ""}
                    {v.class_count} 个角色)
                  </option>
                ))}
            </select>
          </div>

          <label
            className="skip-annotated-label"
            title="开启后跳过本地已有人工标注的图片（直接加载已有标注，不重复调用 AI Worker 推理）"
          >
            <input
              type="checkbox"
              checked={controller.skipAnnotated}
              onChange={(e) => controller.setSkipAnnotated(e.target.checked)}
              disabled={scanBusy}
            />
            <span>跳过已标注</span>
          </label>

          <button
            className="ghost-button"
            onClick={() => {
              if (confirmDiscard()) onChooseLibrary();
            }}
            disabled={scanBusy}
          >
            <FolderOpen size={15} />
            {library ? "更换图库" : "选择图库"}
          </button>
          <button
            className="ghost-button"
            onClick={() => {
              setDialogFolder(null);
              setDirectoryDialogMode("scan");
            }}
            disabled={scanBusy || worker !== "healthy"}
            title="从扫描目录树或磁盘中选择一个目录识别"
          >
            <FolderOpen size={15} />
            选择目录识别
          </button>
          <button
            className="ghost-button"
            onClick={() => void choosePreview()}
            disabled={scanBusy}
          >
            <FileImage size={15} />
            预览单图
          </button>

          {scanState === "running" || scanState === "pausing" ? (
            <>
              <button
                className="ghost-button"
                onClick={() => void controller.pause()}
                disabled={scanState !== "running"}
              >
                <Clock3 size={15} />
                {scanState === "pausing" ? "等待暂停…" : "暂停"}
              </button>
              <button
                className="ghost-button danger-button"
                onClick={() => void controller.cancel()}
              >
                <X size={15} />
                终止
              </button>
            </>
          ) : scanState === "paused" ? (
            <>
              <button
                className="ghost-button"
                onClick={() => void controller.resume()}
              >
                <Play size={15} />
                继续
              </button>
              <button
                className="ghost-button danger-button"
                onClick={() => void controller.cancel()}
              >
                <X size={15} />
                终止
              </button>
            </>
          ) : scanState === "cancelling" ? (
            <button className="ghost-button danger-button" disabled>
              <X size={15} />
              正在终止…
            </button>
          ) : (
            <button
              className="primary-button"
              disabled={!library || worker !== "healthy"}
              title={
                !library
                  ? "请先选择图库"
                  : worker !== "healthy"
                    ? "请先连接内置推理引擎"
                    : "扫描所选文件夹中的全部图片"
              }
              onClick={() => void controller.start(confirmDiscard)}
            >
              <Play size={15} />
              {scanState === "completed" ||
              scanState === "cancelled" ||
              scanState === "failed"
                ? "再次扫描"
                : "开始扫描"}
            </button>
          )}
        </div>
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          onChange={handleBrowserPreview}
        />
      </section>

      {scanError && (
        <div className="workbench-notice error">
          <AlertCircle size={15} />
          <span>{scanError}</span>
          <button aria-label="关闭错误" onClick={() => setScanError(null)}>
            <X size={15} />
          </button>
        </div>
      )}
      {annotationError && (
        <div className="workbench-notice error">
          <AlertCircle size={15} />
          <span>{annotationError}</span>
          <button aria-label="关闭标注错误" onClick={() => setAnnotationError(null)}>
            <X size={15} />
          </button>
        </div>
      )}
      {notice && (
        <div className="workbench-notice">
          <Info size={15} />
          <span>{notice}</span>
          <button aria-label="关闭提示" onClick={() => setNotice(null)}>
            <X size={15} />
          </button>
        </div>
      )}

      {scanBusy && progress && (
        <section className="scan-progress panel-card" aria-live="polite">
          <div className="scan-progress-header">
            <div>
              <div className="eyebrow">LIVE SCAN</div>
              <h3>{progress.message || "正在处理图片"}</h3>
            </div>
            <strong>
              {progress.current} / {progress.total || "—"}
            </strong>
          </div>
          <div className="progress-track">
            <div
              className="progress-bar"
              style={{
                width: `${
                  progress.total
                    ? Math.min(100, (progress.current / progress.total) * 100)
                    : 0
                }%`,
              }}
            />
          </div>
          <p>{progress.path || "正在准备本地图库…"}</p>
        </section>
      )}

      {/* 主工作区：左边只显示大图画布，右边整合检查器与图片胶片 */}
      <div className="workbench-layout-unified">
        <div className="workbench-main-col">
          <AnnotationCanvas
            imageSrc={displaySource}
            imageSize={selectedResult?.image_size}
            annotations={annotations}
            activeAnnotationId={activeAnnotationId}
            onSelectAnnotation={(item) => {
              setActiveAnnotationId(item.clientId);
              setLabelDraft(item.label_name);
              setActivePersonId(item.personId ?? null);
            }}
            onUpdateAnnotation={updateAnnotationBox}
            onAddAnnotation={handleAddAnnotation}
            editorMode={editorMode}
            onToggleEditor={() => {
              if (editorMode) {
                if (!dirty || globalThis.confirm("放弃当前图片未保存的人工标注？")) {
                  resetEditor(initialAnnotationsRef.current, annotationRevision);
                  setEditorMode(false);
                }
              } else {
                setEditorMode(true);
              }
            }}
            onUndo={undo}
            onRedo={redo}
            onDeleteActive={() =>
              activeAnnotationId && removeAnnotation(activeAnnotationId)
            }
            onSelectCandidateIndex={handleSelectCandidateIndex}
            isDemo={isDemo}
            onPrevImage={handlePrevImage}
            onNextImage={handleNextImage}
            hasPrevImage={hasPrevImage}
            hasNextImage={hasNextImage}
            onContextMenu={(e) => {
              if (selectedResult?.path) {
                setContextMenu({
                  x: e.clientX,
                  y: e.clientY,
                  path: selectedResult.path,
                });
              }
            }}
          />
        </div>

        {/* 右侧统一属性检查器 (整合人脸列表、置信度、候选点击赋名、标注表单) */}
        <UnifiedInspector
          people={peopleForCurrentImage}
          activePersonId={activePersonId}
          onSelectPerson={(id) => {
            setActivePersonId(id);
            const annot = annotationForPerson({ id });
            if (annot) {
              setActiveAnnotationId(annot.clientId);
              setLabelDraft(annot.label_name);
            }
          }}
          filter={filter}
          onChangeFilter={setFilter}
          editorMode={editorMode}
          onToggleEditor={() => {
            if (editorMode) {
              if (!dirty || globalThis.confirm("放弃当前图片未保存的人工标注？")) {
                resetEditor(initialAnnotationsRef.current, annotationRevision);
                setEditorMode(false);
              }
            } else {
              setEditorMode(true);
            }
          }}
          onAddNewBox={() => {
            handleAddAnnotation({ x: 0.35, y: 0.25, width: 0.3, height: 0.35 });
          }}
          activeAnnotation={activeAnnotation ?? null}
          labelDraft={labelDraft}
          onChangeLabelDraft={setLabelDraft}
          onCommitLabel={commitLabel}
          onAssignCandidate={assignCandidate}
          onRemoveAnnotation={removeAnnotation}
          identities={identities}
          canUndo={historyIndex > 0}
          canRedo={historyIndex < history.length - 1}
          dirty={dirty}
          onUndo={undo}
          onRedo={redo}
          onCancelEdits={() => {
            resetEditor(initialAnnotationsRef.current, annotationRevision);
            setEditorMode(false);
          }}
          onSaveAnnotations={() => void saveAnnotations()}
          saveLoading={annotationLoading}
          personalModelReady={personalStatus?.ready}
          verifiedSampleCount={personalStatus?.verified_sample_count}
          eligibleClassCount={personalStatus?.eligible_class_count}
          totalClassCount={personalStatus?.class_count}
          onOpenPersonalModel={onOpenPersonalModel}
          onQuickTrain={handleQuickTrain}
          trainingLoading={quickTraining}
          footer={
            visibleScanResults.length > 0 ? (
              <Filmstrip
                results={visibleScanResults}
                selectedPath={selectedResult?.path ?? null}
                onSelectResult={selectResult}
                onDeleteScan={() => controller.deleteScan(scanRootPath)}
                modelVersionName={scan?.model_name}
                onOpenDirectoryTree={() => {
                  setDialogFolder(null);
                  setDirectoryDialogMode("filter");
                }}
                onContextMenu={(path, e) =>
                  setContextMenu({ x: e.clientX, y: e.clientY, path })
                }
              />
            ) : undefined
          }
        />
      </div>

      {directoryDialogMode && (
        <div
          className="modal-backdrop directory-tree-modal-backdrop"
          onClick={() => setDirectoryDialogMode(null)}
        >
          <div
            className="directory-tree-modal"
            role="dialog"
            aria-modal="true"
            aria-label={
              directoryDialogMode === "filter"
                ? "选择胶片显示目录"
                : "选择识别目录"
            }
            onClick={(event) => event.stopPropagation()}
          >
            <ScanFolderTree
              results={scanResults}
              libraryPath={scanRootPath}
              selectedPath={selectedResult?.path ?? null}
              selectedFolderPath={dialogFolder?.relativePath ?? selectedFolderPath}
              foldersOnly
              onSelectFolder={setDialogFolder}
              onSelectResult={selectResult}
              onClose={() => setDirectoryDialogMode(null)}
            />
            <div className="directory-tree-modal-actions">
              <span>
                {dialogFolder?.name ??
                  (directoryDialogMode === "scan"
                    ? "当前扫描根目录"
                    : "全部目录")}
              </span>
              <div>
                {directoryDialogMode === "scan" && (
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => void chooseDirectoryToScan()}
                  >
                    <FolderOpen size={14} /> 从磁盘选择
                  </button>
                )}
                <button
                  type="button"
                  className="primary-button"
                  disabled={directoryDialogMode === "scan" && !scanRootPath}
                  onClick={applyDirectoryDialog}
                >
                  {directoryDialogMode === "filter" ? "显示此目录" : "识别此目录"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {contextMenu && (
        <ImageContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          imagePath={contextMenu.path}
          onClose={() => setContextMenu(null)}
          onShowInFolder={handleShowInFolder}
          onMoveFile={handleMoveSingleFile}
          onDeleteFile={handleDeleteSingleFile}
        />
      )}
    </div>
  );
}
