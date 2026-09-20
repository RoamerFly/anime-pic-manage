import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  FileSearch,
  Filter,
  FolderArchive,
  FolderOpen,
  HardDrive,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
  StopCircle,
} from "lucide-react";
import type {
  AppSettings,
  BatchMoveResult,
  LibrarySelection,
  SimilarityDecisionAction,
  SimilarityDecisionResult,
  SimilarityGroup,
  SimilarityGroupItem,
  SimilarityGroupType,
  SimilarityItemDecision,
  SimilarityScanPayload,
  SimilarityScanProgress,
  SimilarityScanState,
} from "@anime-pic-manage/shared-types";
import { ImageContextMenu } from "./ContextMenu";
import { SimilarityGroupCard, formatBytes } from "./SimilarityGroupCard";
import {
  SimilarityDeleteConfirmModal,
  SimilarityImagePreviewModal,
} from "./SimilarityCompareModal";
import { invokeCore } from "../lib/tauri";

/** Groups rendered per batch: a huge result set must not freeze the WebView. */
const GROUP_RENDER_STEP = 30;
/** Progress arrives per extraction batch; throttle repaints to ~8 fps. */
const PROGRESS_FLUSH_MS = 120;

function normalizePath(value?: string | null): string {
  return (value ?? "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

function directoryMatches(
  scanned?: string | null,
  current?: string | null,
): boolean {
  // Older progress payloads carry no directory; treat them as matching.
  if (!scanned) return true;
  if (!current) return false;
  return normalizePath(scanned) === normalizePath(current);
}

export function SimilarityWorkbench({
  library,
  onChooseLibrary,
  libraryLocked = false,
}: {
  library: LibrarySelection | null;
  onChooseLibrary: () => void;
  libraryLocked?: boolean;
}) {
  const [threshold, setThreshold] = useState<number>(0.85);
  const [includeSubfolders, setIncludeSubfolders] = useState<boolean>(true);
  const [archiveDirName, setArchiveDirName] = useState<string>("_duplicates");

  const [scanning, setScanning] = useState<boolean>(false);
  const [progress, setProgress] = useState<SimilarityScanProgress | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [groups, setGroups] = useState<SimilarityGroup[]>([]);
  const [totalScanned, setTotalScanned] = useState<number>(0);
  const [filterType, setFilterType] = useState<"all" | SimilarityGroupType>("all");
  const [resultRevision, setResultRevision] = useState<number>(0);
  const cacheLoadRevision = useRef<number>(0);
  const [visibleGroupCount, setVisibleGroupCount] = useState<number>(GROUP_RENDER_STEP);
  const progressFlushRef = useRef<number>(0);
  const pendingProgressRef = useRef<SimilarityScanProgress | null>(null);

  const [executing, setExecuting] = useState<boolean>(false);
  const [decisionResult, setDecisionResult] = useState<SimilarityDecisionResult | null>(null);
  const [showConfirmModal, setShowConfirmModal] = useState<boolean>(false);
  const [previewImage, setPreviewImage] = useState<SimilarityGroupItem | null>(null);

  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; path: string } | null>(null);

  const handleShowInFolder = useCallback(async (path: string) => {
    const res = await invokeCore<void>("show_item_in_folder", "system.explorer.show", { path });
    if (res.error) {
      setScanError("无法打开所在文件夹: " + res.error.message);
    }
  }, []);

  const handleMoveSingleFile = useCallback(async (path: string) => {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择图片移动目标文件夹",
    });
    if (typeof selected === "string" && selected.trim()) {
      const dest = selected.trim();
      const res = await invokeCore<BatchMoveResult>("move_library_files", "library.files.move", {
        sources: [path],
        destination_directory: dest,
      });
      if (res.error) {
        setScanError("移动文件失败: " + res.error.message);
      } else if (res.payload) {
        setNotice(`已成功将图片移动到: ${dest}`);
        setGroups((prev) =>
          prev
            .map((g) => ({
              ...g,
              items: g.items.filter((it) => it.path !== path),
            }))
            .filter((g) => g.items.length >= 2),
        );
      }
    }
  }, []);

  const handleDeleteSingleFile = useCallback(async (path: string) => {
    const res = await invokeCore<void>("delete_library_file", "library.file.delete", { path });
    if (res.error) {
      setScanError("删除失败: " + res.error.message);
    } else {
      setNotice("已删除文件");
      setGroups((prev) =>
        prev
          .map((g) => ({
            ...g,
            items: g.items.filter((it) => it.path !== path),
          }))
          .filter((g) => g.items.length >= 2),
      );
    }
  }, []);

  const loadLatestSimilarityResult = useCallback(async (clearCurrent = false) => {
    const loadRevision = ++cacheLoadRevision.current;
    const libraryPath = library?.path;
    if (clearCurrent) {
      setGroups([]);
      setTotalScanned(0);
      setVisibleGroupCount(GROUP_RENDER_STEP);
    }
    if (!libraryPath) {
      return;
    }

    const res = await invokeCore<SimilarityScanPayload | null>(
      "get_latest_similarity_result",
      "similarity.group.list",
      { path: libraryPath },
    );
    if (loadRevision !== cacheLoadRevision.current) return;
    if (res.error) {
      setScanError(res.error.message);
      return;
    }
    if (res.payload) {
      setGroups(res.payload.groups);
      setTotalScanned(res.payload.total_scanned);
      setResultRevision((revision) => revision + 1);
      return;
    }

    // Read-only compatibility fallback for results created before file storage.
    const legacy = await invokeCore<SimilarityGroup[]>(
      "get_cached_similarity_groups",
      "similarity.group.list",
    );
    if (loadRevision !== cacheLoadRevision.current || !legacy.payload) return;
    const libraryGroups = legacy.payload.filter((group) =>
      group.items.some((item) => item.path.startsWith(libraryPath)),
    );
    setGroups(libraryGroups);
    setResultRevision((revision) => revision + 1);
  }, [library?.path]);

  // Apply the saved similarity defaults once, before any scan can start.
  useEffect(() => {
    let active = true;
    const loadDefaults = async () => {
      try {
        const response = await invokeCore<AppSettings>(
          "get_app_settings",
          "settings.get",
        );
        if (!active) return;
        if (response.payload) {
          setThreshold(response.payload.similarity_threshold);
          setIncludeSubfolders(response.payload.similarity_include_subfolders);
          if (response.payload.similarity_archive_dir.trim()) {
            setArchiveDirName(response.payload.similarity_archive_dir);
          }
        }
      } catch {
        // Browser preview or an unavailable core keeps the built-in defaults.
      }
    };
    void loadDefaults();
    return () => {
      active = false;
    };
  }, []);

  // Re-attach to a scan that is still running in the backend, otherwise load
  // the latest stored result. Loading a previous result file while a scan was
  // running made switching away and back feel frozen.
  useEffect(() => {
    let active = true;
    const attach = async () => {
      try {
        const response = await invokeCore<SimilarityScanState>(
          "get_similarity_scan_state",
          "similarity.scan.state",
        );
        if (!active) return;
        const snapshot = response.payload;
        if (snapshot?.running && directoryMatches(snapshot.directory, library?.path)) {
          setScanning(true);
          setProgress({
            phase: snapshot.phase || "extracting",
            current: snapshot.current,
            total: snapshot.total,
            path: snapshot.path ?? undefined,
            message: snapshot.message || "正在继续相似度扫描...",
            directory: snapshot.directory ?? undefined,
          });
          return;
        }
      } catch {
        if (!active) return;
      }
      setScanning(false);
      await loadLatestSimilarityResult(true);
    };
    void attach();
    return () => {
      active = false;
    };
  }, [library?.path, loadLatestSimilarityResult]);

  // Listen for real-time progress events from the backend. The `active` guard
  // matters because `listen()` resolves asynchronously: without it, leaving the
  // page mid-scan leaks a listener that keeps updating an unmounted component.
  useEffect(() => {
    let active = true;
    let unlisten: (() => void) | undefined;
    void listen<SimilarityScanProgress>("similarity://progress", (event) => {
      if (!active) return;
      const payload = event.payload;
      const terminal =
        payload.phase === "completed" ||
        payload.phase === "cancelled" ||
        payload.phase === "error";
      if (terminal) {
        pendingProgressRef.current = null;
        setProgress(payload);
        setScanning(false);
        if (payload.phase !== "error" && directoryMatches(payload.directory, library?.path)) {
          void loadLatestSimilarityResult(true);
        }
        return;
      }
      pendingProgressRef.current = payload;
      const now = Date.now();
      if (now - progressFlushRef.current < PROGRESS_FLUSH_MS) return;
      progressFlushRef.current = now;
      setProgress(payload);
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
  }, [library?.path, loadLatestSimilarityResult]);

  // Keep manual keep/archive/delete choices in the current result file.
  useEffect(() => {
    if (!library || scanning || executing || groups.length === 0) return;
    const decisions = groups.flatMap((group) =>
      group.items
        .filter((item) => item.decision)
        .map((item) => ({
          path: item.path,
          action: item.decision!,
        })),
    );
    if (decisions.length === 0) return;
    const timer = window.setTimeout(() => {
      void invokeCore<boolean>(
        "save_similarity_result_decisions",
        "similarity.decision.apply",
        { path: library.path, decisions },
      ).then((response) => {
        if (response.error) setScanError(response.error.message);
      });
    }, 350);
    return () => window.clearTimeout(timer);
  }, [groups, library, scanning, executing]);

  const handleStartScan = useCallback(async () => {
    if (!library) return;
    // Invalidate cached cards before the backend extends the asset protocol scope.
    // Otherwise a failed image element can retain its old src until this page remounts.
    cacheLoadRevision.current += 1;
    setGroups([]);
    setTotalScanned(0);
    setVisibleGroupCount(GROUP_RENDER_STEP);
    setScanning(true);
    setScanError(null);
    setNotice(null);
    setProgress({
      phase: "starting",
      current: 0,
      total: 0,
      path: undefined,
      message: "准备扫描图库相似度...",
    });
    setDecisionResult(null);

    const res = await invokeCore<SimilarityScanPayload>(
      "scan_similarity_preview",
      "similarity.scan.start",
      {
        args: {
          path: library.path,
          threshold,
          include_subfolders: includeSubfolders,
        },
      },
    );

    setScanning(false);
    if (res.error) {
      setScanError(res.error.message || "相似度扫描失败");
    } else if (res.payload) {
      setGroups(res.payload.groups);
      setTotalScanned(res.payload.total_scanned);
      setResultRevision((revision) => revision + 1);
      if (res.payload.cancelled) {
        setNotice("扫描已取消，已展示已扫描的部分相似结果。");
      } else {
        setNotice(
          `扫描完成！在 ${res.payload.total_scanned} 张图片中发现了 ${res.payload.groups.length} 组相似图片，可释放约 ${formatBytes(res.payload.potential_space_saved)} 空间。`,
        );
      }
    }
  }, [library, threshold, includeSubfolders]);

  const handleCancelScan = useCallback(async () => {
    await invokeCore("cancel_similarity_scan", "similarity.scan.cancel");
    setNotice("已发送取消指令，等待停止...");
  }, []);

  // Modify individual item decision
  const handleItemDecisionChange = useCallback(
    (groupId: string, itemPath: string, decision: SimilarityDecisionAction) => {
      setGroups((prevGroups) =>
        prevGroups.map((g) => {
          if (g.group_id !== groupId) return g;
          return {
            ...g,
            items: g.items.map((it) => (it.path === itemPath ? { ...it, decision } : it)),
          };
        }),
      );
    },
    [],
  );

  // Group quick action: Apply recommended
  const handleApplyGroupRecommended = useCallback((groupId: string) => {
    setGroups((prev) =>
      prev.map((g) => {
        if (g.group_id !== groupId) return g;
        return {
          ...g,
          items: g.items.map((it) => ({
            ...it,
            decision: it.is_recommended ? "keep" : "archive",
          })),
        };
      }),
    );
  }, []);

  // Group quick action: Keep all
  const handleGroupKeepAll = useCallback((groupId: string) => {
    setGroups((prev) =>
      prev.map((g) => {
        if (g.group_id !== groupId) return g;
        return {
          ...g,
          items: g.items.map((it) => ({ ...it, decision: "keep" })),
        };
      }),
    );
  }, []);

  // Group quick action: Archive all non-recommended
  const handleGroupArchiveAll = useCallback((groupId: string) => {
    setGroups((prev) =>
      prev.map((g) => {
        if (g.group_id !== groupId) return g;
        return {
          ...g,
          items: g.items.map((it) => ({
            ...it,
            decision: it.is_recommended ? "keep" : "archive",
          })),
        };
      }),
    );
  }, []);

  // Group quick action: Delete all non-recommended
  const handleGroupDeleteAll = useCallback((groupId: string) => {
    setGroups((prev) =>
      prev.map((g) => {
        if (g.group_id !== groupId) return g;
        return {
          ...g,
          items: g.items.map((it) => ({
            ...it,
            decision: it.is_recommended ? "keep" : "delete",
          })),
        };
      }),
    );
  }, []);

  // Batch action across all groups: Apply recommended for all
  const handleApplyAllRecommended = useCallback(() => {
    setGroups((prev) =>
      prev.map((g) => ({
        ...g,
        items: g.items.map((it) => ({
          ...it,
          decision: it.is_recommended ? "keep" : "archive",
        })),
      })),
    );
  }, []);

  // Batch action: Reset all to keep
  const handleResetAllKeep = useCallback(() => {
    setGroups((prev) =>
      prev.map((g) => ({
        ...g,
        items: g.items.map((it) => ({ ...it, decision: "keep" })),
      })),
    );
  }, []);

  // Summary of decisions
  const decisionSummary = useMemo(() => {
    let keepCount = 0;
    let archiveCount = 0;
    let deleteCount = 0;
    let freedBytes = 0;

    for (const g of groups) {
      for (const it of g.items) {
        if (it.decision === "keep") keepCount++;
        else if (it.decision === "archive") {
          archiveCount++;
        } else if (it.decision === "delete") {
          deleteCount++;
          freedBytes += it.file_size;
        }
      }
    }
    return { keepCount, archiveCount, deleteCount, freedBytes };
  }, [groups]);

  // Execute decisions
  const handleExecuteDecisions = useCallback(async () => {
    setShowConfirmModal(false);
    setExecuting(true);
    setScanError(null);
    setNotice(null);

    const decisions: SimilarityItemDecision[] = [];
    for (const g of groups) {
      for (const it of g.items) {
        if (it.decision) {
          decisions.push({
            path: it.path,
            action: it.decision,
          });
        }
      }
    }

    if (decisions.length === 0) {
      setExecuting(false);
      setNotice("还没有选择需要保存的处理结果。");
      return;
    }

    const res = await invokeCore<SimilarityDecisionResult>(
      "apply_similarity_decisions",
      "similarity.decision.apply",
      {
        request: {
          archive_directory_name: archiveDirName.trim() || "_duplicates",
          result_directory: library?.path,
          decisions,
        },
      },
    );

    setExecuting(false);
    if (res.error) {
      setScanError(res.error.message || "执行处理失败");
    } else if (res.payload) {
      setDecisionResult(res.payload);
      const completionMessage =
        decisionSummary.archiveCount === 0 && decisionSummary.deleteCount === 0
          ? `已保存 ${res.payload.success_count} 项保留结果。`
          : `治理执行完成：成功处理 ${res.payload.success_count} 个文件，删除释放 ${formatBytes(res.payload.freed_bytes)}。`;
      setNotice(completionMessage);
      if (res.payload.failed_count > 0) {
        const firstError = res.payload.errors[0];
        setScanError(
          `${res.payload.failed_count} 个文件处理失败${firstError ? `：${firstError.path}（${firstError.error}）` : ""}`,
        );
      }

      await loadLatestSimilarityResult(false);
    }
  }, [
    groups,
    library?.path,
    archiveDirName,
    decisionSummary.archiveCount,
    decisionSummary.deleteCount,
    loadLatestSimilarityResult,
  ]);

  // Filter groups
  const filteredGroups = useMemo(() => {
    if (filterType === "all") return groups;
    return groups.filter((g) => g.group_type === filterType);
  }, [groups, filterType]);

  // Render incrementally: a library with thousands of duplicate groups must not
  // block the WebView while a scan is still reporting progress.
  const visibleGroups = useMemo(
    () => filteredGroups.slice(0, visibleGroupCount),
    [filteredGroups, visibleGroupCount],
  );
  const remainingGroups = Math.max(0, filteredGroups.length - visibleGroups.length);

  useEffect(() => {
    setVisibleGroupCount(GROUP_RENDER_STEP);
  }, [filterType, resultRevision]);

  const handleCardContextMenu = useCallback(
    (event: React.MouseEvent, path: string) => {
      event.preventDefault();
      event.stopPropagation();
      setContextMenu({ x: event.clientX, y: event.clientY, path });
    },
    [],
  );

  const totalDuplicates = useMemo(() => {
    return groups.reduce((acc, g) => acc + Math.max(0, g.items.length - 1), 0);
  }, [groups]);

  return (
    <div className="page-stack similarity-workbench">
      {/* 图库与扫描配置工具条 */}
      <section className="library-toolbar similarity-toolbar panel-card">
        <div className="similarity-toolbar-row">
          <div className="library-identity">
            <span className="toolbar-icon">
              <FolderOpen size={18} />
            </span>
            <div>
              <span className="eyebrow">TARGET FOLDER</span>
              <strong>{library?.display_name ?? "尚未选择图库"}</strong>
              <small>{library?.path ?? "请先选择图库以开始相似度分析"}</small>
            </div>
          </div>

          <div className="similarity-controls">
            <button
              className="ghost-button"
              onClick={onChooseLibrary}
              disabled={libraryLocked || scanning}
              title={libraryLocked ? "任务进行中" : "更换本地图库文件夹"}
            >
              <FolderOpen size={15} />
              {library ? "更换目录" : "选择目录"}
            </button>

            <div
              className="threshold-selector"
              title="相似度阈值：推荐 85% 识别差分与微小压缩，95% 以上仅匹配几乎完全重复"
            >
              <span className="control-label">匹配灵敏度:</span>
              <input
                type="range"
                min="0.70"
                max="0.98"
                step="0.01"
                value={threshold}
                disabled={scanning}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="threshold-slider"
              />
              <span className="threshold-value">{Math.round(threshold * 100)}%</span>
            </div>

            <label className="checkbox-label" title="是否递归遍历所有子文件夹">
              <input
                type="checkbox"
                checked={includeSubfolders}
                disabled={scanning}
                onChange={(e) => setIncludeSubfolders(e.target.checked)}
              />
              <span>包含子目录</span>
            </label>

            {scanning ? (
              <button className="danger-button" onClick={handleCancelScan}>
                <StopCircle size={15} /> 取消扫描
              </button>
            ) : (
              <button
                className="primary-button"
                onClick={handleStartScan}
                disabled={!library || libraryLocked}
              >
                <Play size={15} /> 开始相似分析
              </button>
            )}
          </div>
        </div>

        {/* 实时进度条 */}
        {scanning && progress && (
          <div className="similarity-progress-panel">
            <div className="progress-header">
              <span className="progress-phase">
                <Loader2 size={14} className="spin" /> {progress.message}
              </span>
              <span className="progress-stats">
                {progress.total > 0 ? `${progress.current} / ${progress.total}` : ""}
              </span>
            </div>
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* 提示与报错横幅 */}
        {scanError && (
          <div className="banner-callout banner-error">
            <AlertTriangle size={16} />
            <span>{scanError}</span>
          </div>
        )}
        {notice && (
          <div className="banner-callout banner-info">
            <CheckCircle2 size={16} />
            <span>{notice}</span>
          </div>
        )}
      </section>

      {/* 结果概览与过滤器 */}
      {groups.length > 0 ? (
        <section className="similarity-results-container">
          {/* 数据总览与筛选栏 */}
          <div className="similarity-overview-bar panel-card">
            <div className="overview-stats">
              <div className="stat-badge">
                <Layers size={14} />
                <span>相似分组:</span>
                <strong>{groups.length} 组</strong>
              </div>
              <div className="stat-badge">
                <Sparkles size={14} />
                <span>冗余/差分图片:</span>
                <strong>{totalDuplicates} 张</strong>
              </div>
              <div className="stat-badge highlight">
                <HardDrive size={14} />
                <span>可释放空间:</span>
                <strong>{formatBytes(decisionSummary.freedBytes)}</strong>
              </div>
            </div>

            {/* 批量操作与分类过滤 */}
            <div className="overview-actions">
              <button
                className="ghost-button compact"
                onClick={handleApplyAllRecommended}
                title="所有分组自动将最佳画质设为保留，其余设为隔离"
              >
                <Sparkles size={13} /> 全部应用推荐
              </button>
              <button
                className="ghost-button compact"
                onClick={handleResetAllKeep}
                title="重置所有图片决策为保留"
              >
                <RotateCcw size={13} /> 全部重置保留
              </button>

              <div className="filter-divider" />

              <button
                className={`filter-tab ${filterType === "all" ? "active" : ""}`}
                onClick={() => setFilterType("all")}
              >
                <Filter size={13} /> 全部 ({groups.length})
              </button>
              <button
                className={`filter-tab ${filterType === "exact_duplicate" ? "active" : ""}`}
                onClick={() => setFilterType("exact_duplicate")}
              >
                完全重复 ({groups.filter((g) => g.group_type === "exact_duplicate").length})
              </button>
              <button
                className={`filter-tab ${filterType === "variation" ? "active" : ""}`}
                onClick={() => setFilterType("variation")}
              >
                差分变体 ({groups.filter((g) => g.group_type === "variation").length})
              </button>
              <button
                className={`filter-tab ${filterType === "similar" ? "active" : ""}`}
                onClick={() => setFilterType("similar")}
              >
                近似图 ({groups.filter((g) => g.group_type === "similar").length})
              </button>
            </div>
          </div>

          {/* 相似分组卡片列表 */}
          <div className="similarity-groups-list">
            {visibleGroups.map((group, groupIdx) => (
              <SimilarityGroupCard
                key={`${resultRevision}:${group.group_id}`}
                group={group}
                groupIdx={groupIdx}
                onApplyRecommended={handleApplyGroupRecommended}
                onKeepAll={handleGroupKeepAll}
                onArchiveAll={handleGroupArchiveAll}
                onDeleteAll={handleGroupDeleteAll}
                onItemDecisionChange={handleItemDecisionChange}
                onPreviewImage={setPreviewImage}
                onContextMenu={handleCardContextMenu}
              />
            ))}
            {remainingGroups > 0 && (
              <button
                className="ghost-button similarity-load-more"
                onClick={() =>
                  setVisibleGroupCount((count) => count + GROUP_RENDER_STEP)
                }
              >
                显示更多分组（剩余 {remainingGroups} 组）
              </button>
            )}
          </div>
        </section>
      ) : (
        <section className="similarity-empty-state panel-card">
          <FileSearch size={36} className="empty-icon" />
          <h3>尚未发现相似图片</h3>
          <p>
            {library
              ? "点击上方「开始相似分析」，引擎将自动为您建立图像感知特征并进行组群聚类。"
              : "请先在上方选择一个本地图片库目录。"}
          </p>
        </section>
      )}

      {/* 底部浮动治理操作条 */}
      {groups.length > 0 && (
        <aside className="similarity-governance-bar panel-card">
          <div className="governance-summary">
            <span className="gov-metric">
              保留: <strong>{decisionSummary.keepCount}</strong>
            </span>
            <span className="gov-metric">
              隔离归档: <strong>{decisionSummary.archiveCount}</strong>
            </span>
            <span className="gov-metric danger">
              永久删除: <strong>{decisionSummary.deleteCount}</strong>
            </span>
            <span className="gov-metric highlight">
              删除可释放: <strong>{formatBytes(decisionSummary.freedBytes)}</strong>
            </span>
          </div>

          <div className="governance-actions">
            <div className="archive-folder-input" title="归档隔离目标子文件夹名称">
              <FolderArchive size={15} />
              <input
                type="text"
                value={archiveDirName}
                onChange={(e) => setArchiveDirName(e.target.value)}
                placeholder="_duplicates"
              />
            </div>

            <button
              className="primary-button"
              disabled={
                executing ||
                (decisionSummary.keepCount === 0 &&
                  decisionSummary.archiveCount === 0 &&
                  decisionSummary.deleteCount === 0)
              }
              onClick={() => {
                if (decisionSummary.deleteCount > 0) {
                  setShowConfirmModal(true);
                } else {
                  void handleExecuteDecisions();
                }
              }}
            >
              {executing ? (
                <>
                  <Loader2 size={15} className="spin" /> 执行中...
                </>
              ) : (
                <>
                  <ArrowRight size={15} />
                  {decisionSummary.archiveCount === 0 && decisionSummary.deleteCount === 0
                    ? "保存保留结果"
                    : "执行整理治理"}
                </>
              )}
            </button>
          </div>
        </aside>
      )}

      {/* 永久删除二次确认模态框 */}
      <SimilarityDeleteConfirmModal
        show={showConfirmModal}
        deleteCount={decisionSummary.deleteCount}
        freedBytes={decisionSummary.freedBytes}
        onClose={() => setShowConfirmModal(false)}
        onConfirm={() => void handleExecuteDecisions()}
      />

      {/* 大图快速对比弹窗 */}
      <SimilarityImagePreviewModal
        item={previewImage}
        onClose={() => setPreviewImage(null)}
      />

      {/* 图片独立右键菜单 */}
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
