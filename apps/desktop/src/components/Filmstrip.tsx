import { useEffect, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import {
  ChevronLeft,
  ChevronRight,
  FolderTree,
  Maximize2,
  Trash2,
  Users,
  X,
} from "lucide-react";
import type { LibraryScanResult } from "@anime-pic-manage/shared-types";
import { isTauriRuntime, normalizePreviewPath } from "../lib/tauri";

interface FilmstripProps {
  results: LibraryScanResult[];
  selectedPath: string | null;
  onSelectResult: (result: LibraryScanResult) => void;
  onDeleteScan?: () => void;
  modelVersionName?: string;
  onOpenDirectoryTree?: () => void;
  onContextMenu?: (path: string, e: React.MouseEvent) => void;
}

export function Filmstrip({
  results,
  selectedPath,
  onSelectResult,
  onDeleteScan,
  modelVersionName,
  onOpenDirectoryTree,
  onContextMenu,
}: FilmstripProps) {
  const [previewResult, setPreviewResult] = useState<LibraryScanResult | null>(null);

  useEffect(() => {
    if (!previewResult) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPreviewResult(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [previewResult]);

  if (results.length === 0) return null;

  const selectedIndex = results.findIndex((r) => r.path === selectedPath);
  const handlePrev = () => {
    if (selectedIndex > 0) {
      onSelectResult(results[selectedIndex - 1]);
    }
  };
  const handleNext = () => {
    if (selectedIndex >= 0 && selectedIndex < results.length - 1) {
      onSelectResult(results[selectedIndex + 1]);
    }
  };

  const totalPeople = results.reduce((acc, r) => acc + r.people.length, 0);
  const previewFileName = previewResult
    ? previewResult.path.split(/[\\/]/).pop() || "图片"
    : "";
  const previewSrc = previewResult
    ? isTauriRuntime()
      ? convertFileSrc(normalizePreviewPath(previewResult.path))
      : previewResult.path
    : "";

  return (
    <section className="filmstrip-container panel-card" aria-label="扫描图片底片栏">
      <div className="filmstrip-header">
        <div className="filmstrip-header-meta">
          <span className="eyebrow">IMAGE FILMSTRIP</span>
          <strong>已扫描 {results.length} 张图片</strong>
          <span className="filmstrip-badge">
            <Users size={12} /> {totalPeople} 个人物
          </span>
          {modelVersionName && (
            <span className="filmstrip-model-pill">模型: {modelVersionName}</span>
          )}
          {onOpenDirectoryTree && (
            <button
              type="button"
              className="ghost-button filmstrip-tree-button"
              onClick={onOpenDirectoryTree}
              title="选择要在胶片中显示的目录"
            >
              <FolderTree size={12} /> 目录树
            </button>
          )}
        </div>
        <div className="filmstrip-controls">
          <button
            type="button"
            className="icon-button"
            onClick={handlePrev}
            disabled={selectedIndex <= 0}
            title="上一张 (键盘 ←)"
            aria-label="上一张图片"
          >
            <ChevronLeft size={16} />
          </button>
          <span className="filmstrip-pagination">
            {selectedIndex >= 0 ? selectedIndex + 1 : 1} / {results.length}
          </span>
          <button
            type="button"
            className="icon-button"
            onClick={handleNext}
            disabled={selectedIndex < 0 || selectedIndex >= results.length - 1}
            title="下一张 (键盘 →)"
            aria-label="下一张图片"
          >
            <ChevronRight size={16} />
          </button>
          {onDeleteScan && (
            <button
              type="button"
              className="ghost-button danger-button filmstrip-delete-btn"
              onClick={() => {
                if (globalThis.confirm("确定要删除并清空当前图库的本次扫描结果吗？")) {
                  onDeleteScan();
                }
              }}
              title="删除本次扫描结果并恢复待扫描状态"
            >
              <Trash2 size={13} />
              删除结果
            </button>
          )}
        </div>
      </div>

      <div className="filmstrip-scroll-area">
        {results.map((result, index) => {
          const isSelected = result.path === selectedPath;
          const fileName = result.path.split(/[\\/]/).pop() || `图片 ${index + 1}`;
          const imgSrc = isTauriRuntime()
            ? convertFileSrc(normalizePreviewPath(result.path))
            : result.path;
          const hasReview = result.people.some((p) => p.status === "needs_review");
          const allHigh = result.people.length > 0 && result.people.every((p) => p.status === "high_confidence");
          const statusDotClass = hasReview ? "review" : allHigh ? "high" : result.people.length === 0 ? "none" : "unknown";

          return (
            <div
              key={result.path}
              role="button"
              tabIndex={0}
              className={`filmstrip-thumb-card ${isSelected ? "selected" : ""}`}
              onClick={() => onSelectResult(result)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectResult(result);
                }
              }}
              onContextMenu={(e) => {
                if (onContextMenu) {
                  e.preventDefault();
                  e.stopPropagation();
                  onContextMenu(result.path, e);
                }
              }}
              title={`${fileName} (${result.people.length} 个人物) - 右键可打开位置/移动/删除`}
            >
              <div className="filmstrip-thumb-media">
                <img
                  src={imgSrc}
                  alt={fileName}
                  loading="lazy"
                  onError={(e) => {
                    (e.currentTarget as HTMLElement).style.display = "none";
                  }}
                />
                <span className={`filmstrip-status-dot ${statusDotClass}`} />
                <button
                  type="button"
                  className="filmstrip-expand-btn"
                  aria-label={`大图查看 ${fileName}`}
                  title="大图查看"
                  onClick={(event) => {
                    event.stopPropagation();
                    setPreviewResult(result);
                  }}
                >
                  <Maximize2 size={11} />
                </button>
              </div>
              <div className="filmstrip-thumb-info">
                <span className="filmstrip-thumb-name">{fileName}</span>
                <small className="filmstrip-thumb-people">
                  {result.people.length > 0 ? `${result.people.length} 人物` : "无人物"}
                </small>
              </div>
            </div>
          );
        })}
      </div>

      {previewResult && (
        <div className="modal-backdrop" onClick={() => setPreviewResult(null)}>
          <div
            className="filmstrip-lightbox"
            role="dialog"
            aria-modal="true"
            aria-label={`大图查看：${previewFileName}`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="filmstrip-lightbox-header">
              <strong>{previewFileName}</strong>
              <small>{previewResult.path}</small>
              <button
                type="button"
                className="icon-close"
                onClick={() => setPreviewResult(null)}
                aria-label="关闭大图"
              >
                <X size={18} />
              </button>
            </div>
            <div className="filmstrip-lightbox-body">
              <img src={previewSrc} alt={previewFileName} />
            </div>
            <div className="filmstrip-lightbox-footer">
              <span>
                {previewResult.image_size[0]} × {previewResult.image_size[1]} px
              </span>
              <span>
                {previewResult.people.length > 0
                  ? `${previewResult.people.length} 个人物`
                  : "无人物"}
              </span>
              <span className="muted">按 Esc 或点击空白处关闭</span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
