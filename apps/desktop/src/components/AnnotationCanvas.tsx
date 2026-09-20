import React, { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ChevronLeft, ChevronRight, Edit3, Image as ImageIcon, Plus, Sparkles, X } from "lucide-react";
import type { NormalizedBoundingBox } from "@anime-pic-manage/shared-types";
import type { EditableAnnotation, PersonStatus } from "./UnifiedInspector";

type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

interface DragState {
  mode: "move" | "resize" | "create";
  id?: string;
  handle?: ResizeHandle;
  start: { x: number; y: number };
  origin?: NormalizedBoundingBox;
  last?: NormalizedBoundingBox;
  base?: EditableAnnotation[];
  pointerId?: number;
}

const MIN_BBOX_SIZE = 0.02;

interface AnnotationCanvasProps {
  imageSrc?: string;
  imageSize?: [number, number];
  annotations: EditableAnnotation[];
  activeAnnotationId: string | null;
  onSelectAnnotation: (item: EditableAnnotation) => void;
  onUpdateAnnotation: (id: string, box: NormalizedBoundingBox) => void;
  onAddAnnotation: (box: NormalizedBoundingBox) => void;
  editorMode: boolean;
  onToggleEditor: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onDeleteActive?: () => void;
  onSelectCandidateIndex?: (index: number) => void;
  isDemo?: boolean;
  onPrevImage?: () => void;
  onNextImage?: () => void;
  hasPrevImage?: boolean;
  hasNextImage?: boolean;
  onContextMenu?: (e: React.MouseEvent) => void;
}

export function AnnotationCanvas({
  imageSrc,
  imageSize,
  annotations,
  activeAnnotationId,
  onSelectAnnotation,
  onUpdateAnnotation,
  onAddAnnotation,
  editorMode,
  onToggleEditor,
  onUndo,
  onRedo,
  onDeleteActive,
  onSelectCandidateIndex,
  isDemo = false,
  onPrevImage,
  onNextImage,
  hasPrevImage = false,
  hasNextImage = false,
  onContextMenu,
}: AnnotationCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const resizeDragRef = useRef<{ startY: number; startHeight: number } | null>(null);
  const [imageLoadError, setImageLoadError] = useState<string | null>(null);
  const [creatingMode, setCreatingMode] = useState(false);
  const [draftBox, setDraftBox] = useState<NormalizedBoundingBox | null>(null);
  const DEFAULT_CANVAS_HEIGHT = 340;
  const [viewportHeight, setViewportHeight] = useState<number>(() => {
    try {
      const saved = localStorage.getItem("anime_pic_canvas_height");
      const parsed = saved ? Number(saved) : DEFAULT_CANVAS_HEIGHT;
      return Number.isFinite(parsed) && parsed >= 240 && parsed <= 1200 ? parsed : DEFAULT_CANVAS_HEIGHT;
    } catch {
      return DEFAULT_CANVAS_HEIGHT;
    }
  });

  const handleResizePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    resizeDragRef.current = {
      startY: e.clientY,
      startHeight: viewportHeight,
    };
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch { /* noop */ }
  };

  const handleResizePointerMove = (e: React.PointerEvent) => {
    if (!resizeDragRef.current) return;
    const delta = e.clientY - resizeDragRef.current.startY;
    const nextHeight = Math.max(240, Math.min(1000, resizeDragRef.current.startHeight + delta));
    setViewportHeight(nextHeight);
  };

  const handleResizePointerUp = (e: React.PointerEvent) => {
    if (!resizeDragRef.current) return;
    resizeDragRef.current = null;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch { /* noop */ }
    try {
      localStorage.setItem("anime_pic_canvas_height", String(viewportHeight));
    } catch { /* noop */ }
  };

  const handleResetResize = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setViewportHeight(DEFAULT_CANVAS_HEIGHT);
    try {
      localStorage.setItem("anime_pic_canvas_height", String(DEFAULT_CANVAS_HEIGHT));
    } catch { /* noop */ }
  };

  useEffect(() => {
    setImageLoadError(null);
  }, [imageSrc]);

  // 根据当前 img 元素的实际显示位置精确计算 0..1 归一化坐标
  const getNormalizedPoint = useCallback((clientX: number, clientY: number): { x: number; y: number } | null => {
    const img = imgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;

    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
    return { x, y };
  }, []);

  const handlePointerDown = (event: React.PointerEvent) => {
    if (!editorMode || !imageSrc) return;
    const point = getNormalizedPoint(event.clientX, event.clientY);
    if (!point) return;

    const target = event.target as HTMLElement;
    // 如果点击的是图片或容器空白处，且处于创建模式（或者正在按着创建）
    const isCanvasClick =
      target === containerRef.current ||
      target === imgRef.current ||
      target.classList.contains("preview-canvas") ||
      target.classList.contains("canvas-stage") ||
      target.classList.contains("canvas-viewport");
    if (creatingMode || isCanvasClick) {
      event.preventDefault();
      event.stopPropagation();
      const id = `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const blankBox: NormalizedBoundingBox = { x: point.x, y: point.y, width: 0, height: 0 };

      dragRef.current = {
        mode: "create",
        id,
        start: point,
        origin: blankBox,
        last: blankBox,
        base: [...annotations],
        pointerId: event.pointerId,
      };

      try {
        containerRef.current?.setPointerCapture(event.pointerId);
      } catch { /* noop */ }
    }
  };

  const handleBoxPointerDown = (
    event: React.PointerEvent,
    item: EditableAnnotation,
    handle?: ResizeHandle,
  ) => {
    if (!editorMode) {
      onSelectAnnotation(item);
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    onSelectAnnotation(item);

    const point = getNormalizedPoint(event.clientX, event.clientY);
    if (!point) return;

    dragRef.current = {
      mode: handle ? "resize" : "move",
      id: item.clientId,
      handle,
      start: point,
      origin: { ...item.bbox },
      last: { ...item.bbox },
      base: [...annotations],
      pointerId: event.pointerId,
    };

    try {
      containerRef.current?.setPointerCapture(event.pointerId);
    } catch { /* noop */ }
  };

  const handlePointerMove = (event: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !drag.id) return;
    const point = getNormalizedPoint(event.clientX, event.clientY);
    if (!point) return;

    let nextBox: NormalizedBoundingBox;
    if (drag.mode === "create") {
      nextBox = {
        x: Math.min(drag.start.x, point.x),
        y: Math.min(drag.start.y, point.y),
        width: Math.abs(point.x - drag.start.x),
        height: Math.abs(point.y - drag.start.y),
      };
    } else if (drag.mode === "resize" && drag.origin && drag.handle) {
      const dx = point.x - drag.start.x;
      const dy = point.y - drag.start.y;
      let left = drag.origin.x;
      let right = drag.origin.x + drag.origin.width;
      let top = drag.origin.y;
      let bottom = drag.origin.y + drag.origin.height;

      if (drag.handle.includes("w")) left = Math.max(0, Math.min(right - MIN_BBOX_SIZE, left + dx));
      if (drag.handle.includes("e")) right = Math.min(1, Math.max(left + MIN_BBOX_SIZE, right + dx));
      if (drag.handle.includes("n")) top = Math.max(0, Math.min(bottom - MIN_BBOX_SIZE, top + dy));
      if (drag.handle.includes("s")) bottom = Math.min(1, Math.max(top + MIN_BBOX_SIZE, bottom + dy));

      nextBox = { x: left, y: top, width: Math.max(MIN_BBOX_SIZE, right - left), height: Math.max(MIN_BBOX_SIZE, bottom - top) };
    } else if (drag.origin) {
      const dx = point.x - drag.start.x;
      const dy = point.y - drag.start.y;
      nextBox = {
        ...drag.origin,
        x: Math.max(0, Math.min(1 - drag.origin.width, drag.origin.x + dx)),
        y: Math.max(0, Math.min(1 - drag.origin.height, drag.origin.y + dy)),
      };
    } else {
      return;
    }

    drag.last = nextBox;
    if (drag.mode === "create") {
      setDraftBox(nextBox);
    } else {
      onUpdateAnnotation(drag.id, nextBox);
    }
  };

  const handlePointerUp = (event: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    try {
      containerRef.current?.releasePointerCapture(event.pointerId);
    } catch { /* noop */ }

    if (drag.mode === "create") {
      setCreatingMode(false);
      setDraftBox(null);
      if (drag.last && drag.last.width >= MIN_BBOX_SIZE && drag.last.height >= MIN_BBOX_SIZE) {
        onAddAnnotation(drag.last);
      }
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (!editorMode) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      onUndo?.();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      onRedo?.();
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      onDeleteActive?.();
      return;
    }
    if (event.key === "n" || event.key === "N") {
      event.preventDefault();
      setCreatingMode((prev) => !prev);
      return;
    }
    // 快捷键 1~5 键选择候选
    if (["1", "2", "3", "4", "5"].includes(event.key)) {
      const idx = Number.parseInt(event.key, 10) - 1;
      onSelectCandidateIndex?.(idx);
    }
  };

  // 添加默认居中框
  const handleAddDefaultBox = () => {
    const defaultBox: NormalizedBoundingBox = {
      x: 0.35,
      y: 0.25,
      width: 0.3,
      height: 0.35,
    };
    onAddAnnotation(defaultBox);
  };

  return (
    <div
      ref={containerRef}
      className={`annotation-canvas-wrapper panel-card ${editorMode ? "editing" : ""} ${creatingMode ? "cursor-crosshair" : ""}`}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      aria-label="图片标注与预览工作区"
    >
      {/* 顶部浮动工具条 */}
      <div className="canvas-header-bar">
        <div className="canvas-stats">
          <span className="soft-badge blue">
            <ImageIcon size={12} /> {annotations.length} 个识别框
          </span>
          {creatingMode && (
            <span className="soft-badge orange animate-pulse">
              绘框模式：请在图片上拖拽绘制识别框
            </span>
          )}
        </div>
        <div className="canvas-actions">
          {editorMode && (
            <>
              <button
                type="button"
                className={`ghost-button canvas-tool-btn ${creatingMode ? "active" : ""}`}
                onClick={() => {
                  setCreatingMode(!creatingMode);
                  setDraftBox(null);
                }}
                title="进入手映画框模式 (快捷键 N)"
              >
                <Plus size={13} /> {creatingMode ? "取消画框" : "画框"}
              </button>
              <button
                type="button"
                className="ghost-button canvas-tool-btn"
                onClick={handleAddDefaultBox}
                title="直接在图片中央置入新识别框"
              >
                + 添加居中框
              </button>
            </>
          )}
          <button
            type="button"
            className={`ghost-button canvas-mode-btn ${editorMode ? "active" : ""}`}
            onClick={onToggleEditor}
          >
            {editorMode ? <><X size={13} /> 退出标注</> : <><Edit3 size={13} /> 编辑标注</>}
          </button>
        </div>
      </div>

      {/* 主图片展示区：固定/受控高度，杜绝切换不同长宽比图片引起的抖动 */}
      <div
        className="canvas-viewport"
        style={{ height: `${viewportHeight}px` }}
        onContextMenu={(e) => {
          if (!creatingMode && onContextMenu) {
            e.preventDefault();
            e.stopPropagation();
            onContextMenu(e);
          }
        }}
      >
        {/* 悬浮左右快速切图箭头 */}
        {onPrevImage && (
          <button
            type="button"
            className="canvas-nav-arrow prev"
            onClick={(e) => {
              e.stopPropagation();
              onPrevImage();
            }}
            disabled={!hasPrevImage}
            title="上一张 (键盘 ←)"
            aria-label="上一张图片"
          >
            <ChevronLeft size={24} />
          </button>
        )}
        {onNextImage && (
          <button
            type="button"
            className="canvas-nav-arrow next"
            onClick={(e) => {
              e.stopPropagation();
              onNextImage();
            }}
            disabled={!hasNextImage}
            title="下一张 (键盘 →)"
            aria-label="下一张图片"
          >
            <ChevronRight size={24} />
          </button>
        )}

        {imageSrc ? (
          <div
            className="canvas-stage"
            style={{
              aspectRatio:
                imageSize && imageSize[0] > 0 && imageSize[1] > 0
                  ? `${imageSize[0]} / ${imageSize[1]}`
                  : undefined,
            }}
          >
            <img
              ref={imgRef}
              src={imageSrc}
              alt="待识别图片"
              className="canvas-image"
              draggable={false}
              onError={() => setImageLoadError("图片加载失败，请确认文件路径仍然有效。")}
            />

            {/* 选框层：完全限定在 canvas-stage 内部，坐标百分比与图片真实边界 100% 对齐 */}
            {annotations.map((item, index) => {
              const isSelected = item.clientId === activeAnnotationId;
              const statusTone: PersonStatus = item.source === "manual" ? "review" : "high";

              return (
                <div
                  key={item.clientId}
                  className={`canvas-detection-box ${statusTone} ${isSelected ? "selected" : ""} ${editorMode ? "draggable" : ""}`}
                  style={{
                    left: `${item.bbox.x * 100}%`,
                    top: `${item.bbox.y * 100}%`,
                    width: `${item.bbox.width * 100}%`,
                    height: `${item.bbox.height * 100}%`,
                  }}
                  onPointerDown={(e) => handleBoxPointerDown(e, item)}
                  onClick={() => onSelectAnnotation(item)}
                >
                  <span className="detection-box-label">
                    {String(index + 1).padStart(2, "0")} · {item.label_name || "未分类"}
                  </span>

                  {/* 8 方向调整手柄 */}
                  {editorMode && isSelected && (
                    <>
                      {(["nw", "n", "ne", "e", "se", "s", "sw", "w"] as ResizeHandle[]).map((handle) => (
                        <button
                          key={handle}
                          type="button"
                          className={`canvas-resize-handle ${handle}`}
                          aria-label={`调整${handle}方向`}
                          onPointerDown={(e) => handleBoxPointerDown(e, item, handle)}
                        />
                      ))}
                    </>
                  )}
                </div>
              );
            })}

            {draftBox && (
              <div
                className="canvas-detection-box draft"
                aria-hidden="true"
                style={{
                  left: `${draftBox.x * 100}%`,
                  top: `${draftBox.y * 100}%`,
                  width: `${draftBox.width * 100}%`,
                  height: `${draftBox.height * 100}%`,
                }}
              >
                <span className="detection-box-label">新识别框</span>
              </div>
            )}

            {isDemo && <div className="preview-watermark">DEMO PREVIEW</div>}
          </div>
        ) : (
          <div className="canvas-empty-state">
            <Sparkles size={28} />
            <p>请在下方选择一张图片开始预览与复核</p>
          </div>
        )}

        {/* 右下角手动拉伸展示框高度手柄 */}
        <div
          className="canvas-viewport-resize-handle"
          onPointerDown={handleResizePointerDown}
          onPointerMove={handleResizePointerMove}
          onPointerUp={handleResizePointerUp}
          onPointerCancel={handleResizePointerUp}
          onDoubleClick={handleResetResize}
          title="按住鼠标拖拽调整展示框高度；双击重置为默认高度"
        >
          <span className="resize-handle-grip" />
        </div>
      </div>

      {imageLoadError && (
        <div className="canvas-error-alert" role="alert">
          <AlertCircle size={14} /> {imageLoadError}
        </div>
      )}

      {/* 底部小图例 */}
      <div className="canvas-footer-legend">
        <span><span className="legend-dot high" /> 高置信度</span>
        <span><span className="legend-dot review" /> 待复核/已标注</span>
        <span><span className="legend-dot unknown" /> 未识别</span>
        <span className="canvas-hotkeys-hint">
          快捷键：N 画框 · Delete 删框 · Ctrl+Z 撤销 · 1~5 赋名
        </span>
      </div>
    </div>
  );
}
