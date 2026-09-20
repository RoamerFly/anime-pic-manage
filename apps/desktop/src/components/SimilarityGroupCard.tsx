import React from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { Archive, Check, Film, Folder, Maximize2, Sparkles, Trash2 } from "lucide-react";
import type {
  SimilarityDecisionAction,
  SimilarityGroup,
  SimilarityGroupItem,
} from "@anime-pic-manage/shared-types";
import { isTauriRuntime, normalizePreviewPath } from "../lib/tauri";

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function getParentDirectory(path: string): string {
  const lastSeparator = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  if (lastSeparator < 0) return "所在目录未知";
  if (lastSeparator === 0) return path.slice(0, 1);
  if (lastSeparator === 2 && path[1] === ":") return path.slice(0, 3);
  return path.slice(0, lastSeparator);
}

export interface SimilarityGroupCardProps {
  group: SimilarityGroup;
  groupIdx: number;
  onApplyRecommended: (groupId: string) => void;
  onKeepAll: (groupId: string) => void;
  onArchiveAll: (groupId: string) => void;
  onDeleteAll: (groupId: string) => void;
  onItemDecisionChange: (
    groupId: string,
    itemPath: string,
    decision: SimilarityDecisionAction,
  ) => void;
  onPreviewImage: (item: SimilarityGroupItem) => void;
  onContextMenu: (e: React.MouseEvent, path: string) => void;
}

export const SimilarityGroupCard = React.memo(function SimilarityGroupCard({
  group,
  groupIdx,
  onApplyRecommended,
  onKeepAll,
  onArchiveAll,
  onDeleteAll,
  onItemDecisionChange,
  onPreviewImage,
  onContextMenu,
}: SimilarityGroupCardProps) {
  const groupTypeLabel =
    group.group_type === "exact_duplicate"
      ? "完全重复"
      : group.group_type === "variation"
        ? "差分变体"
        : "近似相似";
  const groupTypeTone =
    group.group_type === "exact_duplicate"
      ? "tag-red"
      : group.group_type === "variation"
        ? "tag-orange"
        : "tag-blue";

  return (
    <div className="similarity-group-card">
      <div className="similarity-group-header">
        <div className="group-title-left">
          <span className="group-index-badge">#{groupIdx + 1}</span>
          <span className={`group-type-badge ${groupTypeTone}`}>{groupTypeLabel}</span>
          <span className="group-meta-text">
            相似度: <strong>{Math.round(group.average_similarity * 100)}%</strong> ·{" "}
            {group.items.length} 张图片
          </span>
        </div>

        <div className="group-actions-right">
          <button
            className="group-action-btn primary"
            onClick={() => onApplyRecommended(group.group_id)}
            title="自动将最高画质设为保留，其余设为隔离归档"
          >
            <Sparkles size={13} /> 推荐配置
          </button>
          <button
            className="group-action-btn ghost"
            onClick={() => onKeepAll(group.group_id)}
            title="全部保留"
          >
            全部保留
          </button>
          <button
            className="group-action-btn ghost"
            onClick={() => onArchiveAll(group.group_id)}
            title="除推荐项外全部隔离归档"
          >
            <Archive size={13} /> 隔离非推荐
          </button>
          <button
            className="group-action-btn ghost danger"
            onClick={() => onDeleteAll(group.group_id)}
            title="除推荐项外全部永久删除"
          >
            <Trash2 size={13} /> 删除非推荐
          </button>
        </div>
      </div>

      {/* 组内图片并排卡片 */}
      <div className="group-items-row">
        {group.items.map((item) => {
          const fileName = item.path.split(/[/\\]/).pop() ?? item.path;
          const parentDirectory = getParentDirectory(item.path);
          const imgSrc = isTauriRuntime()
            ? convertFileSrc(normalizePreviewPath(item.path))
            : item.path;
          const [w, h] = item.dimensions;

          return (
            <div
              key={item.path}
              className={`similarity-item-card ${item.is_recommended ? "recommended" : ""} ${
                item.decision === "delete"
                  ? "decision-delete"
                  : item.decision === "archive"
                    ? "decision-archive"
                    : "decision-keep"
              }`}
              onContextMenu={(e) => onContextMenu(e, item.path)}
              title={`${fileName} - 右键可打开位置/移动/删除`}
            >
              {item.is_recommended && (
                <div
                  className="recommended-badge"
                  title={item.recommend_reason || "最佳画质版本"}
                >
                  <Sparkles size={12} /> 推荐保留
                </div>
              )}

              {item.is_animated && (
                <div
                  className="animated-badge"
                  title={`动图共 ${item.frame_count ?? 1} 帧，已比较第 ${
                    (item.sampled_frames ?? [0]).map((index) => index + 1).join("/")
                  } 帧`}
                >
                  <Film size={11} /> 动图 {item.frame_count} 帧 · 比对首/中/末
                </div>
              )}

              <div
                className="item-thumbnail-wrapper"
                role="button"
                tabIndex={0}
                aria-label={`查看大图：${fileName}`}
                onClick={() => onPreviewImage(item)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onPreviewImage(item);
                  }
                }}
              >
                <img
                  src={imgSrc}
                  alt={fileName}
                  className="item-thumbnail"
                  loading="lazy"
                />
                <span
                  className="item-zoom-btn"
                  title="点击放大对比"
                  aria-hidden="true"
                >
                  <Maximize2 size={14} />
                </span>
              </div>

              <div className="item-details">
                <div className="item-filename" title={item.path}>
                  {fileName}
                </div>
                <div className="item-directory" title={parentDirectory}>
                  <Folder size={11} />
                  <span>{parentDirectory}</span>
                </div>
                <div className="item-specs-grid">
                  <span className={`spec-item ${item.is_recommended ? "highlight-spec" : ""}`}>
                    分辨率: <strong>{w}x{h}</strong>
                  </span>
                  <span className="spec-item">
                    体积: <strong>{formatBytes(item.file_size)}</strong>
                  </span>
                  <span className="spec-item">
                    格式: <strong>{item.format.toUpperCase()}</strong>
                  </span>
                  <span className="spec-item" title="拉普拉斯边缘方差清晰度评分">
                    清晰度: <strong>{item.clarity_score}</strong>
                  </span>
                </div>

                {item.recommend_reason && (
                  <div className="recommend-reason-text">{item.recommend_reason}</div>
                )}

                {/* 治理动作单选切换 */}
                <div className="item-decision-selector">
                  <button
                    className={`decision-chip keep ${item.decision === "keep" ? "active" : ""}`}
                    onClick={() => onItemDecisionChange(group.group_id, item.path, "keep")}
                  >
                    <Check size={12} /> 保留
                  </button>
                  <button
                    className={`decision-chip archive ${item.decision === "archive" ? "active" : ""}`}
                    onClick={() => onItemDecisionChange(group.group_id, item.path, "archive")}
                  >
                    <Archive size={12} /> 隔离
                  </button>
                  <button
                    className={`decision-chip delete ${item.decision === "delete" ? "active" : ""}`}
                    onClick={() => onItemDecisionChange(group.group_id, item.path, "delete")}
                  >
                    <Trash2 size={12} /> 删除
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});
