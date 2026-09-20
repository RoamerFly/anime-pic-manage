import { convertFileSrc } from "@tauri-apps/api/core";
import { AlertTriangle, X } from "lucide-react";
import type { SimilarityGroupItem } from "@anime-pic-manage/shared-types";
import { isTauriRuntime, normalizePreviewPath } from "../lib/tauri";
import { formatBytes } from "./SimilarityGroupCard";

export function SimilarityDeleteConfirmModal({
  show,
  deleteCount,
  freedBytes,
  onClose,
  onConfirm,
}: {
  show: boolean;
  deleteCount: number;
  freedBytes: number;
  onClose: () => void;
  onConfirm: () => void;
}) {
  if (!show) return null;

  return (
    <div className="modal-backdrop">
      <div className="confirm-modal-card">
        <div className="modal-header">
          <div className="modal-title-danger">
            <AlertTriangle size={20} />
            <h3>确认执行永久删除？</h3>
          </div>
          <button className="icon-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p>
            您选择了对 <strong>{deleteCount} 张图片</strong> 执行物理删除，并将释放{" "}
            <strong>{formatBytes(freedBytes)}</strong> 空间。
          </p>
          <div className="danger-alert-box">
            <strong>注意：</strong>
            此操作将直接将文件从硬盘中永久移除，不可撤销！若想安全备份，建议使用「隔离归档」动作移动到隔离文件夹。
          </div>
        </div>
        <div className="modal-footer">
          <button className="ghost-button" onClick={onClose}>
            返回修改
          </button>
          <button className="danger-button" onClick={onConfirm}>
            确认永久删除
          </button>
        </div>
      </div>
    </div>
  );
}

export function SimilarityImagePreviewModal({
  item,
  onClose,
}: {
  item: SimilarityGroupItem | null;
  onClose: () => void;
}) {
  if (!item) return null;

  const fileName = item.path.split(/[/\\]/).pop();
  const imgSrc = isTauriRuntime() ? convertFileSrc(normalizePreviewPath(item.path)) : item.path;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="image-preview-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`查看大图：${fileName ?? "图片"}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="preview-modal-header">
          <div className="preview-filename">{fileName}</div>
          <button className="icon-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <div className="preview-modal-body">
          <img src={imgSrc} alt="Preview" className="large-preview-img" />
        </div>
        <div className="preview-modal-footer">
          <span>
            分辨率: {item.dimensions[0]}x{item.dimensions[1]}
          </span>
          <span>体积: {formatBytes(item.file_size)}</span>
          <span>清晰度评分: {item.clarity_score}</span>
          <span>路径: {item.path}</span>
        </div>
      </div>
    </div>
  );
}
