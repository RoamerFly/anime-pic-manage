import { useEffect, useRef } from "react";
import { ExternalLink, FolderInput, Trash2 } from "lucide-react";

export interface ContextMenuProps {
  x: number;
  y: number;
  imagePath: string;
  onClose: () => void;
  onShowInFolder: (path: string) => void;
  onMoveFile: (path: string) => void;
  onDeleteFile: (path: string) => void;
}

export function ImageContextMenu({
  x,
  y,
  imagePath,
  onClose,
  onShowInFolder,
  onMoveFile,
  onDeleteFile,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handlePointerDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  // Adjust coordinates so the menu stays inside viewport
  const menuWidth = 168;
  const menuHeight = 120;
  const safeX = Math.max(8, Math.min(x, (window.innerWidth || 800) - menuWidth - 8));
  const safeY = Math.max(8, Math.min(y, (window.innerHeight || 600) - menuHeight - 8));

  const filename = imagePath.replace(/\\/g, "/").split("/").pop() || imagePath;

  return (
    <div
      ref={menuRef}
      className="custom-context-menu"
      style={{ left: `${safeX}px`, top: `${safeY}px` }}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="context-menu-header" title={imagePath}>
        <span>{filename}</span>
      </div>
      <button
        className="context-menu-item"
        onClick={() => {
          onShowInFolder(imagePath);
          onClose();
        }}
      >
        <ExternalLink size={14} />
        <span>打开文件位置</span>
      </button>
      <button
        className="context-menu-item"
        onClick={() => {
          onMoveFile(imagePath);
          onClose();
        }}
      >
        <FolderInput size={14} />
        <span>移动文件…</span>
      </button>
      <div className="context-menu-divider" />
      <button
        className="context-menu-item danger"
        onClick={() => {
          onDeleteFile(imagePath);
          onClose();
        }}
      >
        <Trash2 size={14} />
        <span>删除文件</span>
      </button>
    </div>
  );
}
