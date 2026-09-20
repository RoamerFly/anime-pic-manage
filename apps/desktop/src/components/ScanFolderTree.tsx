import { useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronsDownUp,
  ChevronsUpDown,
  Folder,
  FolderOpen,
  Image as ImageIcon,
  Search,
  Users,
  X,
} from "lucide-react";
import type { LibraryScanResult } from "@anime-pic-manage/shared-types";
import { isTauriRuntime } from "../lib/tauri";

export interface FolderTreeNode {
  id: string;
  name: string;
  relativePath: string;
  isFolder: boolean;
  subFolders: FolderTreeNode[];
  files: LibraryScanResult[];
  totalFileCount: number;
  totalPeopleCount: number;
  hasReview: boolean;
  allHighConfidence: boolean;
}

interface ScanFolderTreeProps {
  results: LibraryScanResult[];
  libraryPath?: string;
  selectedPath: string | null;
  onSelectResult: (result: LibraryScanResult) => void;
  onClose?: () => void;
  onContextMenu?: (path: string, e: React.MouseEvent) => void;
  foldersOnly?: boolean;
  selectedFolderPath?: string;
  onSelectFolder?: (node: FolderTreeNode) => void;
}

function normalizePath(p: string): string {
  return p.replace(/\\/g, "/");
}

export function resultsForFolder(
  results: LibraryScanResult[],
  libraryPath: string | undefined,
  relativePath: string,
): LibraryScanResult[] {
  const root = normalizePath(libraryPath ?? "").replace(/\/+$/, "");
  const relative = normalizePath(relativePath).replace(/^\/+|\/+$/g, "");
  const prefix = relative ? `${root}/${relative}/` : root ? `${root}/` : "";

  return [...results]
    .filter((result) => !prefix || normalizePath(result.path).startsWith(prefix))
    .sort((a, b) => {
      const pathA = normalizePath(a.path).slice(prefix.length);
      const pathB = normalizePath(b.path).slice(prefix.length);
      return pathA.localeCompare(pathB, "zh-CN", { numeric: true });
    });
}

export function buildFolderTree(results: LibraryScanResult[], libraryPath?: string): FolderTreeNode {
  const normLibPath = libraryPath ? normalizePath(libraryPath).replace(/\/+$/, "") : "";
  const rootName = normLibPath ? normLibPath.split("/").pop() || "图库根目录" : "图库根目录";

  const root: FolderTreeNode = {
    id: "root",
    name: rootName,
    relativePath: "",
    isFolder: true,
    subFolders: [],
    files: [],
    totalFileCount: 0,
    totalPeopleCount: 0,
    hasReview: false,
    allHighConfidence: true,
  };

  const folderMap = new Map<string, FolderTreeNode>();
  folderMap.set("", root);

  for (const item of results) {
    const normItemPath = normalizePath(item.path);
    let rel = normItemPath;
    if (normLibPath && normItemPath.startsWith(normLibPath)) {
      rel = normItemPath.slice(normLibPath.length).replace(/^\/+/, "");
    }
    const segments = rel.split("/");
    segments.pop(); // remove fileName

    let currentRelPath = "";
    let parentNode = root;

    for (const segment of segments) {
      currentRelPath = currentRelPath ? `${currentRelPath}/${segment}` : segment;
      let folderNode = folderMap.get(currentRelPath);
      if (!folderNode) {
        folderNode = {
          id: currentRelPath,
          name: segment,
          relativePath: currentRelPath,
          isFolder: true,
          subFolders: [],
          files: [],
          totalFileCount: 0,
          totalPeopleCount: 0,
          hasReview: false,
          allHighConfidence: true,
        };
        folderMap.set(currentRelPath, folderNode);
        parentNode.subFolders.push(folderNode);
      }
      parentNode = folderNode;
    }

    parentNode.files.push(item);
  }

  // 计算汇总统计与排序
  function finalizeNode(node: FolderTreeNode): void {
    let fileCount = node.files.length;
    let peopleCount = node.files.reduce((acc, f) => acc + f.people.length, 0);
    let hasRev = node.files.some((f) => f.people.some((p) => p.status === "needs_review"));
    let allHigh = node.files.length > 0 && node.files.every((f) => f.people.length > 0 && f.people.every((p) => p.status === "high_confidence"));

    for (const sub of node.subFolders) {
      finalizeNode(sub);
      fileCount += sub.totalFileCount;
      peopleCount += sub.totalPeopleCount;
      if (sub.hasReview) hasRev = true;
      if (!sub.allHighConfidence) allHigh = false;
    }

    node.totalFileCount = fileCount;
    node.totalPeopleCount = peopleCount;
    node.hasReview = hasRev;
    node.allHighConfidence = allHigh;

    node.subFolders.sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
    node.files.sort((a, b) => {
      const nameA = a.path.split(/[\\/]/).pop() || "";
      const nameB = b.path.split(/[\\/]/).pop() || "";
      return nameA.localeCompare(nameB, "zh-CN");
    });
  }

  finalizeNode(root);
  return root;
}

export function ScanFolderTree({
  results,
  libraryPath,
  selectedPath,
  onSelectResult,
  onClose,
  onContextMenu,
  foldersOnly = false,
  selectedFolderPath = "",
  onSelectFolder,
}: ScanFolderTreeProps) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(() => new Set(["root"]));
  const [filterText, setFilterText] = useState("");

  const treeRoot = useMemo(() => buildFolderTree(results, libraryPath), [results, libraryPath]);

  const toggleFolder = (folderId: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) {
        next.delete(folderId);
      } else {
        next.add(folderId);
      }
      return next;
    });
  };

  const expandAll = () => {
    const all = new Set<string>();
    function collect(node: FolderTreeNode) {
      all.add(node.id);
      node.subFolders.forEach(collect);
    }
    collect(treeRoot);
    setExpandedFolders(all);
  };

  const collapseAll = () => {
    setExpandedFolders(new Set(["root"]));
  };

  // 统计文件夹总数
  const totalFolderCount = useMemo(() => {
    let count = 0;
    function walk(node: FolderTreeNode) {
      if (node.id !== "root") count++;
      node.subFolders.forEach(walk);
    }
    walk(treeRoot);
    return count;
  }, [treeRoot]);

  const renderNode = (node: FolderTreeNode, depth = 0) => {
    const isExpanded = expandedFolders.has(node.id) || filterText.trim().length > 0;
    const isRoot = node.id === "root";
    const paddingLeft = `${depth * 18 + 8}px`;

    // 如果有过滤词，过滤不相关的节点
    const search = filterText.trim().toLowerCase();
    const matchesFilter =
      !search ||
      node.name.toLowerCase().includes(search) ||
      node.files.some((f) => f.path.toLowerCase().includes(search));

    if (!matchesFilter && node.subFolders.every((s) => !s.name.toLowerCase().includes(search))) {
      return null;
    }

    return (
      <div key={node.id} className="folder-tree-node-group">
        {/* 文件夹行 */}
        <div
          className={`folder-tree-row folder-row ${isRoot ? "root-row" : ""} ${selectedFolderPath === node.relativePath ? "selected" : ""}`}
          style={{ paddingLeft }}
          onClick={() => {
            if (onSelectFolder) onSelectFolder(node);
            else toggleFolder(node.id);
          }}
          title={`文件夹: ${node.name} (${node.totalFileCount} 张图片)`}
        >
          <button
            type="button"
            className="folder-tree-toggle-icon"
            onClick={(event) => {
              event.stopPropagation();
              toggleFolder(node.id);
            }}
            aria-label={isExpanded ? `折叠 ${node.name}` : `展开 ${node.name}`}
          >
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          <span className="folder-icon">
            {isExpanded ? <FolderOpen size={16} className="text-amber" /> : <Folder size={16} className="text-amber" />}
          </span>
          <span className="folder-name">{node.name}</span>
          <div className="folder-badges">
            <span className="folder-badge file-count">
              <ImageIcon size={11} /> {node.totalFileCount}
            </span>
            {node.totalPeopleCount > 0 && (
              <span className="folder-badge people-count">
                <Users size={11} /> {node.totalPeopleCount}
              </span>
            )}
            {node.hasReview ? (
              <span className="status-dot-sm review" title="含待复核图片" />
            ) : node.allHighConfidence && node.totalPeopleCount > 0 ? (
              <span className="status-dot-sm high" title="全部高置信度" />
            ) : null}
          </div>
        </div>

        {/* 展开内容 */}
        {isExpanded && (
          <div className="folder-tree-children">
            {/* 子文件夹 */}
            {node.subFolders.map((sub) => renderNode(sub, depth + 1))}

            {/* 该文件夹下的图片列表 */}
            {!foldersOnly && node.files.map((file) => {
              const isSelected = file.path === selectedPath;
              const fileName = file.path.split(/[\\/]/).pop() || "image";
              if (search && !fileName.toLowerCase().includes(search)) {
                return null;
              }

              const hasReview = file.people.some((p) => p.status === "needs_review");
              const allHigh = file.people.length > 0 && file.people.every((p) => p.status === "high_confidence");
              const imgSrc = isTauriRuntime() ? convertFileSrc(file.path) : undefined;

              return (
                <div
                  key={file.path}
                  className={`folder-tree-row file-row ${isSelected ? "selected" : ""}`}
                  style={{ paddingLeft: `${(depth + 1) * 18 + 8}px` }}
                  onClick={() => onSelectResult(file)}
                  onContextMenu={(e) => {
                    if (onContextMenu) {
                      e.preventDefault();
                      e.stopPropagation();
                      onContextMenu(file.path, e);
                    }
                  }}
                  title={`${file.path} - 右键可打开位置/移动/删除`}
                >
                  <div className="tree-file-thumb">
                    {imgSrc ? (
                      <img src={imgSrc} alt="" loading="lazy" />
                    ) : (
                      <ImageIcon size={14} />
                    )}
                  </div>
                  <span className="tree-file-name">{fileName}</span>
                  <div className="tree-file-meta">
                    {file.people.length === 0 ? (
                      <span className="meta-pill none">未识别人物</span>
                    ) : hasReview ? (
                      <span className="meta-pill review">
                        <AlertCircle size={10} /> {file.people.length} 人物需复核
                      </span>
                    ) : allHigh ? (
                      <span className="meta-pill high">
                        <CheckCircle2 size={10} /> {file.people.length} 人物就绪
                      </span>
                    ) : (
                      <span className="meta-pill normal">
                        <Users size={10} /> {file.people.length} 人物
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="scan-folder-tree-container panel-card" aria-label="扫描目录树状层级">
      {/* 头部：标题与快捷操作 */}
      <div className="folder-tree-header">
        <div className="folder-tree-title-group">
          <div className="eyebrow">SCANNED DIRECTORIES</div>
          <h4>原始文件夹层级 ({totalFolderCount} 个文件夹 · {results.length} 张图片)</h4>
        </div>
        <div className="folder-tree-actions">
          <button
            type="button"
            className="ghost-button btn-xs"
            onClick={expandAll}
            title="展开全部文件夹"
          >
            <ChevronsUpDown size={13} /> 展开全部
          </button>
          <button
            type="button"
            className="ghost-button btn-xs"
            onClick={collapseAll}
            title="折叠至根目录"
          >
            <ChevronsDownUp size={13} /> 全部折叠
          </button>
          {onClose && (
            <button
              type="button"
              className="icon-button btn-xs"
              onClick={onClose}
              title="切换回底片栏"
              aria-label="关闭树状图"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* 搜索框 */}
      <div className="folder-tree-search-bar">
        <Search size={14} className="search-icon" />
        <input
          type="text"
          className="folder-tree-search-input"
          placeholder="按文件夹名或文件名快速过滤…"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
        />
        {filterText && (
          <button
            type="button"
            className="clear-search-btn"
            onClick={() => setFilterText("")}
            title="清空搜索"
          >
            <X size={12} />
          </button>
        )}
      </div>

      {/* 树主体 */}
      <div className="folder-tree-scroll-area">
        {results.length === 0 ? (
          <div className="folder-tree-empty">
            <FolderOpen size={28} />
            <p>尚未扫描图片。请在上方选择图库并开始扫描。</p>
          </div>
        ) : (
          renderNode(treeRoot)
        )}
      </div>
    </div>
  );
}
