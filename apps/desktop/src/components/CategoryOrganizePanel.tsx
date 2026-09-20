import { useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { convertFileSrc } from "@tauri-apps/api/core";
import {
  AlertCircle,
  CheckSquare,
  CircleCheck,
  FolderInput,
  FolderOpen,
  Layers,
  LoaderCircle,
  Search,
  Square,
  UsersRound,
  X,
} from "lucide-react";
import type { BatchMoveResult, LibraryScanResult } from "@anime-pic-manage/shared-types";
import { invokeCore, isTauriRuntime } from "../lib/tauri";

export interface CategoryOrganizePanelProps {
  scanResults: LibraryScanResult[];
  libraryPath?: string | null;
  onFilesMoved?: (moved: Array<{ source: string; destination: string }>) => void;
}

interface CategoryFileItem {
  path: string;
  filename: string;
  imageSize: [number, number];
  characterCount: number;
  confidence: number;
}

interface CategoryGroup {
  name: string;
  files: CategoryFileItem[];
}

export function CategoryOrganizePanel({
  scanResults,
  libraryPath,
  onFilesMoved,
}: CategoryOrganizePanelProps) {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [categorySearch, setCategorySearch] = useState("");
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [targetDirectory, setTargetDirectory] = useState<string>(libraryPath ?? "");
  const [createSubfolder, setCreateSubfolder] = useState<boolean>(true);
  const [moving, setMoving] = useState<boolean>(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Group scan results by recognized characters
  const categories = useMemo(() => {
    const map = new Map<string, Map<string, CategoryFileItem>>();

    for (const result of scanResults) {
      const filename = result.path.replace(/\\/g, "/").split("/").pop() || result.path;
      if (!result.people || result.people.length === 0) {
        const cat = "未检出角色";
        if (!map.has(cat)) map.set(cat, new Map());
        map.get(cat)!.set(result.path, {
          path: result.path,
          filename,
          imageSize: result.image_size,
          characterCount: 0,
          confidence: 0,
        });
        continue;
      }

      let hasRecognizedPerson = false;
      for (const person of result.people) {
        const charName =
          person.top1?.display_name?.trim() ||
          person.top1?.character_tag?.trim();
        if (charName) {
          hasRecognizedPerson = true;
          if (!map.has(charName)) map.set(charName, new Map());
          const existing = map.get(charName)!.get(result.path);
          const conf = person.top1?.confidence ?? 0;
          if (!existing || conf > existing.confidence) {
            map.get(charName)!.set(result.path, {
              path: result.path,
              filename,
              imageSize: result.image_size,
              characterCount: result.people.length,
              confidence: conf,
            });
          }
        }
      }

      if (!hasRecognizedPerson) {
        const cat = "未识别角色";
        if (!map.has(cat)) map.set(cat, new Map());
        map.get(cat)!.set(result.path, {
          path: result.path,
          filename,
          imageSize: result.image_size,
          characterCount: result.people.length,
          confidence: 0,
        });
      }
    }

    const groups: CategoryGroup[] = Array.from(map.entries()).map(([name, filesMap]) => ({
      name,
      files: Array.from(filesMap.values()),
    }));

    // Sort by file count descending
    groups.sort((a, b) => b.files.length - a.files.length);
    return groups;
  }, [scanResults]);

  // Filter categories by search
  const filteredCategories = useMemo(() => {
    const query = categorySearch.trim().toLowerCase();
    if (!query) return categories;
    return categories.filter((c) => c.name.toLowerCase().includes(query));
  }, [categories, categorySearch]);

  // Current active category
  const activeCategoryName = useMemo(() => {
    if (selectedCategory && categories.some((c) => c.name === selectedCategory)) {
      return selectedCategory;
    }
    return categories[0]?.name ?? null;
  }, [categories, selectedCategory]);

  const activeGroup = useMemo(() => {
    return categories.find((c) => c.name === activeCategoryName) ?? null;
  }, [categories, activeCategoryName]);

  const activeFiles = activeGroup?.files ?? [];

  // Toggle single file selection
  const toggleSelect = (path: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  // Select all in current category
  const handleSelectAll = () => {
    setSelectedPaths(new Set(activeFiles.map((f) => f.path)));
  };

  // Deselect all
  const handleDeselectAll = () => {
    setSelectedPaths(new Set());
  };

  // Invert selection
  const handleInvertSelect = () => {
    setSelectedPaths((prev) => {
      const next = new Set<string>();
      for (const f of activeFiles) {
        if (!prev.has(f.path)) {
          next.add(f.path);
        }
      }
      return next;
    });
  };

  // Choose destination directory
  const chooseTargetDirectory = async () => {
    if (!isTauriRuntime()) return;
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择文件整理目标文件夹",
    });
    if (typeof selected === "string" && selected.trim()) {
      setTargetDirectory(selected);
    }
  };

  // Perform batch move
  const handleMoveFiles = async () => {
    if (selectedPaths.size === 0) return;
    const dest = targetDirectory.trim();
    if (!dest) {
      setError("请先选择目标存储目录。");
      return;
    }

    const subfolder = createSubfolder && activeCategoryName ? activeCategoryName : null;
    const destDesc = subfolder ? `${dest} / ${subfolder}` : dest;

    const confirmed = globalThis.confirm(
      `确定将选中的 ${selectedPaths.size} 个图片文件移动到以下目录吗？\n\n${destDesc}\n\n移动后，已保存在数据库的人工标注路径将自动同步更新。`
    );
    if (!confirmed) return;

    setMoving(true);
    setError(null);
    setNotice(null);

    const sources = Array.from(selectedPaths);
    const response = await invokeCore<BatchMoveResult>("move_library_files", "library.files.move", {
      sources,
      destination_directory: dest,
      subfolder_by_category: subfolder,
    });

    setMoving(false);
    if (response.error || !response.payload) {
      setError(response.error?.message ?? "批量移动文件失败。");
      return;
    }

    const result = response.payload;
    if (result.success_count > 0) {
      setNotice(
        `成功移动 ${result.success_count} 个文件至 ${destDesc}！` +
          (result.failed_count > 0 ? `（${result.failed_count} 个文件移动失败）` : "")
      );
      setSelectedPaths(new Set());
      if (onFilesMoved) {
        onFilesMoved(result.moved);
      }
    } else if (result.failed_count > 0) {
      setError(
        `文件移动失败: ${result.errors[0]?.error ?? "未知错误"}`
      );
    }
  };

  if (scanResults.length === 0) {
    return (
      <section className="category-organize-empty panel-card">
        <div className="empty-icon-wrap">
          <Layers size={36} />
        </div>
        <h3>当前尚无识别结果</h3>
        <p>
          请先在「识别与复核」页面点击「开始扫描」，完成图片角色识别后，系统将自动汇总各角色的识别分类，支持多选并一键归类移动至指定文件夹。
        </p>
      </section>
    );
  }

  return (
    <div className="category-organize-container">
      {notice && (
        <div className="workbench-notice" role="status">
          <CircleCheck size={16} />
          <span>{notice}</span>
          <button aria-label="关闭提示" onClick={() => setNotice(null)}>
            <X size={15} />
          </button>
        </div>
      )}
      {error && (
        <div className="workbench-notice error" role="alert">
          <AlertCircle size={16} />
          <span>{error}</span>
          <button aria-label="关闭错误" onClick={() => setError(null)}>
            <X size={15} />
          </button>
        </div>
      )}

      <div className="category-organize-layout">
        <aside className="category-sidebar panel-card">
          <div className="category-sidebar-header">
            <div className="eyebrow">CATEGORIES</div>
            <h3>识别分类 ({categories.length})</h3>
            <div className="category-search-box">
              <Search size={14} />
              <input
                type="text"
                placeholder="搜索角色分类…"
                value={categorySearch}
                onChange={(e) => setCategorySearch(e.target.value)}
              />
              {categorySearch && (
                <button onClick={() => setCategorySearch("")} aria-label="清除搜索">
                  <X size={12} />
                </button>
              )}
            </div>
          </div>

          <div className="category-list" role="list">
            {filteredCategories.map((cat) => {
              const isActive = cat.name === activeCategoryName;
              return (
                <button
                  key={cat.name}
                  className={`category-item-btn${isActive ? " active" : ""}`}
                  onClick={() => {
                    setSelectedCategory(cat.name);
                    setSelectedPaths(new Set());
                  }}
                  role="listitem"
                >
                  <span className="category-name" title={cat.name}>
                    <UsersRound size={14} />
                    {cat.name}
                  </span>
                  <span className="category-count-badge">{cat.files.length}</span>
                </button>
              );
            })}
          </div>
        </aside>

        <main className="category-main panel-card">
          <div className="category-action-bar">
            <div className="category-header-info">
              <h3>
                {activeCategoryName}
                <small>（共 {activeFiles.length} 张图片）</small>
              </h3>
              <div className="selection-stats">
                已选中 <strong>{selectedPaths.size}</strong> / {activeFiles.length} 项
              </div>
            </div>

            <div className="category-selection-btns">
              <button
                type="button"
                className="ghost-button"
                onClick={handleSelectAll}
                disabled={activeFiles.length === 0}
              >
                <CheckSquare size={14} /> 全选
              </button>
              <button
                type="button"
                className="ghost-button"
                onClick={handleInvertSelect}
                disabled={activeFiles.length === 0}
              >
                <Square size={14} /> 反选
              </button>
              <button
                type="button"
                className="ghost-button"
                onClick={handleDeselectAll}
                disabled={selectedPaths.size === 0}
              >
                <X size={14} /> 取消
              </button>
            </div>
          </div>

          <div className="category-move-bar">
            <div className="destination-input-group">
              <span className="dest-label">目标目录:</span>
              <input
                type="text"
                className="dest-path-input"
                placeholder="选择或输入目标文件夹绝对路径…"
                value={targetDirectory}
                onChange={(e) => setTargetDirectory(e.target.value)}
              />
              <button
                type="button"
                className="ghost-button"
                onClick={() => void chooseTargetDirectory()}
              >
                <FolderOpen size={14} /> 浏览
              </button>
            </div>

            <div className="move-options-group">
              <label className="checkbox-label" title="在目标文件夹下为当前角色自动建立同名子文件夹">
                <input
                  type="checkbox"
                  checked={createSubfolder}
                  onChange={(e) => setCreateSubfolder(e.target.checked)}
                />
                <span>创建分类子目录 (/{activeCategoryName})</span>
              </label>

              <button
                type="button"
                className="primary-button move-btn"
                disabled={selectedPaths.size === 0 || !targetDirectory.trim() || moving}
                onClick={() => void handleMoveFiles()}
              >
                {moving ? (
                  <>
                    <LoaderCircle size={15} className="spin" /> 正在移动…
                  </>
                ) : (
                  <>
                    <FolderInput size={15} /> 移动所选图片 ({selectedPaths.size})
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="category-image-grid">
            {activeFiles.map((file) => {
              const isSelected = selectedPaths.has(file.path);
              const imgSrc = convertFileSrc(file.path);
              return (
                <div
                  key={file.path}
                  className={`category-image-card${isSelected ? " selected" : ""}`}
                  onClick={() => toggleSelect(file.path)}
                  title={`点击选择/取消选择: ${file.path}`}
                >
                  <div className="card-checkbox">
                    {isSelected ? (
                      <CheckSquare size={18} className="checked-icon" />
                    ) : (
                      <Square size={18} className="unchecked-icon" />
                    )}
                  </div>
                  <div className="card-image-wrap">
                    <img src={imgSrc} alt={file.filename} loading="lazy" />
                  </div>
                  <div className="card-meta">
                    <span className="card-filename" title={file.filename}>
                      {file.filename}
                    </span>
                    {file.confidence > 0 && (
                      <span
                        className={`card-conf-badge ${
                          file.confidence >= 0.85
                            ? "high"
                            : file.confidence >= 0.6
                            ? "mid"
                            : "low"
                        }`}
                      >
                        {Math.round(file.confidence * 100)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </main>
      </div>
    </div>
  );
}
