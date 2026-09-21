import {
  AlertCircle,
  ChevronRight,
  Clock3,
  FolderOpen,
  Play,
  ScanSearch,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import type { WorkerStatus } from "@anime-pic-manage/shared-types";
import { isScanActive } from "../lib/scan-state";
import { normalizeDisplayPath } from "../lib/tauri";
import type { WorkspaceProps } from "../types";

export function workerLabel(status: WorkerStatus): string {
  return {
    healthy: "运行正常",
    starting: "正在启动",
    stopped: "已停止",
    unavailable: "未连接",
    error: "需要检查",
  }[status];
}

export function WorkspacePage({
  status,
  loading,
  library,
  onChooseLibrary,
  recentScan,
  onNavigate,
  controller,
}: WorkspaceProps) {
  const workerStatus = status?.worker.status ?? "unavailable";
  const active = isScanActive(controller.state);
  const progress = controller.progress;
  const percent = progress?.total
    ? Math.min(100, Math.round((progress.current / progress.total) * 100))
    : 0;
  const stateLabel = {
    idle: "待开始",
    running: "识别中",
    pausing: "即将暂停",
    paused: "已暂停",
    cancelling: "正在终止",
    completed: "已完成",
    cancelled: "已取消",
    failed: "失败",
  }[controller.state];
  const startDisabled = !library || workerStatus !== "healthy" || active;
  const startTitle = !library
    ? "请先选择图库"
    : workerStatus !== "healthy"
      ? "请先连接内置推理引擎"
      : active
        ? "当前已有扫描任务"
        : "扫描所选文件夹中的全部图片";

  return (
    <div className="page-stack workspace-page">
      <section
        className="workspace-overview workspace-dashboard-overview"
        aria-labelledby="workspace-overview-title"
      >
        <div className="workspace-overview-copy">
          <div className="tag">
            <WandSparkles size={14} />
            本地工作流 · 安全可复核
          </div>
          <h2 id="workspace-overview-title">图片工作台</h2>
          <p>
            选择图库后，从角色识别或相似计算开始。结果只在本机处理，原图不会被自动移动、改名或删除。
          </p>
          <button
            className="ghost-button"
            onClick={onChooseLibrary}
            disabled={active}
            title={active ? "扫描进行中，请先终止任务或等待完成" : undefined}
          >
            <FolderOpen size={16} />
            {library ? "切换图库" : "选择图库"}
          </button>
        </div>
        <div className="workspace-overview-stats" aria-label="工作台状态">
          <div>
            <span>当前图库</span>
            <strong>{library?.display_name ?? "尚未选择"}</strong>
            <small>
              {library?.path ? normalizeDisplayPath(library.path) : "选择目录后开始"}
            </small>
          </div>
          <div>
            <span>推理引擎</span>
            <strong>{loading ? "检查中" : workerLabel(workerStatus)}</strong>
            <small>{status?.worker.message ?? "等待桌面核心响应"}</small>
          </div>
          <div>
            <span>数据库</span>
            <strong>
              {loading
                ? "检查中"
                : status?.database.status === "ready"
                  ? "已就绪"
                  : "需检查"}
            </strong>
            <small>{status?.database.message ?? "等待桌面核心响应"}</small>
          </div>
        </div>
      </section>

      <section
        className="workspace-task-grid"
        aria-labelledby="workspace-quick-start-title"
      >
        <div className="workspace-scan-control panel-card">
          <div className="panel-header">
            <div>
              <div className="eyebrow">ROLE RECOGNITION</div>
              <h3 id="workspace-quick-start-title">角色识别</h3>
            </div>
            <span
              className={`soft-badge ${
                controller.state === "failed"
                  ? "orange"
                  : controller.state === "completed"
                    ? "green"
                    : "blue"
              }`}
            >
              {stateLabel}
            </span>
          </div>
          <div className="workspace-scan-body">
            <div className="workspace-scan-status">
              <strong>
                {active && progress
                  ? progress.message || "正在处理图片"
                  : controller.state === "completed"
                    ? "最近一次扫描已完成"
                    : controller.state === "cancelled"
                      ? "扫描已取消，已保留返回结果"
                      : controller.state === "failed"
                        ? controller.error ?? "扫描失败，请检查内置推理引擎"
                        : "选择图库后开始真实识别"}
              </strong>
              <small>
                {active && progress?.path
                  ? progress.path
                  : library?.path
                    ? normalizeDisplayPath(library.path)
                    : "尚未选择图库"}
              </small>
              {active && (
                <div className="workspace-progress-inline">
                  <div className="progress-track">
                    <div className="progress-bar" style={{ width: `${percent}%` }} />
                  </div>
                  <span>
                    {progress?.current ?? 0} / {progress?.total || "—"}{" "}
                    {progress?.total ? `（${percent}%）` : ""}
                  </span>
                </div>
              )}
            </div>
            <div className="workspace-scan-actions">
              <button
                className="primary-button"
                onClick={() => void controller.start()}
                disabled={startDisabled}
                title={startTitle}
              >
                <Play size={15} />
                {controller.state === "completed" ||
                controller.state === "cancelled" ||
                controller.state === "failed"
                  ? "再次扫描"
                  : "开始扫描"}
              </button>
              {(controller.state === "running" || controller.state === "pausing") && (
                <button
                  className="ghost-button"
                  onClick={() => void controller.pause()}
                  disabled={controller.state !== "running"}
                >
                  <Clock3 size={15} />
                  {controller.state === "pausing" ? "等待暂停…" : "暂停"}
                </button>
              )}
              {controller.state === "paused" && (
                <button
                  className="ghost-button"
                  onClick={() => void controller.resume()}
                >
                  <Play size={15} />
                  继续
                </button>
              )}
              {active && (
                <button
                  className="ghost-button danger-button"
                  onClick={() => void controller.cancel()}
                  disabled={controller.state === "cancelling"}
                >
                  <X size={15} />
                  {controller.state === "cancelling" ? "正在终止…" : "终止"}
                </button>
              )}
              <button
                className="ghost-button"
                onClick={() => onNavigate("/recognition")}
                disabled={!controller.latestScan && !recentScan}
                title={
                  !controller.latestScan && !recentScan
                    ? "完成一次角色识别扫描后可查看详细结果"
                    : "查看角色识别详细结果"
                }
              >
                查看详细结果
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
          {controller.error && (
            <div className="workbench-notice error" role="alert">
              <AlertCircle size={14} />
              {controller.error}
            </div>
          )}
        </div>

        <button
          className="quick-start-card similarity"
          onClick={() => onNavigate("/similarity")}
        >
          <span className="quick-start-icon">
            <ScanSearch size={19} />
          </span>
          <span>
            <strong>相似图片分析</strong>
            <small>感知哈希与特征聚类 · 自动去重治理</small>
          </span>
          <ChevronRight size={17} />
        </button>
      </section>

      <section
        className="workspace-recent panel-card"
        aria-labelledby="workspace-recent-title"
      >
        <div className="panel-header">
          <div>
            <div className="eyebrow">RECENT ACTIVITY</div>
            <h3 id="workspace-recent-title">最近一次结果</h3>
          </div>
          {recentScan && (
            <div className="panel-header-actions">
              <span className="soft-badge green">已完成</span>
              <button
                type="button"
                className="ghost-button danger-button compact-btn"
                onClick={() => {
                  if (globalThis.confirm("确定删除最近扫描记录吗？")) {
                    controller.deleteScan();
                  }
                }}
                title="删除此扫描结果"
              >
                <Trash2 size={12} /> 清除
              </button>
            </div>
          )}
        </div>
        {recentScan ? (
          <div className="recent-scan-summary">
            <div>
              <span>图库</span>
              <strong>{normalizeDisplayPath(recentScan.directory)}</strong>
            </div>
            <div>
              <span>处理进度</span>
              <strong>
                {recentScan.processed} / {recentScan.totalDiscovered} 张
              </strong>
            </div>
            <div>
              <span>识别结果</span>
              <strong>
                {recentScan.resultCount} 张 · {recentScan.personCount} 个人物
              </strong>
            </div>
            <div>
              <span>使用的模型</span>
              <strong>
                {recentScan.modelName ??
                  (recentScan.modelVersion
                    ? `个人模型 v${recentScan.modelVersion}`
                    : "通用基础模型")}
              </strong>
            </div>
            <button
              className="ghost-button"
              onClick={() => onNavigate("/recognition")}
            >
              查看详细结果
              <ChevronRight size={14} />
            </button>
          </div>
        ) : (
          <div className="workspace-recent-empty">
            <Clock3 size={18} />
            <span>
              还没有已完成的扫描任务。在角色识别中开始扫描后，结果会持久保存在本机。
            </span>
          </div>
        )}
      </section>
    </div>
  );
}
