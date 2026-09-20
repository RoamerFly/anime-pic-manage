import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Archive,
  Check,
  CircleCheck,
  Clock3,
  LoaderCircle,
  Play,
  RefreshCw,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import type {
  ActivatePersonalModelPayload,
  DeletePersonalModelPayload,
  PersonalTrainingStatus,
} from "@anime-pic-manage/shared-types";
import { invokeCore } from "../lib/tauri";
import {
  formatActivePersonalModel,
  formatPersonalVersion,
  findActivePersonalModelVersion,
  hasPersonalModelArtifact,
  isBuiltinCompatibilityVersion,
  isUsablePersonalModelVersion,
  personalVersionStatusLabel,
  personalVersionStatusTone,
  readPersonalTrainingStatus,
  type PersonalTrainingView,
} from "../lib/personal-model";

const emptyPersonalTrainingStatus: PersonalTrainingView = {
  status: "idle",
  message: "正在读取个人模型状态…",
  verified_sample_count: 0,
  verified_class_count: 0,
  eligible_class_count: 0,
  class_count: 0,
  ready: false,
  readiness_message: "正在读取个人模型状态…",
  active_version: null,
  versions: [],
  settings: {},
  class_progress: [],
  updated_at: null,
  last_error: null,
};

function formatMetricValue(
  key: string,
  value: number | string | null | undefined,
): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
  const percentage =
    /accuracy|precision|recall|f1|score|置信|准确|精确|召回/i.test(key) ||
    (value >= 0 && value <= 1);
  return percentage
    ? `${Math.round(value * 1000) / 10}%`
    : `${Math.round(value * 100) / 100}`;
}

function formatTrainingTime(value: string): string {
  if (!value) return "时间未知";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
}

export function PersonalModelPage({
  onSwitchToScan,
}: {
  onSwitchToScan?: (version?: number) => void;
}) {
  const [status, setStatus] = useState<PersonalTrainingView | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [operation, setOperation] = useState<
    "training" | "activating" | "deleting" | null
  >(null);
  const [operatingVersion, setOperatingVersion] = useState<number | null>(null);
  const [trainMode, setTrainMode] = useState<"new" | "overwrite">("new");
  const [overwriteTargetVersion, setOverwriteTargetVersion] = useState<
    number | null
  >(null);

  const overwritableVersions = useMemo(() => {
    return (status?.versions ?? []).filter(
      (v) => v.version >= 2 && !isBuiltinCompatibilityVersion(v),
    );
  }, [status?.versions]);

  useEffect(() => {
    if (overwritableVersions.length > 0 && overwriteTargetVersion === null) {
      setOverwriteTargetVersion(overwritableVersions[0].version);
    }
  }, [overwritableVersions, overwriteTargetVersion]);

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    setError(null);
    const response = await invokeCore<PersonalTrainingStatus>(
      "get_personal_training_status",
      "personal.training.status",
    );
    if (response.error || !response.payload) {
      setError(
        response.error?.message ??
          "无法读取个人模型状态。请确认桌面核心已连接。",
      );
    } else {
      const next = readPersonalTrainingStatus(response.payload);
      if (next) setStatus(next);
      else setError("个人模型状态格式不完整，暂时无法展示版本历史。");
    }
    if (!quiet) setLoading(false);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runTraining = async () => {
    if (operation) return;
    const isOverwrite = trainMode === "overwrite";
    if (isOverwrite && (!overwriteTargetVersion || overwriteTargetVersion < 2)) {
      setError(
        "请选择要覆盖的个人模型版本。系统基础模型(v1)受保护，无法覆盖。",
      );
      return;
    }
    setOperation("training");
    setError(null);
    setNotice(null);
    const response = await invokeCore<PersonalTrainingStatus>(
      "train_personal_model",
      "personal.training.start",
      {
        autoActivate: true,
        auto_activate: true,
        overwriteVersion: isOverwrite ? overwriteTargetVersion : null,
        overwrite_version: isOverwrite ? overwriteTargetVersion : null,
      },
    );
    const next = response.payload
      ? readPersonalTrainingStatus(response.payload)
      : null;
    if (response.error) {
      setError(response.error.message || "个人模型训练失败。");
    } else {
      if (next) setStatus(next);
      if (isOverwrite) {
        setNotice(
          `训练已完成！已就地覆盖更新个人模型 personal-v${overwriteTargetVersion} 并保持激活。`,
        );
      } else {
        setNotice(
          `训练已完成！生成了新的个人模型 personal-v${next?.active_version ?? ""} 并已设为默认激活。原有历史版本完整保留在列表中。`,
        );
      }
      await refresh(true);
    }
    setOperation(null);
  };

  const runActivate = async (version: number) => {
    if (operation) return;
    setOperation("activating");
    setOperatingVersion(version);
    setError(null);
    setNotice(null);
    const response = await invokeCore<PersonalTrainingStatus>(
      "activate_personal_model",
      "personal.training.activate",
      { version } satisfies ActivatePersonalModelPayload,
    );
    const next = response.payload
      ? readPersonalTrainingStatus(response.payload)
      : null;
    if (response.error) {
      setError(response.error.message || "激活版本失败。");
    } else {
      if (next) setStatus(next);
      setNotice(`版本 personal-v${version} 已设为当前默认激活版本。`);
      await refresh(true);
    }
    setOperation(null);
    setOperatingVersion(null);
  };

  const runDelete = async (version: number) => {
    if (operation) return;
    if (
      !globalThis.confirm(
        `确定要永久删除模型 personal-v${version} 吗？删除后不可恢复。`,
      )
    )
      return;
    setOperation("deleting");
    setOperatingVersion(version);
    setError(null);
    setNotice(null);
    const response = await invokeCore<PersonalTrainingStatus>(
      "delete_personal_model",
      "personal.training.delete",
      { version } satisfies DeletePersonalModelPayload,
    );
    const next = response.payload
      ? readPersonalTrainingStatus(response.payload)
      : null;
    if (response.error) {
      setError(response.error.message || "删除模型失败。");
    } else {
      if (next) setStatus(next);
      setNotice(`已成功删除模型 personal-v${version}。`);
      await refresh(true);
    }
    setOperation(null);
    setOperatingVersion(null);
  };

  const current = status ?? emptyPersonalTrainingStatus;
  const activeVersion = findActivePersonalModelVersion(current);

  return (
    <div className="page-stack personal-model-page">
      {error && (
        <div className="workbench-notice error" role="alert">
          <AlertCircle size={15} />
          <span>{error}</span>
          <button aria-label="关闭错误" onClick={() => setError(null)}>
            <X size={15} />
          </button>
        </div>
      )}
      {notice && (
        <div className="workbench-notice" role="status">
          <CircleCheck size={15} />
          <span>{notice}</span>
          <button aria-label="关闭提示" onClick={() => setNotice(null)}>
            <X size={15} />
          </button>
        </div>
      )}

      <section className="personal-model-summary panel-card" aria-label="个人模型概览">
        <div className="panel-header">
          <div>
            <div className="eyebrow">PERSONAL MODEL OVERVIEW</div>
            <h3>训练概览</h3>
          </div>
          <button
            className="icon-button"
            onClick={() => void refresh()}
            disabled={loading || operation !== null}
            title="刷新状态"
            aria-label="刷新状态"
          >
            <RefreshCw size={16} className={loading ? "spin" : ""} />
          </button>
        </div>
        <div className="personal-stats">
          <div>
            <span>已验证样本</span>
            <strong>
              {loading && !status ? (
                <LoaderCircle size={18} className="spin" />
              ) : (
                current.verified_sample_count
              )}
            </strong>
            <small>可用于学习的标注框</small>
          </div>
          <div>
            <span>分类数量</span>
            <strong>
              {loading && !status ? (
                <LoaderCircle size={18} className="spin" />
              ) : (
                current.verified_class_count
              )}
            </strong>
            <small>已确认 {current.eligible_class_count} 个满足条件</small>
          </div>
          <div>
            <span>训练就绪</span>
            <strong className={current.ready ? "text-green" : "text-orange"}>
              {current.ready ? "可以训练新版本" : "尚未就绪"}
            </strong>
            <small>{current.readiness_message}</small>
          </div>
          <div>
            <span>当前默认激活版本</span>
            <strong className="version-value">
              {formatActivePersonalModel(activeVersion)}
            </strong>
            <small>
              {activeVersion ? "识别默认使用的版本" : "当前使用基础通用模型"}
            </small>
          </div>
        </div>
      </section>

      {/* 角色训练样本达标明细进度面板 */}
      <section className="personal-model-progress panel-card" aria-label="角色训练样本进度">
        <div className="panel-header">
          <div>
            <div className="eyebrow">CLASS READINESS</div>
            <h3>各角色样本达标明细</h3>
          </div>
          <span className={`soft-badge ${current.ready ? "green" : "orange"}`}>
            {current.eligible_class_count} /{" "}
            {Math.max(2, current.verified_class_count)} 角色达标（至少需2个）
          </span>
        </div>
        <p className="panel-intro-text">
          每个角色需至少累积 3 个已标注样本，且至少有 2 个角色达标，系统即可训练出高精度的轻量向量分类器。
        </p>
        {current.class_progress.length === 0 ? (
          <div className="personal-progress-empty">
            <small>
              尚未提取到有效特征样本。请在角色识别界面调整并保存标注以累积学习样本。
            </small>
          </div>
        ) : (
          <div className="personal-class-grid">
            {current.class_progress.map((item) => {
              const percent = Math.min(
                100,
                Math.round((item.sample_count / item.min_required) * 100),
              );
              return (
                <div
                  className={`personal-class-card ${
                    item.eligible ? "eligible" : "pending"
                  }`}
                  key={item.identity_id}
                >
                  <div className="personal-class-header">
                    <strong>{item.display_name}</strong>
                    {item.eligible ? (
                      <span className="status-pill green">
                        <Check size={12} /> 达标
                      </span>
                    ) : (
                      <span className="status-pill orange">
                        缺 {Math.max(0, item.min_required - item.sample_count)} 个
                      </span>
                    )}
                  </div>
                  <div className="personal-class-progress-bar">
                    <div
                      className={`personal-class-progress-fill ${
                        item.eligible ? "fill-green" : "fill-orange"
                      }`}
                      style={{ width: `${percent}%` }}
                    />
                  </div>
                  <div className="personal-class-footer">
                    <small>
                      {item.sample_count} / {item.min_required} 样本
                    </small>
                    <small>{percent}%</small>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 训练操作区 */}
      <section className="personal-model-actions panel-card">
        <div>
          <div className="eyebrow">MODEL ACTIONS</div>
          <h3>模型生成与重训策略</h3>
          <p>
            训练时可生成独立新版本，或选择性覆盖您自己训练的已有模型（初始基础模型受保护，不可覆盖）。
          </p>
        </div>

        <div className="train-mode-selector">
          <label
            className={`train-mode-option${
              trainMode === "new" ? " selected" : ""
            }`}
          >
            <input
              type="radio"
              name="trainMode"
              value="new"
              checked={trainMode === "new"}
              onChange={() => setTrainMode("new")}
              disabled={operation !== null}
            />
            <div className="train-mode-info">
              <strong>保存为独立新版本 (默认)</strong>
              <span>
                版本号自增并设为当前激活版本，历史旧版本完整保留在版本库中。
              </span>
            </div>
          </label>

          <label
            className={`train-mode-option${
              trainMode === "overwrite" ? " selected" : ""
            }${overwritableVersions.length === 0 ? " disabled" : ""}`}
          >
            <input
              type="radio"
              name="trainMode"
              value="overwrite"
              checked={trainMode === "overwrite"}
              onChange={() => setTrainMode("overwrite")}
              disabled={overwritableVersions.length === 0 || operation !== null}
            />
            <div className="train-mode-info">
              <strong>覆盖已有个人模型</strong>
              <span>
                {overwritableVersions.length === 0
                  ? "（当前暂无自己训练的个人模型，请先训练首个个人版本）"
                  : "更新所选个人模型的特征权重与指标，不生成新版本号。"}
              </span>
            </div>
          </label>

          {trainMode === "overwrite" && overwritableVersions.length > 0 && (
            <div className="overwrite-target-select">
              <span className="overwrite-label">选择要覆盖的个人模型:</span>
              <select
                value={overwriteTargetVersion ?? ""}
                onChange={(e) => setOverwriteTargetVersion(Number(e.target.value))}
                disabled={operation !== null}
              >
                {overwritableVersions.map((v) => (
                  <option key={v.version} value={v.version}>
                    personal-v{v.version} (
                    {v.status === "active" ? "当前激活 · " : ""}
                    {v.class_count} 个角色 · {v.sample_count} 个样本)
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="personal-action-buttons">
          <button
            className="primary-button"
            onClick={() => void runTraining()}
            disabled={loading || !current.ready || operation !== null}
            title={
              !current.ready
                ? current.readiness_message
                : trainMode === "overwrite"
                  ? `覆盖更新 personal-v${overwriteTargetVersion}`
                  : "使用当前样本训练新版本并自动设为默认激活"
            }
          >
            {operation === "training" ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Sparkles size={15} />
            )}
            {operation === "training"
              ? "训练中…"
              : trainMode === "overwrite"
                ? `覆盖更新 personal-v${overwriteTargetVersion}`
                : "立即训练新版本"}
          </button>
          {onSwitchToScan && (
            <button
              type="button"
              className="ghost-button"
              onClick={() => onSwitchToScan(activeVersion?.version)}
              title="返回角色识别工作台开始扫描"
            >
              <Play size={15} />
              前往工作台扫描
            </button>
          )}
        </div>
      </section>

      {/* 模型版本库 */}
      <section className="personal-model-history panel-card">
        <div className="panel-header">
          <div>
            <div className="eyebrow">VERSION REPOSITORY</div>
            <h3>个人模型版本库</h3>
          </div>
          <span className="soft-badge blue">{current.versions.length} 个版本</span>
        </div>
        {current.versions.length === 0 ? (
          <div className="personal-history-empty">
            <Archive size={23} />
            <strong>还没有个人模型版本</strong>
            <p>
              {current.readiness_message ||
                "先在识别工作台保存人工标注，样本达到条件后即可开始训练新版本。"}
            </p>
          </div>
        ) : (
          <div className="personal-version-list" role="list">
            {current.versions.map((version) => {
              const isCompatibility = isBuiltinCompatibilityVersion(version);
              const isUsable = isUsablePersonalModelVersion(version);
              const isActive = version.status === "active";
              const canActivate =
                !isCompatibility &&
                !isActive &&
                version.eligible_for_activation === true &&
                hasPersonalModelArtifact(version.artifact);
              const canDelete = !isCompatibility && !isActive;
              const statusTone = isCompatibility
                ? "gray"
                : personalVersionStatusTone(version.status);
              const versionLabel = formatPersonalVersion(version.version);
              const statusLabel = isCompatibility
                ? "基础兼容"
                : personalVersionStatusLabel(version.status);
              const versionWarnings = [
                ...(version.warnings ?? []),
                ...(version.error_message ? [version.error_message] : []),
              ];

              return (
                <article
                  className="personal-version-card"
                  key={version.version}
                  role="listitem"
                >
                  <div className="personal-version-heading">
                    <div>
                      <strong>{versionLabel}</strong>
                      <span className={`personal-version-status ${statusTone}`}>
                        <span />
                        {statusLabel}
                      </span>
                    </div>
                    <span className="personal-version-time">
                      <Clock3 size={12} />
                      {formatTrainingTime(version.created_at)}
                    </span>
                  </div>
                  <div className="personal-version-meta">
                    <span>
                      <b>算法</b>
                      {version.algorithm}
                    </span>
                    <span>
                      <b>样本</b>
                      {version.sample_count}
                    </span>
                    <span>
                      <b>分类</b>
                      {version.class_count}
                    </span>
                    {version.metrics &&
                      Object.entries(version.metrics).length > 0 && (
                        <span className="personal-version-metrics">
                          <b>指标</b>
                          {Object.entries(version.metrics)
                            .slice(0, 3)
                            .map(([k, v]) => (
                              <em key={k}>
                                {k} {formatMetricValue(k, v)}
                              </em>
                            ))}
                        </span>
                      )}
                  </div>
                  {versionWarnings.length > 0 && (
                    <div className="personal-version-warnings">
                      <AlertCircle size={13} />
                      <span>{versionWarnings.join("；")}</span>
                    </div>
                  )}
                  <div className="personal-version-actions">
                    <button
                      className="ghost-button"
                      disabled={!canActivate || operation !== null}
                      onClick={() => void runActivate(version.version)}
                      title={
                        isActive
                          ? "当前已是默认激活版本"
                          : canActivate
                            ? `将 ${versionLabel} 设为默认激活版本`
                            : "该版本目前不可激活"
                      }
                      aria-label={`激活版本 ${versionLabel}`}
                    >
                      {operation === "activating" &&
                      operatingVersion === version.version ? (
                        <LoaderCircle size={13} className="spin" />
                      ) : (
                        <CircleCheck size={13} />
                      )}
                      {isCompatibility
                        ? "基础模型"
                        : isActive
                          ? "当前激活"
                          : "设为激活"}
                    </button>
                    {isUsable && onSwitchToScan && (
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => onSwitchToScan(version.version)}
                        title={`返回工作台并使用 ${versionLabel} 进行识别扫描`}
                      >
                        <Play size={13} />
                        使用此版本扫描
                      </button>
                    )}
                    {canDelete && (
                      <button
                        type="button"
                        className="ghost-button danger-button compact-btn"
                        disabled={operation !== null}
                        onClick={() => void runDelete(version.version)}
                        title={`永久删除 ${versionLabel}`}
                      >
                        {operation === "deleting" &&
                        operatingVersion === version.version ? (
                          <LoaderCircle size={12} className="spin" />
                        ) : (
                          <Trash2 size={12} />
                        )}
                        删除
                      </button>
                    )}
                    {isUsable && (
                      <span className="personal-active-note">
                        <Check size={13} />
                        扫描可选
                      </span>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
