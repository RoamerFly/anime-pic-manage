import React, { type ReactNode } from "react";
import {
  BrainCircuit,
  Check,
  ChevronRight,
  Edit3,
  Info,
  PlusCircle,
  Redo2,
  Save,
  Trash2,
  Undo2,
  UsersRound,
  X,
  Zap,
} from "lucide-react";
import type { AnnotationSource, CharacterIdentity, NormalizedBoundingBox } from "@anime-pic-manage/shared-types";

export type ResultFilter = "all" | "high" | "review" | "unknown";
export type PersonStatus = "high" | "review" | "unknown";

export interface Candidate {
  label: string;
  score: number;
  source?: string | null;
  baseScore?: number | null;
  personalScore?: number | null;
  sampleCount?: number | null;
  reason?: string | null;
}

export interface InspectorPerson {
  id: string;
  title: string;
  confidence: number;
  status: PersonStatus;
  candidates: Candidate[];
  margin?: number;
  consistency?: number;
  reason?: string | null;
  personIndex: number;
}

export interface EditableAnnotation {
  clientId: string;
  personId?: string;
  id?: string;
  label_name: string;
  identity_id?: string | null;
  bbox: NormalizedBoundingBox;
  source: AnnotationSource;
  manually_adjusted: boolean;
}

interface UnifiedInspectorProps {
  people: InspectorPerson[];
  activePersonId: string | null;
  onSelectPerson: (id: string) => void;
  filter: ResultFilter;
  onChangeFilter: (f: ResultFilter) => void;
  // 编辑相关
  editorMode: boolean;
  onToggleEditor: () => void;
  onAddNewBox: () => void;
  activeAnnotation: EditableAnnotation | null;
  labelDraft: string;
  onChangeLabelDraft: (val: string) => void;
  onCommitLabel: (label?: string) => void;
  onAssignCandidate?: (personId: string, label: string) => void;
  onRemoveAnnotation: (id: string) => void;
  identities: CharacterIdentity[];
  // 撤销重做保存
  canUndo: boolean;
  canRedo: boolean;
  dirty: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onCancelEdits: () => void;
  onSaveAnnotations: () => void;
  saveLoading: boolean;
  // 个人模型微调状态入口
  personalModelReady?: boolean;
  verifiedSampleCount?: number;
  eligibleClassCount?: number;
  totalClassCount?: number;
  onOpenPersonalModel?: () => void;
  onQuickTrain?: () => void;
  trainingLoading?: boolean;
  footer?: ReactNode;
}

export function UnifiedInspector({
  people,
  activePersonId,
  onSelectPerson,
  filter,
  onChangeFilter,
  editorMode,
  onToggleEditor,
  onAddNewBox,
  activeAnnotation,
  labelDraft,
  onChangeLabelDraft,
  onCommitLabel,
  onAssignCandidate,
  onRemoveAnnotation,
  identities,
  canUndo,
  canRedo,
  dirty,
  onUndo,
  onRedo,
  onCancelEdits,
  onSaveAnnotations,
  saveLoading,
  personalModelReady,
  verifiedSampleCount,
  eligibleClassCount,
  totalClassCount,
  onOpenPersonalModel,
  onQuickTrain,
  trainingLoading,
  footer,
}: UnifiedInspectorProps) {
  const counts = {
    all: people.length,
    high: people.filter((p) => p.status === "high").length,
    review: people.filter((p) => p.status === "review").length,
    unknown: people.filter((p) => p.status === "unknown").length,
  };

  const visiblePeople = filter === "all" ? people : people.filter((p) => p.status === filter);
  const activePerson = people.find((p) => p.id === activePersonId) || visiblePeople[0];

  const formatScore = (score: number | undefined) => `${Math.round((score ?? 0) * 100)}%`;
  // The candidate that matches the currently assigned label is the effective
  // choice, so it is highlighted instead of the removed confidence ring card.
  const assignedLabel = (activeAnnotation?.label_name ?? labelDraft ?? "")
    .trim()
    .toLocaleLowerCase();

  return (
    <aside className="unified-inspector panel-card" aria-label="属性与复核检查器">
      <div className="unified-inspector-content">
      {/* 顶部标题与模式切换 */}
      <div className="inspector-top-header">
        <div>
          <div className="eyebrow">INSPECTOR & REVIEW</div>
          <h3>识别检查与标注</h3>
        </div>
        <div className="inspector-top-actions">
          <button
            type="button"
            className={`ghost-button annotation-toggle ${editorMode ? "active" : ""}`}
            onClick={onToggleEditor}
            title={editorMode ? "退出标注编辑模式" : "进入画框与分类编辑"}
          >
            {editorMode ? <><X size={13} /> 退出编辑</> : <><Edit3 size={13} /> 标注模式</>}
          </button>
        </div>
      </div>

      {/* 人物过滤 Tab */}
      <div className="filter-tabs" role="tablist" aria-label="人物状态过滤">
        <button
          type="button"
          className={`filter-tab ${filter === "all" ? "active" : ""}`}
          onClick={() => onChangeFilter("all")}
        >
          全部 <span>{counts.all}</span>
        </button>
        <button
          type="button"
          className={`filter-tab ${filter === "high" ? "active" : ""}`}
          onClick={() => onChangeFilter("high")}
        >
          高置信 <span>{counts.high}</span>
        </button>
        <button
          type="button"
          className={`filter-tab ${filter === "review" ? "active" : ""}`}
          onClick={() => onChangeFilter("review")}
        >
          待复核 <span>{counts.review}</span>
        </button>
        <button
          type="button"
          className={`filter-tab ${filter === "unknown" ? "active" : ""}`}
          onClick={() => onChangeFilter("unknown")}
        >
          未识别 <span>{counts.unknown}</span>
        </button>
      </div>

      {/* 人物选择横向滑块/列表 */}
      {visiblePeople.length > 0 ? (
        <div className="inspector-people-pills" role="tablist">
          {visiblePeople.map((person) => {
            const isSelected = person.id === activePerson?.id;
            return (
              <button
                key={person.id}
                type="button"
                className={`inspector-person-pill ${person.status} ${isSelected ? "selected" : ""}`}
                onClick={() => onSelectPerson(person.id)}
              >
                <span className={`person-pill-dot ${person.status}`} />
                <span className="person-pill-name">{person.title}</span>
                <span className="person-pill-score">{formatScore(person.confidence)}</span>
              </button>
            );
          })}
          {editorMode && (
            <button
              type="button"
              className="inspector-person-pill add-new"
              onClick={onAddNewBox}
              title="新增人物识别框 (快捷键 N)"
            >
              <PlusCircle size={13} />
              <span>新加框</span>
            </button>
          )}
        </div>
      ) : (
        <div className="inspector-empty-hint">当前筛选无对应人物</div>
      )}

      {/* 当前选定人物的识别详情 */}
      {activePerson && (
        <section className="inspector-section" aria-label="当前人物详情">
          {/* Top-5 候选紧凑网格 */}
          <div className="inspector-candidate-section">
            <div className="inspector-subheading">
              <span>TOP-5 候选</span>
              <small className="muted">快捷键 1~5 一键指派</small>
            </div>
            <div className="inspector-candidate-list">
              {activePerson.candidates.slice(0, 5).map((candidate, idx) => {
                const isAssigned =
                  assignedLabel.length > 0 &&
                  candidate.label.trim().toLocaleLowerCase() === assignedLabel;
                return (
                  <button
                    key={`${candidate.label}-${idx}`}
                    type="button"
                    className={`inspector-candidate-row${isAssigned ? " assigned" : ""}`}
                    onClick={() => {
                      if (onAssignCandidate) {
                        onAssignCandidate(activePerson.id, candidate.label);
                      } else {
                        onChangeLabelDraft(candidate.label);
                        onCommitLabel(candidate.label);
                      }
                    }}
                    title={
                      isAssigned
                        ? `当前已指派为: ${candidate.label}`
                        : `点击将当前识别框指定为: ${candidate.label}`
                    }
                  >
                    <span className="candidate-name-text" title={candidate.label}>
                      {isAssigned && <Check size={11} />}
                      {candidate.label}
                    </span>
                    <strong className="candidate-score">{formatScore(candidate.score)}</strong>
                  </button>
                );
              })}
            </div>
          </div>
        </section>
      )}

      {/* 标注编辑与改名表单 (在同一面板无需跳到底部) */}
      {editorMode && (
        <section className="inspector-edit-box">
          <div className="inspector-edit-header">
            <div className="eyebrow">LABEL ASSIGNMENT</div>
            <span className="soft-badge blue">
              {activeAnnotation?.source === "manual" ? "人工标注" : "模型草稿"}
            </span>
          </div>

          <div className="inspector-field-group">
            <label className="inspector-field">
              <span>分类名称</span>
              <input
                value={labelDraft}
                list="inspector-identity-options"
                onChange={(e) => onChangeLabelDraft(e.target.value)}
                onBlur={() => onCommitLabel()}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    onCommitLabel();
                    e.currentTarget.blur();
                  }
                }}
                placeholder="输入角色名或点击上方候选"
              />
              <datalist id="inspector-identity-options">
                {identities.map((id) => (
                  <option key={id.id} value={id.display_name}>
                    {id.sample_count !== undefined ? `${id.sample_count} 个样本` : ""}
                  </option>
                ))}
              </datalist>
            </label>
          </div>

          {activeAnnotation && (
            <div className="inspector-bbox-info">
              <span>
                位置: X {Math.round(activeAnnotation.bbox.x * 100)}% / Y {Math.round(activeAnnotation.bbox.y * 100)}%
              </span>
              <span>
                尺寸: {Math.round(activeAnnotation.bbox.width * 100)}% × {Math.round(activeAnnotation.bbox.height * 100)}%
              </span>
            </div>
          )}

          {/* 快捷操作栏 */}
          <div className="inspector-edit-actions">
            <button
              type="button"
              className="ghost-button"
              onClick={onUndo}
              disabled={!canUndo}
              title="撤销 (Ctrl+Z)"
            >
              <Undo2 size={13} />
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={onRedo}
              disabled={!canRedo}
              title="重做 (Ctrl+Y)"
            >
              <Redo2 size={13} />
            </button>
            {activeAnnotation && (
              <button
                type="button"
                className="ghost-button danger-button"
                onClick={() => onRemoveAnnotation(activeAnnotation.clientId)}
                title="删除当前选框 (Delete)"
              >
                <Trash2 size={13} />
                删除框
              </button>
            )}
            <button
              type="button"
              className="primary-button inspector-save-btn"
              onClick={onSaveAnnotations}
              disabled={!dirty || saveLoading}
              title="保存当前图片调整的标注框与分类 (Ctrl+S)"
            >
              <Save size={13} />
              保存
            </button>
          </div>
        </section>
      )}

      {/* 底部个人模型状态徽章入口与独立再训练按钮 */}
      <div className="inspector-footer-model">
        <div className="inspector-model-badge">
          <BrainCircuit size={15} />
          <div>
            <strong>个人学习模型</strong>
            <small>
              已保存 {verifiedSampleCount ?? 0} 个调整样本
              {eligibleClassCount !== undefined && totalClassCount !== undefined
                ? ` · ${eligibleClassCount}/${totalClassCount} 角色达标`
                : personalModelReady
                  ? " · 可以再训练"
                  : " · 样本累积中"}
            </small>
          </div>
        </div>
        <div className="inspector-footer-actions">
          {onQuickTrain && (
            <button
              type="button"
              className={`inspector-quick-train-btn ${personalModelReady ? "primary-button" : "ghost-button"}`}
              onClick={onQuickTrain}
              disabled={trainingLoading || !personalModelReady}
              title={
                personalModelReady
                  ? "将目前保存的所有用户调整结果一次性送入再训练并应用新模型"
                  : `已保存 ${verifiedSampleCount ?? 0} 个样本。再训练需至少 2 个角色达标（各至少 3 张图），当前已达标 ${eligibleClassCount ?? 0} 个角色。`
              }
            >
              <Zap size={12} />
              {trainingLoading ? "再训练中..." : "再训练"}
            </button>
          )}
          {onOpenPersonalModel && (
            <button
              type="button"
              className="ghost-button inspector-model-btn"
              onClick={onOpenPersonalModel}
              title="查看与管理全部模型版本"
            >
              版本 <ChevronRight size={13} />
            </button>
          )}
        </div>
      </div>
      </div>
      {footer && <div className="inspector-filmstrip">{footer}</div>}
    </aside>
  );
}
