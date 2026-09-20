import type { PersonalModelVersion, PersonalModelVersionStatus, TrainingClassProgress } from "@anime-pic-manage/shared-types";

/** A UI-friendly projection of the canonical nested core status. */
export interface PersonalTrainingView {
  status: string;
  message: string;
  verified_sample_count: number;
  verified_class_count: number;
  eligible_class_count: number;
  class_count: number;
  ready: boolean;
  readiness_message: string;
  active_version: number | null;
  versions: PersonalModelVersion[];
  settings: Record<string, unknown>;
  class_progress: TrainingClassProgress[];
  updated_at?: string | null;
  last_error?: string | null;
}

/** The compatibility row installed by the database is not a trained model. */
export const BASE_MODEL_UNTRAINED_LABEL = "基础模型（未训练）";

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function hasPersonalModelArtifact(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return false;
}

/**
 * A version is usable by the recognition pipeline when it has a valid artifact,
 * has not failed, and is eligible for activation. Any version meeting these
 * criteria (draft, active, or archived) can be selected for scanning.
 */
export function isUsablePersonalModelVersion(version: PersonalModelVersion | null | undefined): version is PersonalModelVersion {
  return Boolean(
    version
      && version.status !== "failed"
      && version.eligible_for_activation !== false
      && hasPersonalModelArtifact(version.artifact),
  );
}

/** Identify the database's builtin compatibility row for explicit UI display. */
export function isBuiltinCompatibilityVersion(version: PersonalModelVersion): boolean {
  return version.id?.toLowerCase().startsWith("builtin-") === true
    || (
      version.status === "active"
      && version.sample_count === 0
      && version.class_count === 0
      && version.eligible_for_activation !== true
      && !hasPersonalModelArtifact(version.artifact)
    );
}

function readVersionNumber(value: unknown): number | null {
  const number = finiteNumber(value);
  if (number !== null) return Math.trunc(number);
  if (typeof value !== "string") return null;
  const match = value.match(/(?:personal-v|personal-|v)?(\d+)$/i);
  return match ? Number.parseInt(match[1], 10) : null;
}

export function formatReadinessMessage(ready: boolean, reasons: string[], fallback: string): string {
  const concreteReasons = reasons
    .map((reason) => reason.trim())
    .filter((reason) => reason.length > 0);
  if (!ready && concreteReasons.length > 0) return concreteReasons.join("；");
  return fallback.trim() || (ready ? "样本已满足训练条件。" : "样本尚未满足训练条件。");
}

/**
 * Converts the Rust status shape into the small projection consumed by the page.
 * It also accepts the earlier flat status shape so an upgraded UI can talk to an
 * older local core while it is being replaced.
 */
export function readPersonalTrainingStatus(value: unknown): PersonalTrainingView | null {
  const root = asRecord(value);
  if (!root) return null;
  const rootPayload = asRecord(root.payload);
  const rootData = asRecord(root.data);
  const rootStatus = asRecord(root.status);
  const raw = rootStatus ?? rootData ?? rootPayload ?? root;
  const readiness = asRecord(raw.readiness);
  const versionsValue = Array.isArray(raw.versions) ? raw.versions : [];
  const ready = typeof readiness?.ready === "boolean" ? readiness.ready : typeof raw.ready === "boolean" ? raw.ready : null;
  if (ready === null || !Array.isArray(raw.versions)) return null;

  const verifiedSamples = finiteNumber(readiness?.verified_sample_count) ?? finiteNumber(raw.verified_sample_count) ?? 0;
  const verifiedClasses = finiteNumber(readiness?.verified_class_count) ?? finiteNumber(raw.class_count) ?? 0;
  const eligibleClasses = finiteNumber(readiness?.eligible_class_count) ?? verifiedClasses;
  const reasons = Array.isArray(readiness?.reasons) ? readiness.reasons.filter((reason): reason is string => typeof reason === "string") : [];
  const message = typeof raw.message === "string" ? raw.message : typeof raw.readiness_message === "string" ? raw.readiness_message : reasons.join("；");
  const readinessMessage = formatReadinessMessage(ready, reasons, message);
  const activeVersion = readVersionNumber(raw.active_version);
  const versions: PersonalModelVersion[] = versionsValue.flatMap((item): PersonalModelVersion[] => {
    const version = asRecord(item);
    const number = readVersionNumber(version?.version);
    if (!version || number === null) return [];
    const metricsRecord = asRecord(version.metrics);
    const metrics: PersonalModelVersion["metrics"] = metricsRecord
      ? Object.fromEntries(Object.entries(metricsRecord).filter(([, metric]) => metric === null || typeof metric === "number" || typeof metric === "string")) as PersonalModelVersion["metrics"]
      : null;
    return [{
      id: typeof version.id === "string" ? version.id : undefined,
      version: number,
      status: typeof version.status === "string" ? version.status : "draft",
      algorithm: typeof version.algorithm === "string" ? version.algorithm : "未记录",
      sample_count: finiteNumber(version.sample_count) ?? 0,
      class_count: finiteNumber(version.class_count) ?? 0,
      artifact: version.artifact ?? null,
      metrics,
      warnings: Array.isArray(version.warnings) ? version.warnings.filter((warning): warning is string => typeof warning === "string") : null,
      eligible_for_activation: version.eligible_for_activation === true,
      error_code: typeof version.error_code === "string" ? version.error_code : null,
      error_message: typeof version.error_message === "string" ? version.error_message : null,
      created_at: typeof version.created_at === "string" ? version.created_at : "",
      updated_at: typeof version.updated_at === "string" ? version.updated_at : null,
      activated_at: typeof version.activated_at === "string" ? version.activated_at : null,
      last_error: typeof version.last_error === "string" ? version.last_error : null,
    }];
  });
  const classProgressRaw = Array.isArray(readiness?.class_progress)
    ? readiness.class_progress
    : Array.isArray(raw.class_progress)
      ? raw.class_progress
      : [];
  const classProgress: TrainingClassProgress[] = classProgressRaw.flatMap((item) => {
    const rec = asRecord(item);
    if (!rec || typeof rec.identity_id !== "string") return [];
    return [{
      identity_id: rec.identity_id,
      display_name: typeof rec.display_name === "string" ? rec.display_name : rec.identity_id,
      sample_count: finiteNumber(rec.sample_count) ?? 0,
      min_required: finiteNumber(rec.min_required) ?? 0,
      eligible: rec.eligible === true,
    }];
  });

  return {
    status: typeof raw.status === "string" ? raw.status : "unknown",
    message,
    verified_sample_count: Math.max(0, verifiedSamples),
    verified_class_count: Math.max(0, verifiedClasses),
    eligible_class_count: Math.max(0, eligibleClasses),
    class_count: Math.max(0, verifiedClasses),
    ready,
    readiness_message: readinessMessage,
    active_version: activeVersion,
    versions,
    settings: asRecord(raw.settings) ?? {},
    class_progress: classProgress,
    updated_at: typeof raw.updated_at === "string" ? raw.updated_at : null,
    last_error: typeof raw.last_error === "string" ? raw.last_error : null,
  };
}

export function findActivePersonalModelVersion(
  view: Pick<PersonalTrainingView, "active_version" | "versions">,
): PersonalModelVersion | null {
  const activeById = view.versions.find((version) =>
    version.version === view.active_version && isUsablePersonalModelVersion(version),
  );
  return activeById ?? view.versions.find((version) => isUsablePersonalModelVersion(version)) ?? null;
}

export function formatPersonalVersion(version: number | null | undefined): string {
  return version === null || version === undefined ? "尚未激活" : `personal-v${version}`;
}

export function formatActivePersonalModel(version: PersonalModelVersion | null | undefined): string {
  return isUsablePersonalModelVersion(version) ? formatPersonalVersion(version.version) : BASE_MODEL_UNTRAINED_LABEL;
}

export function personalVersionStatusLabel(status: PersonalModelVersionStatus): string {
  const labels: Record<string, string> = { draft: "草稿", active: "已激活", archived: "已归档", failed: "失败" };
  return labels[status] ?? status;
}

export function personalVersionStatusTone(status: PersonalModelVersionStatus): string {
  return status === "active" ? "green" : status === "failed" ? "red" : status === "draft" ? "blue" : "gray";
}
