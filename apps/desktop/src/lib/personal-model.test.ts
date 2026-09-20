import { describe, expect, it } from "vitest";
import {
  BASE_MODEL_UNTRAINED_LABEL,
  formatActivePersonalModel,
  formatPersonalVersion,
  findActivePersonalModelVersion,
  isBuiltinCompatibilityVersion,
  isUsablePersonalModelVersion,
  readPersonalTrainingStatus,
} from "./personal-model";

describe("personal model status protocol", () => {
  it("parses the nested Rust status shape and keeps numeric versions", () => {
    const parsed = readPersonalTrainingStatus({
      status: "ready",
      message: "已有版本可用。",
      readiness: {
        verified_sample_count: 8,
        verified_class_count: 3,
        eligible_class_count: 2,
        ready: true,
        reasons: [],
        class_progress: [
          { identity_id: "char-1", display_name: "Character 1", sample_count: 5, min_required: 3, eligible: true },
          { identity_id: "char-2", display_name: "Character 2", sample_count: 3, min_required: 3, eligible: true },
          { identity_id: "char-3", display_name: "Character 3", sample_count: 1, min_required: 3, eligible: false },
        ],
      },
      settings: { min_samples_per_class: 3 },
      active_version: 3,
      versions: [
        { id: "model-3", version: 3, status: "active", algorithm: "prototype-centroid", sample_count: 8, class_count: 3, metrics: { accuracy: 0.91 }, warnings: [], eligible_for_activation: false, created_at: "2026-09-03T08:00:00Z" },
        { id: "model-4", version: 4, status: "draft", algorithm: "prototype-centroid", sample_count: 8, class_count: 3, metrics: {}, warnings: ["验证集较小"], eligible_for_activation: true, created_at: "2026-09-03T09:00:00Z" },
        { id: "model-5", version: 5, status: "draft", algorithm: "prototype-centroid", sample_count: 1, class_count: 1, metrics: {}, warnings: [], eligible_for_activation: false, created_at: "2026-09-03T10:00:00Z" },
      ],
      updated_at: "2026-09-03T10:00:00Z",
      last_error: null,
    });

    expect(parsed).not.toBeNull();
    expect(parsed?.verified_sample_count).toBe(8);
    expect(parsed?.verified_class_count).toBe(3);
    expect(parsed?.eligible_class_count).toBe(2);
    expect(parsed?.active_version).toBe(3);
    expect(parsed?.versions.map((version) => version.version)).toEqual([3, 4, 5]);
    expect(parsed?.versions[1].eligible_for_activation).toBe(true);
    expect(parsed?.versions[2].eligible_for_activation).toBe(false);
    expect(formatPersonalVersion(parsed?.active_version)).toBe("personal-v3");
    expect(parsed?.class_progress).toHaveLength(3);
    expect(parsed?.class_progress[0].eligible).toBe(true);
    expect(parsed?.class_progress[2].eligible).toBe(false);
  });

  it("keeps a legacy flat payload usable without allowing an unqualified draft", () => {
    const parsed = readPersonalTrainingStatus({
      verified_sample_count: 3,
      class_count: 1,
      ready: true,
      readiness_message: "旧核心状态",
      active_version: "personal-v2",
      versions: [{ version: "personal-v2", status: "draft", algorithm: "legacy", sample_count: 3, class_count: 1, created_at: "" }],
    });
    expect(parsed?.active_version).toBe(2);
    expect(parsed?.readiness_message).toBe("旧核心状态");
    expect(parsed?.versions[0].eligible_for_activation).toBe(false);
  });

  it("does not treat the builtin compatibility row as an active personal model", () => {
    const parsed = readPersonalTrainingStatus({
      status: "idle",
      message: "已验证样本尚不足以训练个人模型。",
      readiness: {
        verified_sample_count: 0,
        verified_class_count: 0,
        eligible_class_count: 0,
        ready: false,
        reasons: ["至少需要 2 个分类，当前为 0 个", "尚未有可用于训练的分类"],
      },
      active_version: 1,
      versions: [{
        id: "builtin-v1",
        version: 1,
        status: "active",
        algorithm: "prototype_centroid_v1",
        sample_count: 0,
        class_count: 0,
        artifact: null,
        eligible_for_activation: false,
        created_at: "",
      }],
    });

    expect(parsed).not.toBeNull();
    expect(findActivePersonalModelVersion(parsed!)).toBeNull();
    expect(formatActivePersonalModel(null)).toBe(BASE_MODEL_UNTRAINED_LABEL);
    expect(isBuiltinCompatibilityVersion(parsed!.versions[0])).toBe(true);
  });

  it("allows active, draft, and archived versions with non-empty artifact and eligibility", () => {
    const base = {
      version: 2,
      status: "active" as const,
      algorithm: "normalized_centroid_v1",
      sample_count: 6,
      class_count: 2,
      artifact: { prototypes: [{ identity_id: "alice" }] },
      eligible_for_activation: true,
      created_at: "",
    };
    expect(isUsablePersonalModelVersion(base)).toBe(true);
    expect(isUsablePersonalModelVersion({ ...base, status: "draft" })).toBe(true);
    expect(isUsablePersonalModelVersion({ ...base, status: "archived" })).toBe(true);
    expect(isUsablePersonalModelVersion({ ...base, status: "failed" })).toBe(false);
    expect(isUsablePersonalModelVersion({ ...base, eligible_for_activation: false })).toBe(false);
    expect(isUsablePersonalModelVersion({ ...base, artifact: null })).toBe(false);
    expect(isUsablePersonalModelVersion({ ...base, artifact: {} })).toBe(false);
  });

  it("shows concrete readiness reasons instead of the generic root message", () => {
    const parsed = readPersonalTrainingStatus({
      status: "idle",
      message: "已验证样本尚不足以训练个人模型。",
      readiness: {
        verified_sample_count: 3,
        verified_class_count: 1,
        eligible_class_count: 0,
        ready: false,
        reasons: ["至少需要 2 个分类，当前为 1 个", "每个分类至少需要 3 个样本，当前 0/1 个分类达标"],
      },
      settings: { min_samples_per_class: 3 },
      active_version: 1,
      versions: [],
    });

    expect(parsed?.readiness_message).toBe("至少需要 2 个分类，当前为 1 个；每个分类至少需要 3 个样本，当前 0/1 个分类达标");
    expect(parsed?.readiness_message).not.toContain("已验证样本尚不足");
  });
});
