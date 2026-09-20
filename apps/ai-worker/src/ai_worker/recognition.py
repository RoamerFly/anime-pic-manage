"""Model-independent score filtering and multi-scale decision logic."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .images import CropType


@dataclass(frozen=True, slots=True)
class LabelScore:
    tag: str
    score: float
    display_name: str | None = None
    category: str = "character"

    def __post_init__(self) -> None:
        if not self.tag.strip():
            raise ValueError("标签不能为空")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("标签分数必须在 0 到 1 之间")

    def as_dict(self) -> dict[str, object]:
        return {
            "character_tag": self.tag,
            "display_name": self.display_name,
            "confidence": self.score,
            "category": self.category,
        }


EMBEDDING_DIMENSION = 2048


def normalize_embedding(
    value: object, *, dimension: int = EMBEDDING_DIMENSION
) -> tuple[float, ...]:
    """Validate and L2-normalize an embedding crossing the worker boundary.

    Embeddings are deliberately represented as tuples in the core so callers
    cannot accidentally mutate an in-flight vector.  A zero vector is not a
    useful prototype and is rejected rather than silently producing NaNs.
    """

    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValueError(f"embedding 必须是 {dimension} 维数字数组")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError("embedding 必须是数字数组")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding 必须是数字数组") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError("embedding 不能包含非有限数")
    norm = math.sqrt(sum(item * item for item in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding 不能是零向量")
    return tuple(item / norm for item in values)


@dataclass(frozen=True, slots=True)
class PersonalPrototype:
    """A user-confirmed identity prototype used during recognition."""

    identity_id: str
    display_name: str
    embedding: tuple[float, ...]
    sample_count: int

    @classmethod
    def from_value(cls, value: object, index: int = 0) -> PersonalPrototype:
        if not isinstance(value, Mapping):
            raise ValueError(f"personal_prototypes[{index}] 必须是对象")
        identity_id = value.get("identity_id")
        display_name = value.get("display_name")
        sample_count = value.get("sample_count")
        if not isinstance(identity_id, str) or not identity_id.strip():
            raise ValueError(f"personal_prototypes[{index}].identity_id 必须是非空字符串")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"personal_prototypes[{index}].display_name 必须是非空字符串")
        if len(identity_id.strip()) > 256 or len(display_name.strip()) > 256:
            raise ValueError(
                f"personal_prototypes[{index}] 的标签长度不能超过 256 个字符"
            )
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
            raise ValueError(f"personal_prototypes[{index}].sample_count 必须是正整数")
        try:
            embedding = normalize_embedding(value.get("embedding"))
        except ValueError as exc:
            raise ValueError(f"personal_prototypes[{index}].embedding 无效: {exc}") from exc
        return cls(identity_id.strip(), display_name.strip(), embedding, sample_count)


def parse_personal_prototypes(value: object) -> tuple[PersonalPrototype, ...]:
    """Parse the optional prototype array with duplicate-id protection."""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("personal_prototypes 必须是数组")
    prototypes = tuple(
        PersonalPrototype.from_value(item, index) for index, item in enumerate(value)
    )
    ids = [prototype.identity_id for prototype in prototypes]
    if len(set(ids)) != len(ids):
        raise ValueError("personal_prototypes 不能包含重复 identity_id")
    return prototypes


@dataclass(frozen=True, slots=True)
class PersonalMatch:
    prototype: PersonalPrototype
    score: float


def score_personal_prototypes(
    embedding: Sequence[float], prototypes: Iterable[PersonalPrototype]
) -> tuple[PersonalMatch, ...]:
    """Return cosine similarities in deterministic descending order."""

    normalized = normalize_embedding(embedding)
    matches = tuple(
        PersonalMatch(
            prototype,
            max(
                -1.0,
                min(
                    1.0,
                    sum(
                        left * right
                        for left, right in zip(normalized, prototype.embedding, strict=True)
                    ),
                ),
            ),
        )
        for prototype in prototypes
    )
    return tuple(sorted(matches, key=lambda item: (-item.score, item.prototype.identity_id)))


def _coerce_score(value: LabelScore | Mapping[str, object] | Sequence[object]) -> LabelScore:
    if isinstance(value, LabelScore):
        return value
    if isinstance(value, Mapping):
        tag = value.get("tag", value.get("character_tag"))
        score = value.get("score", value.get("confidence"))
        if not isinstance(tag, str) or not isinstance(score, (int, float)):
            raise ValueError("标签分数必须包含 tag 和 score")
        display_name = value.get("display_name")
        category = value.get("category", "character")
        return LabelScore(
            tag,
            float(score),
            display_name if isinstance(display_name, str) else None,
            str(category),
        )
    if len(value) < 2:
        raise ValueError("标签分数序列至少需要 tag 和 score")
    score_value = value[1]
    if not isinstance(score_value, (int, float)):
        raise ValueError("标签分数序列的 score 必须是数字")
    return LabelScore(str(value[0]), float(score_value))


def top_k_scores(
    scores: Iterable[LabelScore | Mapping[str, object] | Sequence[object]],
    k: int = 5,
    *,
    allowed_tags: Iterable[str] | None = None,
    category: str | None = "character",
) -> list[LabelScore]:
    """Return deterministic top-k scores after optional whitelist filtering."""

    if k <= 0:
        return []
    allowed = set(allowed_tags) if allowed_tags is not None else None
    selected: list[LabelScore] = []
    for raw in scores:
        item = _coerce_score(raw)
        if allowed is not None and item.tag not in allowed:
            continue
        if category is not None and item.category != category:
            continue
        selected.append(item)
    # Stable lexical tie-breaks make JSON/CSV exports and benchmark results
    # reproducible across Python/platform versions.
    selected.sort(key=lambda item: (-item.score, item.tag))
    return selected[:k]


def filter_top_k(
    scores: Iterable[LabelScore | Mapping[str, object] | Sequence[object]],
    allowed_tags: Iterable[str],
    k: int = 5,
) -> list[LabelScore]:
    """Compatibility alias for whitelist-aware Top-K filtering."""

    return top_k_scores(scores, k, allowed_tags=allowed_tags)


class DecisionStatus(StrEnum):
    HIGH_CONFIDENCE = "high_confidence"
    NEEDS_REVIEW = "needs_review"
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Configurable policy thresholds, not an accuracy guarantee."""

    crop_weights: Mapping[CropType | str, float] = field(
        default_factory=lambda: {
            CropType.HEAD: 0.40,
            CropType.BUST: 0.40,
            CropType.LARGE: 0.20,
        }
    )
    auto_threshold: float = 0.80
    unrecognized_threshold: float = 0.35
    min_margin: float = 0.15
    min_consistency: float = 0.67
    consistency_bonus: float = 0.03
    conflict_penalty: float = 0.04
    force_review_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.crop_weights or any(weight < 0 for weight in self.crop_weights.values()):
            raise ValueError("裁剪权重必须包含至少一个非负权重")
        if sum(self.crop_weights.values()) <= 0:
            raise ValueError("裁剪权重总和必须大于 0")
        for value in (self.auto_threshold, self.unrecognized_threshold, self.min_consistency):
            if not 0 <= value <= 1:
                raise ValueError("阈值必须在 0 到 1 之间")
        if self.min_margin < 0:
            raise ValueError("margin 不能为负数")


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    tag: str
    score: float
    rank: int
    display_name: str | None
    crop_support: int
    base_score: float | None = None
    personal_score: float | None = None
    source: str = "base_model"
    sample_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "character_tag": self.tag,
            "display_name": self.display_name,
            "confidence": self.score,
            "rank": self.rank,
            "crop_support": self.crop_support,
            "base_score": self.base_score,
            "personal_score": self.personal_score,
            "source": self.source,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class FusionResult:
    status: DecisionStatus
    top1: FusedCandidate | None
    top2: FusedCandidate | None
    margin: float
    consistency: float
    candidates: tuple[FusedCandidate, ...]
    reason: str

    @property
    def top1_score(self) -> float:
        return self.top1.score if self.top1 else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "top1": self.top1.as_dict() if self.top1 else None,
            "top2": self.top2.as_dict() if self.top2 else None,
            "margin": self.margin,
            "consistency": self.consistency,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "reason": self.reason,
        }


def _crop_type(value: CropType | str) -> CropType:
    try:
        return CropType(value)
    except ValueError:
        raise ValueError(f"未知裁剪类型: {value}") from None


def fuse_multiscale(
    predictions: Mapping[
        CropType | str, Iterable[LabelScore | Mapping[str, object] | Sequence[object]]
    ],
    *,
    config: FusionConfig | None = None,
    allowed_tags: Iterable[str] | None = None,
    top_k: int = 5,
) -> FusionResult:
    """Fuse crop predictions and classify as high-confidence/review/unrecognized.

    Scores are weighted by the configured crops that actually produced a
    prediction, then normalized by their active weight.  The consistency bonus
    rewards agreement among crop Top-1 values; disagreement applies the
    conflict penalty once.  Both adjustments are capped to [0, 1].
    """

    policy = config or FusionConfig()
    allowed = set(allowed_tags) if allowed_tags is not None else None
    aggregates: dict[str, float] = {}
    names: dict[str, str | None] = {}
    support: dict[str, set[CropType]] = {}
    crop_winners: list[str] = []
    active_weight = 0.0
    for raw_crop, raw_predictions in predictions.items():
        crop = _crop_type(raw_crop)
        weight = policy.crop_weights.get(crop, policy.crop_weights.get(crop.value, 0.0))
        if weight <= 0:
            continue
        ranked = top_k_scores(raw_predictions, top_k, allowed_tags=allowed)
        if not ranked:
            continue
        active_weight += weight
        crop_winners.append(ranked[0].tag)
        for prediction in ranked:
            aggregates[prediction.tag] = (
                aggregates.get(prediction.tag, 0.0) + weight * prediction.score
            )
            names.setdefault(prediction.tag, prediction.display_name)
            support.setdefault(prediction.tag, set()).add(crop)
    if not aggregates or active_weight <= 0:
        return FusionResult(
            DecisionStatus.UNRECOGNIZED, None, None, 0.0, 0.0, (), "没有可用的角色候选"
        )

    consistency_by_tag = {
        tag: sum(winner == tag for winner in crop_winners) / len(crop_winners) for tag in aggregates
    }
    disagreement = len(set(crop_winners)) > 1
    candidates = [
        FusedCandidate(
            tag,
            max(
                0.0,
                min(
                    1.0,
                    aggregates[tag] / active_weight
                    + policy.consistency_bonus * consistency_by_tag[tag]
                    - (policy.conflict_penalty if disagreement and tag != crop_winners[0] else 0.0),
                ),
            ),
            0,
            names[tag],
            len(support[tag]),
        )
        for tag in aggregates
    ]
    candidates.sort(key=lambda item: (-item.score, item.tag))
    ranked_candidates = tuple(
        FusedCandidate(
            item.tag,
            item.score,
            index,
            item.display_name,
            item.crop_support,
            base_score=item.score,
        )
        for index, item in enumerate(candidates, start=1)
    )
    top1 = ranked_candidates[0]
    top2 = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    margin = top1.score - (top2.score if top2 else 0.0)
    consistency = consistency_by_tag[top1.tag]
    if top1.score < policy.unrecognized_threshold:
        status = DecisionStatus.UNRECOGNIZED
        reason = "Top-1 分数低于未识别阈值"
    elif (
        top1.score >= policy.auto_threshold
        and margin >= policy.min_margin
        and consistency >= policy.min_consistency
        and top1.tag not in policy.force_review_tags
    ):
        status = DecisionStatus.HIGH_CONFIDENCE
        reason = "分数、margin 和多尺度一致性均达到策略阈值"
    else:
        status = DecisionStatus.NEEDS_REVIEW
        reason = "分数、margin、一致性或强制复核规则未满足自动分类条件"
    return FusionResult(status, top1, top2, margin, consistency, ranked_candidates, reason)


def combine_embeddings(
    embeddings: Mapping[CropType | str, Sequence[float]],
    *,
    weights: Mapping[CropType | str, float] | None = None,
) -> tuple[float, ...]:
    """Combine multi-scale embeddings with weighted averaging and L2 normalize."""

    if not embeddings:
        raise ValueError("至少需要一个 embedding")
    crop_weights = weights or {
        CropType.HEAD: 0.40,
        CropType.BUST: 0.40,
        CropType.LARGE: 0.20,
    }
    total_weight = 0.0
    values = [0.0] * EMBEDDING_DIMENSION
    for raw_crop, raw_embedding in embeddings.items():
        crop = _crop_type(raw_crop)
        weight = crop_weights.get(crop, crop_weights.get(crop.value, 0.0))
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError("embedding 裁剪权重必须是非负有限数字")
        if weight <= 0:
            continue
        vector = normalize_embedding(raw_embedding)
        total_weight += float(weight)
        for index, item in enumerate(vector):
            values[index] += float(weight) * item
    if total_weight <= 0:
        raise ValueError("embedding 裁剪权重总和必须大于 0")
    return normalize_embedding(tuple(item / total_weight for item in values))


def fuse_personal_prototypes(
    base: FusionResult,
    embedding: Sequence[float],
    prototypes: Iterable[PersonalPrototype],
    *,
    top_k: int = 5,
    personal_weight: float = 0.55,
    strict_threshold: float = 0.86,
    min_auto_samples: int = 3,
    min_personal_margin: float = 0.08,
) -> FusionResult:
    """Fuse user-confirmed prototypes into a base multi-scale result.

    Prototypes are allowed to introduce a label absent from the base model.
    A new or weakly-supported personal label is always marked for review; it
    can become high-confidence only after at least three samples and a strict
    cosine/margin check.
    """

    if not 0.0 <= personal_weight <= 1.0:
        raise ValueError("personal_weight 必须位于 0 到 1")
    if not 0.0 <= strict_threshold <= 1.0:
        raise ValueError("strict_threshold 必须位于 0 到 1")
    if min_auto_samples < 1:
        raise ValueError("min_auto_samples 必须为正整数")
    if min_personal_margin < 0:
        raise ValueError("min_personal_margin 不能为负数")
    parsed = tuple(prototypes)
    if not parsed:
        return base
    matches = score_personal_prototypes(embedding, parsed)
    by_id = {match.prototype.identity_id: match for match in matches}
    base_by_tag = {candidate.tag: candidate for candidate in base.candidates}
    combined: list[FusedCandidate] = []
    for candidate in base.candidates:
        match = by_id.get(candidate.tag)
        if match is None:
            combined.append(candidate)
            continue
        personal_score = match.score
        combined_score = ((1.0 - personal_weight) * candidate.score) + (
            personal_weight * max(0.0, personal_score)
        )
        combined.append(
            FusedCandidate(
                candidate.tag,
                max(0.0, min(1.0, combined_score)),
                0,
                match.prototype.display_name or candidate.display_name,
                candidate.crop_support,
                base_score=candidate.score,
                personal_score=personal_score,
                source="fused",
                sample_count=match.prototype.sample_count,
            )
        )
    for match in matches:
        if match.prototype.identity_id in base_by_tag:
            continue
        combined.append(
            FusedCandidate(
                match.prototype.identity_id,
                max(0.0, min(1.0, match.score)),
                0,
                match.prototype.display_name,
                0,
                base_score=None,
                personal_score=match.score,
                source="personal_model",
                sample_count=match.prototype.sample_count,
            )
        )
    combined.sort(key=lambda item: (-item.score, item.tag))
    ranked = tuple(
        FusedCandidate(
            item.tag,
            item.score,
            index,
            item.display_name,
            item.crop_support,
            base_score=item.base_score,
            personal_score=item.personal_score,
            source=item.source,
            sample_count=item.sample_count,
        )
        for index, item in enumerate(combined, start=1)
    )
    visible = ranked[: max(1, top_k)] if ranked else ()
    top1 = visible[0] if visible else None
    top2 = visible[1] if len(visible) > 1 else None
    margin = top1.score - (top2.score if top2 else 0.0) if top1 else 0.0
    if top1 is None:
        return base
    if top1.personal_score is None:
        return FusionResult(
            base.status, top1, top2, margin, base.consistency, visible, base.reason
        )
    if top1.sample_count is None or top1.sample_count < min_auto_samples:
        status = DecisionStatus.NEEDS_REVIEW
        reason = f"个人原型样本不足 {min_auto_samples} 个，需要人工复核"
    elif top1.personal_score <= strict_threshold:
        status = DecisionStatus.NEEDS_REVIEW
        reason = "个人原型相似度未超过严格自动分类阈值"
    elif margin < min_personal_margin:
        status = DecisionStatus.NEEDS_REVIEW
        reason = "个人原型 Top-1 与其他候选差距不足，需要人工复核"
    else:
        status = DecisionStatus.HIGH_CONFIDENCE
        reason = "个人原型样本数、相似度和 margin 均达到严格阈值"
    return FusionResult(status, top1, top2, margin, base.consistency, visible, reason)


def fuse_predictions(*args: object, **kwargs: object) -> FusionResult:
    """Alias retained for callers that use the shorter fusion name."""

    return fuse_multiscale(*args, **kwargs)  # type: ignore[arg-type]
