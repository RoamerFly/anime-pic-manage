"""Deterministic local personal-model training and evaluation.

The personal model is intentionally a small, persisted classifier rather than
an implicit neural-network fine-tune.  It stores one L2-normalized centroid
per identity and a calibrated cosine threshold/margin.  This makes the model
portable, reproducible, and safe to train in the local worker without writing
to source images or requiring a network connection.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .recognition import (
    EMBEDDING_DIMENSION,
    PersonalPrototype,
    normalize_embedding,
    parse_personal_prototypes,
    score_personal_prototypes,
)

ARTIFACT_SCHEMA_VERSION = "1.0"
ALGORITHM = "normalized_centroid_v1"
MAX_TRAINING_SAMPLES = 10_000
DEFAULT_MIN_SAMPLES_PER_CLASS = 3
DEFAULT_MIN_CLASSES = 2
DEFAULT_MIN_VALIDATION_COUNT = 1
DEFAULT_MIN_ACCURACY = 0.90
DEFAULT_STRICT_THRESHOLD = 0.86
DEFAULT_MIN_MARGIN = 0.08


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One user-confirmed embedding and its stable identity label."""

    identity_id: str
    display_name: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Validated controls for calibration and activation eligibility."""

    min_samples_per_class: int = DEFAULT_MIN_SAMPLES_PER_CLASS
    min_classes: int = DEFAULT_MIN_CLASSES
    min_validation_count: int = DEFAULT_MIN_VALIDATION_COUNT
    min_accuracy: float = DEFAULT_MIN_ACCURACY
    strict_threshold: float | None = None
    min_margin: float | None = None


@dataclass(frozen=True, slots=True)
class PersonalModelArtifact:
    """Validated persisted model snapshot accepted by recognition.image."""

    schema_version: str
    algorithm: str
    version: str
    prototypes: tuple[PersonalPrototype, ...]
    strict_threshold: float
    min_margin: float

    @classmethod
    def from_value(cls, value: object) -> PersonalModelArtifact:
        if not isinstance(value, Mapping):
            raise ValueError("personal_model 必须是对象")
        schema_version = value.get("schema_version")
        if schema_version not in (ARTIFACT_SCHEMA_VERSION, 1, "1"):
            raise ValueError(
                f"personal_model.schema_version 不受支持: {schema_version!r}"
            )
        algorithm = value.get("algorithm")
        if algorithm != ALGORITHM:
            raise ValueError(
                f"personal_model.algorithm 必须是 {ALGORITHM!r}"
            )
        version = value.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("personal_model.version 必须是非空字符串")
        try:
            prototypes = _parse_artifact_prototypes(value.get("prototypes"))
            strict_threshold = _bounded_number(
                value.get("strict_threshold"),
                "personal_model.strict_threshold",
                lower=0.0,
                upper=1.0,
            )
            min_margin = _bounded_number(
                value.get("min_margin"),
                "personal_model.min_margin",
                lower=0.0,
                upper=1.0,
            )
        except ValueError:
            raise
        return cls(
            ARTIFACT_SCHEMA_VERSION,
            ALGORITHM,
            version.strip(),
            prototypes,
            strict_threshold,
            min_margin,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "version": self.version,
            "prototypes": [
                {
                    "identity_id": prototype.identity_id,
                    "display_name": prototype.display_name,
                    "embedding": list(prototype.embedding),
                    "sample_count": prototype.sample_count,
                }
                for prototype in self.prototypes
            ],
            "strict_threshold": self.strict_threshold,
            "min_margin": self.min_margin,
        }


def _bounded_number(
    value: object,
    field_name: str,
    *,
    lower: float,
    upper: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是数字")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{field_name} 必须是 {lower:g} 到 {upper:g} 之间的有限数字")
    return result


def _label(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    result = value.strip()
    if len(result) > 256:
        raise ValueError(f"{field_name} 长度不能超过 256 个字符")
    return result


def parse_training_samples(value: object) -> tuple[TrainingSample, ...]:
    """Validate JSON training samples and normalize every 2048-D vector."""

    if not isinstance(value, list) or not value:
        raise ValueError("samples 必须是非空对象数组")
    if len(value) > MAX_TRAINING_SAMPLES:
        raise ValueError(f"samples 一次最多支持 {MAX_TRAINING_SAMPLES} 个样本")
    samples: list[TrainingSample] = []
    names: dict[str, str] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"samples[{index}] 必须是对象")
        identity_id = _label(raw.get("identity_id"), f"samples[{index}].identity_id")
        display_name = _label(raw.get("display_name"), f"samples[{index}].display_name")
        previous_name = names.setdefault(identity_id, display_name)
        if previous_name != display_name:
            raise ValueError(
                f"samples[{index}].display_name 与 identity_id={identity_id!r} 的已有名称不一致"
            )
        try:
            embedding = normalize_embedding(raw.get("embedding"))
        except ValueError as exc:
            raise ValueError(f"samples[{index}].embedding 无效: {exc}") from exc
        samples.append(TrainingSample(identity_id, display_name, embedding))
    return tuple(samples)


def parse_training_config(value: object) -> TrainingConfig:
    """Parse optional training controls with conservative defaults."""

    if value is None:
        return TrainingConfig()
    if not isinstance(value, Mapping):
        raise ValueError("config 必须是对象")

    def positive_int(name: str, default: int) -> int:
        raw = value.get(name, default)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1 or raw > 10_000:
            raise ValueError(f"config.{name} 必须是 1 到 10000 之间的整数")
        return cast(int, raw)

    min_samples_per_class = positive_int(
        "min_samples_per_class", DEFAULT_MIN_SAMPLES_PER_CLASS
    )
    min_classes = positive_int("min_classes", DEFAULT_MIN_CLASSES)
    min_validation_count = positive_int(
        "min_validation_count", DEFAULT_MIN_VALIDATION_COUNT
    )
    min_accuracy = _bounded_number(
        value.get("min_accuracy", DEFAULT_MIN_ACCURACY),
        "config.min_accuracy",
        lower=0.0,
        upper=1.0,
    )

    strict_value = value.get("strict_threshold")
    strict_threshold = (
        None
        if strict_value is None
        else _bounded_number(
            strict_value,
            "config.strict_threshold",
            lower=0.0,
            upper=1.0,
        )
    )
    margin_value = value.get("min_margin")
    min_margin = (
        None
        if margin_value is None
        else _bounded_number(margin_value, "config.min_margin", lower=0.0, upper=1.0)
    )
    return TrainingConfig(
        min_samples_per_class,
        min_classes,
        min_validation_count,
        min_accuracy,
        strict_threshold,
        min_margin,
    )


def _parse_artifact_prototypes(value: object) -> tuple[PersonalPrototype, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("personal_model.prototypes 必须是非空对象数组")
    if len(value) > MAX_TRAINING_SAMPLES:
        raise ValueError(
            f"personal_model.prototypes 一次最多支持 {MAX_TRAINING_SAMPLES} 个原型"
        )
    try:
        prototypes = parse_personal_prototypes(value)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return tuple(sorted(prototypes, key=lambda item: item.identity_id))


def parse_personal_model(value: object) -> PersonalModelArtifact:
    """Public strict parser used by recognition.image and evaluate."""

    return PersonalModelArtifact.from_value(value)


def _centroid(samples: Sequence[TrainingSample]) -> tuple[float, ...]:
    if not samples:
        raise ValueError("无法为空类别计算 prototype")
    values = [0.0] * EMBEDDING_DIMENSION
    for sample in samples:
        for index, item in enumerate(sample.embedding):
            values[index] += item
    try:
        return normalize_embedding(values)
    except ValueError as exc:
        raise ValueError("类别 embedding 的平均向量为零，无法生成稳定 prototype") from exc


def _group_samples(
    samples: Sequence[TrainingSample],
) -> dict[str, list[TrainingSample]]:
    grouped: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.identity_id].append(sample)
    return {identity_id: grouped[identity_id] for identity_id in sorted(grouped)}


def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
    # Both sides have already been validated and normalized, but clamp the
    # result to make the persisted metrics robust to tiny floating-point drift.
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


def _leave_one_out(
    samples: Sequence[TrainingSample],
    grouped: Mapping[str, Sequence[TrainingSample]],
) -> tuple[list[dict[str, object]], list[float], list[float], list[float]]:
    """Build deterministic held-out rows and similarity statistics."""

    if len(grouped) < 2:
        return [], [], [], []
    rows: list[dict[str, object]] = []
    positive: list[float] = []
    negative: list[float] = []
    margins: list[float] = []
    for sample in samples:
        own_samples = grouped[sample.identity_id]
        if len(own_samples) < 2:
            # A singleton class cannot be evaluated without using the sample
            # as its own prototype, which would leak the label.
            continue
        own_centroid = _centroid(
            [candidate for candidate in own_samples if candidate is not sample]
        )
        candidate_scores: list[tuple[str, float]] = []
        for identity_id, class_samples in grouped.items():
            prototype = (
                own_centroid
                if identity_id == sample.identity_id
                else _centroid(class_samples)
            )
            candidate_scores.append((identity_id, _similarity(sample.embedding, prototype)))
        candidate_scores.sort(key=lambda item: (-item[1], item[0]))
        top_score = candidate_scores[0][1]
        second_score = candidate_scores[1][1]
        margin = top_score - second_score
        own_score = next(
            score for identity_id, score in candidate_scores if identity_id == sample.identity_id
        )
        other_scores = [
            score for identity_id, score in candidate_scores if identity_id != sample.identity_id
        ]
        positive.append(own_score)
        negative.append(max(other_scores))
        margins.append(margin)
        rows.append(
            {
                "identity_id": sample.identity_id,
                "predicted_identity_id": candidate_scores[0][0],
                "correct": candidate_scores[0][0] == sample.identity_id,
                "score": top_score,
                "margin": margin,
            }
        )
    return rows, positive, negative, margins


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile 至少需要一个值")
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def _calibrate_threshold(
    positive: Sequence[float],
    negative: Sequence[float],
    explicit: float | None,
) -> float:
    if explicit is not None:
        return explicit
    if positive and negative:
        # When classes separate cleanly, the midpoint leaves equal room on
        # both sides.  A conservative floor prevents weak one-off examples
        # from being activated by a low threshold.
        midpoint = (min(positive) + max(negative)) / 2.0
        return max(0.50, min(0.995, midpoint))
    if positive:
        return max(DEFAULT_STRICT_THRESHOLD, min(0.995, min(positive)))
    return DEFAULT_STRICT_THRESHOLD


def _calibrate_margin(margins: Sequence[float], explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    if not margins:
        return DEFAULT_MIN_MARGIN
    # Use a lower-tail margin, so one unusually easy sample cannot make the
    # entire personal model too permissive.  Keep the value in the range
    # accepted by recognition fusion.
    lower_tail = max(0.0, _percentile(margins, 0.10))
    return max(0.02, min(0.50, lower_tail * 0.50))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metrics(
    samples: Sequence[TrainingSample],
    grouped: Mapping[str, Sequence[TrainingSample]],
    rows: Sequence[Mapping[str, object]],
    positive: Sequence[float],
    negative: Sequence[float],
    margins: Sequence[float],
) -> dict[str, object]:
    correct_count = sum(bool(row["correct"]) for row in rows)
    validation_count = len(rows)
    return {
        "sample_count": len(samples),
        "class_count": len(grouped),
        "validation_count": validation_count,
        "correct_count": correct_count,
        "accuracy": (
            correct_count / validation_count if validation_count else None
        ),
        "samples_per_class": {
            identity_id: len(grouped[identity_id]) for identity_id in sorted(grouped)
        },
        "intra_similarity_mean": _mean(positive),
        "intra_similarity_min": min(positive) if positive else None,
        "inter_similarity_mean": _mean(negative),
        "inter_similarity_max": max(negative) if negative else None,
        "validation_margin_mean": _mean(margins),
        "validation_margin_min": min(margins) if margins else None,
    }


def train_personal_model(
    version: str,
    raw_samples: object,
    raw_config: object = None,
) -> dict[str, object]:
    """Create a deterministic centroid artifact and activation assessment."""

    model_version = _label(version, "version")
    # Canonicalize input order so the same labelled dataset produces the same
    # centroids and calibration metrics even when the caller enumerates files
    # in a different order.
    samples = tuple(
        sorted(
            parse_training_samples(raw_samples),
            key=lambda sample: (sample.identity_id, sample.embedding),
        )
    )
    config = parse_training_config(raw_config)
    grouped = _group_samples(samples)
    prototypes = tuple(
        PersonalPrototype(
            identity_id,
            class_samples[0].display_name,
            _centroid(class_samples),
            len(class_samples),
        )
        for identity_id, class_samples in grouped.items()
    )
    rows, positive, negative, margins = _leave_one_out(samples, grouped)
    strict_threshold = _calibrate_threshold(
        positive, negative, config.strict_threshold
    )
    min_margin = _calibrate_margin(margins, config.min_margin)
    metrics = _metrics(samples, grouped, rows, positive, negative, margins)

    warnings: list[str] = []
    sample_counts = metrics["samples_per_class"]
    assert isinstance(sample_counts, dict)
    if len(grouped) < config.min_classes:
        warnings.append(
            f"类别数 {len(grouped)} 小于自动激活要求 {config.min_classes}"
        )
    underrepresented = sorted(
        identity_id
        for identity_id, count in sample_counts.items()
        if isinstance(count, int) and count < config.min_samples_per_class
    )
    if underrepresented:
        warnings.append(
            "以下类别样本不足 "
            f"{config.min_samples_per_class} 个: {', '.join(underrepresented)}"
        )
    validation_count = metrics["validation_count"]
    if isinstance(validation_count, int) and validation_count < config.min_validation_count:
        warnings.append(
            f"留一验证样本数 {validation_count} 小于要求 {config.min_validation_count}"
        )
    accuracy = metrics["accuracy"]
    if isinstance(accuracy, float) and accuracy < config.min_accuracy:
        warnings.append(
            f"留一验证准确率 {accuracy:.3f} 低于要求 {config.min_accuracy:.3f}"
        )
    elif accuracy is None:
        warnings.append("当前数据无法进行跨类别留一验证")

    eligible = not warnings
    artifact = PersonalModelArtifact(
        ARTIFACT_SCHEMA_VERSION,
        ALGORITHM,
        model_version,
        prototypes,
        strict_threshold,
        min_margin,
    )
    # Keep the summary fields at the response root as well as inside metrics.
    # This lets older desktop runtimes validate the response without knowing
    # the newer metrics envelope yet.
    return {
        "artifact": artifact.as_dict(),
        "metrics": metrics,
        "sample_count": metrics["sample_count"],
        "class_count": metrics["class_count"],
        "algorithm": ALGORITHM,
        "warnings": warnings,
        "eligible_for_activation": eligible,
    }


def evaluate_personal_model(
    artifact_value: object,
    raw_samples: object,
) -> dict[str, object]:
    """Evaluate an artifact on explicitly supplied, labelled samples.

    Training itself uses leave-one-out metrics.  This endpoint is for a
    caller-provided holdout set and therefore does not silently reuse samples
    from the artifact, which intentionally stores no original image data.
    """

    artifact = parse_personal_model(artifact_value)
    samples = parse_training_samples(raw_samples)
    rows: list[dict[str, object]] = []
    score_margins: list[float] = []
    threshold_pass_count = 0
    for sample in samples:
        matches = score_personal_prototypes(sample.embedding, artifact.prototypes)
        if not matches:
            continue
        top = matches[0]
        second_score = matches[1].score if len(matches) > 1 else 0.0
        margin = top.score - second_score
        accepted = (
            top.score >= artifact.strict_threshold and margin >= artifact.min_margin
        )
        threshold_pass_count += int(accepted)
        score_margins.append(margin)
        rows.append(
            {
                "identity_id": sample.identity_id,
                "predicted_identity_id": top.prototype.identity_id,
                "correct": top.prototype.identity_id == sample.identity_id,
                "score": top.score,
                "margin": margin,
                "accepted": accepted,
            }
        )
    correct_count = sum(bool(row["correct"]) for row in rows)
    validation_count = len(rows)
    warnings: list[str] = []
    unknown = sorted(
        {
            sample.identity_id
            for sample in samples
            if sample.identity_id not in {p.identity_id for p in artifact.prototypes}
        }
    )
    if unknown:
        warnings.append("评估集包含模型中不存在的类别: " + ", ".join(unknown))
    if not validation_count:
        warnings.append("评估集没有可用样本")
    if validation_count and correct_count != validation_count:
        warnings.append("评估集存在分类错误，模型不满足全部样本的准确性要求")
    if validation_count and threshold_pass_count != validation_count:
        warnings.append("评估集存在未达到 strict_threshold 或 min_margin 的样本")
    metrics: dict[str, object] = {
        "sample_count": len(samples),
        "class_count": len({sample.identity_id for sample in samples}),
        "validation_count": validation_count,
        "correct_count": correct_count,
        "accuracy": correct_count / validation_count if validation_count else None,
        "threshold_pass_count": threshold_pass_count,
        "threshold_pass_rate": (
            threshold_pass_count / validation_count if validation_count else None
        ),
        "validation_margin_mean": _mean(score_margins),
        "validation_margin_min": min(score_margins) if score_margins else None,
    }
    return {
        "version": artifact.version,
        "metrics": metrics,
        "warnings": warnings,
        "eligible_for_activation": not warnings and validation_count > 0,
        "rows": rows,
    }
