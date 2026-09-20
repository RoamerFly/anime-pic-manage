"""Reference-image matching for characters the classifier cannot label.

A model's tag list decides what it can name; rare characters are simply absent
(`character_alpha` is missing from the shipped recognizers). This module keeps a
small library of feature vectors built from images the user already corrected,
so those characters become recognisable without retraining anything.

Two backends are supported:

* ``ccip`` – the CCIP character-similarity model (independent of the recognizer,
  measured AUC 1.000 on the local collection);
* ``embedding`` – cosine similarity over the active recognizer's embedding
  (no extra download, but lower separation).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import WorkerError

REFERENCE_SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = ("ccip", "embedding")
# Defaults come from the upstream metrics plus the local leave-one-out sweep:
# CCIP differences were ~0.025 (same character) vs ~0.27 (different), and the
# library default threshold sits between them. The embedding backend needs a
# much higher cosine threshold.
DEFAULT_THRESHOLDS = {"ccip": 0.17847511429108218, "embedding": 0.77}
MAX_REFERENCES_PER_IDENTITY = 64
MAX_IDENTITIES = 200


def normalize_vector(values: Sequence[float]) -> list[float]:
    total = sum(value * value for value in values) ** 0.5
    if total <= 0.0:
        raise WorkerError("REFERENCE_INVALID", "参考向量为空，无法归一化")
    return [float(value) / total for value in values]


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    identity_id: str
    display_name: str
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ReferenceLibrary:
    backend: str
    threshold: float
    entries: tuple[ReferenceEntry, ...]
    built_at: str | None = None

    @classmethod
    def from_value(cls, value: object) -> ReferenceLibrary:
        if not isinstance(value, Mapping):
            raise WorkerError("REFERENCE_INVALID", "参考库必须是对象")
        backend = str(value.get("backend", "ccip")).strip().lower()
        if backend not in SUPPORTED_BACKENDS:
            raise WorkerError(
                "REFERENCE_INVALID",
                f"参考库 backend 必须是 {'/'.join(SUPPORTED_BACKENDS)}",
            )
        raw_threshold = value.get("threshold")
        threshold = (
            float(raw_threshold)
            if isinstance(raw_threshold, (int, float)) and not isinstance(raw_threshold, bool)
            else DEFAULT_THRESHOLDS[backend]
        )
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise WorkerError("REFERENCE_INVALID", "参考库缺少 entries 数组或没有可用参考图")
        entries: list[ReferenceEntry] = []
        for item in raw_entries:
            if not isinstance(item, Mapping):
                raise WorkerError("REFERENCE_INVALID", "参考库条目必须是对象")
            identity_id = item.get("identity_id")
            if not isinstance(identity_id, str) or not identity_id.strip():
                raise WorkerError("REFERENCE_INVALID", "参考库条目缺少 identity_id")
            display_name = item.get("display_name")
            raw_vectors = item.get("vectors")
            if not isinstance(raw_vectors, list) or not raw_vectors:
                raise WorkerError("REFERENCE_INVALID", "参考库条目缺少 vectors")
            vectors: list[tuple[float, ...]] = []
            for raw_vector in raw_vectors:
                if not isinstance(raw_vector, list) or not raw_vector:
                    raise WorkerError("REFERENCE_INVALID", "参考向量必须是数字数组")
                try:
                    vectors.append(
                        tuple(float(value) for value in raw_vector)  # type: ignore[arg-type]
                    )
                except (TypeError, ValueError) as exc:
                    raise WorkerError("REFERENCE_INVALID", "参考向量含非数字", str(exc)) from exc
            entries.append(
                ReferenceEntry(
                    identity_id=identity_id,
                    display_name=display_name if isinstance(display_name, str) else identity_id,
                    vectors=tuple(vectors),
                )
            )
        built_at = value.get("built_at")
        return cls(
            backend=backend,
            threshold=threshold,
            entries=tuple(entries),
            built_at=built_at if isinstance(built_at, str) else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> ReferenceLibrary:
        reference_path = Path(path).expanduser()
        try:
            raw = json.loads(reference_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkerError(
                "REFERENCE_INVALID",
                f"参考库无法读取: {reference_path.name}",
                str(exc),
            ) from exc
        return cls.from_value(raw)

    @property
    def reference_count(self) -> int:
        return sum(len(entry.vectors) for entry in self.entries)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


def match_library(
    library: ReferenceLibrary,
    vector: Sequence[float],
) -> dict[str, Any] | None:
    """Best reference match for one feature vector, or None when empty."""

    if not library.entries:
        return None
    best: tuple[float, ReferenceEntry] | None = None
    if library.backend == "ccip":
        # CCIP works on the raw feature vectors; the library ships its own
        # metric model, so the distance is derived from the same features.
        # imgutils only accepts numpy arrays here — a list is interpreted as an
        # image path and fails with "Unknown image type".
        import numpy as np
        from imgutils.metrics import ccip_difference

        query = np.asarray(vector, dtype=np.float32)
        for entry in library.entries:
            for candidate in entry.vectors:
                difference = float(
                    ccip_difference(np.asarray(candidate, dtype=np.float32), query)
                )
                if best is None or difference < best[0]:
                    best = (difference, entry)
        if best is None:
            return None
        difference, entry = best
        similarity = max(0.0, 1.0 - difference)
        return {
            "identity_id": entry.identity_id,
            "display_name": entry.display_name,
            "similarity": round(similarity, 4),
            "distance": round(difference, 4),
            "matched": difference <= library.threshold,
            "backend": library.backend,
            "threshold": library.threshold,
        }

    normalized = normalize_vector(vector)
    for entry in library.entries:
        for candidate in entry.vectors:
            similarity = _cosine(candidate, normalized)
            if best is None or similarity > best[0]:
                best = (similarity, entry)
    if best is None:
        return None
    similarity, entry = best
    return {
        "identity_id": entry.identity_id,
        "display_name": entry.display_name,
        "similarity": round(similarity, 4),
        "distance": round(1.0 - similarity, 4),
        "matched": similarity >= library.threshold,
        "backend": library.backend,
        "threshold": library.threshold,
    }


def build_library(
    samples: Sequence[Mapping[str, object]],
    *,
    backend: str,
    threshold: float | None = None,
    max_per_identity: int = MAX_REFERENCES_PER_IDENTITY,
    extract: Callable[[Mapping[str, object]], Sequence[float]],
    built_at: str | None = None,
) -> dict[str, Any]:
    """Group sample features into a serialisable reference library."""

    backend = backend.strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise WorkerError(
            "INVALID_PAYLOAD",
            f"backend 必须是 {'/'.join(SUPPORTED_BACKENDS)}",
        )
    grouped: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for sample in samples:
        identity_id = sample.get("identity_id")
        path = sample.get("path")
        if not isinstance(identity_id, str) or not identity_id.strip():
            failures.append({"path": str(path), "reason": "缺少 identity_id"})
            continue
        if not isinstance(path, str) or not path.strip():
            failures.append({"path": str(path), "reason": "缺少图片路径"})
            continue
        entry = grouped.setdefault(
            identity_id,
            {
                "identity_id": identity_id,
                "display_name": str(sample.get("display_name") or identity_id),
                "vectors": [],
            },
        )
        if len(grouped) > MAX_IDENTITIES:
            raise WorkerError("INVALID_PAYLOAD", f"参考库角色数不能超过 {MAX_IDENTITIES}")
        if len(entry["vectors"]) >= max_per_identity:
            continue
        try:
            vector = list(extract(sample))
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - report per-file failures
            failures.append({"path": path, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if backend == "embedding":
            vector = normalize_vector(vector)
        entry["vectors"].append(vector)

    kept = [entry for entry in grouped.values() if entry["vectors"]]
    library = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "backend": backend,
        "threshold": float(
            threshold if threshold is not None else DEFAULT_THRESHOLDS[backend]
        ),
        "entries": kept,
    }
    if built_at:
        library["built_at"] = built_at
    return {
        "library": library,
        "identities": len(kept),
        "references": sum(len(entry["vectors"]) for entry in kept),
        "skipped": len(failures),
        "failures": failures[:10],
    }
