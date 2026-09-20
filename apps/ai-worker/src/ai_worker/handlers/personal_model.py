"""Personal model training and evaluation handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..errors import WorkerError
from ..personal_model import evaluate_personal_model, train_personal_model


def handle_personal_model_train(
    payload: Mapping[str, object],
    *,
    required_string_fn: Callable[[Mapping[str, object], str], str],
) -> dict[str, Any]:
    """Train a deterministic centroid snapshot from labelled embeddings."""
    version = required_string_fn(payload, "version")
    try:
        return train_personal_model(
            version,
            payload.get("samples"),
            payload.get("config"),
        )
    except ValueError as exc:
        raise WorkerError("INVALID_PAYLOAD", str(exc)) from exc


def handle_personal_model_evaluate(payload: Mapping[str, object]) -> dict[str, Any]:
    """Evaluate a persisted snapshot against a caller-provided holdout set."""
    artifact = payload.get("personal_model", payload.get("artifact"))
    if artifact is None:
        raise WorkerError(
            "INVALID_PAYLOAD",
            "personal_model.evaluate 需要 personal_model 或 artifact",
        )
    try:
        return evaluate_personal_model(artifact, payload.get("samples"))
    except ValueError as exc:
        raise WorkerError("INVALID_PAYLOAD", str(exc)) from exc
