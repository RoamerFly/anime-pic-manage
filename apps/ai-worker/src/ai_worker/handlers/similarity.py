"""Similarity feature extraction, clustering, and dataset scanning handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..errors import WorkerError
from ..images import enumerate_images
from ..similarity import cluster_features, extract_image_features, extract_image_features_batch

MAX_SIMILARITY_WORKERS = 64


def parse_max_workers(payload: Mapping[str, object]) -> int | None:
    """Read the optional per-request feature extraction parallelism."""

    value = payload.get("max_workers")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise WorkerError("INVALID_PAYLOAD", "max_workers 必须是整数")
    if not 1 <= value <= MAX_SIMILARITY_WORKERS:
        raise WorkerError(
            "INVALID_PAYLOAD",
            f"max_workers 必须在 1 到 {MAX_SIMILARITY_WORKERS} 之间",
        )
    return value


def handle_similarity_features_extract(
    payload: Mapping[str, object],
    *,
    required_string_fn: Callable[[Mapping[str, object], str], str],
) -> dict[str, Any]:
    if "path" in payload:
        path = required_string_fn(payload, "path")
        features = extract_image_features(path)
        return {"feature": features}
    if "paths" in payload:
        paths = payload.get("paths")
        if not isinstance(paths, list):
            raise WorkerError("INVALID_PAYLOAD", "paths 必须是字符串列表")
        features_list, errors = extract_image_features_batch(
            [str(p) for p in paths],
            max_workers=parse_max_workers(payload),
        )
        return {"features": features_list, "errors": errors}
    raise WorkerError("INVALID_PAYLOAD", "缺少 path 或 paths 参数")


def handle_similarity_cluster(payload: Mapping[str, object]) -> dict[str, Any]:
    features = payload.get("features")
    if not isinstance(features, list):
        raise WorkerError("INVALID_PAYLOAD", "features 必须是列表")
    threshold_val = payload.get("threshold", 0.85)
    try:
        threshold = float(threshold_val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        threshold = 0.85
    return cluster_features(features, threshold=threshold)


def handle_similarity_scan(
    payload: Mapping[str, object],
    *,
    required_string_fn: Callable[[Mapping[str, object], str], str],
) -> dict[str, Any]:
    threshold_val = payload.get("threshold", 0.85)
    try:
        threshold = float(threshold_val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        threshold = 0.85

    paths: list[str] = []
    if "paths" in payload and isinstance(payload["paths"], list):
        paths = [str(p) for p in payload["paths"]]
    elif "directory" in payload:
        dir_path = required_string_fn(payload, "directory")
        include_sub = bool(payload.get("include_subfolders", True))
        paths = [str(p) for p in enumerate_images(dir_path, recursive=include_sub)]

    features, errors = extract_image_features_batch(
        paths,
        max_workers=parse_max_workers(payload),
    )

    result = cluster_features(features, threshold=threshold)
    result["errors"] = errors
    return result
