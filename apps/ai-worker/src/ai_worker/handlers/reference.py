"""Build a reference library from corrected samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

from ..errors import WorkerError
from ..images import DetectionBox, make_multiscale_crops, read_image
from ..reference import DEFAULT_THRESHOLDS, build_library
from .recognition import _reference_feature


def _crop_sample(service: Any, sample: Mapping[str, object]) -> Image.Image:
    """Single-character crop for one corrected sample.

    The annotation box is the person box; the reference vector therefore uses
    the same "bust" crop the live pipeline feeds the recognizer.
    """

    path = sample.get("path")
    if not isinstance(path, str) or not path.strip():
        raise WorkerError("INVALID_PAYLOAD", "参考样本缺少 path")
    try:
        image = read_image(path)
    except FileNotFoundError as exc:
        raise WorkerError("IMAGE_NOT_FOUND", f"图片不存在: {path}", str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise WorkerError("IMAGE_READ_FAILED", f"图片读取失败: {path}", str(exc)) from exc
    bbox = sample.get("bbox")
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
    ):
        x, y, width, height = (float(value) for value in bbox)
        if width > 0 and height > 0:
            box = DetectionBox(
                x1=x * image.width,
                y1=y * image.height,
                x2=(x + width) * image.width,
                y2=(y + height) * image.height,
            ).normalized(image.width, image.height)
            crops = make_multiscale_crops(image, box)
            crop = crops.get("bust")
            if crop is None:
                for key, value in crops.items():
                    if getattr(key, "value", key) == "bust":
                        crop = value
                        break
            if crop is not None:
                return crop
    return image


def handle_reference_build(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    backend = str(payload.get("backend", "ccip")).strip().lower()
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise WorkerError("INVALID_PAYLOAD", "reference.build 需要非空 samples 数组")
    recognizer = None
    if backend == "embedding":
        manifest = service._select_manifest("character_recognizer", None)  # noqa: SLF001
        recognizer = service._recognizer_adapter(manifest)  # noqa: SLF001
    max_per_identity = payload.get("max_per_identity", 32)
    if not isinstance(max_per_identity, int) or max_per_identity <= 0:
        max_per_identity = 32
    threshold = payload.get("threshold")
    threshold_value = (
        float(threshold)
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
        else DEFAULT_THRESHOLDS.get(backend)
    )

    def extract(sample: Mapping[str, object]) -> list[float]:
        crop = _crop_sample(service, sample)
        return _reference_feature(service, recognizer, backend, crop)

    result = build_library(
        [item for item in raw_samples if isinstance(item, Mapping)],
        backend=backend,
        threshold=threshold_value,
        max_per_identity=max_per_identity,
        extract=extract,
    )
    return result
