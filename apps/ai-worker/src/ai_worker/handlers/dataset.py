"""Dataset export handler (LoRA training material)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..dataset import export_lora_dataset
from ..errors import WorkerError


def handle_dataset_export(
    payload: Mapping[str, object],
    *,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    try:
        return export_lora_dataset(payload, on_progress=on_progress)
    except WorkerError:
        raise
    except OSError as exc:
        raise WorkerError("DATASET_IO_FAILED", f"写入训练集失败: {exc}") from exc
