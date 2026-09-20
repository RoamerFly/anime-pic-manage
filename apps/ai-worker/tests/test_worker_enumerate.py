from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import ai_worker.worker as worker_module
from ai_worker.protocol import SCHEMA_VERSION, WorkerRequest
from ai_worker.worker import WorkerService


def _request(root: Path, payload: dict[str, object]) -> WorkerRequest:
    return WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "request-enumerate",
            "task_id": "task-enumerate",
            "message_type": "images.enumerate",
            "payload": {"directory": str(root), **payload},
        }
    )


def test_enumeration_limit_inspects_only_prefix_but_reports_total(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6), "white").save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes(b"not an image")
    Image.new("RGB", (10, 4), "black").save(tmp_path / "c.png")
    service = WorkerService(models_root=tmp_path / "missing")

    limited = service.handle(_request(tmp_path, {"limit": 1}))
    assert limited.error is None
    assert limited.payload["count"] == 1
    assert limited.payload["total_count"] == 3
    assert limited.payload["returned_count"] == 1
    assert len(limited.payload["images"]) == 1
    assert limited.payload["images"][0]["path"].endswith("a.png")
    # The invalid b.png is outside the inspected prefix and therefore does
    # not delay the preview or appear as an inspection error.
    assert limited.payload["errors"] == []

    full = service.handle(_request(tmp_path, {}))
    assert full.error is None
    assert full.payload["count"] == 2
    assert full.payload["total_count"] == 3
    assert full.payload["returned_count"] == 2
    assert len(full.payload["errors"]) == 1


def test_enumeration_limit_rejects_invalid_values(tmp_path: Path) -> None:
    service = WorkerService(models_root=tmp_path / "missing")

    for invalid in (0, 51, True, "1"):
        response = service.handle(_request(tmp_path, {"limit": invalid}))
        assert response.error is not None
        assert response.error["code"] == "INVALID_PAYLOAD"


def test_enumeration_maps_header_pixel_budget_failures_to_structured_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "too-large-header.png"
    image_path.write_bytes(b"placeholder")
    service = WorkerService(models_root=tmp_path / "missing")

    def reject_image(_: Path) -> object:
        raise ValueError("图片像素数超过安全上限")

    monkeypatch.setattr(worker_module, "inspect_image", reject_image)

    response = service.handle(_request(tmp_path, {"limit": 1}))

    assert response.error is None
    assert response.payload["total_count"] == 1
    assert response.payload["images"] == []
    assert response.payload["errors"] == [
        {"path": str(image_path), "message": "图片像素数超过安全上限"}
    ]
