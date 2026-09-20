from __future__ import annotations

import json
from pathlib import Path

from ai_worker.models import ModelCatalog
from ai_worker.protocol import SCHEMA_VERSION, WorkerRequest
from ai_worker.worker import WorkerService, process_line


def test_protocol_round_trip_and_health() -> None:
    service = WorkerService(models_root=Path("不存在的模型目录"))
    request = {
        "schema_version": SCHEMA_VERSION,
        "request_id": "request-1",
        "task_id": "task-1",
        "message_type": "health",
        "payload": {},
        "error": None,
    }
    response = json.loads(process_line(json.dumps(request), service))
    assert response["schema_version"] == SCHEMA_VERSION
    assert response["request_id"] == "request-1"
    assert response["payload"]["status"] == "ok"
    assert response["error"] is None


def test_protocol_malformed_request_is_structured_error() -> None:
    response = json.loads(process_line("not-json", WorkerService(models_root="missing")))
    assert set(response) == {
        "schema_version",
        "request_id",
        "task_id",
        "message_type",
        "payload",
        "error",
    }
    assert response["error"]["code"] == "INVALID_REQUEST"


def test_shutdown_is_explicit() -> None:
    service = WorkerService(models_root="missing")
    request = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "r",
            "task_id": "t",
            "message_type": "shutdown",
            "payload": {},
        }
    )
    response = service.handle(request)
    assert response.payload == {"ok": True}
    assert service.stopping


def test_model_catalog_without_assets_is_not_installed(tmp_path: Path) -> None:
    # A catalog without manifests remains empty and does not pretend a model exists.
    catalog = ModelCatalog(tmp_path / "models")
    assert catalog.list_models() == []


def test_model_catalog_reports_missing_weight_without_fake_ready_state(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "example"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "example-recognizer",
                "kind": "character_recognizer",
                "adapter": "animetimm_onnx",
                "format": "onnx",
                "version": "test",
                "model_file": "model.onnx",
                "labels_file": "selected_tags.csv",
                "preprocess_file": "preprocess.json",
                "sha256": "",
                "input": {"width": 4, "height": 4},
                "supported_devices": ["cpu"],
                "input_layout": "nchw",
                "output_activation": "sigmoid",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "selected_tags.csv").write_text(
        "tag,category\ncharacter_alpha,character\n", encoding="utf-8"
    )
    (model_dir / "preprocess.json").write_text(
        json.dumps(
            {
                "size": 4,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "layout": "nchw",
                "output_activation": "sigmoid",
            }
        ),
        encoding="utf-8",
    )
    models = ModelCatalog(tmp_path / "models").list_models()
    assert models[0]["status"] == "not_installed"
    assert models[0]["error"]["code"] == "MODEL_NOT_INSTALLED"
