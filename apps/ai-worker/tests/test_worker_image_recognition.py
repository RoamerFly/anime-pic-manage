from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from ai_worker.images import DetectionBox
from ai_worker.models import ModelManifest
from ai_worker.protocol import SCHEMA_VERSION, WorkerRequest
from ai_worker.recognition import LabelScore
from ai_worker.worker import WorkerService


class _FakeCatalog:
    def __init__(self, manifests: list[Any]) -> None:
        self._manifests = {manifest.id: manifest for manifest in manifests}
        self.list_calls = 0

    def list_models(self) -> list[dict[str, object]]:
        self.list_calls += 1
        return [
            {"id": manifest.id, "kind": manifest.kind, "status": "installed"}
            for manifest in self._manifests.values()
        ]

    def get(self, model_id: str) -> Any:
        return self._manifests[model_id]


class _FakeDetector:
    def __init__(self, manifest: Any, detections: list[DetectionBox]) -> None:
        self.manifest = manifest
        self.detections = detections
        self.unload_count = 0

    def health(self) -> dict[str, object]:
        return {"id": self.manifest.id, "status": "ready"}

    def predict(self, image: Image.Image) -> list[DetectionBox]:
        assert image.size == (40, 30)
        return self.detections

    def unload(self) -> None:
        self.unload_count += 1


class _FakeRecognizer:
    def __init__(self, manifest: Any, *, error: Exception | None = None) -> None:
        self.manifest = manifest
        self.error = error
        self.calls: list[tuple[tuple[int, int], int]] = []
        self.unload_count = 0
        self._prediction_count = 0

    def health(self) -> dict[str, object]:
        return {"id": self.manifest.id, "status": "ready"}

    def predict(self, image: Image.Image, top_k: int = 5) -> list[LabelScore]:
        self.calls.append((image.size, top_k))
        self._prediction_count += 1
        if self.error is not None:
            raise self.error
        candidates = [
            LabelScore("character_alpha", 0.95, "角色甲"),
            LabelScore("character_beta", 0.10, "角色乙"),
        ]
        if self._prediction_count % 3 == 0:
            candidates[1] = LabelScore("character_gamma", 0.09, "角色丙")
        return candidates

    def unload(self) -> None:
        self.unload_count += 1


def _manifest(model_id: str, kind: str) -> Any:
    # The worker only needs id/kind after the fake catalog has selected the
    # manifest; fake objects keep these tests independent of model assets.
    return type("FakeManifest", (), {"id": model_id, "kind": kind})()


def _request(path: Path, message_type: str = "recognition.image") -> WorkerRequest:
    return WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "request-1",
            "task_id": "task-1",
            "message_type": message_type,
            "payload": {"path": str(path), "top_k": 2},
        }
    )


def _service(
    detector: _FakeDetector,
    recognizer: _FakeRecognizer | None,
) -> WorkerService:
    detector_manifest = detector.manifest
    manifests = [detector_manifest]
    if recognizer is not None:
        manifests.append(recognizer.manifest)
    service = WorkerService(models_root="missing")
    service.models = _FakeCatalog(manifests)  # type: ignore[assignment]
    adapters: dict[str, Any] = {detector_manifest.id: detector}
    if recognizer is not None:
        adapters[recognizer.manifest.id] = recognizer

    def factory(manifest: ModelManifest) -> Any:
        return adapters[manifest.id]

    service._adapter_factory = factory  # type: ignore[assignment]
    return service


def test_recognition_image_runs_pipeline_and_reuses_cached_adapters(tmp_path: Path) -> None:
    path = tmp_path / "input.png"
    Image.new("RGB", (40, 30), "white").save(path)
    source_bytes = path.read_bytes()
    detector = _FakeDetector(
        _manifest("detector", "head_detector"),
        [DetectionBox(4, 5, 20, 24, 0.88)],
    )
    recognizer = _FakeRecognizer(_manifest("recognizer", "character_recognizer"))
    service = _service(detector, recognizer)

    response = service.handle(_request(path))
    second_response = service.handle(_request(path, "image.recognize"))

    assert response.error is None
    result = response.payload
    assert result["path"] == str(path)
    assert result["image_size"] == [40, 30]
    assert len(result["people"]) == 1
    person = result["people"][0]
    assert person["person_index"] == 0
    assert person["box"]["confidence"] == 0.88
    assert set(person["crops"]) == {"head", "bust", "large"}
    assert person["top1"]["character_tag"] == "character_alpha"
    assert person["top2"]["character_tag"] == "character_beta"
    assert person["status"] == "high_confidence"
    assert len(person["candidates"]) == 2
    assert second_response.error is None
    assert len(recognizer.calls) == 6
    assert all(top_k == 5 for _, top_k in recognizer.calls)
    assert service.models.list_calls == 2

    # The response is JSON-safe and the source image remains byte-for-byte
    # untouched because recognition only reads it and uses detached crops.
    json.dumps(response.payload)
    assert path.read_bytes() == source_bytes
    service.close()
    assert detector.unload_count == 1
    assert recognizer.unload_count == 1


def test_recognition_image_without_people_skips_character_model(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    Image.new("RGB", (40, 30), "black").save(path)
    detector = _FakeDetector(_manifest("detector", "head_detector"), [])
    service = _service(detector, None)

    response = service.handle(_request(path))

    assert response.error is None
    assert response.payload == {"path": str(path), "image_size": [40, 30], "people": []}
    service.close()
    assert detector.unload_count == 1


def test_recognition_image_maps_read_and_model_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    service = _service(
        _FakeDetector(_manifest("detector", "head_detector"), []),
        None,
    )
    missing_response = service.handle(_request(missing))
    assert missing_response.error is not None
    assert missing_response.error["code"] == "IMAGE_NOT_FOUND"

    path = tmp_path / "input.png"
    Image.new("RGB", (40, 30), "white").save(path)
    detector = _FakeDetector(
        _manifest("detector", "head_detector"),
        [DetectionBox(1, 1, 10, 10, 0.9)],
    )
    recognizer = _FakeRecognizer(
        _manifest("recognizer", "character_recognizer"),
        error=ValueError("mock inference failure"),
    )
    model_error_response = _service(detector, recognizer).handle(_request(path))
    assert model_error_response.error is not None
    assert model_error_response.error["code"] == "RECOGNITION_FAILED"


class _VerifiedManifest:
    def __init__(self, model_id: str, kind: str) -> None:
        self.id = model_id
        self.kind = kind
        self.verify_sidecars_calls = 0
        self.verify_model_file_calls = 0
        self.validate_calls = 0

    def verify_sidecars(self) -> None:
        self.verify_sidecars_calls += 1

    def verify_model_file(self) -> None:
        self.verify_model_file_calls += 1

    def validate(self) -> None:
        self.validate_calls += 1


def test_explicit_manifest_selection_caches_verification(tmp_path: Path) -> None:
    manifest = _VerifiedManifest("detector", "head_detector")
    service = WorkerService(models_root=tmp_path / "missing")
    catalog = _FakeCatalog([manifest])
    service.models = catalog  # type: ignore[assignment]

    first = service._select_manifest("head_detector", "detector")
    second = service._select_manifest("head_detector", "detector")

    assert first is manifest
    assert second is manifest
    assert catalog.list_calls == 0
    assert manifest.verify_sidecars_calls == 1
    assert manifest.verify_model_file_calls == 1
    assert manifest.validate_calls == 1
