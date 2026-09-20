from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from ai_worker.images import DetectionBox
from ai_worker.protocol import SCHEMA_VERSION, WorkerRequest
from ai_worker.recognition import (
    EMBEDDING_DIMENSION,
    DecisionStatus,
    PersonalPrototype,
    fuse_multiscale,
    fuse_personal_prototypes,
    normalize_embedding,
    parse_personal_prototypes,
)
from ai_worker.worker import WorkerService


def _vector(index: int = 0) -> list[float]:
    values = [0.0] * EMBEDDING_DIMENSION
    values[index] = 1.0
    return values


def test_personal_prototypes_are_strict_and_normalized() -> None:
    prototype = parse_personal_prototypes(
        [
            {
                "identity_id": "custom-a",
                "display_name": "自定义甲",
                "embedding": [2.0, *_vector()[1:]],
                "sample_count": 3,
            }
        ]
    )[0]
    assert prototype.sample_count == 3
    assert math.isclose(prototype.embedding[0], 1.0)
    assert math.isclose(math.sqrt(sum(item * item for item in prototype.embedding)), 1.0)

    for bad in (
        {"identity_id": "x", "display_name": "X", "embedding": _vector(), "sample_count": 0},
        {
            "identity_id": "x",
            "display_name": "X",
            "embedding": [1.0, 2.0],
            "sample_count": 3,
        },
    ):
        try:
            parse_personal_prototypes([bad])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid prototype should be rejected")


def test_personal_fusion_can_add_label_but_single_sample_needs_review() -> None:
    base = fuse_multiscale({"head": [{"tag": "base", "score": 0.2}]})
    result = fuse_personal_prototypes(
        base,
        _vector(),
        (
            PersonalPrototype("custom", "自定义", tuple(_vector()), 1),
        ),
        top_k=2,
    )
    assert result.top1 is not None and result.top1.tag == "custom"
    assert result.top1.source == "personal_model"
    assert result.top1.personal_score == 1.0
    assert result.status is DecisionStatus.NEEDS_REVIEW


def test_personal_fusion_three_samples_can_be_high_confidence() -> None:
    base = fuse_multiscale({"head": [{"tag": "base", "score": 0.1}]})
    result = fuse_personal_prototypes(
        base,
        _vector(),
        (
            PersonalPrototype("custom", "自定义", tuple(_vector()), 3),
        ),
    )
    assert result.top1 is not None and result.top1.tag == "custom"
    assert result.status is DecisionStatus.HIGH_CONFIDENCE


class _FakeCatalog:
    def __init__(self, manifests: list[Any]) -> None:
        self.manifests = {item.id: item for item in manifests}

    def list_models(self) -> list[dict[str, object]]:
        return [
            {"id": item.id, "kind": item.kind, "status": "installed"}
            for item in self.manifests.values()
        ]

    def get(self, model_id: str) -> Any:
        return self.manifests[model_id]


class _FakeDetector:
    def __init__(self, manifest: Any) -> None:
        self.manifest = manifest

    def predict(self, image: Image.Image) -> list[DetectionBox]:
        return [DetectionBox(4, 4, 20, 20, 0.9)]

    def unload(self) -> None:
        pass


class _FakeRecognizer:
    def __init__(self, manifest: Any) -> None:
        self.manifest = manifest
        self.embedding_calls = 0

    def predict(self, image: Image.Image, top_k: int = 5) -> list[Any]:
        return [{"tag": "base", "score": 0.1}]

    def predict_with_embedding(
        self, image: Image.Image, top_k: int = 5
    ) -> tuple[list[Any], tuple[float, ...]]:
        self.embedding_calls += 1
        return self.predict(image, top_k), tuple(_vector())

    def unload(self) -> None:
        pass


def _worker(tmp_path: Path) -> tuple[WorkerService, _FakeRecognizer]:
    detector_manifest = type("Manifest", (), {"id": "detector", "kind": "head_detector"})()
    recognizer_manifest = type(
        "Manifest", (), {"id": "recognizer", "kind": "character_recognizer"}
    )()
    detector = _FakeDetector(detector_manifest)
    recognizer = _FakeRecognizer(recognizer_manifest)
    service = WorkerService(models_root=tmp_path / "missing")
    service.models = _FakeCatalog([detector_manifest, recognizer_manifest])  # type: ignore[assignment]
    service._adapter_factory = (  # type: ignore[assignment]
        lambda manifest: detector if manifest.kind == "head_detector" else recognizer
    )
    return service, recognizer


def test_embedding_message_accepts_multiple_normalized_annotations_and_is_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (40, 30), "white").save(path)
    original = path.read_bytes()
    service, recognizer = _worker(tmp_path)
    request = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "embedding-request",
            "task_id": "embedding-task",
            "message_type": "recognition.embedding",
            "payload": {
                "path": str(path),
                "annotations": [
                    {"annotation_id": "a", "bbox": [0.1, 0.1, 0.5, 0.8]},
                    {"annotation_id": "b", "bbox": {"x1": 0.5, "y1": 0.1, "x2": 0.9, "y2": 0.8}},
                ],
            },
        }
    )
    response = service.handle(request)
    assert response.error is None
    assert response.payload["count"] == 2
    assert response.payload["embedding_dimension"] == EMBEDDING_DIMENSION
    embeddings = response.payload["embeddings"]
    assert isinstance(embeddings, list)
    assert [item["annotation_id"] for item in embeddings] == ["a", "b"]
    assert all(len(item["embedding"]) == EMBEDDING_DIMENSION for item in embeddings)
    assert recognizer.embedding_calls == 6
    assert path.read_bytes() == original
    service.close()


def test_recognition_image_uses_personal_prototype_and_validates_bbox(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (40, 30), "white").save(path)
    service, recognizer = _worker(tmp_path)
    request = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "recognition-request",
            "task_id": "recognition-task",
            "message_type": "recognition.image",
            "payload": {
                "path": str(path),
                "top_k": 2,
                "personal_prototypes": [
                    {
                        "identity_id": "custom",
                        "display_name": "自定义",
                        "embedding": _vector(),
                        "sample_count": 1,
                    }
                ],
            },
        }
    )
    response = service.handle(request)
    assert response.error is None
    person = response.payload["people"][0]
    assert person["top1"]["character_tag"] == "custom"
    assert person["status"] == DecisionStatus.NEEDS_REVIEW.value
    assert person["top1"]["source"] == "personal_model"
    assert recognizer.embedding_calls == 3

    bad = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "bad-request",
            "task_id": "bad-task",
            "message_type": "recognition.embedding",
            "payload": {"path": str(path), "bbox": [0.1, 0.2, 1.1, 0.9]},
        }
    )
    bad_response = service.handle(bad)
    assert bad_response.error is not None
    assert bad_response.error["code"] == "INVALID_PAYLOAD"
    service.close()


def test_normalize_embedding_rejects_non_finite_and_zero_vectors() -> None:
    bad = [0.0] * EMBEDDING_DIMENSION
    for values in (bad, [math.nan] + bad[1:]):
        try:
            normalize_embedding(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding should be rejected")
