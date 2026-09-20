from __future__ import annotations

import math
from typing import Any

import pytest
from PIL import Image

from ai_worker.images import DetectionBox
from ai_worker.personal_model import (
    ALGORITHM,
    ARTIFACT_SCHEMA_VERSION,
    evaluate_personal_model,
    parse_personal_model,
    train_personal_model,
)
from ai_worker.protocol import SCHEMA_VERSION, WorkerRequest
from ai_worker.recognition import EMBEDDING_DIMENSION
from ai_worker.worker import WorkerService


def _vector(index: int) -> list[float]:
    return [1.0 if position == index else 0.0 for position in range(EMBEDDING_DIMENSION)]


def _samples() -> list[dict[str, object]]:
    return [
        {"identity_id": "a", "display_name": "甲", "embedding": _vector(0)},
        {"identity_id": "a", "display_name": "甲", "embedding": _vector(0)},
        {"identity_id": "a", "display_name": "甲", "embedding": _vector(0)},
        {"identity_id": "b", "display_name": "乙", "embedding": _vector(1)},
        {"identity_id": "b", "display_name": "乙", "embedding": _vector(1)},
        {"identity_id": "b", "display_name": "乙", "embedding": _vector(1)},
    ]


def test_training_returns_deterministic_centroid_artifact_and_metrics() -> None:
    first = train_personal_model("personal-v1", _samples())
    second = train_personal_model("personal-v1", list(reversed(_samples())))

    assert first == second
    assert first["eligible_for_activation"] is True
    assert first["warnings"] == []
    artifact = first["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["algorithm"] == ALGORITHM
    assert artifact["version"] == "personal-v1"
    assert artifact["strict_threshold"] == 0.5
    assert artifact["min_margin"] == 0.5
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["sample_count"] == 6
    assert metrics["class_count"] == 2
    assert metrics["validation_count"] == 6
    assert metrics["accuracy"] == 1.0
    assert [item["identity_id"] for item in artifact["prototypes"]] == ["a", "b"]


def test_training_can_return_a_draft_but_not_activate_insufficient_data() -> None:
    result = train_personal_model(
        "draft-v1",
        [
            {"identity_id": "only", "display_name": "仅有类别", "embedding": _vector(0)},
        ],
    )

    assert result["artifact"]
    assert result["eligible_for_activation"] is False
    assert any("类别数" in warning for warning in result["warnings"])
    assert any("样本不足" in warning for warning in result["warnings"])
    assert any("留一验证" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding", [0.0] * (EMBEDDING_DIMENSION - 1)),
        ("embedding", [math.nan, *_vector(0)[1:]]),
        ("identity_id", ""),
        ("display_name", ""),
        ("embedding", [True, *_vector(0)[1:]]),
        ("embedding", ["1.0", *_vector(0)[1:]]),
    ],
)
def test_training_strictly_validates_labels_and_embeddings(field: str, value: object) -> None:
    sample: dict[str, object] = {
        "identity_id": "a",
        "display_name": "甲",
        "embedding": _vector(0),
    }
    sample[field] = value
    with pytest.raises(ValueError):
        train_personal_model("bad", [sample])


def test_artifact_validation_and_holdout_evaluation() -> None:
    result = train_personal_model("v-eval", _samples())
    artifact = result["artifact"]
    parsed = parse_personal_model(artifact)
    assert parsed.version == "v-eval"

    evaluation = evaluate_personal_model(artifact, _samples())
    assert evaluation["version"] == "v-eval"
    assert evaluation["eligible_for_activation"] is True
    assert evaluation["metrics"]["accuracy"] == 1.0
    assert all(row["accepted"] is True for row in evaluation["rows"])

    invalid: dict[str, Any] = dict(artifact)
    invalid["algorithm"] = "neural_finetune"
    with pytest.raises(ValueError, match="algorithm"):
        parse_personal_model(invalid)
    invalid = dict(artifact)
    invalid["strict_threshold"] = float("inf")
    with pytest.raises(ValueError, match="strict_threshold"):
        parse_personal_model(invalid)


def test_worker_routes_training_and_evaluation_without_loading_models(tmp_path: Any) -> None:
    service = WorkerService(models_root=tmp_path / "no-models")
    training_request = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "train-request",
            "task_id": "train-task",
            "message_type": "personal_model.train",
            "payload": {"version": "worker-v1", "samples": _samples()},
        }
    )
    response = service.handle(training_request)
    assert response.error is None
    artifact = response.payload["artifact"]

    evaluation_request = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "evaluate-request",
            "task_id": "evaluate-task",
            "message_type": "personal_model.evaluate",
            "payload": {"personal_model": artifact, "samples": _samples()},
        }
    )
    evaluated = service.handle(evaluation_request)
    assert evaluated.error is None
    assert evaluated.payload["metrics"]["accuracy"] == 1.0
    service.close()


class _ArtifactDetector:
    def __init__(self, manifest: Any) -> None:
        self.manifest = manifest

    def predict(self, image: Image.Image) -> list[DetectionBox]:
        return [DetectionBox(4, 4, 20, 20, 0.9)]

    def unload(self) -> None:
        pass


class _ArtifactRecognizer:
    def __init__(self, manifest: Any) -> None:
        self.manifest = manifest

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict[str, object]]:
        return [{"tag": "base", "score": 0.1}]

    def predict_with_embedding(
        self, image: Image.Image, top_k: int = 5
    ) -> tuple[list[dict[str, object]], list[float]]:
        return self.predict(image, top_k), _vector(0)

    def unload(self) -> None:
        pass


def test_recognition_image_accepts_artifact_thresholds_and_returns_version(tmp_path: Any) -> None:
    path = tmp_path / "recognition.png"
    Image.new("RGB", (40, 30), "white").save(path)
    detector_manifest = type("Manifest", (), {"id": "detector", "kind": "head_detector"})()
    recognizer_manifest = type(
        "Manifest", (), {"id": "recognizer", "kind": "character_recognizer"}
    )()
    detector = _ArtifactDetector(detector_manifest)
    recognizer = _ArtifactRecognizer(recognizer_manifest)
    service = WorkerService(models_root=tmp_path / "no-models")
    service.models = type(
        "Catalog",
        (),
        {
            "list_models": lambda self: [
                {"id": "detector", "kind": "head_detector", "status": "installed"},
                {
                    "id": "recognizer",
                    "kind": "character_recognizer",
                    "status": "installed",
                },
            ],
            "get": lambda self, model_id: {
                "detector": detector_manifest,
                "recognizer": recognizer_manifest,
            }[model_id],
        },
    )()  # type: ignore[assignment]
    service._adapter_factory = lambda manifest: (  # type: ignore[assignment]
        detector if manifest.kind == "head_detector" else recognizer
    )
    artifact = train_personal_model("artifact-v7", _samples())["artifact"]
    request = WorkerRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": "recognition-request",
            "task_id": "recognition-task",
            "message_type": "recognition.image",
            "payload": {"path": str(path), "personal_model": artifact},
        }
    )
    response = service.handle(request)
    assert response.error is None
    assert response.payload["model_version"] == "artifact-v7"
    person = response.payload["people"][0]
    assert person["model_version"] == "artifact-v7"
    assert person["top1"]["source"] == "personal_model"
    service.close()
