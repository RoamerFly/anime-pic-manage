from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_worker.errors import WorkerError
from ai_worker.handlers.models import handle_model_delete, handle_model_install
from ai_worker.models_manifest import PreprocessSpec


class _Catalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.refreshed = 0

    def refresh(self) -> None:
        self.refreshed += 1

    def get(self, model_id: str):
        metadata = json.loads(
            (self.root / "recognizer" / model_id / "metadata.json").read_text()
        )
        return type("Manifest", (), {"kind": metadata["kind"], "id": model_id})()


class _Service:
    def __init__(self, root: Path) -> None:
        self.models = _Catalog(root)


def test_camie_style_stages_are_accepted() -> None:
    spec = PreprocessSpec._from_stages(
        [
            {"type": "pad_to_size", "size": [512, 512], "background_color": [0, 0, 0]},
            {"type": "to_tensor"},
        ],
        expected_size=(512, 512),
        activation="sigmoid",
        layout="nchw",
    )

    assert spec.width == 512 and spec.height == 512
    assert spec.activation == "sigmoid"
    assert spec.crop_mode == "pipeline"
    assert spec.pipeline is not None and len(spec.pipeline) == 2


def test_stages_must_match_the_declared_input_size() -> None:
    with pytest.raises(Exception) as error:
        PreprocessSpec._from_stages(
            [{"type": "pad_to_size", "size": [256, 256]}, {"type": "to_tensor"}],
            expected_size=(512, 512),
            activation="sigmoid",
            layout="nchw",
        )

    assert "pad_to_size" in str(error.value)


def test_stages_reject_unknown_steps() -> None:
    with pytest.raises(Exception) as error:
        PreprocessSpec._from_stages(
            [{"type": "pad_to_size", "size": [512, 512]}, {"type": "teleport"}],
            expected_size=(512, 512),
            activation="sigmoid",
            layout="nchw",
        )

    assert "未知步骤" in str(error.value)


def test_install_writes_manifest_and_generated_sidecars(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"onnx weights")
    service = _Service(models_root)
    payload = {
        "id": "camie-test",
        "kind": "character_recognizer",
        "repo_id": "deepghs/camie_tagger_onnx",
        "files": [{"path": "initial/model.onnx", "name": "model.onnx"}],
        "metadata": {
            "id": "camie-test",
            "kind": "character_recognizer",
            "adapter": "camie_onnx",
            "format": "onnx",
            "version": "initial",
            "model_file": "model.onnx",
            "labels_file": "selected_tags.csv",
            "preprocess_file": "preprocess.json",
            "input": {"width": 512, "height": 512},
            "supported_devices": ["cpu"],
        },
        "generate": {
            "categories.json": [{"name": "character", "category": 4}],
            "thresholds.csv": "name,category,threshold\ncharacter,4,0.25\n",
        },
    }

    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename: str(weights),
        raising=False,
    )
    try:
        result = handle_model_install(service, payload)
    finally:
        monkey.undo()

    target = models_root / "recognizer" / "camie-test"
    assert result["status"] == "installed"
    assert (target / "model.onnx").is_file()
    assert (target / "categories.json").is_file()
    assert "character,4,0.25" in (target / "thresholds.csv").read_text(encoding="utf-8")
    manifest = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert len(manifest["sha256"]) == 64
    assert service.models.refreshed == 1


def test_install_rejects_path_traversal(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    service = _Service(models_root)

    with pytest.raises(WorkerError) as error:
        handle_model_install(
            service,
            {
                "id": "x",
                "kind": "character_recognizer",
                "repo_id": "r",
                "files": [{"path": "a", "name": "../escape.onnx"}],
                "metadata": {"model_file": "escape.onnx"},
            },
        )

    assert error.value.code == "INVALID_PAYLOAD"


def test_delete_removes_the_model_directory(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    target = models_root / "recognizer" / "camie-test"
    target.mkdir(parents=True)
    (target / "metadata.json").write_text(
        json.dumps({"kind": "character_recognizer"}), encoding="utf-8"
    )
    service = _Service(models_root)

    result = handle_model_delete(service, {"id": "camie-test"})

    assert result["status"] == "deleted"
    assert not target.exists()
    assert service.models.refreshed == 1
