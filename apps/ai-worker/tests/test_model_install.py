from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from huggingface_hub import constants

from ai_worker.errors import WorkerError
from ai_worker.handlers.models import (
    handle_model_cache_delete,
    handle_model_cache_prefetch,
    handle_model_cache_status,
    handle_model_delete,
    handle_model_install,
)
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


def _seed_cache_file(
    cache_dir: Path, repo_id: str, filename: str, content: bytes
) -> Path:
    """Write one file in the exact layout ``huggingface_hub`` uses.

    Building the real structure (blobs + an ``main`` ref + the snapshot file)
    keeps the test honest: the handler has to resolve the file the same way the
    pipeline does, instead of trusting a mock.
    """

    folder = cache_dir / f"models--{repo_id.replace('/', '--')}"
    digest = hashlib.sha256(content).hexdigest()
    blob = folder / "blobs" / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    (folder / "refs").mkdir(parents=True, exist_ok=True)
    (folder / "refs" / "main").write_text(digest, encoding="utf-8")
    snapshot = folder / "snapshots" / digest / Path(filename)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(content)
    return snapshot


def test_cache_status_reports_missing_partial_and_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "hub"
    _seed_cache_file(cache_dir, "owner/repo", "a/model.onnx", b"x" * 10)
    _seed_cache_file(cache_dir, "other/labels", "selected_tags.csv", b"y" * 4)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_dir))
    service = _Service(tmp_path / "models")

    result = handle_model_cache_status(
        service,
        {
            "entries": [
                {
                    "id": "wd14",
                    "repo_id": "owner/repo",
                    "files": [
                        {"path": "a/model.onnx", "name": "model.onnx"},
                        {
                            "repo_id": "other/labels",
                            "path": "selected_tags.csv",
                            "name": "selected_tags.csv",
                        },
                    ],
                },
                {
                    "id": "partial",
                    "repo_id": "owner/repo",
                    "files": [
                        {"path": "a/model.onnx", "name": "model.onnx"},
                        {"path": "a/missing.onnx", "name": "missing.onnx"},
                    ],
                },
                {
                    "id": "absent",
                    "repo_id": "owner/nowhere",
                    "files": [{"path": "x.onnx", "name": "x.onnx"}],
                },
            ]
        },
    )

    assert result["cache_dir"] == str(cache_dir)
    assert result["models_dir"] == str(service.models.root)
    by_id = {entry["id"]: entry for entry in result["entries"]}

    assert by_id["wd14"]["status"] == "installed"
    assert by_id["wd14"]["total_bytes"] == 14
    assert by_id["wd14"]["repo_ids"] == ["other/labels", "owner/repo"]
    assert by_id["wd14"]["files"][1]["path"].endswith("selected_tags.csv")

    assert by_id["partial"]["status"] == "partial"
    assert by_id["partial"]["total_bytes"] == 10
    assert by_id["partial"]["files"][1]["present"] is False

    assert by_id["absent"]["status"] == "missing"
    assert by_id["absent"]["total_bytes"] == 0
    assert all(item["path"] is None for item in by_id["absent"]["files"])


def test_cache_status_accepts_a_single_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_dir))

    result = handle_model_cache_status(
        _Service(tmp_path / "models"),
        {
            "id": "ccip",
            "repo_id": "owner/repo",
            "files": [{"path": "a.onnx", "name": "a.onnx"}],
        },
    )

    assert [entry["id"] for entry in result["entries"]] == ["ccip"]
    assert result["entries"][0]["status"] == "missing"


def test_cache_status_rejects_an_entry_without_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    with pytest.raises(WorkerError) as error:
        handle_model_cache_status(
            _Service(tmp_path / "models"), {"id": "x", "repo_id": "owner/repo"}
        )

    assert error.value.code == "INVALID_PAYLOAD"


def test_cache_prefetch_downloads_every_declared_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_dir))
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str, cache_dir: str) -> str:
        calls.append((repo_id, filename))
        return str(
            _seed_cache_file(Path(cache_dir), repo_id, filename, f"{repo_id}:{filename}".encode())
        )

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    result = handle_model_cache_prefetch(
        _Service(tmp_path / "models"),
        {
            "id": "wd14",
            "repo_id": "owner/repo",
            "files": [
                {"path": "a/model.onnx", "name": "model.onnx"},
                {
                    "repo_id": "other/labels",
                    "path": "selected_tags.csv",
                    "name": "selected_tags.csv",
                },
            ],
        },
    )

    assert calls == [("owner/repo", "a/model.onnx"), ("other/labels", "selected_tags.csv")]
    entry = result["entries"][0]
    assert entry["status"] == "installed"
    assert entry["total_bytes"] > 0
    assert {item["name"] for item in entry["files"]} == {"model.onnx", "selected_tags.csv"}


def test_cache_prefetch_reports_download_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    def failing_download(**_: object) -> str:
        raise OSError("网络不可达")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failing_download)

    with pytest.raises(WorkerError) as error:
        handle_model_cache_prefetch(
            _Service(tmp_path / "models"),
            {
                "id": "wd14",
                "repo_id": "owner/repo",
                "files": [{"path": "a.onnx", "name": "a.onnx"}],
            },
        )

    assert error.value.code == "MODEL_DOWNLOAD_FAILED"
    assert "a.onnx" in error.value.message


def test_cache_delete_removes_the_repository_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "hub"
    _seed_cache_file(cache_dir, "owner/repo", "a/model.onnx", b"x" * 32)
    _seed_cache_file(cache_dir, "owner/keep", "b.onnx", b"y" * 8)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_dir))

    result = handle_model_cache_delete(
        _Service(tmp_path / "models"),
        {"id": "wd14", "repo_ids": ["owner/repo"]},
    )

    assert result["status"] == "deleted"
    assert result["repo_ids"] == ["owner/repo"]
    assert result["freed_bytes"] > 0
    assert not (cache_dir / "models--owner--repo").exists()
    assert (cache_dir / "models--owner--keep").is_dir()


def test_cache_delete_requires_repository_ids(tmp_path: Path) -> None:
    with pytest.raises(WorkerError) as error:
        handle_model_cache_delete(_Service(tmp_path / "models"), {"id": "wd14"})

    assert error.value.code == "INVALID_PAYLOAD"
