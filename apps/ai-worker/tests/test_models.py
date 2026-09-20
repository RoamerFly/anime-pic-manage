from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_worker.models as models
from ai_worker.errors import ModelError, ModelManifestError
from ai_worker.models import (
    AnimeTimmONNXAdapter,
    ModelManifest,
    PreprocessSpec,
    make_onnx_session_options,
)


def _pipeline() -> list[dict[str, object]]:
    return [
        {
            "background_color": "white",
            "interpolation": "bilinear",
            "size": [512, 512],
            "type": "pad_to_size",
        },
        {
            "antialias": True,
            "interpolation": "bilinear",
            "max_size": None,
            "size": 384,
            "type": "resize",
        },
        {"size": [384, 384], "type": "center_crop"},
        {"type": "maybe_to_tensor"},
        {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "type": "normalize",
        },
    ]


def _write_character_manifest(
    root: Path,
    *,
    tag_count: int = 2,
    supported_devices: tuple[str, ...] = ("cpu",),
) -> ModelManifest:
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_tags.csv").write_text(
        "tag,category\ncharacter_alpha,4\ncharacter_beta,4\n",
        encoding="utf-8",
    )
    (root / "preprocess.json").write_text(json.dumps({"test": _pipeline()}), encoding="utf-8")
    (root / "categories.json").write_text(
        json.dumps([{"category": 4, "name": "character"}]), encoding="utf-8"
    )
    (root / "thresholds.csv").write_text(
        "category,name,alpha,threshold\n4,character,1.0,0.49\n",
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "id": "test-character-model",
                "kind": "character_recognizer",
                "adapter": "animetimm_onnx",
                "format": "onnx",
                "version": "test",
                "model_file": "model.onnx",
                "labels_file": "selected_tags.csv",
                "preprocess_file": "preprocess.json",
                "sha256": "",
                "input": {"width": 384, "height": 384},
                "supported_devices": list(supported_devices),
                "input_name": "input",
                "output_name": "prediction",
                "input_layout": "nchw",
                "output_activation": "none",
                "categories_file": "categories.json",
                "thresholds_file": "thresholds.csv",
                "tag_count": tag_count,
            }
        ),
        encoding="utf-8",
    )
    return ModelManifest.from_file(root / "metadata.json")


def test_realistic_test_pipeline_is_preserved_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "preprocess.json"
    path.write_text(json.dumps({"test": _pipeline()}), encoding="utf-8")

    spec = PreprocessSpec.from_json_file(path, expected_size=(384, 384))

    assert spec.width == 384
    assert spec.height == 384
    assert spec.layout == "nchw"
    assert spec.activation == "none"
    assert [step["type"] for step in spec.pipeline or ()] == [
        "pad_to_size",
        "resize",
        "center_crop",
        "maybe_to_tensor",
        "normalize",
    ]


def test_test_pipeline_rejects_wrong_order_or_size(tmp_path: Path) -> None:
    path = tmp_path / "preprocess.json"
    pipeline = _pipeline()
    pipeline[2] = {"size": [448, 448], "type": "center_crop"}
    path.write_text(json.dumps({"test": pipeline}), encoding="utf-8")

    with pytest.raises(ModelManifestError, match="center_crop 尺寸"):
        PreprocessSpec.from_json_file(path, expected_size=(384, 384))


def test_manifest_requires_explicit_input_and_output_names(tmp_path: Path) -> None:
    manifest = _write_character_manifest(tmp_path)
    raw = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    raw.pop("input_name")
    (tmp_path / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    without_input = ModelManifest.from_file(tmp_path / "metadata.json")

    with pytest.raises(ModelManifestError, match="input_name"):
        without_input.validate()
    assert manifest.options["output_name"] == "prediction"


def test_character_sidecars_validate_category_tag_count_and_threshold(tmp_path: Path) -> None:
    manifest = _write_character_manifest(tmp_path)

    labels, preprocess = manifest.validate()

    assert len(labels) == 2
    assert preprocess.width == 384

    bad_manifest = _write_character_manifest(tmp_path / "bad", tag_count=3)
    with pytest.raises(ModelManifestError, match="tag_count"):
        bad_manifest.validate()


def test_character_sidecars_require_character_threshold_row(tmp_path: Path) -> None:
    manifest = _write_character_manifest(tmp_path)
    (tmp_path / "thresholds.csv").write_text(
        "category,name,alpha,threshold\n0,general,1.0,0.33\n", encoding="utf-8"
    )

    with pytest.raises(ModelManifestError, match="thresholds.csv"):
        manifest.validate()


def test_session_output_dimension_is_checked_without_loading_weights(tmp_path: Path) -> None:
    manifest = _write_character_manifest(tmp_path)
    labels, preprocess = manifest.validate()
    adapter = AnimeTimmONNXAdapter(manifest)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input", shape=["batch", 3, "height", "width"])],
        get_outputs=lambda: [SimpleNamespace(name="prediction", shape=["batch", 2])],
    )

    adapter._validate_session_contract(session, labels, preprocess)

    bad_session = SimpleNamespace(
        get_inputs=session.get_inputs,
        get_outputs=lambda: [SimpleNamespace(name="prediction", shape=["batch", 3])],
    )
    with pytest.raises(ModelError, match="输出维度"):
        adapter._validate_session_contract(bad_session, labels, preprocess)


@pytest.mark.parametrize(
    ("cpu_count", "expected_threads"),
    [(1, 1), (2, 1), (8, 4), (None, 1)],
)
def test_onnx_session_options_bound_cpu_threads(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, expected_threads: int
) -> None:
    class FakeSessionOptions:
        pass

    class FakeOrt:
        SessionOptions = FakeSessionOptions
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

    monkeypatch.setattr(models.os, "cpu_count", lambda: cpu_count)

    options = make_onnx_session_options(FakeOrt)

    assert options.intra_op_num_threads == expected_threads
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
    assert options.graph_optimization_level == "all"
    assert options.enable_mem_pattern is True
    assert options.enable_cpu_mem_arena is True


def test_activate_passes_configured_options_to_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    captured: dict[str, object] = {}

    class FakeSessionOptions:
        pass

    class FakeSession:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input", shape=["batch", 3, 384, 384])]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="prediction", shape=["batch", 2])]

    class FakeOrt:
        SessionOptions = FakeSessionOptions
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

        @staticmethod
        def InferenceSession(
            path: str, *, sess_options: object, providers: list[str]
        ) -> FakeSession:
            captured.update(path=path, sess_options=sess_options, providers=providers)
            return FakeSession()

    monkeypatch.setattr(models.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (SimpleNamespace(), FakeOrt))

    adapter.activate()

    options = captured["sess_options"]
    assert isinstance(options, FakeSessionOptions)
    assert options.intra_op_num_threads == 4
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
    assert options.graph_optimization_level == "all"
    assert captured["providers"] == ["CPUExecutionProvider"]


def _activate_with_providers(
    monkeypatch: pytest.MonkeyPatch,
    adapter: AnimeTimmONNXAdapter,
    providers: list[str],
    *,
    failing_providers: tuple[str, ...] = (),
) -> list[list[str]]:
    """Activate ``adapter`` against a fake ONNX Runtime and record its attempts."""

    captured: list[list[str]] = []

    class FakeSessionOptions:
        pass

    class FakeSession:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input", shape=["batch", 3, 384, 384])]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="prediction", shape=["batch", 2])]

    class FakeOrt:
        SessionOptions = FakeSessionOptions
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

        @staticmethod
        def get_available_providers() -> list[str]:
            return list(providers)

        @staticmethod
        def InferenceSession(
            path: str, *, sess_options: object, providers: list[str]
        ) -> FakeSession:
            if providers[0] in failing_providers:
                raise RuntimeError(f"provider unavailable: {providers[0]}")
            captured.append(list(providers))
            return FakeSession()

    monkeypatch.setattr(models.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (SimpleNamespace(), FakeOrt))
    adapter.activate()
    return captured


def test_cuda_device_uses_cuda_execution_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path, supported_devices=("cpu", "cuda"))
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    adapter.set_device("cuda")

    captured = _activate_with_providers(
        monkeypatch,
        adapter,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert captured == [["CUDAExecutionProvider", "CPUExecutionProvider"]]
    assert adapter.device_info()["active_device"] == "cuda"
    assert adapter.device_info()["requested_device"] == "cuda"


def test_auto_device_falls_back_to_cpu_when_cuda_session_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path, supported_devices=("cpu", "cuda"))
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    adapter.set_device("auto")

    captured = _activate_with_providers(
        monkeypatch,
        adapter,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        failing_providers=("CUDAExecutionProvider",),
    )

    assert captured == [["CPUExecutionProvider"]]
    assert adapter.device_info()["active_device"] == "cpu"


def test_auto_device_stays_on_cpu_when_manifest_declares_cpu_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    adapter.set_device("auto")

    captured = _activate_with_providers(
        monkeypatch,
        adapter,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert captured == [["CPUExecutionProvider"]]
    assert adapter.device_info()["active_device"] == "cpu"


def test_explicit_cuda_without_provider_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path, supported_devices=("cpu", "cuda"))
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    adapter.set_device("cuda")

    class FakeOrt:
        SessionOptions = object
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(adapter, "_import_runtime", lambda: (SimpleNamespace(), FakeOrt))

    with pytest.raises(ModelError) as error:
        adapter.activate()

    assert error.value.code == "MODEL_DEVICE_UNAVAILABLE"
    assert adapter.device_info()["active_device"] is None


def test_explicit_cuda_requires_manifest_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    adapter.set_device("cuda")

    with pytest.raises(ModelError) as error:
        adapter.activate()

    assert error.value.code == "MODEL_DEVICE_UNSUPPORTED"


def test_switching_device_drops_the_cached_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path, supported_devices=("cpu", "cuda"))
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    _activate_with_providers(monkeypatch, adapter, ["CPUExecutionProvider"])
    assert adapter.device_info()["active_device"] == "cpu"

    adapter.set_device("cuda")

    assert adapter.device_info()["requested_device"] == "cuda"
    assert adapter.device_info()["active_device"] is None
    assert adapter._session is None


def test_onnx_thread_override_is_bounded_and_reloads_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    captured: dict[str, object] = {}

    class FakeSessionOptions:
        pass

    class FakeSession:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input", shape=["batch", 3, 384, 384])]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="prediction", shape=["batch", 2])]

    class FakeOrt:
        SessionOptions = FakeSessionOptions
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

        @staticmethod
        def InferenceSession(
            path: str, *, sess_options: object, providers: list[str]
        ) -> FakeSession:
            captured["sess_options"] = sess_options
            return FakeSession()

    monkeypatch.setattr(models.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (SimpleNamespace(), FakeOrt))

    adapter.set_onnx_threads(6)
    adapter.activate()
    options = captured["sess_options"]
    assert isinstance(options, FakeSessionOptions)
    assert options.intra_op_num_threads == 6

    # Out-of-range values are clamped instead of failing inference.
    adapter.set_onnx_threads(999)
    assert adapter.device_info()["requested_threads"] == 16
    assert adapter._session is None


def test_automatic_thread_budget_stays_conservative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_character_manifest(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"test model")
    adapter = AnimeTimmONNXAdapter(manifest)
    captured: dict[str, object] = {}

    class FakeSessionOptions:
        pass

    class FakeOrt:
        SessionOptions = FakeSessionOptions
        ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

        @staticmethod
        def InferenceSession(
            path: str, *, sess_options: object, providers: list[str]
        ) -> SimpleNamespace:
            captured["sess_options"] = sess_options
            return SimpleNamespace()

    monkeypatch.setattr(models.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(adapter, "_import_runtime", lambda: (SimpleNamespace(), FakeOrt))
    monkeypatch.setattr(adapter, "_validate_session_contract", lambda *args: None)

    adapter.activate()

    options = captured["sess_options"]
    assert isinstance(options, FakeSessionOptions)
    assert options.intra_op_num_threads == 4
