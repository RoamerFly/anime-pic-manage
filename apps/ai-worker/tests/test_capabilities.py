from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace
from typing import TextIO, cast

import pytest

from ai_worker.capabilities import (
    REQUIRED_FEATURE_MODULES,
    apply_cuda_runtime_dir,
    probe_compute_capability,
    probe_cuda_runtime,
    probe_runtime_capabilities,
)
from ai_worker.errors import ModelError, WorkerError
from ai_worker.protocol import make_request
from ai_worker.worker import WorkerService, configure_protocol_streams


class _FakeStream:
    def __init__(self) -> None:
        self.options: dict[str, object] | None = None

    def reconfigure(self, **options: object) -> None:
        self.options = options


def test_runtime_capabilities_imports_required_modules_without_model_loading() -> None:
    report = probe_runtime_capabilities()

    assert report["status"] == "ok"
    assert report["ready"] is True
    assert report["dghs_imgutils"]["distribution"] == "dghs-imgutils"
    assert report["dghs_imgutils"]["version"]
    assert report["onnxruntime"]["version"]
    assert set(report["features"]) == set(REQUIRED_FEATURE_MODULES)
    assert all(item["status"] == "ready" for item in report["features"].values())


def test_runtime_capabilities_returns_structured_import_errors() -> None:
    def failing_import(module_name: str) -> object:
        if module_name == "imgutils.data":
            raise ImportError("simulated missing feature")
        return object()

    report = probe_runtime_capabilities(module_importer=failing_import)

    assert report["status"] == "unavailable"
    assert report["ready"] is False
    failed = report["features"]["imgutils.data"]
    assert failed["status"] == "unavailable"
    assert failed["error"]["code"] == "MODULE_IMPORT_FAILED"
    assert failed["error"]["retryable"] is False
    assert report["errors"]


def test_runtime_capabilities_is_a_worker_protocol_request() -> None:
    response = WorkerService(models_root="missing").handle(make_request("runtime.capabilities"))

    assert response.error is None
    assert response.message_type == "runtime.capabilities.result"
    assert response.payload["dghs_imgutils"]["distribution"] == "dghs-imgutils"


def test_missing_distribution_is_structured() -> None:
    def missing_version(distribution: str) -> str:
        if distribution == "dghs-imgutils":
            raise importlib.metadata.PackageNotFoundError(distribution)
        return "1.0"

    report = probe_runtime_capabilities(distribution_version=missing_version)

    dependency = report["dghs_imgutils"]
    assert dependency["status"] == "unavailable"
    assert dependency["error"]["code"] == "DEPENDENCY_NOT_INSTALLED"


def test_runtime_capabilities_reports_execution_providers() -> None:
    report = probe_runtime_capabilities()

    compute = report["compute"]
    assert "CPUExecutionProvider" in compute["available_providers"]
    assert isinstance(compute["cuda_available"], bool)
    assert compute["distribution"] in {"onnxruntime", "onnxruntime-gpu"}


def test_compute_capability_detects_gpu_runtime() -> None:
    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def fake_import(module_name: str) -> object:
        return FakeOrt if module_name == "onnxruntime" else object()

    compute = probe_compute_capability(
        module_importer=fake_import,
        distribution_version=lambda _: "1.20.1",
    )

    assert compute["cuda_available"] is True
    assert compute["available_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert compute["distribution"] == "onnxruntime-gpu"
    assert compute["error"] is None


def test_compute_capability_reports_provider_probe_failure() -> None:
    def failing_import(module_name: str) -> object:
        raise ImportError(module_name)

    compute = probe_compute_capability(module_importer=failing_import)

    assert compute["cuda_available"] is False
    assert compute["available_providers"] == ["CPUExecutionProvider"]
    assert compute["error"]["code"] == "PROVIDER_PROBE_FAILED"


def test_cuda_runtime_probe_names_the_missing_libraries() -> None:
    def loader(name: str) -> object:
        if name == "cudnn64_9.dll":
            # Frozen builds surface load failures as their own ImportError
            # subclass, so the probe must tolerate both shapes.
            raise ImportError("找不到指定的模块。")
        return SimpleNamespace(_name=f"C:/Windows/System32/{name}")

    report = probe_cuda_runtime(loader=loader)

    assert report["status"] == "missing"
    assert report["missing"] == ["cuDNN 9"]
    assert "cuDNN 9" in report["message"]
    assert [item["loaded"] for item in report["libraries"]] == [True, True, True, False]


def test_cuda_runtime_probe_reports_a_complete_runtime() -> None:
    report = probe_cuda_runtime(loader=lambda name: SimpleNamespace(_name=name))

    assert report["status"] == "ready"
    assert report["missing"] == []
    assert "可加载" in report["message"]


def test_configured_cuda_directory_is_used_as_a_fallback(tmp_path) -> None:
    dll = tmp_path / "cudnn64_9.dll"
    dll.write_bytes(b"fake")

    def loader(name: str) -> object:
        if name.endswith("cudnn64_9.dll") and name == str(dll):
            return SimpleNamespace(_name=name)
        raise OSError("not on PATH")

    applied = apply_cuda_runtime_dir(str(tmp_path))
    try:
        assert applied == str(tmp_path)
        report = probe_cuda_runtime(loader=loader)
        assert report["configured_dir"] == str(tmp_path)
        entry = next(
            item for item in report["libraries"] if item["name"] == "cudnn64_9.dll"
        )
        assert entry["loaded"] is True
        assert entry["path"] == str(dll)
        assert "cuDNN 9" not in report["missing"]
    finally:
        apply_cuda_runtime_dir(None)


def test_missing_cuda_directory_is_ignored() -> None:
    assert apply_cuda_runtime_dir("D:/definitely/not/here") is None
    assert apply_cuda_runtime_dir("") is None


def test_cuda_runtime_probe_is_skipped_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ai_worker.capabilities.os.name", "posix")

    report = probe_cuda_runtime(loader=lambda name: SimpleNamespace(_name=name))

    assert report["status"] == "unsupported"


def test_compute_capability_includes_the_cuda_runtime_report() -> None:
    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    compute = probe_compute_capability(
        module_importer=lambda name: FakeOrt if name == "onnxruntime" else object(),
        distribution_version=lambda _: "1.20.1",
    )

    assert "cuda_runtime" in compute
    assert isinstance(compute["cuda_usable"], bool)


def test_worker_accepts_and_reports_compute_device() -> None:
    service = WorkerService(models_root="missing", compute_device="cpu")

    assert service.compute_device == "cpu"
    report = service.handle(make_request("runtime.device"))
    assert report.error is None
    assert report.payload["requested_device"] == "cpu"
    assert "available_providers" in report.payload

    invalid = service.handle(
        make_request("recognition.image", {"image_path": "missing.png", "device": "tpu"})
    )
    assert invalid.error is not None
    assert invalid.error["code"] == "INVALID_PAYLOAD"


def test_worker_rejects_unknown_compute_device() -> None:
    service = WorkerService(models_root="missing")

    with pytest.raises(WorkerError) as error:
        service.set_compute_device("tpu")

    assert error.value.code == "INVALID_PAYLOAD"


def test_worker_propagates_compute_device_to_cached_adapters() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def set_device(self, device: str) -> None:
            self.devices.append(device)

    service = WorkerService(models_root="missing", compute_device="cpu")
    adapter = FakeAdapter()
    service._adapters["fake-model"] = adapter  # type: ignore[assignment]

    service.set_compute_device("CUDA")

    assert service.compute_device == "cuda"
    assert adapter.devices == ["cuda"]

    # Re-selecting the same device must not churn cached sessions.
    service.set_compute_device("cuda")
    assert adapter.devices == ["cuda"]


def test_worker_applies_onnx_threads_from_request() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.threads: list[int] = []

        def set_onnx_threads(self, threads: int) -> None:
            self.threads.append(threads)

    service = WorkerService(models_root="missing")
    adapter = FakeAdapter()
    service._adapters["fake-model"] = adapter  # type: ignore[assignment]

    service._apply_requested_device({"onnx_threads": 6})

    assert service.onnx_threads == 6
    assert adapter.threads == [6]

    invalid = service.handle(
        make_request("recognition.image", {"image_path": "missing.png", "onnx_threads": 0})
    )
    assert invalid.error is not None
    assert invalid.error["code"] == "INVALID_PAYLOAD"


def test_compute_probe_reports_the_providers_a_session_really_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.activated = False

        def activate(self) -> None:
            self.activated = True

        def device_info(self) -> dict[str, object]:
            return {
                "model_id": "fake-model",
                "requested_device": "auto",
                "active_device": "cuda",
                "active_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            }

    service = WorkerService(models_root="missing", compute_device="auto")
    adapter = FakeAdapter()
    monkeypatch.setattr(
        service, "_probe_manifests", lambda: [SimpleNamespace(id="fake-model")]
    )
    monkeypatch.setattr(service, "_adapter", lambda _manifest: adapter)

    report = service._compute_device_report(probe=True)

    assert adapter.activated is True
    assert report["cuda_session_ready"] is True
    assert report["models"][0]["active_device"] == "cuda"
    assert report["errors"] == []


def test_compute_probe_reports_activation_failure_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAdapter:
        def activate(self) -> None:
            raise ModelError(
                "MODEL_DEVICE_UNAVAILABLE",
                "当前推理依赖未提供 CUDA 执行提供器",
            )

        def device_info(self) -> dict[str, object]:
            return {
                "model_id": "fake-model",
                "requested_device": "cuda",
                "active_device": None,
                "active_providers": [],
            }

    service = WorkerService(models_root="missing", compute_device="cuda")
    monkeypatch.setattr(
        service, "_probe_manifests", lambda: [SimpleNamespace(id="fake-model")]
    )
    monkeypatch.setattr(service, "_adapter", lambda _manifest: FailingAdapter())

    report = service._compute_device_report(probe=True)

    assert report["cuda_session_ready"] is False
    assert report["errors"][0]["model_id"] == "fake-model"
    assert report["errors"][0]["code"] == "MODEL_DEVICE_UNAVAILABLE"


def test_protocol_streams_force_utf8_and_preserve_compatibility() -> None:
    input_stream = _FakeStream()
    output_stream = _FakeStream()
    error_stream = _FakeStream()
    configure_protocol_streams(
        cast(TextIO, input_stream),
        cast(TextIO, output_stream),
        cast(TextIO, error_stream),
    )

    # The request stream tolerates a leading BOM written by Windows shells.
    assert input_stream.options == {"encoding": "utf-8-sig", "errors": "strict"}
    assert output_stream.options == {
        "encoding": "utf-8",
        "errors": "strict",
        "write_through": True,
    }
    assert error_stream.options == output_stream.options

    # StringIO is representative of test/embedded streams without a
    # TextIOWrapper.reconfigure method and must remain accepted.
    import io

    configure_protocol_streams(cast(TextIO, io.StringIO()), cast(TextIO, io.StringIO()))
