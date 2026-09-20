"""Runtime dependency and feature capability checks.

The check intentionally imports only the modules used by the recognition
pipeline.  It does not enumerate models, create an ONNX session, or download
anything, so it is safe to call while the desktop settings page is probing a
Worker runtime.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import os
from collections.abc import Callable
from typing import Any

ModuleImporter = Callable[[str], object]
DistributionVersion = Callable[[str], str]

REQUIRED_FEATURE_MODULES: dict[str, str] = {
    "imgutils.preprocess.pillow": "imgutils.preprocess.pillow",
    "imgutils.generic.yolo": "imgutils.generic.yolo",
    "imgutils.data": "imgutils.data",
    # Used by the LoRA training-set export to caption cropped images.
    "imgutils.tagging": "imgutils.tagging",
}

# CPU and GPU ONNX Runtime wheels install the same ``onnxruntime`` module but
# register different distribution names, so both have to be probed.
ONNXRUNTIME_DISTRIBUTIONS: tuple[str, ...] = ("onnxruntime-gpu", "onnxruntime")
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"

# onnxruntime-gpu 1.20 needs the CUDA 12 and cuDNN 9 runtimes on PATH. Listing
# the CUDA provider only proves the wheel was built with it; if these libraries
# cannot be loaded the session silently falls back to CPU, which is the single
# most confusing GPU failure mode, so they are probed explicitly.
CUDA_RUNTIME_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("cudart64_12.dll", "CUDA Runtime 12"),
    ("cublas64_12.dll", "cuBLAS 12"),
    ("cublasLt64_12.dll", "cuBLASLt 12"),
    ("cudnn64_9.dll", "cuDNN 9"),
)

# Set once at startup from ANIME_PIC_CUDA_RUNTIME_DIR so the probe can explain
# where the libraries were expected to come from.
_CUDA_RUNTIME_DIR: str | None = None


def apply_cuda_runtime_dir(directory: str | None) -> str | None:
    """Make a user-provided CUDA/cuDNN folder visible to ONNX Runtime.

    Installing CUDA system-wide is not always desirable (and this project must
    not modify the user's system), so the desktop can point at a folder that
    already contains the CUDA 12 / cuDNN 9 DLLs. Both the ``PATH`` entry and
    ``os.add_dll_directory`` are registered because ONNX Runtime and ``ctypes``
    resolve their dependencies differently.
    """

    global _CUDA_RUNTIME_DIR
    value = (directory or "").strip()
    if not value:
        _CUDA_RUNTIME_DIR = None
        return None
    path = os.path.abspath(value)
    if not os.path.isdir(path):
        _CUDA_RUNTIME_DIR = None
        return None
    if os.name == "nt":
        current = os.environ.get("PATH", "")
        if path not in current.split(os.pathsep):
            os.environ["PATH"] = path + os.pathsep + current
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError):  # pragma: no cover - platform specific
            pass
    _CUDA_RUNTIME_DIR = path
    return path


def cuda_runtime_dir() -> str | None:
    return _CUDA_RUNTIME_DIR


def _error(code: str, message: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "detail": detail,
        "retryable": False,
    }


def _probe_distribution(
    distribution: str,
    version: DistributionVersion,
) -> dict[str, Any]:
    try:
        installed_version = version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        return {
            "distribution": distribution,
            "version": None,
            "status": "unavailable",
            "error": _error(
                "DEPENDENCY_NOT_INSTALLED",
                f"未安装 Python 发行包 {distribution}",
                str(exc) or None,
            ),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must remain structured
        return {
            "distribution": distribution,
            "version": None,
            "status": "unavailable",
            "error": _error(
                "DISTRIBUTION_VERSION_FAILED",
                f"无法读取发行包 {distribution} 的版本",
                f"{type(exc).__name__}: {exc}",
            ),
        }
    return {
        "distribution": distribution,
        "version": installed_version,
        "status": "ready",
        "error": None,
    }


def _probe_module(module_name: str, importer: ModuleImporter) -> dict[str, Any]:
    try:
        importer(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostics must remain structured
        return {
            "module": module_name,
            "status": "unavailable",
            "error": _error(
                "MODULE_IMPORT_FAILED",
                f"无法导入必需模块 {module_name}",
                f"{type(exc).__name__}: {exc}",
            ),
        }
    return {"module": module_name, "status": "ready", "error": None}


def _probe_runtime_distribution(
    distributions: tuple[str, ...],
    version: DistributionVersion,
) -> dict[str, Any]:
    reports = [_probe_distribution(name, version) for name in distributions]
    for report in reports:
        if report["status"] == "ready":
            return report
    return reports[-1]


def _probe_execution_providers(
    module_importer: ModuleImporter,
) -> tuple[list[str], dict[str, Any] | None]:
    try:
        runtime = module_importer("onnxruntime")
    except Exception as exc:  # noqa: BLE001 - diagnostics must remain structured
        return [CPU_EXECUTION_PROVIDER], _error(
            "PROVIDER_PROBE_FAILED",
            "无法读取 ONNX Runtime 执行提供器",
            f"{type(exc).__name__}: {exc}",
        )
    getter = getattr(runtime, "get_available_providers", None)
    if not callable(getter):
        return [CPU_EXECUTION_PROVIDER], None
    try:
        providers = [str(provider) for provider in getter()]
    except Exception as exc:  # noqa: BLE001 - diagnostics must remain structured
        return [CPU_EXECUTION_PROVIDER], _error(
            "PROVIDER_PROBE_FAILED",
            "无法读取 ONNX Runtime 执行提供器",
            f"{type(exc).__name__}: {exc}",
        )
    return providers or [CPU_EXECUTION_PROVIDER], None


def probe_compute_capability(
    *,
    module_importer: ModuleImporter = importlib.import_module,
    distribution_version: DistributionVersion = importlib.metadata.version,
) -> dict[str, Any]:
    """Report the ONNX Runtime build and its available execution providers."""

    distribution = _probe_runtime_distribution(ONNXRUNTIME_DISTRIBUTIONS, distribution_version)
    providers, provider_error = _probe_execution_providers(module_importer)
    cuda_runtime = probe_cuda_runtime()
    return {
        "distribution": distribution["distribution"] if distribution["status"] == "ready" else None,
        "available_providers": providers,
        "cuda_available": CUDA_EXECUTION_PROVIDER in providers,
        "cuda_runtime": cuda_runtime,
        "cuda_usable": CUDA_EXECUTION_PROVIDER in providers
        and cuda_runtime["status"] == "ready",
        "error": provider_error,
    }


def probe_cuda_runtime(*, loader: Callable[[str], object] | None = None) -> dict[str, Any]:
    """Check whether the CUDA/cuDNN libraries onnxruntime-gpu loads are present.

    ``onnxruntime`` advertises ``CUDAExecutionProvider`` whenever the wheel was
    built with CUDA support, even on machines where cuDNN is missing; the
    session then quietly falls back to CPU. Loading each library reports the real
    state, and the names make the fix obvious ("install cuDNN 9").
    """

    if os.name != "nt":
        return {
            "status": "unsupported",
            "libraries": [],
            "missing": [],
            "configured_dir": _CUDA_RUNTIME_DIR,
            "message": "当前平台不做 CUDA 动态库探测。",
        }
    load = loader or ctypes.WinDLL  # type: ignore[attr-defined]
    fallback_dir = _CUDA_RUNTIME_DIR
    libraries: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, label in CUDA_RUNTIME_LIBRARIES:
        entry: dict[str, Any] = {"name": name, "label": label, "loaded": False}
        try:
            handle = load(name)
        except (OSError, ImportError) as exc:
            # Not on PATH; retry from the configured folder before giving up.
            resolved = os.path.join(fallback_dir, name) if fallback_dir else None
            if resolved and os.path.isfile(resolved):
                try:
                    handle = load(resolved)
                except (OSError, ImportError) as nested:
                    entry["detail"] = f"{type(nested).__name__}: {nested}"
                    missing.append(label)
                    libraries.append(entry)
                    continue
                entry["loaded"] = True
                entry["path"] = resolved
                libraries.append(entry)
                continue
            entry["detail"] = f"{type(exc).__name__}: {exc}"
            missing.append(label)
        else:
            entry["loaded"] = True
            entry["path"] = getattr(handle, "_name", None)
        libraries.append(entry)
    if missing:
        message = (
            "缺少 " + "、".join(missing) + "：ONNX Runtime 会自动回落到 CPU。"
            "请安装 CUDA 12 + cuDNN 9，或在设置里指定包含这些 DLL 的目录。"
        )
    else:
        message = "CUDA 12 与 cuDNN 9 运行时均可加载。"
    return {
        "status": "missing" if missing else "ready",
        "libraries": libraries,
        "missing": missing,
        "configured_dir": _CUDA_RUNTIME_DIR,
        "message": message,
    }


def probe_runtime_capabilities(
    *,
    module_importer: ModuleImporter = importlib.import_module,
    distribution_version: DistributionVersion = importlib.metadata.version,
) -> dict[str, Any]:
    """Return a JSON-serializable, model-free runtime capability report."""

    dghs = _probe_distribution("dghs-imgutils", distribution_version)
    onnxruntime = _probe_runtime_distribution(ONNXRUNTIME_DISTRIBUTIONS, distribution_version)
    # Import the runtime itself as well as reading its distribution metadata;
    # metadata alone can remain present after a broken installation.
    onnxruntime_import = _probe_module("onnxruntime", module_importer)
    if onnxruntime["status"] == "ready" and onnxruntime_import["status"] != "ready":
        onnxruntime = {
            **onnxruntime,
            "status": "unavailable",
            "version": None,
            "error": onnxruntime_import["error"],
        }

    features = {
        feature: _probe_module(module_name, module_importer)
        for feature, module_name in REQUIRED_FEATURE_MODULES.items()
    }
    ready = all(
        dependency["status"] == "ready"
        for dependency in (dghs, onnxruntime)
    ) and all(feature["status"] == "ready" for feature in features.values())
    errors = [
        dependency["error"]
        for dependency in (dghs, onnxruntime)
        if dependency["error"] is not None
    ] + [feature["error"] for feature in features.values() if feature["error"] is not None]
    return {
        "status": "ok" if ready else "unavailable",
        "ready": ready,
        "dghs_imgutils": dghs,
        "onnxruntime": onnxruntime,
        "compute": probe_compute_capability(
            module_importer=module_importer,
            distribution_version=distribution_version,
        ),
        "features": features,
        "errors": errors,
    }
