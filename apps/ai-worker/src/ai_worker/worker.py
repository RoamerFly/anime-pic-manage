"""JSON Lines worker process and V0.1 service dispatch."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TextIO, cast
from uuid import uuid4

from PIL.Image import Image as PILImage

from . import __version__
from .adapters import COMPUTE_DEVICES, MAX_ONNX_INTRA_OP_THREADS, normalize_compute_device
from .capabilities import (
    apply_cuda_runtime_dir,
    probe_compute_capability,
    probe_runtime_capabilities,
)
from .errors import ModelError, ModelNotInstalledError, ProtocolError, WorkerError
from .handlers import (
    handle_character_set_intersection,
    handle_crops,
    handle_dataset_export,
    handle_enumerate,
    handle_export,
    handle_fusion,
    handle_model_delete,
    handle_model_install,
    handle_personal_model_evaluate,
    handle_personal_model_train,
    handle_recognition_embedding,
    handle_recognition_image,
    handle_similarity_cluster,
    handle_similarity_features_extract,
    handle_similarity_scan,
    normalized_bbox,
    parse_embedding_annotations,
    parse_fusion_config,
    predict_with_embedding,
)
from .images import (
    DetectionBox,
    enumerate_images_with_total,
    inspect_image,
)
from .logging_utils import configure_logging
from .models import Adapter, ModelCatalog, ModelManifest, create_adapter
from .protocol import SCHEMA_VERSION, WorkerRequest, WorkerResponse
from .recognition import FusionConfig, LabelScore


class _DetectorAdapter(Protocol):
    """Runtime shape required from a head detector adapter."""

    def predict(self, image: PILImage) -> list[DetectionBox]: ...
    def unload(self) -> None: ...


class _RecognizerAdapter(Protocol):
    """Runtime shape required from a character recognizer adapter."""

    def predict(self, image: PILImage, top_k: int = 5) -> list[LabelScore]: ...
    def unload(self) -> None: ...


def _configure_text_stream(
    stream: TextIO | None,
    *,
    write_through: bool,
    encoding: str = "utf-8",
) -> None:
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return

    options: dict[str, object] = {"encoding": encoding, "errors": "strict"}
    if write_through:
        options["write_through"] = True

    try:
        reconfigure(**options)
    except (AttributeError, ValueError):
        pass


def configure_protocol_streams(
    input_stream: TextIO, output_stream: TextIO, error_stream: TextIO | None = None
) -> None:
    """Ensure worker protocol streams use UTF-8 and write-through stdout."""

    # `utf-8-sig` on the request stream accepts (and strips) a leading byte
    # order mark. Windows PowerShell 5.1 prepends one when it writes the first
    # request line, which would otherwise fail JSON parsing.
    _configure_text_stream(input_stream, write_through=False, encoding="utf-8-sig")
    _configure_text_stream(output_stream, write_through=True)
    _configure_text_stream(error_stream, write_through=True)


class WorkerService:
    """Dispatches validated IPC requests to pipeline components."""

    def __init__(
        self,
        models_root: str | Path | None = None,
        *,
        character_set_path: str | Path | None = None,
        adapter_factory: Callable[[ModelManifest], Adapter] = create_adapter,
        log_level: str | int | None = None,
        compute_device: str | None = None,
    ) -> None:
        self.logger = logging.getLogger("ai_worker.service")
        if log_level is not None:
            configure_logging(log_level)
        else:
            configure_logging()
        package_root = Path(__file__).resolve().parent
        configured_models = os.environ.get("ANIME_PIC_MODELS")
        # Must happen before any ONNX session is created so the CUDA provider
        # can find a user-provided CUDA 12 / cuDNN 9 folder.
        configured_cuda = apply_cuda_runtime_dir(
            os.environ.get("ANIME_PIC_CUDA_RUNTIME_DIR")
        )
        if configured_cuda:
            self.logger.info(
                "using configured CUDA runtime directory",
                extra={"cuda_runtime_dir": configured_cuda},
            )
        self.models = ModelCatalog(models_root or configured_models or package_root / "models")
        self.character_set_path = Path(
            character_set_path or package_root / "resources" / "character_set.csv"
        )
        self._adapter_factory = adapter_factory
        self._adapters: dict[str, Adapter] = {}
        self._reference_cache: dict[tuple[str, int], Any] = {}
        self._selected_manifests: dict[tuple[str, str | None], ModelManifest] = {}
        self.compute_device = (
            normalize_compute_device(compute_device)
            or normalize_compute_device(os.environ.get("ANIME_PIC_COMPUTE_DEVICE"))
            or "auto"
        )
        # ``None`` keeps the conservative automatic budget; an explicit value
        # (settings page or ANIME_PIC_ONNX_THREADS) overrides it.
        self.onnx_threads: int | None = None
        configured_threads = os.environ.get("ANIME_PIC_ONNX_THREADS", "").strip().casefold()
        if configured_threads and configured_threads != "auto":
            try:
                parsed_threads = int(configured_threads)
            except ValueError:
                parsed_threads = 0
            if 1 <= parsed_threads <= MAX_ONNX_INTRA_OP_THREADS:
                self.onnx_threads = parsed_threads
        self.stopping = False
        try:
            self.models.list_models()
        except Exception:  # noqa: BLE001
            pass

    def handle(self, request: WorkerRequest) -> WorkerResponse:
        try:
            payload = self._dispatch(request)
            return WorkerResponse.ok(request, payload)
        except WorkerError as exc:
            return WorkerResponse.failure(request, exc)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "unexpected internal worker failure",
                extra={
                    "request_id": request.request_id,
                    "task_id": request.task_id,
                    "message_type": request.message_type,
                },
            )
            return WorkerResponse.failure(
                request,
                WorkerError("INTERNAL_ERROR", "Worker 处理请求时发生内部错误", str(exc), True),
            )

    def _dispatch(self, request: WorkerRequest) -> dict[str, Any]:
        message_type = request.message_type.casefold()
        if message_type in {"runtime.capabilities", "runtime_capabilities"}:
            return probe_runtime_capabilities()
        if message_type in {"runtime.device", "runtime_device", "runtime.compute"}:
            return self._compute_device_report(probe=bool(request.payload.get("probe")))
        if message_type == "health":
            return {
                "status": "stopping" if self.stopping else "ok",
                "worker_version": __version__,
                "schema_version": SCHEMA_VERSION,
                "model_count": len(self.models.list_models()),
                "compute_device": self.compute_device,
            }
        if message_type == "shutdown":
            self.close()
            self.stopping = True
            return {"ok": True}
        if message_type in {"model.list", "model_list"}:
            self.models.refresh()
            self._selected_manifests.clear()
            return {"models": self.models.list_models()}
        if message_type in {"model.health", "model_health"}:
            model_id = self._required_string(request.payload, "model_id")
            return self.models.health(model_id)
        if message_type in {"model.install", "model_install"}:
            return handle_model_install(self, request.payload)
        if message_type in {"model.delete", "model_delete"}:
            return handle_model_delete(self, request.payload)
        if message_type in {
            "library.images.list",
            "images.enumerate",
            "image.list",
            "library.scan.preview",
        }:
            return self._enumerate(request.payload)
        if message_type in {"crops.create", "recognition.crops"}:
            return self._crops(request.payload)
        if message_type in {"recognition.image", "image.recognize"}:
            self._apply_requested_device(request.payload)
            return self._recognition_image(request.payload)
        if message_type in {"reference.build", "reference.library.build"}:
            return self._build_reference_library(request.payload)
        if message_type in {"recognition.embedding", "embedding.extract"}:
            self._apply_requested_device(request.payload)
            return self._recognition_embedding(request.payload)
        if message_type in {"personal_model.train", "personal-model.train"}:
            return self._personal_model_train(request.payload)
        if message_type in {"personal_model.evaluate", "personal-model.evaluate"}:
            return self._personal_model_evaluate(request.payload)
        if message_type in {"recognition.fuse", "recognition.fusion"}:
            return self._fusion(request.payload)
        if message_type in {
            "character_set.intersection",
            "character-set.intersection",
            "recognition.filter",
        }:
            return self._character_set_intersection(request.payload)
        if message_type in {"similarity.features.extract", "similarity.features"}:
            return self._similarity_features_extract(request.payload)
        if message_type in {"similarity.cluster", "similarity.clustering"}:
            return self._similarity_cluster(request.payload)
        if message_type in {"similarity.scan", "similarity.scan.start"}:
            return self._similarity_scan(request.payload)
        if message_type in {"dataset.export", "dataset.lora.export"}:
            return handle_dataset_export(request.payload)
        if message_type in {"benchmark.inventory", "benchmark.run"}:
            from .benchmark import collect_benchmark_inventory

            root = self._required_string(request.payload, "root")
            return collect_benchmark_inventory(root)
        raise WorkerError("UNKNOWN_MESSAGE_TYPE", f"不支持的消息类型: {request.message_type}")

    def _apply_requested_device(self, payload: Mapping[str, object]) -> None:
        value = payload.get("device")
        if value is not None:
            if not isinstance(value, str) or normalize_compute_device(value) is None:
                raise WorkerError(
                    "INVALID_PAYLOAD",
                    f"device 必须是 {'、'.join(COMPUTE_DEVICES)} 之一",
                )
            self.set_compute_device(value)

        threads = payload.get("onnx_threads")
        if threads is not None:
            if (
                not isinstance(threads, int)
                or isinstance(threads, bool)
                or not 1 <= threads <= MAX_ONNX_INTRA_OP_THREADS
            ):
                raise WorkerError(
                    "INVALID_PAYLOAD",
                    f"onnx_threads 必须是 1 到 {MAX_ONNX_INTRA_OP_THREADS} 之间的整数",
                )
            self.set_onnx_threads(threads)

    def set_compute_device(self, device: str) -> None:
        """Switch the execution device used by every cached model adapter."""

        normalized = normalize_compute_device(device)
        if normalized is None:
            raise WorkerError("INVALID_PAYLOAD", f"不支持的推理设备: {device}")
        if normalized == self.compute_device:
            return
        self.compute_device = normalized
        for adapter in self._adapters.values():
            setter = getattr(adapter, "set_device", None)
            if callable(setter):
                setter(normalized)

    def set_onnx_threads(self, threads: int) -> None:
        """Switch the ONNX CPU thread budget used by every cached adapter."""

        bounded = max(1, min(MAX_ONNX_INTRA_OP_THREADS, int(threads)))
        if bounded == self.onnx_threads:
            return
        self.onnx_threads = bounded
        for adapter in self._adapters.values():
            setter = getattr(adapter, "set_onnx_threads", None)
            if callable(setter):
                setter(bounded)

    def _adapter_device_info(self, adapter: object) -> dict[str, object] | None:
        device_info = getattr(adapter, "device_info", None)
        if not callable(device_info):
            return None
        return cast(dict[str, object], device_info())

    def _probe_manifests(self) -> list[ModelManifest]:
        """Installed models used for a real CUDA session test, at most one per kind."""

        manifests: list[ModelManifest] = []
        for kind in ("head_detector", "character_recognizer"):
            try:
                manifests.append(self._select_manifest(kind, None))
            except WorkerError:
                continue
        return manifests

    def _compute_device_report(self, *, probe: bool = False) -> dict[str, Any]:
        """Report the compute providers, optionally proving them with a real session.

        ``probe`` loads the installed models with the configured device and
        returns the execution providers each session actually ended up with.
        That is the only trustworthy answer to "is CUDA really being used?",
        because a build can advertise CUDA while the CUDA/cuDNN runtime fails to
        load at session creation time.
        """

        report = probe_compute_capability()
        report["requested_device"] = self.compute_device
        models: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []

        if not probe:
            for adapter in self._adapters.values():
                info = self._adapter_device_info(adapter)
                if info is not None:
                    models.append(info)
        else:
            for manifest in self._probe_manifests():
                probed_adapter: object | None = None
                try:
                    probed_adapter = self._adapter(manifest)
                    activate = getattr(probed_adapter, "activate", None)
                    if callable(activate):
                        activate()
                except Exception as exc:  # noqa: BLE001 - report instead of failing the probe
                    code = getattr(exc, "code", "MODEL_ACTIVATION_FAILED")
                    errors.append(
                        {
                            "model_id": manifest.id,
                            "code": str(code),
                            "error": str(exc),
                        }
                    )
                if probed_adapter is not None:
                    info = self._adapter_device_info(probed_adapter)
                    if info is not None:
                        models.append(info)

        report["models"] = models
        report["errors"] = errors
        report["cuda_session_ready"] = any(
            item.get("active_device") == "cuda" for item in models
        )
        return report

    def _recognition_image(self, payload: Mapping[str, object]) -> dict[str, Any]:
        # A reference library is referenced by path and cached, so the (large)
        # vector payload never travels with every recognition request.
        resolved = dict(payload)
        library = self._load_reference_library(payload.get("reference_library_path"))
        if library is not None:
            resolved["reference_library"] = library
        return handle_recognition_image(self, resolved)

    def _load_reference_library(self, path_value: object) -> Any:
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        from .reference import ReferenceLibrary

        path = Path(path_value).expanduser()
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None
        cache_key = (str(path), stamp)
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            library = ReferenceLibrary.load(path)
        except WorkerError as exc:
            if exc.code == "REFERENCE_INVALID":
                raise
            return None
        self._reference_cache = {cache_key: library}
        return library

    def _build_reference_library(self, payload: Mapping[str, object]) -> dict[str, Any]:
        from .handlers.reference import handle_reference_build

        return handle_reference_build(self, payload)

    @staticmethod
    def _predict_with_embedding(
        recognizer: _RecognizerAdapter,
        image: PILImage,
        top_k: int,
    ) -> tuple[list[LabelScore], tuple[float, ...]]:
        return predict_with_embedding(recognizer, image, top_k)

    def _personal_model_train(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_personal_model_train(payload, required_string_fn=self._required_string)

    def _personal_model_evaluate(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_personal_model_evaluate(payload)

    def _recognition_embedding(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_recognition_embedding(self, payload)

    @classmethod
    def _embedding_annotations(
        cls, payload: Mapping[str, object]
    ) -> list[tuple[str | None, dict[str, float]]]:
        return parse_embedding_annotations(payload)

    @staticmethod
    def _normalized_bbox(value: object, index: int) -> dict[str, float]:
        return normalized_bbox(value, index)

    def _select_manifest(self, kind: str, requested_id: str | None) -> ModelManifest:
        cache_key = (kind, requested_id)
        cached = self._selected_manifests.get(cache_key)
        if cached is not None:
            return cached

        if requested_id is not None:
            manifest = self.models.get(requested_id)
            if manifest.kind != kind:
                raise WorkerError(
                    "MODEL_KIND_MISMATCH",
                    f"模型 {requested_id} 的 kind={manifest.kind}，需要 {kind}",
                )
            self._verify_installed(manifest)
            self._selected_manifests[cache_key] = manifest
            return manifest

        candidates: list[ModelManifest] = []
        for item in self.models.list_models():
            if item.get("kind") != kind or item.get("status") != "installed":
                continue
            model_id = item.get("id")
            if isinstance(model_id, str):
                candidates.append(self.models.get(model_id))
        if not candidates:
            raise ModelNotInstalledError(f"没有已安装的 {kind} 模型")
        manifest = min(candidates, key=lambda item: item.id)
        self._selected_manifests[cache_key] = manifest
        return manifest

    @staticmethod
    def _verify_installed(manifest: ModelManifest) -> None:
        try:
            manifest.verify_sidecars()
            manifest.verify_model_file()
            manifest.validate()
        except ModelError:
            raise
        except (OSError, ValueError) as exc:
            raise ModelNotInstalledError(f"模型 {manifest.id} 不可用", str(exc)) from exc

    def _adapter(self, manifest: ModelManifest) -> Adapter:
        cached = self._adapters.get(manifest.id)
        if cached is not None:
            return cached
        try:
            adapter = self._adapter_factory(manifest)
        except ModelError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WorkerError(
                "MODEL_LOAD_FAILED", f"模型 {manifest.id} 加载失败", str(exc), True
            ) from exc
        setter = getattr(adapter, "set_device", None)
        if callable(setter):
            setter(self.compute_device)
        if self.onnx_threads is not None:
            thread_setter = getattr(adapter, "set_onnx_threads", None)
            if callable(thread_setter):
                thread_setter(self.onnx_threads)
        self._adapters[manifest.id] = adapter
        return adapter

    def _detector_adapter(self, manifest: ModelManifest) -> _DetectorAdapter:
        return cast(_DetectorAdapter, self._adapter(manifest))

    def _recognizer_adapter(self, manifest: ModelManifest) -> _RecognizerAdapter:
        return cast(_RecognizerAdapter, self._adapter(manifest))

    def _requested_model_id(self, payload: Mapping[str, object], role: str) -> str | None:
        role_key = f"{role}_model_id"
        value = payload.get(role_key, payload.get(f"{role}_id"))
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise WorkerError("INVALID_PAYLOAD", f"{role_key} 必须是非空字符串")
            return value

        generic = payload.get("model_id")
        if generic is None:
            return None
        if not isinstance(generic, str) or not generic.strip():
            raise WorkerError("INVALID_PAYLOAD", "model_id 必须是非空字符串")
        try:
            manifest = self.models.get(generic)
        except ModelNotInstalledError:
            return generic
        expected_kind = "head_detector" if role == "detector" else "character_recognizer"
        return generic if manifest.kind == expected_kind else None

    @staticmethod
    def _top_k(value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
            raise WorkerError("INVALID_PAYLOAD", "top_k 必须是 1 到 100 之间的整数")
        return value

    @staticmethod
    def _enumeration_limit(value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 50:
            raise WorkerError("INVALID_PAYLOAD", "limit 必须是 1 到 50 之间的整数")
        return value

    def close(self) -> None:
        for model_id, adapter in tuple(self._adapters.items()):
            try:
                adapter_with_lifecycle = cast(_DetectorAdapter, adapter)
                adapter_with_lifecycle.unload()
            except Exception:  # noqa: BLE001
                self.logger.warning("failed to unload model adapter", extra={"model_id": model_id})
        self._adapters.clear()
        self._selected_manifests.clear()

    def _enumerate(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_enumerate(
            self,
            payload,
            inspect_fn=inspect_image,
            enumerate_fn=enumerate_images_with_total,
        )

    def _crops(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_crops(self, payload)

    def _fusion(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_fusion(self, payload)

    def _character_set_intersection(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_character_set_intersection(self, payload)

    def _export(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_export(self, payload)

    def _fusion_config(self, raw: object) -> FusionConfig:
        return parse_fusion_config(raw)

    @staticmethod
    def _required_string(
        payload: Mapping[str, object], key: str, *, fallback_key: str | None = None
    ) -> str:
        value = payload.get(key)
        if value is None and fallback_key:
            value = payload.get(fallback_key)
        if not isinstance(value, str) or not value.strip():
            raise WorkerError("INVALID_PAYLOAD", f"缺少非空字段 {key}")
        return value

    @staticmethod
    def _boolean(payload: Mapping[str, object], key: str, default: bool) -> bool:
        value = payload.get(key, default)
        if not isinstance(value, bool):
            raise WorkerError("INVALID_PAYLOAD", f"{key} 必须是布尔值")
        return value

    @staticmethod
    def _number(payload: Mapping[str, object], key: str) -> float:
        value = payload.get(key)
        if not isinstance(value, (int, float)):
            raise WorkerError("INVALID_PAYLOAD", f"box.{key} 必须是数字")
        return float(value)

    def _similarity_features_extract(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_similarity_features_extract(payload, required_string_fn=self._required_string)

    def _similarity_cluster(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_similarity_cluster(payload)

    def _similarity_scan(self, payload: Mapping[str, object]) -> dict[str, Any]:
        return handle_similarity_scan(payload, required_string_fn=self._required_string)


def process_line(line: str, service: WorkerService) -> str:
    try:
        request = WorkerRequest.from_json_line(line)
    except ProtocolError as exc:
        request = WorkerRequest(SCHEMA_VERSION, str(uuid4()), str(uuid4()), "protocol", {}, None)
        return WorkerResponse.failure(request, exc).to_json_line()
    return service.handle(request).to_json_line()


def run_json_lines(input_stream: TextIO, output_stream: TextIO, service: WorkerService) -> None:
    try:
        for line in input_stream:
            if not line.strip():
                continue
            output_stream.write(process_line(line, service) + "\n")
            output_stream.flush()
            if service.stopping:
                break
    finally:
        service.close()


def main() -> int:
    configure_protocol_streams(sys.stdin, sys.stdout, sys.stderr)
    service = WorkerService()
    run_json_lines(sys.stdin, sys.stdout, service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
