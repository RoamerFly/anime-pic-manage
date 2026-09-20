"""ONNX adapters and lazy runtime model execution."""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image, ImageOps

from .errors import ModelError, ModelManifestError, ModelNotInstalledError
from .images import DetectionBox
from .models_manifest import ModelLabel, ModelManifest, PreprocessSpec, sha256_file
from .recognition import EMBEDDING_DIMENSION, LabelScore, normalize_embedding, top_k_scores

CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
COMPUTE_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
MAX_ONNX_INTRA_OP_THREADS = 16


class Adapter(Protocol):
    manifest: ModelManifest

    def health(self) -> dict[str, object]: ...


def available_execution_providers(ort: Any) -> tuple[str, ...]:
    """Return the execution providers exposed by the installed ONNX Runtime build."""

    getter = getattr(ort, "get_available_providers", None)
    if not callable(getter):
        return (CPU_EXECUTION_PROVIDER,)
    try:
        providers = tuple(str(provider) for provider in getter())
    except Exception:  # noqa: BLE001 - provider probing must never break inference
        return (CPU_EXECUTION_PROVIDER,)
    return providers or (CPU_EXECUTION_PROVIDER,)


def normalize_compute_device(value: object) -> str | None:
    """Return the canonical device name, or ``None`` when it is not supported."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in COMPUTE_DEVICES else None


def bounded_onnx_intra_op_threads(cpu_count: int | None) -> int:
    """Reserve at least one logical core for the desktop UI and worker IPC."""

    if cpu_count is None or cpu_count < 1:
        return 1
    return max(1, min(4, cpu_count - 1))


def make_onnx_session_options(
    ort: Any,
    *,
    cpu_count: int | None = None,
    intra_op_threads: int | None = None,
) -> Any:
    """Build the shared, conservative CPU/memory policy for ONNX sessions."""

    options = ort.SessionOptions()
    if intra_op_threads is None:
        options.intra_op_num_threads = bounded_onnx_intra_op_threads(
            os.cpu_count() if cpu_count is None else cpu_count
        )
    else:
        # A user-selected thread count is honored directly (still bounded) so
        # the settings page can trade latency against leaving cores free.
        options.intra_op_num_threads = max(
            1, min(MAX_ONNX_INTRA_OP_THREADS, intra_op_threads)
        )
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Keep ORT's arena and memory-pattern reuse enabled: sessions are cached and
    # reused by the adapter, so this avoids repeated allocation without allowing
    # an unbounded thread pool to compete with the desktop UI.
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    return options


class ONNXAdapter:
    """Common lazy ONNX Runtime lifecycle.

    Importing the worker does not import numpy/onnxruntime.  Activation is the
    only point at which optional model dependencies are required.
    """

    manifest: ModelManifest

    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest
        self._session: Any | None = None
        self._labels: tuple[ModelLabel, ...] = ()
        self._preprocess: PreprocessSpec | None = None
        self._numpy: Any | None = None
        self._requested_device = "cpu"
        self._requested_threads: int | None = None
        self._active_device: str | None = None
        self._active_providers: tuple[str, ...] = ()

    def _import_runtime(self) -> tuple[Any, Any]:
        try:
            np = importlib.import_module("numpy")
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ModelNotInstalledError(
                f"模型 {self.manifest.id} 缺少 ONNX Runtime 依赖，请安装 worker 的 model 依赖",
                str(exc),
            ) from exc
        return np, ort

    def set_device(self, device: str) -> None:
        """Remember the preferred execution device and drop a stale session."""

        normalized = normalize_compute_device(device)
        if normalized is None:
            raise ModelError(
                "MODEL_DEVICE_UNSUPPORTED",
                f"不支持的推理设备: {device}",
                retryable=False,
            )
        if normalized == self._requested_device:
            return
        self.unload()
        self._requested_device = normalized

    def set_onnx_threads(self, threads: int) -> None:
        """Set the ONNX intra-op thread budget, reloading a stale session."""

        bounded = max(1, min(MAX_ONNX_INTRA_OP_THREADS, int(threads)))
        if bounded == self._requested_threads:
            return
        self.unload()
        self._requested_threads = bounded

    def device_info(self) -> dict[str, object]:
        return {
            "model_id": self.manifest.id,
            "requested_device": self._requested_device,
            "requested_threads": self._requested_threads,
            "active_device": self._active_device,
            "active_providers": list(self._active_providers),
            "supported_devices": list(self.manifest.supported_devices),
        }

    def activate(self, *, device: str | None = None) -> None:
        requested = self._requested_device if device is None else device
        normalized = normalize_compute_device(requested)
        if normalized is None:
            raise ModelError(
                "MODEL_DEVICE_UNSUPPORTED",
                f"不支持的推理设备: {requested}",
                retryable=False,
            )
        if normalized == "cuda" and "cuda" not in self.manifest.supported_devices:
            raise ModelError(
                "MODEL_DEVICE_UNSUPPORTED",
                f"模型 {self.manifest.id} 未声明 CUDA 支持，请更新模型元数据后重试",
                retryable=False,
            )
        self.manifest.verify_model_file()
        labels, preprocess = self.manifest.validate()
        np, ort = self._import_runtime()
        cuda_available = CUDA_EXECUTION_PROVIDER in available_execution_providers(ort)
        if normalized == "cuda" and not cuda_available:
            raise ModelError(
                "MODEL_DEVICE_UNAVAILABLE",
                "当前推理依赖未提供 CUDA 执行提供器，请安装 GPU 版运行时或改用自动/CPU。",
                retryable=False,
            )
        wants_cuda = normalized == "cuda" or (
            normalized == "auto" and cuda_available and "cuda" in self.manifest.supported_devices
        )
        attempts: list[tuple[str, list[str]]] = []
        if wants_cuda:
            attempts.append(("cuda", [CUDA_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER]))
            if normalized == "auto":
                # Auto mode keeps working on machines where the CUDA provider is
                # advertised but the CUDA/cuDNN DLLs cannot actually be loaded.
                attempts.append(("cpu", [CPU_EXECUTION_PROVIDER]))
        else:
            attempts.append(("cpu", [CPU_EXECUTION_PROVIDER]))

        session_options = make_onnx_session_options(ort, intra_op_threads=self._requested_threads)
        last_error: Exception | None = None
        for active_device, providers in attempts:
            try:
                session = ort.InferenceSession(
                    str(self.manifest.model_path()),
                    sess_options=session_options,
                    providers=providers,
                )
            except Exception as exc:  # noqa: BLE001 - third-party boundary
                last_error = exc
                continue
            self._validate_session_contract(session, labels, preprocess)
            self._session = session
            self._numpy = np
            self._labels = labels
            self._preprocess = preprocess
            self._requested_device = normalized
            self._active_device = active_device
            self._active_providers = tuple(providers)
            return
        raise ModelError(
            "MODEL_LOAD_FAILED", f"模型 {self.manifest.id} 加载失败", str(last_error), True
        )

    def unload(self) -> None:
        self._session = None
        self._labels = ()
        self._preprocess = None
        self._numpy = None
        self._active_device = None
        self._active_providers = ()

    def health(self) -> dict[str, object]:
        try:
            self.manifest.verify_sidecars()
            self.manifest.verify_model_file()
            self.manifest.validate()
        except ModelNotInstalledError as exc:
            return {"id": self.manifest.id, "status": "not_installed", "error": exc.as_dict()}
        except ModelError as exc:
            return {"id": self.manifest.id, "status": "invalid", "error": exc.as_dict()}
        try:
            self._import_runtime()
        except ModelNotInstalledError as exc:
            return {"id": self.manifest.id, "status": "dependency_missing", "error": exc.as_dict()}
        return {
            "id": self.manifest.id,
            "status": "ready" if self._session is not None else "installed",
            "kind": self.manifest.kind,
            "version": self.manifest.version,
            "supported_devices": list(self.manifest.supported_devices),
            "requested_device": self._requested_device,
            "requested_threads": self._requested_threads,
            "active_device": self._active_device,
            "active_providers": list(self._active_providers),
        }

    def _input_tensor(self, image: Image.Image) -> Any:
        if self._numpy is None or self._preprocess is None:
            self.activate()
        assert self._numpy is not None and self._preprocess is not None
        spec = self._preprocess
        rgb = image.convert(spec.color_mode)
        if spec.crop_mode == "center_crop":
            source_ratio = rgb.width / rgb.height
            target_ratio = spec.width / spec.height
            if source_ratio > target_ratio:
                crop_width = int(rgb.height * target_ratio)
                offset = (rgb.width - crop_width) // 2
                rgb = rgb.crop((offset, 0, offset + crop_width, rgb.height))
            else:
                crop_height = int(rgb.width / target_ratio)
                offset = (rgb.height - crop_height) // 2
                rgb = rgb.crop((0, offset, rgb.width, offset + crop_height))
        resized = ImageOps.fit(rgb, (spec.width, spec.height), method=Image.Resampling.BICUBIC)
        array = self._numpy.asarray(resized, dtype=self._numpy.float32) / 255.0
        array = (
            array - self._numpy.asarray(spec.mean, dtype=self._numpy.float32)
        ) / self._numpy.asarray(spec.std, dtype=self._numpy.float32)
        if spec.layout == "nchw":
            array = self._numpy.transpose(array, (2, 0, 1))
        return self._numpy.expand_dims(array, axis=0).astype(self._numpy.float32)

    def _run(self, image: Image.Image) -> Any:
        output_name = self.manifest.options.get("output_name")
        if not isinstance(output_name, str) or not output_name:
            raise ModelManifestError("模型元数据必须明确 output_name")
        return self._run_outputs(image, (output_name,))[output_name]

    def _run_outputs(self, image: Image.Image, output_names: tuple[str, ...]) -> dict[str, Any]:
        """Run one inference and return the requested named outputs."""

        if self._session is None:
            self.activate()
        assert self._session is not None
        input_name = self.manifest.options.get("input_name")
        if not isinstance(input_name, str) or not input_name:
            raise ModelManifestError("模型元数据必须明确 input_name")
        input_names = {item.name for item in self._session.get_inputs()}
        if input_name not in input_names:
            raise ModelError("MODEL_INPUT_INVALID", f"模型输入名称不存在: {input_name}")
        available_outputs = {item.name for item in self._session.get_outputs()}
        missing_outputs = [name for name in output_names if name not in available_outputs]
        if missing_outputs:
            raise ModelError("MODEL_OUTPUT_INVALID", "模型没有返回输出")
        output = self._session.run(list(output_names), {input_name: self._input_tensor(image)})
        if len(output) != len(output_names):
            raise ModelError("MODEL_OUTPUT_INVALID", "模型返回输出数量不匹配")
        return dict(zip(output_names, output, strict=True))

    def _validate_session_contract(
        self,
        session: Any,
        labels: tuple[ModelLabel, ...],
        preprocess: PreprocessSpec,
    ) -> None:
        """Check names, tensor ranks, and static dimensions before inference."""
        input_name = self.manifest.options.get("input_name")
        output_name = self.manifest.options.get("output_name")
        inputs = {item.name: item for item in session.get_inputs()}
        outputs = {item.name: item for item in session.get_outputs()}
        if not isinstance(input_name, str) or input_name not in inputs:
            raise ModelError("MODEL_INPUT_INVALID", f"模型输入名称不存在: {input_name}")
        if not isinstance(output_name, str) or output_name not in outputs:
            raise ModelError("MODEL_OUTPUT_INVALID", f"模型输出名称不存在: {output_name}")
        input_shape = inputs[input_name].shape
        if not isinstance(input_shape, list):
            raise ModelError("MODEL_INPUT_INVALID", "模型输入 shape 无效")
        if len(input_shape) != 4:
            raise ModelError("MODEL_INPUT_INVALID", "模型输入必须是四维张量")
        layout = preprocess.layout
        channel_index, height_index, width_index = (1, 2, 3) if layout == "nchw" else (3, 1, 2)
        channel = input_shape[channel_index]
        if isinstance(channel, int) and channel != 3:
            raise ModelError("MODEL_INPUT_INVALID", "模型输入通道数必须为 3")
        for index, expected in (
            (height_index, preprocess.height),
            (width_index, preprocess.width),
        ):
            value = input_shape[index]
            if isinstance(value, int) and value != expected:
                raise ModelError("MODEL_INPUT_INVALID", "预处理尺寸与模型输入 shape 不一致")
        output_shape = outputs[output_name].shape
        if not isinstance(output_shape, list):
            raise ModelError("MODEL_OUTPUT_INVALID", "模型输出 shape 无效")
        if self.manifest.kind == "character_recognizer":
            if len(output_shape) != 2:
                raise ModelError("MODEL_OUTPUT_INVALID", "角色模型 prediction 必须是二维张量")
            output_size = output_shape[-1]
            if isinstance(output_size, int) and output_size != len(labels):
                raise ModelError("MODEL_OUTPUT_INVALID", "模型输出维度与标签行数不一致")
        elif self.manifest.kind == "head_detector":
            if len(output_shape) != 3:
                raise ModelError("MODEL_OUTPUT_INVALID", "YOLOv8 检测输出必须是三维张量")
            channel_count = output_shape[1]
            if isinstance(channel_count, int) and channel_count != 4 + len(labels):
                raise ModelError("MODEL_OUTPUT_INVALID", "YOLOv8 检测类别维度与标签数不一致")
        if self.manifest.kind == "character_recognizer":
            embedding_name = self._embedding_output_name()
            if embedding_name is not None:
                if embedding_name not in outputs:
                    raise ModelError(
                        "MODEL_OUTPUT_INVALID", f"模型 embedding 输出名称不存在: {embedding_name}"
                    )
                embedding_shape = outputs[embedding_name].shape
                if not isinstance(embedding_shape, list) or len(embedding_shape) != 2:
                    raise ModelError("MODEL_OUTPUT_INVALID", "模型 embedding 必须是二维张量")
                embedding_size = embedding_shape[-1]
                if isinstance(embedding_size, int) and embedding_size != EMBEDDING_DIMENSION:
                    raise ModelError(
                        "MODEL_OUTPUT_INVALID",
                        f"模型 embedding 维度必须是 {EMBEDDING_DIMENSION}",
                    )

    def _embedding_output_name(self) -> str | None:
        """Resolve the optional named embedding output from model metadata."""

        explicit = self.manifest.options.get("embedding_output_name")
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit.strip():
                raise ModelManifestError("模型 embedding_output_name 必须是非空字符串")
            return explicit
        model_outputs = self.manifest.options.get("model_outputs")
        if isinstance(model_outputs, Mapping) and "embedding" in model_outputs:
            metadata = model_outputs["embedding"]
            if not isinstance(metadata, Mapping):
                raise ModelManifestError("模型 model_outputs.embedding 必须是对象")
            size = metadata.get("size")
            if size is not None and (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size != EMBEDDING_DIMENSION
            ):
                raise ModelManifestError(
                    f"模型 model_outputs.embedding.size 必须是 {EMBEDDING_DIMENSION}"
                )
            return "embedding"
        return None


class AnimeTimmONNXAdapter(ONNXAdapter):
    """AnimeTimm character recognizer with explicit metadata preprocessing."""

    def _input_tensor(self, image: Image.Image) -> Any:
        if self._numpy is None or self._preprocess is None:
            self.activate()
        assert self._numpy is not None and self._preprocess is not None
        if self._preprocess.pipeline is None:
            return super()._input_tensor(image)
        try:
            transform_module = importlib.import_module("imgutils.preprocess.pillow")
            transform_name = "create_pillow_transforms"
            create_pillow_transforms = getattr(transform_module, transform_name)
            transform = create_pillow_transforms([dict(step) for step in self._preprocess.pipeline])
            tensor = transform(image.convert("RGB"))
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModelNotInstalledError(
                f"模型 {self.manifest.id} 缺少 dghs-imgutils 预处理依赖",
                str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ModelManifestError("AnimeTIMM 预处理 pipeline 执行失败", str(exc)) from exc
        array = self._numpy.asarray(tensor, dtype=self._numpy.float32)
        if array.shape != (3, self._preprocess.height, self._preprocess.width):
            raise ModelError("MODEL_INPUT_INVALID", f"预处理输出尺寸不匹配: {array.shape}")
        return self._numpy.expand_dims(array, axis=0)

    def predict(self, image: Image.Image, top_k: int = 5) -> list[LabelScore]:
        if self.manifest.kind != "character_recognizer":
            raise ModelManifestError("AnimeTimm 适配器需要 kind=character_recognizer")
        return self._scores_from_output(self._run(image), top_k)

    def predict_with_embedding(
        self, image: Image.Image, top_k: int = 5
    ) -> tuple[list[LabelScore], tuple[float, ...]]:
        """Return classifier scores and the model embedding from one ONNX run."""

        if self.manifest.kind != "character_recognizer":
            raise ModelManifestError("AnimeTimm 适配器需要 kind=character_recognizer")
        output_name = self.manifest.options.get("output_name")
        embedding_name = self._embedding_output_name()
        if not isinstance(output_name, str) or not output_name:
            raise ModelManifestError("模型元数据必须明确 output_name")
        if embedding_name is None:
            raise ModelError(
                "MODEL_OUTPUT_INVALID",
                "模型未声明 2048 维 embedding 输出，无法进行个性化识别",
            )
        if embedding_name == output_name:
            raise ModelManifestError("模型 output_name 与 embedding_output_name 不能相同")
        outputs = self._run_outputs(image, (output_name, embedding_name))
        scores = self._scores_from_output(outputs[output_name], top_k)
        try:
            embedding = normalize_embedding(outputs[embedding_name].reshape(-1).tolist())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ModelError("MODEL_OUTPUT_INVALID", "模型 embedding 输出无效", str(exc)) from exc
        return scores, embedding

    def _scores_from_output(self, output: Any, top_k: int) -> list[LabelScore]:
        assert self._preprocess is not None
        try:
            values = output.reshape(-1).tolist()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ModelError("MODEL_OUTPUT_INVALID", "模型 prediction 输出无效", str(exc)) from exc
        if len(values) != len(self._labels):
            raise ModelError("MODEL_OUTPUT_INVALID", "模型输出数量与标签数量不一致")
        if self._preprocess.activation == "sigmoid":
            values = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value)))) for value in values]
        elif self._preprocess.activation == "softmax":
            maximum = max(values)
            exps = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in values]
            total = sum(exps)
            values = [value / total for value in exps]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ModelError("MODEL_OUTPUT_INVALID", "模型输出经预处理后不在 0 到 1 范围")
        scores = [
            LabelScore(
                label.tag,
                float(value),
                label.display_name,
                "character" if label.category.casefold() in {"character", "4"} else label.category,
            )
            for label, value in zip(self._labels, values, strict=True)
            if label.category.casefold() in {"character", "4"}
        ]
        return top_k_scores(scores, top_k)


class AnimeHeadONNXAdapter(ONNXAdapter):
    """Adapter for the upstream YOLOv8 head model contract.

    The published model emits ``(batch, 4 + classes, anchors)``.  The
    ``imgutils`` helpers are part of the detector extra and are deliberately
    used here as the contract boundary: they implement the model's exact
    letterbox/RGB encoding and class-aware NMS coordinate restoration.
    """

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._old_size: tuple[float, float] | None = None
        self._new_size: tuple[float, float] | None = None

    def unload(self) -> None:
        super().unload()
        self._old_size = None
        self._new_size = None

    def _input_tensor(self, image: Image.Image) -> Any:
        if self._numpy is None or self._preprocess is None:
            self.activate()
        assert self._numpy is not None and self._preprocess is not None
        align_value = self.manifest.options.get("align", 32)
        if not isinstance(align_value, int) or isinstance(align_value, bool) or align_value <= 0:
            raise ModelManifestError("头部检测 metadata align 必须是正整数")
        allow_dynamic_value = self.manifest.options.get("allow_dynamic", False)
        if not isinstance(allow_dynamic_value, bool):
            raise ModelManifestError("头部检测 metadata allow_dynamic 必须是布尔值")
        try:
            yolo = importlib.import_module("imgutils.generic.yolo")
            rgb_module = importlib.import_module("imgutils.data")
            rgb_encode_name = "rgb_encode"
            preprocess_name = "_image_preprocess"
            rgb_encode = getattr(rgb_module, rgb_encode_name)
            preprocess = getattr(yolo, preprocess_name)
            processed, old_size, new_size = preprocess(
                image.convert("RGB"),
                max_infer_size=(self._preprocess.width, self._preprocess.height),
                allow_dynamic=allow_dynamic_value,
                align=align_value,
            )
            self._old_size = (float(old_size[0]), float(old_size[1]))
            self._new_size = (float(new_size[0]), float(new_size[1]))
            encoded = rgb_encode(processed, order_="CHW", use_float=True)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModelNotInstalledError(
                f"模型 {self.manifest.id} 缺少 dghs-imgutils 检测预处理依赖",
                str(exc),
            ) from exc
        except (TypeError, ValueError, AttributeError) as exc:
            raise ModelError("MODEL_INPUT_INVALID", "头部检测预处理失败", str(exc)) from exc
        return self._numpy.expand_dims(encoded, axis=0).astype(self._numpy.float32)

    def predict(self, image: Image.Image) -> list[DetectionBox]:
        if self.manifest.kind != "head_detector":
            raise ModelManifestError("Anime Head 适配器需要 kind=head_detector")
        output = self._run(image)
        if self._old_size is None or self._new_size is None:
            raise ModelError("MODEL_INPUT_INVALID", "头部检测预处理尺寸状态缺失")
        threshold_value = self.manifest.options.get("conf_threshold", 0.4)
        iou_value = self.manifest.options.get("iou_threshold", 0.7)
        if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
            raise ModelManifestError("头部检测 metadata conf_threshold 必须是数字")
        if not isinstance(iou_value, (int, float)) or isinstance(iou_value, bool):
            raise ModelManifestError("头部检测 metadata iou_threshold 必须是数字")
        threshold = float(threshold_value)
        iou_threshold = float(iou_value)
        if not 0.0 <= threshold <= 1.0 or not 0.0 <= iou_threshold <= 1.0:
            raise ModelManifestError("头部检测 metadata 阈值必须位于 0 到 1")
        labels = [label.tag for label in self._labels]
        try:
            postprocess_name = "_yolo_postprocess"
            yolo_module = importlib.import_module("imgutils.generic.yolo")
            yolo_postprocess = getattr(yolo_module, postprocess_name)
            # ONNX Runtime returns batch-first output; imgutils expects the
            # per-image ``(4 + classes, anchors)`` matrix.
            matrix = output[0] if len(output.shape) == 3 else output
            rows = yolo_postprocess(
                matrix,
                conf_threshold=threshold,
                iou_threshold=iou_threshold,
                old_size=self._old_size,
                new_size=self._new_size,
                labels=labels,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModelNotInstalledError(
                f"模型 {self.manifest.id} 缺少 dghs-imgutils 检测后处理依赖",
                str(exc),
            ) from exc
        except (AssertionError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(
                "MODEL_OUTPUT_INVALID", "头部检测输出不符合 YOLOv8 契约", str(exc)
            ) from exc
        detections: list[DetectionBox] = []
        for box, label, confidence in rows:
            if label not in labels or not isinstance(confidence, (int, float)):
                raise ModelError("MODEL_OUTPUT_INVALID", "头部检测输出标签或置信度无效")
            if not isinstance(box, (tuple, list)) or len(box) != 4:
                raise ModelError("MODEL_OUTPUT_INVALID", "头部检测输出坐标无效")
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError) as exc:
                raise ModelError("MODEL_OUTPUT_INVALID", "头部检测输出坐标包含非数字") from exc
            detections.append(
                DetectionBox(x1, y1, x2, y2, float(confidence)).normalized(
                    image.width, image.height
                )
            )
        return detections


def create_adapter(manifest: ModelManifest) -> Adapter:
    if manifest.kind == "character_recognizer":
        # `camie_onnx` taggers use the same contract as AnimeTimm classifiers:
        # a multi-label ONNX head over a categorical label file, with the
        # preprocessing pipeline executed by imgutils. Only the pipeline shape
        # differs, and that is validated from the manifest.
        return AnimeTimmONNXAdapter(manifest)
    if manifest.kind == "head_detector":
        return AnimeHeadONNXAdapter(manifest)
    raise ModelManifestError(f"不支持的模型 kind: {manifest.kind}")

