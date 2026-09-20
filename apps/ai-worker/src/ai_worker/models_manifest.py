"""Strict model manifest handling and optional ONNX adapter boundaries."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeGuard, cast

from PIL import Image, ImageOps

from .errors import ModelError, ModelManifestError, ModelNotInstalledError
from .images import DetectionBox
from .recognition import EMBEDDING_DIMENSION, LabelScore, normalize_embedding, top_k_scores


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"模型元数据字段 {key} 必须是非空字符串")
    return value


def _safe_relative_path(value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelManifestError(f"模型元数据字段 {key} 必须是模型目录内的相对路径")
    return path


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    width: int
    height: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str
    color_mode: str
    layout: str
    activation: str
    crop_mode: str = "fit"
    pipeline: tuple[Mapping[str, object], ...] | None = None

    @classmethod
    def from_json_file(
        cls,
        path: Path,
        *,
        expected_size: tuple[int, int],
        activation: str | None = None,
        layout: str | None = None,
    ) -> PreprocessSpec:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelManifestError("模型预处理配置无法读取", str(exc)) from exc
        if not isinstance(raw, dict):
            raise ModelManifestError("模型预处理配置根节点必须是对象")
        # AnimeTIMM stores the authoritative evaluation transform under the
        # `test` pipeline.  Never flatten that pipeline by guesswork: validate
        # its declared operation order and use it verbatim at inference time.
        pipeline_value = raw.get("test")
        if isinstance(pipeline_value, list):
            return cls._from_pipeline(
                pipeline_value,
                expected_size=expected_size,
                activation=activation,
                layout=layout,
            )
        # Camie taggers declare a shorter pipeline under `stages`
        # (`pad_to_size` -> `to_tensor`); it is executed by imgutils exactly the
        # same way the upstream tagger runs it.
        stages_value = raw.get("stages")
        if isinstance(stages_value, list):
            return cls._from_stages(
                stages_value,
                expected_size=expected_size,
                activation=activation,
                layout=layout,
            )
        width = raw.get("width")
        height = raw.get("height")
        size = raw.get("size")
        if isinstance(size, int):
            width = height = size
        elif isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) for v in size):
            width, height = size
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ModelManifestError("预处理配置必须声明正整数 width/height 或 size")
        if (width, height) != expected_size:
            raise ModelManifestError(
                f"预处理尺寸 {(width, height)} 与模型 input {expected_size} 不一致"
            )
        mean = _vector(raw.get("mean"), "mean")
        std = _vector(raw.get("std"), "std")
        if any(value <= 0 for value in std):
            raise ModelManifestError("预处理 std 必须为正数")
        interpolation_value = raw.get("interpolation", "bicubic")
        color_mode_value = raw.get("color_mode", raw.get("color", "RGB"))
        layout_value = layout or raw.get("layout")
        activation_value = activation or raw.get("output_activation") or raw.get("activation")
        if not isinstance(interpolation_value, str) or not interpolation_value:
            raise ModelManifestError("预处理 interpolation 必须是字符串")
        if color_mode_value != "RGB":
            raise ModelManifestError("预处理 color_mode 当前必须明确为 RGB")
        if layout_value not in ("nchw", "nhwc"):
            raise ModelManifestError("预处理 layout 必须明确为 nchw 或 nhwc")
        if activation_value not in ("sigmoid", "softmax", "none"):
            raise ModelManifestError(
                "预处理/元数据必须明确 output_activation: sigmoid/softmax/none"
            )
        crop_mode = raw.get("crop_mode", "fit")
        if crop_mode not in ("fit", "center_crop"):
            raise ModelManifestError("预处理 crop_mode 只支持 fit 或 center_crop")
        return cls(
            width,
            height,
            mean,
            std,
            interpolation_value,
            color_mode_value,
            cast(str, layout_value),
            cast(str, activation_value),
            cast(str, crop_mode),
        )

    @classmethod
    def _from_pipeline(
        cls,
        pipeline_value: list[object],
        *,
        expected_size: tuple[int, int],
        activation: str | None,
        layout: str | None,
    ) -> PreprocessSpec:
        steps = [item for item in pipeline_value if isinstance(item, dict)]
        if len(steps) != len(pipeline_value):
            raise ModelManifestError("预处理 test pipeline 必须是对象数组")
        types = [item.get("type") for item in steps]
        required_types = ["pad_to_size", "resize", "center_crop", "maybe_to_tensor", "normalize"]
        if types != required_types:
            raise ModelManifestError(
                f"预处理 test pipeline 必须严格依次包含 {required_types}，实际为 {types}"
            )
        pad_size = steps[0].get("size")
        if not _size_pair(pad_size):
            raise ModelManifestError("预处理 pad_to_size 必须声明二元 size")
        pad_background = steps[0].get("background_color")
        if pad_background != "white":
            raise ModelManifestError("当前只接受模型声明的 white pad_to_size 背景")
        resize_size = steps[1].get("size")
        if not isinstance(resize_size, int) or resize_size <= 0:
            raise ModelManifestError("预处理 resize 必须声明正整数 size")
        resize_interpolation = steps[1].get("interpolation")
        crop_size = steps[2].get("size")
        if not _size_pair(crop_size):
            raise ModelManifestError("预处理 center_crop 必须声明二元 size")
        width, height = int(crop_size[0]), int(crop_size[1])
        if (width, height) != expected_size:
            raise ModelManifestError(
                f"预处理 center_crop 尺寸 {(width, height)} 与模型 input {expected_size} 不一致"
            )
        if not isinstance(resize_interpolation, str) or not resize_interpolation:
            raise ModelManifestError("预处理 resize 必须声明 interpolation")
        mean = _vector(steps[4].get("mean"), "mean")
        std = _vector(steps[4].get("std"), "std")
        if any(value <= 0 for value in std):
            raise ModelManifestError("预处理 std 必须为正数")
        layout_value = layout or "nchw"
        activation_value = activation or "none"
        if layout_value not in ("nchw", "nhwc"):
            raise ModelManifestError("预处理 layout 必须明确为 nchw 或 nhwc")
        if activation_value not in ("sigmoid", "softmax", "none"):
            raise ModelManifestError("模型 output_activation 必须为 sigmoid/softmax/none")
        return cls(
            width,
            height,
            mean,
            std,
            resize_interpolation,
            "RGB",
            layout_value,
            activation_value,
            "pipeline",
            tuple(cast(Mapping[str, object], item) for item in steps),
        )

    @classmethod
    def _from_stages(
        cls,
        pipeline_value: list[object],
        *,
        expected_size: tuple[int, int],
        activation: str | None,
        layout: str | None,
    ) -> PreprocessSpec:
        """Validate the shorter `stages` pipeline used by Camie taggers.

        Only the steps imgutils can execute are accepted, and the padding size
        must match the declared model input. Normalization is intentionally not
        inferred: Camie supplies none, and its `to_tensor` already scales to
        the range the model expects.
        """

        known_steps = {
            "pad_to_size",
            "resize",
            "center_crop",
            "maybe_to_tensor",
            "to_tensor",
            "normalize",
        }
        steps = [item for item in pipeline_value if isinstance(item, dict)]
        if not steps or len(steps) != len(pipeline_value):
            raise ModelManifestError("预处理 stages 必须是非空对象数组")
        types = [str(item.get("type", "")) for item in steps]
        unknown = sorted({name for name in types if name not in known_steps})
        if unknown:
            raise ModelManifestError(f"预处理 stages 含未知步骤: {'、'.join(unknown)}")
        pad_step = next((item for item in steps if item.get("type") == "pad_to_size"), None)
        if pad_step is None:
            raise ModelManifestError("预处理 stages 必须包含 pad_to_size")
        pad_size = pad_step.get("size")
        if not _size_pair(pad_size) or (
            int(pad_size[0]),  # type: ignore[index]
            int(pad_size[1]),  # type: ignore[index]
        ) != expected_size:
            raise ModelManifestError(
                f"预处理 pad_to_size 尺寸必须与模型 input {expected_size} 一致"
            )
        if not any(name in {"to_tensor", "maybe_to_tensor"} for name in types):
            raise ModelManifestError("预处理 stages 必须包含 to_tensor")
        layout_value = layout or "nchw"
        activation_value = activation or "none"
        if layout_value not in ("nchw", "nhwc"):
            raise ModelManifestError("预处理 layout 必须明确为 nchw 或 nhwc")
        if activation_value not in ("sigmoid", "softmax", "none"):
            raise ModelManifestError("模型 output_activation 必须为 sigmoid/softmax/none")
        width, height = expected_size
        return cls(
            width,
            height,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            "lanczos",
            "RGB",
            layout_value,
            activation_value,
            "pipeline",
            tuple(cast(Mapping[str, object], item) for item in steps),
        )


def _size_pair(value: object) -> TypeGuard[list[int] | tuple[int, int]]:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    )


def _vector(value: object, key: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ModelManifestError(f"预处理 {key} 必须是 3 个数字")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ModelManifestError(f"预处理 {key} 必须是 3 个数字") from exc
    if not all(math.isfinite(item) for item in values):
        raise ModelManifestError(f"预处理 {key} 不能包含非有限数")
    return cast(tuple[float, float, float], values)


@dataclass(frozen=True, slots=True)
class ModelLabel:
    tag: str
    category: str
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class ModelManifest:
    id: str
    kind: str
    adapter: str
    format: str
    version: str
    input_width: int
    input_height: int
    model_file: Path
    labels_file: Path
    preprocess_file: Path
    sha256: str
    supported_devices: tuple[str, ...]
    metadata_path: Path
    options: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> ModelManifest:
        metadata_path = Path(path).expanduser()
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelManifestError("模型 metadata.json 无法读取", str(exc)) from exc
        if not isinstance(raw, dict):
            raise ModelManifestError("模型 metadata.json 根节点必须是对象")
        model_id = _required_string(raw, "id")
        kind_value = raw.get("kind", raw.get("type"))
        if not isinstance(kind_value, str) or not kind_value.strip():
            raise ModelManifestError("模型元数据必须声明 kind")
        adapter = _required_string(raw, "adapter")
        model_format = _required_string(raw, "format").casefold()
        if model_format != "onnx":
            raise ModelManifestError("当前只支持 format=onnx 的模型")
        version = _required_string(raw, "version")
        input_value = raw.get("input")
        if not isinstance(input_value, dict):
            raise ModelManifestError("模型元数据 input 必须是对象")
        width = input_value.get("width")
        height = input_value.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ModelManifestError("模型 input 必须声明正整数 width/height")
        model_file = _safe_relative_path(_required_string(raw, "model_file"), "model_file")
        labels_file = _safe_relative_path(_required_string(raw, "labels_file"), "labels_file")
        preprocess_file = _safe_relative_path(
            _required_string(raw, "preprocess_file"), "preprocess_file"
        )
        sha256 = raw.get("sha256")
        if not isinstance(sha256, str):
            raise ModelManifestError("模型元数据 sha256 必须是字符串（无校验值时使用空字符串）")
        if sha256 and (
            len(sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in sha256)
        ):
            raise ModelManifestError("模型 sha256 必须为空或 64 位十六进制字符串")
        devices = raw.get("supported_devices")
        if (
            not isinstance(devices, list)
            or not devices
            or not all(isinstance(device, str) and device for device in devices)
        ):
            raise ModelManifestError("模型元数据 supported_devices 必须是非空字符串数组")
        return cls(
            model_id,
            kind_value,
            adapter,
            model_format,
            version,
            width,
            height,
            model_file,
            labels_file,
            preprocess_file,
            sha256.casefold(),
            tuple(devices),
            metadata_path,
            dict(raw),
        )

    @property
    def directory(self) -> Path:
        return self.metadata_path.parent

    def resolve(self, relative_path: Path) -> Path:
        candidate = (self.directory / relative_path).resolve()
        base = self.directory.resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ModelManifestError("模型资源路径越过模型目录边界") from exc
        return candidate

    def model_path(self) -> Path:
        return self.resolve(self.model_file)

    def labels_path(self) -> Path:
        return self.resolve(self.labels_file)

    def preprocess_path(self) -> Path:
        return self.resolve(self.preprocess_file)

    def verify_sidecars(self) -> None:
        for path, label in (
            (self.labels_path(), "标签文件"),
            (self.preprocess_path(), "预处理文件"),
        ):
            if not path.is_file():
                raise ModelManifestError(f"模型{label}不存在: {path.name}")

    def verify_model_file(self) -> None:
        model_path = self.model_path()
        if not model_path.is_file():
            raise ModelNotInstalledError(
                f"模型 {self.id} 尚未安装，请将 ONNX 权重放入 {model_path.parent}",
                f"missing_file={model_path.name}",
            )
        if self.sha256 and len(self.sha256) != 64:
            raise ModelManifestError("模型 sha256 必须为空或 64 位十六进制字符串")
        if self.sha256:
            actual = sha256_file(model_path)
            if actual != self.sha256:
                raise ModelManifestError("模型 sha256 校验失败，文件可能损坏或版本不匹配")

    def load_labels(self) -> tuple[ModelLabel, ...]:
        self.verify_sidecars()
        labels_path = self.labels_path()
        if self.kind == "character_recognizer" and labels_path.suffix.casefold() != ".csv":
            raise ModelManifestError("角色模型 labels_file 必须是 CSV 标签文件")
        if labels_path.suffix.casefold() == ".json":
            return self._load_json_labels(labels_path)
        try:
            with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    raise ModelManifestError("标签文件缺少表头")
                fieldnames = {name.strip().casefold() for name in reader.fieldnames if name}
                tag_field = next(
                    (name for name in ("tag", "name", "label") if name in fieldnames), None
                )
                if tag_field is None:
                    raise ModelManifestError("标签文件必须包含 tag/name/label 列")
                # Keep original spelling in row lookup while accepting case variants.
                actual_tag_field = next(
                    name for name in reader.fieldnames if name.strip().casefold() == tag_field
                )
                category_field = next(
                    (
                        name
                        for name in reader.fieldnames
                        if name and name.strip().casefold() == "category"
                    ),
                    None,
                )
                if self.kind == "character_recognizer" and category_field is None:
                    raise ModelManifestError("角色模型标签文件必须包含 category 列")
                display_field = next(
                    (
                        name
                        for name in reader.fieldnames
                        if name and name.strip().casefold() in ("display_name", "中文名")
                    ),
                    None,
                )
                labels: list[ModelLabel] = []
                seen: set[str] = set()
                for row in reader:
                    tag = (row.get(actual_tag_field) or "").strip()
                    if not tag:
                        raise ModelManifestError("标签文件包含空标签")
                    if tag in seen:
                        raise ModelManifestError(f"标签文件存在重复标签: {tag}")
                    category = (
                        (row.get(category_field) or "").strip() if category_field else "character"
                    )
                    if not category:
                        raise ModelManifestError(f"标签 {tag} 的 category 不能为空")
                    display_name = (row.get(display_field) or "").strip() if display_field else None
                    labels.append(ModelLabel(tag, category, display_name or None))
                    seen.add(tag)
        except ModelManifestError:
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ModelManifestError("模型标签文件无法解析", str(exc)) from exc
        if not labels:
            raise ModelManifestError("模型标签文件不能为空")
        return tuple(labels)

    def _load_json_labels(self, labels_path: Path) -> tuple[ModelLabel, ...]:
        try:
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelManifestError("模型 JSON 标签文件无法解析", str(exc)) from exc
        values: object = raw.get("labels") if isinstance(raw, dict) else raw
        if not isinstance(values, list) or not values:
            raise ModelManifestError("模型 JSON 标签文件必须包含非空 labels 数组")
        labels: list[ModelLabel] = []
        seen: set[str] = set()
        for index, value in enumerate(values):
            if isinstance(value, str):
                tag = value.strip()
                category = "detector"
                display_name = None
            elif isinstance(value, dict):
                tag_value = value.get("tag", value.get("name", value.get("label")))
                tag = tag_value.strip() if isinstance(tag_value, str) else ""
                category_value = value.get("category", "detector")
                category = category_value.strip() if isinstance(category_value, str) else ""
                display_value = value.get("display_name")
                display_name = display_value.strip() if isinstance(display_value, str) else None
            else:
                tag = ""
                category = ""
                display_name = None
            if not tag or not category:
                raise ModelManifestError(f"模型 JSON 标签第 {index + 1} 项无效")
            if tag in seen:
                raise ModelManifestError(f"标签文件存在重复标签: {tag}")
            labels.append(ModelLabel(tag, category, display_name or None))
            seen.add(tag)
        return tuple(labels)

    def load_preprocess(self) -> PreprocessSpec:
        self.verify_sidecars()
        activation_value = self.options.get("output_activation")
        layout_value = self.options.get("input_layout")
        return PreprocessSpec.from_json_file(
            self.preprocess_path(),
            expected_size=(self.input_width, self.input_height),
            activation=activation_value if isinstance(activation_value, str) else None,
            layout=layout_value if isinstance(layout_value, str) else None,
        )

    def validate(self) -> tuple[tuple[ModelLabel, ...], PreprocessSpec]:
        labels = self.load_labels()
        preprocess = self.load_preprocess()
        if self.kind == "character_recognizer" and self.adapter not in {
            "animetimm_onnx",
            "animetimm",
            "animetimm-resnet101",
            "animetimm-convformer-b36",
            # Camie taggers share the multi-label classifier contract but
            # declare a shorter `stages` preprocessing pipeline.
            "camie_onnx",
            "camie",
        }:
            raise ModelManifestError(f"角色模型 adapter 不受支持: {self.adapter}")
        if self.kind == "head_detector" and self.adapter not in {
            "anime_head_onnx",
            "dghs_anime_head_onnx",
        }:
            raise ModelManifestError(f"头部检测模型 adapter 不受支持: {self.adapter}")
        output_name = self.options.get("output_name")
        input_name = self.options.get("input_name")
        if self.kind in {"character_recognizer", "head_detector"}:
            if not isinstance(input_name, str) or not input_name:
                raise ModelManifestError("模型元数据必须明确 input_name")
            if not isinstance(output_name, str) or not output_name:
                raise ModelManifestError("模型元数据必须明确 output_name")
        if self.kind == "head_detector" and self.options.get("output_format") != "yolo_v8_nms":
            raise ModelManifestError("头部检测 metadata output_format 必须为 yolo_v8_nms")
        if self.kind == "character_recognizer":
            self._validate_character_sidecars(labels)
        return labels, preprocess

    def _validate_character_sidecars(self, labels: tuple[ModelLabel, ...]) -> None:
        """Validate the category/threshold sidecars shipped with AnimeTIMM.

        These files are model-specific contract data, not optional display
        metadata.  Requiring the character category and checking the declared
        tag count prevents a subtly shifted output vector from being treated
        as a valid prediction.
        """
        tag_count = self.options.get("tag_count")
        if tag_count is not None and (
            not isinstance(tag_count, int)
            or isinstance(tag_count, bool)
            or tag_count != len(labels)
        ):
            raise ModelManifestError("角色模型 tag_count 必须与标签行数一致")
        character_category = self.options.get("character_category")
        if character_category is not None and character_category != 4:
            raise ModelManifestError("角色模型 character_category 必须为 4")
        character_labels = [
            label for label in labels if label.category.casefold() in {"character", "4"}
        ]
        if not character_labels:
            raise ModelManifestError("角色模型标签必须包含 category=4/character")
        category_file = self._option_file("categories_file", "角色模型 categories")
        try:
            raw_categories = json.loads(category_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelManifestError("角色模型 categories.json 无法解析", str(exc)) from exc
        if not isinstance(raw_categories, list):
            raise ModelManifestError("角色模型 categories.json 必须是数组")
        has_character_category = any(
            isinstance(item, dict)
            and str(item.get("category", "")).casefold() == "4"
            and str(item.get("name", "")).casefold() == "character"
            for item in raw_categories
        )
        if not has_character_category:
            raise ModelManifestError("角色模型 categories.json 缺少 category=4/character")
        thresholds_file = self._option_file("thresholds_file", "角色模型 thresholds")
        try:
            with thresholds_file.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ModelManifestError("角色模型 thresholds.csv 无法解析", str(exc)) from exc
        character_thresholds = [
            row
            for row in rows
            if (row.get("category") or "").strip() == "4"
            and (row.get("name") or "").strip().casefold() == "character"
        ]
        if len(character_thresholds) != 1:
            raise ModelManifestError("角色模型 thresholds.csv 必须包含唯一 category=4/character 行")
        threshold = character_thresholds[0].get("threshold")
        try:
            threshold_value = float(threshold or "")
        except ValueError as exc:
            raise ModelManifestError("角色模型 character threshold 必须是数字") from exc
        if not 0.0 <= threshold_value <= 1.0:
            raise ModelManifestError("角色模型 character threshold 必须位于 0 到 1")

    def _option_file(self, key: str, label: str) -> Path:
        value = self.options.get(key)
        if not isinstance(value, str) or not value:
            raise ModelManifestError(f"{label} 文件路径必须在 metadata 中声明")
        relative = _safe_relative_path(value, key)
        resolved = self.resolve(relative)
        if not resolved.is_file():
            raise ModelManifestError(f"{label} 文件不存在: {resolved.name}")
        return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

