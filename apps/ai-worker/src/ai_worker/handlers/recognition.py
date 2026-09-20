"""Image recognition, multiscale crops, embedding extraction, and fusion handlers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PIL.Image import Image as PILImage

from ..errors import ModelError, WorkerError
from ..export import export_results
from ..images import (
    DetectionBox,
    crop_box,
    enumerate_images_with_total,
    inspect_image,
    make_multiscale_crops,
    read_image,
)
from ..personal_model import PersonalModelArtifact, parse_personal_model
from ..reference import ReferenceLibrary, match_library, normalize_vector
from ..recognition import (
    EMBEDDING_DIMENSION,
    FusionConfig,
    LabelScore,
    combine_embeddings,
    fuse_multiscale,
    fuse_personal_prototypes,
    normalize_embedding,
    parse_personal_prototypes,
)


def normalized_bbox(value: object, index: int) -> dict[str, float]:
    raw_values: list[object]
    if isinstance(value, Mapping):
        keys = ("x1", "y1", "x2", "y2")
        raw_values = [value.get(key) for key in keys]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        raw_values = list(value)
    else:
        raise WorkerError(
            "INVALID_PAYLOAD",
            f"annotations[{index}].bbox 必须是 [x1,y1,x2,y2] 或坐标对象",
        )
    numeric_values: list[float] = []
    for item in raw_values:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise WorkerError(
                "INVALID_PAYLOAD", f"annotations[{index}].bbox 必须是有限数字"
            )
        numeric_values.append(float(item))
    if not all(math.isfinite(item) for item in numeric_values):
        raise WorkerError("INVALID_PAYLOAD", f"annotations[{index}].bbox 必须是有限数字")
    x1, y1, x2, y2 = numeric_values
    if not all(0.0 <= item <= 1.0 for item in (x1, y1, x2, y2)):
        raise WorkerError(
            "INVALID_PAYLOAD",
            f"annotations[{index}].bbox 必须是 0 到 1 的归一化坐标",
        )
    if x1 >= x2 or y1 >= y2:
        raise WorkerError("INVALID_PAYLOAD", f"annotations[{index}].bbox 必须具有正面积")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def parse_embedding_annotations(
    payload: Mapping[str, object]
) -> list[tuple[str | None, dict[str, float]]]:
    raw_annotations = payload.get("annotations", payload.get("boxes"))
    if raw_annotations is None:
        raw_bbox = payload.get("bbox", payload.get("box"))
        if raw_bbox is None:
            raise WorkerError(
                "INVALID_PAYLOAD",
                "recognition.embedding 需要 bbox 或非空 annotations 数组",
            )
        raw_annotations = [{"bbox": raw_bbox, "annotation_id": payload.get("annotation_id")}]
    if not isinstance(raw_annotations, list) or not raw_annotations:
        raise WorkerError("INVALID_PAYLOAD", "annotations 必须是非空对象数组")
    if len(raw_annotations) > 100:
        raise WorkerError("INVALID_PAYLOAD", "annotations 一次最多支持 100 个标注")
    parsed: list[tuple[str | None, dict[str, float]]] = []
    for index, raw_annotation in enumerate(raw_annotations):
        if isinstance(raw_annotation, Mapping):
            raw_bbox = raw_annotation.get("bbox", raw_annotation.get("box"))
            raw_id = raw_annotation.get("annotation_id")
        else:
            raw_bbox = raw_annotation
            raw_id = None
        if raw_id is not None and (not isinstance(raw_id, str) or not raw_id.strip()):
            raise WorkerError(
                "INVALID_PAYLOAD", f"annotations[{index}].annotation_id 必须是非空字符串"
            )
        bbox = normalized_bbox(raw_bbox, index)
        parsed.append((raw_id.strip() if isinstance(raw_id, str) else None, bbox))
    return parsed


def predict_with_embedding(
    recognizer: Any,
    image: PILImage,
    top_k: int,
) -> tuple[list[LabelScore], tuple[float, ...]]:
    predictor = getattr(recognizer, "predict_with_embedding", None)
    if not callable(predictor):
        raise WorkerError(
            "EMBEDDING_UNAVAILABLE",
            "当前角色识别模型未提供 2048 维 embedding 输出",
        )
    result = predictor(image, top_k)
    if not isinstance(result, tuple) or len(result) != 2:
        raise WorkerError("EMBEDDING_FAILED", "角色识别器返回的 embedding 结果无效")
    scores, raw_embedding = result
    if not isinstance(scores, list):
        raise WorkerError("RECOGNITION_FAILED", "角色识别器返回了无效结果")
    try:
        embedding = normalize_embedding(raw_embedding)
    except ValueError as exc:
        raise WorkerError(
            "EMBEDDING_FAILED", "角色识别器返回的 embedding 无效", str(exc)
        ) from exc
    return scores, embedding


def parse_fusion_config(raw: object) -> FusionConfig:
    if raw is None:
        return FusionConfig()
    if not isinstance(raw, dict):
        raise WorkerError("INVALID_PAYLOAD", "config 必须是对象")
    values: dict[str, Any] = {}
    for name in (
        "detector_score_threshold",
        "detector_iou_threshold",
        "recognizer_top1_threshold",
        "recognizer_top2_margin",
        "consistency_agreement_threshold",
        "multi_person_suppression_margin",
    ):
        if name in raw:
            value = raw[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise WorkerError("INVALID_PAYLOAD", f"配置项 {name} 必须是数字")
            values[name] = float(value)
    if "weights" in raw:
        weights_raw = raw["weights"]
        if not isinstance(weights_raw, dict):
            raise WorkerError("INVALID_PAYLOAD", "配置项 weights 必须是对象")
        weights: dict[str, float] = {}
        for key, value in weights_raw.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise WorkerError("INVALID_PAYLOAD", f"权重 weights[{key}] 必须是数字")
            weights[str(key)] = float(value)
        values["weights"] = weights
    return FusionConfig(**values)


def _reference_feature(
    service: Any,
    recognizer: Any,
    backend: str,
    crop: PILImage,
) -> list[float]:
    """Feature vector for reference matching, per backend."""

    if backend == "ccip":
        try:
            from imgutils.metrics import ccip_extract_feature
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise WorkerError(
                "DEPENDENCY_MISSING",
                "参考匹配需要 dghs-imgutils 的 CCIP 模型支持",
                str(exc),
            ) from exc
        return [float(value) for value in ccip_extract_feature(crop)]
    _, embedding = service._predict_with_embedding(recognizer, crop, 1)  # noqa: SLF001
    return normalize_vector(embedding)


def _apply_reference_match(
    service: Any,
    payload: Mapping[str, object],
    recognizer: Any,
    crops: Mapping[Any, PILImage],
    person_result: dict[str, Any],
) -> None:
    """Promote a low-confidence person when a reference image matches it.

    Only persons the classifier could not settle are considered, so the extra
    model runs on a small minority of crops (and never for high-confidence
    results the base model already resolved).
    """

    library = payload.get("reference_library")
    if not isinstance(library, ReferenceLibrary):
        return
    if person_result.get("status") == "high_confidence":
        return
    crop = None
    for crop_type in ("bust", "large", "head"):
        candidate = crops.get(crop_type)
        if candidate is None:
            for key, value in crops.items():
                if getattr(key, "value", key) == crop_type:
                    candidate = value
                    break
        if candidate is not None:
            crop = candidate
            break
    if crop is None:
        return
    try:
        vector = _reference_feature(service, recognizer, library.backend, crop)
        match = match_library(library, vector)
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001 - reference matching is best effort
        person_result["reference_error"] = f"{type(exc).__name__}: {exc}"
        return
    if match is None:
        return
    person_result["reference"] = match
    if not match.get("matched"):
        return
    if person_result.get("status") in {"unrecognized", "needs_review"}:
        person_result["status"] = "reference_match"
    candidates = person_result.get("candidates")
    if isinstance(candidates, list):
        candidates.insert(
            0,
            {
                "character_tag": match["identity_id"],
                "display_name": match["display_name"],
                "confidence": match["similarity"],
                "rank": 1,
                "crop_support": 1.0,
                "source": "reference",
                "reason": f"参考匹配 {match['similarity']:.0%}",
            },
        )
        for index, candidate in enumerate(candidates, start=1):
            candidate["rank"] = index
        person_result["candidates"] = candidates


def handle_recognition_image(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    path = service._required_string(payload, "path")
    top_k = service._top_k(payload.get("top_k", 5))
    include_embeddings = service._boolean(payload, "include_embeddings", False)
    personal_model: PersonalModelArtifact | None = None
    if payload.get("personal_model") is not None:
        try:
            personal_model = parse_personal_model(payload["personal_model"])
        except ValueError as exc:
            raise WorkerError("INVALID_PAYLOAD", str(exc)) from exc
    try:
        personal_prototypes = parse_personal_prototypes(payload.get("personal_prototypes"))
    except ValueError as exc:
        raise WorkerError("INVALID_PAYLOAD", str(exc)) from exc
    if personal_model is not None and personal_prototypes:
        raise WorkerError(
            "INVALID_PAYLOAD",
            "personal_model 与 personal_prototypes 不能同时提供",
        )
    active_prototypes = (
        personal_model.prototypes if personal_model is not None else personal_prototypes
    )
    model_version = personal_model.version if personal_model is not None else None
    source = Path(path).expanduser()
    try:
        image = read_image(source)
    except FileNotFoundError as exc:
        raise WorkerError("IMAGE_NOT_FOUND", f"图片不存在: {source}", str(exc)) from exc
    except PermissionError as exc:
        raise WorkerError(
            "IMAGE_ACCESS_DENIED", f"没有读取图片的权限: {source}", str(exc)
        ) from exc
    except (OSError, ValueError) as exc:
        raise WorkerError("IMAGE_READ_FAILED", f"图片读取失败: {source}", str(exc)) from exc

    detector_manifest = service._select_manifest(
        "head_detector", service._requested_model_id(payload, "detector")
    )
    detector = service._detector_adapter(detector_manifest)
    try:
        detections = detector.predict(image)
    except ModelError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkerError("DETECTION_FAILED", "头部检测失败", str(exc), True) from exc
    if not isinstance(detections, list):
        raise WorkerError("DETECTION_FAILED", "头部检测器返回了无效结果")

    if not detections:
        result: dict[str, Any] = {
            "path": str(source),
            "image_size": list(image.size),
            "people": [],
        }
        if model_version is not None:
            result["model_version"] = model_version
        return result

    recognizer_manifest = service._select_manifest(
        "character_recognizer", service._requested_model_id(payload, "recognizer")
    )
    recognizer = service._recognizer_adapter(recognizer_manifest)
    inference_top_k = max(top_k, 5)
    need_embedding = include_embeddings or bool(active_prototypes)
    people: list[dict[str, Any]] = []
    for person_index, raw_detection in enumerate(detections):
        if not isinstance(raw_detection, DetectionBox):
            raise WorkerError("DETECTION_FAILED", "头部检测结果包含无效检测框")
        box = raw_detection.normalized(image.width, image.height)
        try:
            crops = make_multiscale_crops(image, box)
        except (ValueError, OSError) as exc:
            raise WorkerError("CROP_FAILED", "生成多尺度裁剪失败", str(exc), True) from exc
        predictions: dict[str, list[LabelScore]] = {}
        crop_embeddings: dict[str, tuple[float, ...]] = {}
        for crop_type, crop in crops.items():
            try:
                if need_embedding:
                    scores, embedding = service._predict_with_embedding(
                        recognizer, crop, inference_top_k
                    )
                    crop_embeddings[crop_type.value] = embedding
                else:
                    scores = recognizer.predict(crop, inference_top_k)
            except ModelError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise WorkerError(
                    "RECOGNITION_FAILED",
                    f"角色识别失败: {crop_type.value}",
                    str(exc),
                    True,
                ) from exc
            if not isinstance(scores, list):
                raise WorkerError("RECOGNITION_FAILED", "角色识别器返回了无效结果")
            predictions[crop_type.value] = scores
        try:
            fusion = fuse_multiscale(predictions, top_k=top_k)
        except (TypeError, ValueError) as exc:
            raise WorkerError("RECOGNITION_FAILED", "角色结果融合失败", str(exc)) from exc
        if active_prototypes:
            try:
                combined_embedding = combine_embeddings(crop_embeddings)
                strict_threshold = (
                    personal_model.strict_threshold
                    if personal_model is not None
                    else 0.86
                )
                min_personal_margin = (
                    personal_model.min_margin
                    if personal_model is not None
                    else 0.08
                )
                fusion = fuse_personal_prototypes(
                    fusion,
                    combined_embedding,
                    active_prototypes,
                    top_k=top_k,
                    strict_threshold=strict_threshold,
                    min_personal_margin=min_personal_margin,
                )
            except (TypeError, ValueError) as exc:
                raise WorkerError("RECOGNITION_FAILED", "个人原型融合失败", str(exc)) from exc
        fusion_result = fusion.as_dict()
        candidates = fusion_result["candidates"]
        if isinstance(candidates, list):
            candidates = candidates[:top_k]
        person_result: dict[str, Any] = {
            "person_index": person_index,
            "box": {
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
                "confidence": box.confidence,
            },
            "crops": {
                crop_type.value: {
                    "box": list(crop_box(box, image.size, crop_type)),
                }
                for crop_type in crops
            },
            "status": fusion_result["status"],
            "top1": fusion_result["top1"],
            "top2": fusion_result["top2"],
            "margin": fusion_result["margin"],
            "consistency": fusion_result["consistency"],
            "candidates": candidates,
        }
        if model_version is not None:
            person_result["model_version"] = model_version
        if include_embeddings:
            try:
                person_result["embedding"] = list(combine_embeddings(crop_embeddings))
            except (TypeError, ValueError) as exc:
                raise WorkerError(
                    "EMBEDDING_FAILED", "合并人物 embedding 失败", str(exc)
                ) from exc
            person_result["embedding_dimension"] = EMBEDDING_DIMENSION
        _apply_reference_match(service, payload, recognizer, crops, person_result)
        people.append(person_result)
    result = {
        "path": str(source),
        "image_size": list(image.size),
        "people": people,
    }
    if model_version is not None:
        result["model_version"] = model_version
    return result


def handle_recognition_embedding(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    path = service._required_string(payload, "path")
    source = Path(path).expanduser()
    try:
        image = read_image(source)
    except FileNotFoundError as exc:
        raise WorkerError("IMAGE_NOT_FOUND", f"图片不存在: {source}", str(exc)) from exc
    except PermissionError as exc:
        raise WorkerError(
            "IMAGE_ACCESS_DENIED", f"没有读取图片的权限: {source}", str(exc)
        ) from exc
    except (OSError, ValueError) as exc:
        raise WorkerError("IMAGE_READ_FAILED", f"图片读取失败: {source}", str(exc)) from exc

    annotations = parse_embedding_annotations(payload)
    recognizer_manifest = service._select_manifest(
        "character_recognizer", service._requested_model_id(payload, "recognizer")
    )
    recognizer = service._recognizer_adapter(recognizer_manifest)
    results: list[dict[str, Any]] = []
    for index, (annotation_id, n_bbox) in enumerate(annotations, start=1):
        box = DetectionBox(
            n_bbox["x1"] * image.width,
            n_bbox["y1"] * image.height,
            n_bbox["x2"] * image.width,
            n_bbox["y2"] * image.height,
        ).normalized(image.width, image.height)
        try:
            crops = make_multiscale_crops(image, box)
            crop_embeddings: dict[str, tuple[float, ...]] = {}
            for crop_type, crop in crops.items():
                _, embedding = service._predict_with_embedding(recognizer, crop, 1)
                crop_embeddings[crop_type.value] = embedding
            embedding = combine_embeddings(crop_embeddings)
        except WorkerError:
            raise
        except ModelError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WorkerError(
                "EMBEDDING_FAILED",
                f"annotation {annotation_id} embedding 提取失败",
                str(exc),
                True,
            ) from exc
        results.append(
            {
                "annotation_id": annotation_id or f"annotation-{index}",
                "bbox": n_bbox,
                "embedding": list(embedding),
                "embedding_dimension": EMBEDDING_DIMENSION,
            }
        )
    result: dict[str, Any] = {
        "path": str(source),
        "image_size": list(image.size),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "count": len(results),
        "embeddings": results,
    }
    if len(results) == 1:
        result["embedding"] = results[0]["embedding"]
    return result


def handle_enumerate(
    service: Any,
    payload: Mapping[str, object],
    *,
    inspect_fn: Callable[[Path], Any] = inspect_image,
    enumerate_fn: Callable[..., tuple[list[Path], int]] = enumerate_images_with_total,
) -> dict[str, Any]:
    directory = service._required_string(payload, "directory", fallback_key="path")
    recursive = service._boolean(payload, "recursive", True)
    include_hidden = service._boolean(payload, "include_hidden", False)
    limit = service._enumeration_limit(payload.get("limit"))
    root = Path(directory).expanduser()
    paths, total_count = enumerate_fn(
        root,
        recursive=recursive,
        include_hidden=include_hidden,
        limit=limit,
    )
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for p in paths:
        try:
            record = inspect_fn(p)
            records.append(
                {
                    "path": str(p),
                    "relative_path": str(p.relative_to(root)),
                    "format": record.format,
                    "width": record.width,
                    "height": record.height,
                }
            )
        except (OSError, ValueError) as exc:
            errors.append({"path": str(p), "message": str(exc)})
    return {
        "directory": str(root),
        "images": records,
        "errors": errors,
        "count": len(records),
        "total_count": total_count,
        "returned_count": len(records),
    }


def handle_crops(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    path = service._required_string(payload, "path")
    image = read_image(path)
    box_value = payload.get("box")
    if not isinstance(box_value, dict):
        raise WorkerError("INVALID_PAYLOAD", "crops.create 需要 box 对象")
    box = DetectionBox(
        service._number(box_value, "x1"),
        service._number(box_value, "y1"),
        service._number(box_value, "x2"),
        service._number(box_value, "y2"),
        float(box_value.get("confidence", 1.0)),
    )
    crops = make_multiscale_crops(image, box)
    return {
        "path": str(Path(path).expanduser()),
        "image_size": list(image.size),
        "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
        "crops": {
            crop_type.value: {
                "size": list(crop.size),
                "box": list(crop_box(box, image.size, crop_type)),
            }
            for crop_type, crop in crops.items()
        },
    }


def handle_fusion(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, dict):
        raise WorkerError("INVALID_PAYLOAD", "recognition.fuse 需要 predictions 对象")
    predictions: dict[str, list[Mapping[str, object]]] = {}
    for crop_type, values in raw_predictions.items():
        if not isinstance(values, list):
            raise WorkerError("INVALID_PAYLOAD", f"裁剪 {crop_type} 的 predictions 必须是数组")
        predictions[str(crop_type)] = [value for value in values if isinstance(value, dict)]
    policy_raw = payload.get("config")
    config = parse_fusion_config(policy_raw)
    allowed_raw = payload.get("allowed_tags")
    if allowed_raw is not None and (
        not isinstance(allowed_raw, list)
        or not all(isinstance(tag, str) for tag in allowed_raw)
    ):
        raise WorkerError("INVALID_PAYLOAD", "allowed_tags 必须是字符串数组")
    top_k_value = payload.get("top_k", 5)
    if not isinstance(top_k_value, int):
        raise WorkerError("INVALID_PAYLOAD", "top_k 必须是整数")
    result = fuse_multiscale(
        predictions,
        config=config,
        allowed_tags=allowed_raw if isinstance(allowed_raw, list) else None,
        top_k=top_k_value,
    )
    return result.as_dict()


def handle_character_set_intersection(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    from ..whitelist import load_character_set

    model_tags = payload.get("model_tags")
    if not isinstance(model_tags, list) or not all(isinstance(tag, str) for tag in model_tags):
        raise WorkerError("INVALID_PAYLOAD", "model_tags 必须是字符串数组")
    configured_path = payload.get("character_set_path")
    source = configured_path if isinstance(configured_path, str) else service.character_set_path
    character_set = load_character_set(source)
    supported, missing = character_set.model_intersection(model_tags)
    return {
        "set_id": character_set.set_id,
        "schema_version": character_set.schema_version,
        "supported_tags": sorted(supported),
        "missing_tags": sorted(missing),
        "supported_count": len(supported),
        "missing_count": len(missing),
    }


def handle_export(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    output_path = service._required_string(payload, "output_path", fallback_key="path")
    results = payload.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise WorkerError("INVALID_PAYLOAD", "results.export 需要对象数组 results")
    export_format = payload.get("format")
    if export_format is not None and not isinstance(export_format, str):
        raise WorkerError("INVALID_PAYLOAD", "format 必须是 json 或 csv")
    try:
        destination = export_results(results, output_path, format=export_format)
    except (OSError, ValueError) as exc:
        raise WorkerError("EXPORT_FAILED", "结果导出失败", str(exc), True) from exc
    return {
        "path": str(destination),
        "format": (export_format or destination.suffix.lstrip(".")).casefold(),
        "count": len(results),
    }
