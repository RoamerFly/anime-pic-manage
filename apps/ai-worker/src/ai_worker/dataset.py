"""LoRA training dataset export: crops, dedupe and WD14 captioning.

The export mirrors the layout the anime community uses with kohya sd-scripts
(and the folders this project's own LoRAs were trained from): one directory per
character holding images plus same-name ``.txt`` captions written as
comma-separated Danbooru tags, with the character trigger word first.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import WorkerError
from .images import read_image
from .similarity import compute_phash, hamming_distance

MAX_EXPORT_IMAGES = 500
MIN_EXPORT_IMAGES = 1
# Must be one of imgutils' tagger profiles; SwinV2_v3 is the current default
# (SwinV2 architecture, v3 tag set).
WD14_MODEL = "SwinV2_v3"

# Quality prefixes commonly prepended to Pony / Illustrious family captions.
QUALITY_PREFIXES: dict[str, str] = {
    "none": "",
    "pony": "score_9,score_8_up,score_7_up",
    "illustrious": "masterpiece,best quality,amazing quality",
    "sd15": "masterpiece,best quality",
}


def _clean_tag(tag: str) -> str:
    return tag.strip().replace("_", " ").lower()


def _parse_csv(value: object, *, clean: bool = True) -> list[str]:
    if not isinstance(value, str):
        return []
    items = [item.strip() for item in re.split(r"[,\n]", value) if item.strip()]
    if clean:
        return [_clean_tag(item) for item in items]
    return [item.lower() for item in items]


@dataclass(frozen=True)
class ExportOptions:
    identity_label: str
    trigger: str
    output_dir: Path
    crop_mode: str
    resolution: int
    max_images: int
    duplicate_distance: int
    quality_preset: str
    extra_tags: list[str]
    remove_tags: list[str]
    drop_character_tags: bool
    general_threshold: float
    character_threshold: float
    repeats: int
    keep_tokens: int
    shuffle_caption: bool
    base_model: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExportOptions:
        def required(name: str) -> str:
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip():
                raise WorkerError("INVALID_PAYLOAD", f"dataset.export 需要非空字段 {name}")
            return value.strip()

        def number(name: str, default: float) -> float:
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WorkerError("INVALID_PAYLOAD", f"{name} 必须是数字")
            return float(value)

        def integer(name: str, default: int, *, lower: int, upper: int) -> int:
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise WorkerError("INVALID_PAYLOAD", f"{name} 必须是整数")
            if not lower <= value <= upper:
                raise WorkerError("INVALID_PAYLOAD", f"{name} 必须在 {lower} 到 {upper} 之间")
            return value

        crop_mode = str(payload.get("crop_mode", "person"))
        if crop_mode not in {"person", "head", "bust", "large", "none"}:
            raise WorkerError("INVALID_PAYLOAD", "crop_mode 只能是 person/head/bust/large/none")
        quality = str(payload.get("quality_preset", "none"))
        if quality not in QUALITY_PREFIXES:
            raise WorkerError(
                "INVALID_PAYLOAD",
                f"quality_preset 只能是 {'、'.join(QUALITY_PREFIXES)}",
            )
        label = re.sub(r'[<>:"/\\|?*]', "_", required("identity_label")).strip(" .")
        if not label:
            raise WorkerError("INVALID_PAYLOAD", "identity_label 不能用作文件夹名")
        return cls(
            identity_label=label,
            trigger=required("trigger"),
            output_dir=Path(required("output_dir")).expanduser(),
            crop_mode=crop_mode,
            resolution=integer("resolution", 1024, lower=384, upper=1536),
            max_images=integer("max_images", 40, lower=MIN_EXPORT_IMAGES, upper=MAX_EXPORT_IMAGES),
            duplicate_distance=integer("duplicate_distance", 4, lower=0, upper=16),
            quality_preset=quality,
            extra_tags=_parse_csv(payload.get("extra_tags"), clean=False),
            remove_tags=_parse_csv(payload.get("remove_tags")),
            drop_character_tags=bool(payload.get("drop_character_tags", True)),
            general_threshold=number("general_threshold", 0.35),
            character_threshold=number("character_threshold", 0.85),
            repeats=integer("repeats", 10, lower=1, upper=100),
            keep_tokens=integer("keep_tokens", 0, lower=0, upper=20),
            shuffle_caption=bool(payload.get("shuffle_caption", True)),
            base_model=str(payload.get("base_model", "")).strip(),
        )


def _crop_for_training(
    image: Image.Image,
    bbox: tuple[float, float, float, float] | None,
    mode: str,
) -> Image.Image:
    """Crop the annotated person (or a part of it) with a small margin."""

    if mode == "none" or bbox is None:
        return image
    width, height = image.size
    x, y, w, h = bbox
    left, top = x * width, y * height
    right, bottom = (x + w) * width, (y + h) * height
    person_height = max(1.0, bottom - top)
    person_width = max(1.0, right - left)
    if mode == "head":
        bottom = top + person_height * 0.45
    elif mode == "bust":
        bottom = top + person_height * 0.7
    elif mode == "large":
        # Slightly wider context around the person for full-body/action poses.
        pad_x, pad_y = person_width * 0.15, person_height * 0.1
        left, right = left - pad_x, right + pad_x
        top, bottom = top - pad_y, bottom + pad_y
    left = max(0.0, min(left, width - 1))
    top = max(0.0, min(top, height - 1))
    right = max(left + 1.0, min(right, float(width)))
    bottom = max(top + 1.0, min(bottom, float(height)))
    return image.crop((int(left), int(top), int(right), int(bottom)))


def _resize_for_training(image: Image.Image, resolution: int) -> Image.Image:
    """Scale so the shorter side matches the bucket resolution."""

    width, height = image.size
    scale = resolution / max(1, min(width, height))
    if abs(scale - 1.0) < 0.02:
        return image
    target = (max(64, int(width * scale)), max(64, int(height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def _caption_for(
    image: Image.Image,
    options: ExportOptions,
    progress: Callable[[str], None] | None = None,
) -> str:
    try:
        # ``imgutils`` ships no type information; import it lazily so the
        # tagging dependency stays optional for environments that never export.
        tagging = importlib.import_module("imgutils.tagging")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise WorkerError(
            "DEPENDENCY_MISSING",
            "缺少 dghs-imgutils 的 WD14 打标依赖，无法自动生成训练 caption",
            str(exc),
        ) from exc

    if progress:
        progress("tagging")
    rating, general, characters = tagging.get_wd14_tags(
        image,
        model_name=WD14_MODEL,
        general_threshold=options.general_threshold,
        character_threshold=options.character_threshold,
        no_underline=True,
        fmt=("rating", "general", "character"),
    )
    tags: list[str] = []
    sources = [general] + ([characters] if not options.drop_character_tags else [])
    for source in sources:
        for tag in source:
            cleaned = _clean_tag(tag)
            if cleaned and cleaned not in tags:
                tags.append(cleaned)

    del rating  # rating_* tags are model-family specific and never captioned
    blocked = set(options.remove_tags)
    # Quality tags (`score_9`, `score_8_up`) and the trigger word keep their
    # canonical underscore form; only WD14 output is rewritten to spaces.
    quality = QUALITY_PREFIXES[options.quality_preset]
    prefix = [item.strip().lower() for item in quality.split(",") if item.strip()]
    head = prefix + [options.trigger.strip().lower()]
    body = [tag for tag in tags if tag not in blocked and tag not in head]
    body.extend(tag for tag in options.extra_tags if tag not in body and tag not in head)
    return ",".join(head + body)


def _dataset_toml(options: ExportOptions, train_dir: Path) -> str:
    caption_extension = ".txt"
    return "\n".join(
        [
            "[general]",
            f"shuffle_caption = {'true' if options.shuffle_caption else 'false'}",
            f"keep_tokens = {options.keep_tokens}",
            "",
            "[[datasets]]",
            f"resolution = {options.resolution}",
            # `batch_size` is deliberately left out: sd-scripts refuses to start
            # when both the dataset config and `--train_batch_size` set it, and
            # the trainer UI controls the batch size on the command line.
            "enable_bucket = true",
            f"min_bucket_reso = {min(512, options.resolution)}",
            f"max_bucket_reso = {max(1024, options.resolution)}",
            "bucket_no_upscale = false",
            "",
            "  [[datasets.subsets]]",
            f"  image_dir = '{train_dir.as_posix()}'",
            f"  num_repeats = {options.repeats}",
            f"  caption_extension = '{caption_extension}'",
            "",
        ]
    )


def export_lora_dataset(
    payload: Mapping[str, object],
    *,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Crop, dedupe, caption and lay out a kohya-compatible dataset."""

    options = ExportOptions.from_payload(payload)
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise WorkerError("INVALID_PAYLOAD", "dataset.export 需要 images 数组")

    train_dir = options.output_dir / options.identity_label
    manifest_path = options.output_dir / "dataset-manifest.json"
    if train_dir.exists():
        stale = [entry for entry in train_dir.iterdir() if entry.is_file()]
        if stale and not manifest_path.is_file():
            # Only ever clean up after ourselves: a non-empty folder without
            # our manifest may hold hand-curated training data.
            raise WorkerError(
                "DATASET_DIR_NOT_EMPTY",
                f"目录 {train_dir} 已存在且不是本工具导出的训练集，"
                "为避免覆盖你自己的文件已停止导出，请更换输出目录或先手动清理。",
            )
        for entry in stale:
            entry.unlink()
    train_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    hashes: list[str] = []
    total = len(raw_images)

    for index, entry in enumerate(raw_images):
        if len(kept) >= options.max_images:
            skipped.append({"path": "…", "reason": f"已达到上限 {options.max_images} 张"})
            break
        if not isinstance(entry, Mapping):
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        if on_progress:
            on_progress("processing", index + 1, total, Path(path).name)
        try:
            image = read_image(path)
        except Exception as exc:  # noqa: BLE001 - report per-file failures
            skipped.append({"path": path, "reason": f"无法读取: {exc}"})
            continue
        bbox = entry.get("bbox")
        box: tuple[float, float, float, float] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                box = tuple(float(value) for value in bbox)  # type: ignore[assignment]
            except (TypeError, ValueError):
                box = None
        cropped = _crop_for_training(image.convert("RGB"), box, options.crop_mode)
        if min(cropped.size) < 128:
            skipped.append({"path": path, "reason": "裁剪后尺寸过小"})
            continue

        digest = compute_phash(cropped)
        duplicate = any(
            hamming_distance(digest, previous) <= options.duplicate_distance
            for previous in hashes
        )
        if duplicate:
            skipped.append({"path": path, "reason": "与已选图片重复"})
            continue

        prepared = _resize_for_training(cropped, options.resolution)
        sequence = len(kept) + 1
        target = train_dir / f"{sequence:03d}.png"
        prepared.save(target, format="PNG")
        caption = _caption_for(prepared, options)
        target.with_suffix(".txt").write_text(caption, encoding="utf-8")
        hashes.append(digest)
        kept.append(
            {
                "source": path,
                "image": str(target),
                "caption": caption,
                "width": prepared.width,
                "height": prepared.height,
            }
        )

    if len(kept) < MIN_EXPORT_IMAGES:
        raise WorkerError(
            "DATASET_EMPTY",
            "没有任何图片通过筛选，无法生成训练集（检查裁剪模式与重复阈值）",
        )

    dataset_toml = options.output_dir / "dataset.toml"
    dataset_toml.write_text(_dataset_toml(options, train_dir), encoding="utf-8")
    manifest = {
        "identity": options.identity_label,
        "trigger": options.trigger,
        "crop_mode": options.crop_mode,
        "resolution": options.resolution,
        "quality_preset": options.quality_preset,
        "repeats": options.repeats,
        "keep_tokens": options.keep_tokens,
        "shuffle_caption": options.shuffle_caption,
        "base_model": options.base_model,
        "kept": kept,
        "skipped": skipped,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if on_progress:
        on_progress("completed", len(kept), len(kept), "训练集已导出")
    return {
        "output_dir": str(options.output_dir),
        "train_dir": str(train_dir),
        "dataset_toml": str(dataset_toml),
        "kept": len(kept),
        "skipped": len(skipped),
        "samples": kept[:5],
        "skip_reasons": skipped[:10],
        "trigger": options.trigger,
    }
