"""Safe local image enumeration, EXIF orientation, and multi-scale crops."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import Image as PILImage

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
    }
)
# Keep enough headroom for common high-resolution artwork while preventing a
# single decoded image from consuming an unbounded amount of process memory.
MAX_IMAGE_PIXELS = 100_000_000


class CropType(StrEnum):
    HEAD = "head"
    BUST = "bust"
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class DetectionBox:
    """A detector box in the EXIF-corrected image coordinate system."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def normalized(self, width: int, height: int) -> DetectionBox:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return DetectionBox(
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
            max(0.0, min(1.0, self.confidence)),
        )


@dataclass(frozen=True, slots=True)
class CropExpansion:
    """Relative expansion around a detector box.

    Values are fractions of the detected width/height and are deliberately
    centralized so a benchmark can record exactly which crop policy it used.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if min(self.left, self.top, self.right, self.bottom) < 0:
            raise ValueError("裁剪扩展比例不能为负数")


DEFAULT_CROP_EXPANSIONS: dict[CropType, CropExpansion] = {
    CropType.HEAD: CropExpansion(0.15, 0.15, 0.15, 0.15),
    CropType.BUST: CropExpansion(0.35, 0.20, 0.35, 0.85),
    CropType.LARGE: CropExpansion(0.75, 0.55, 0.75, 1.35),
}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    relative_path: str
    format: str
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def _image_sort_key(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    return relative.casefold(), relative


def _image_paths(
    root: Path,
    *,
    recursive: bool,
    include_hidden: bool,
) -> Iterator[Path]:
    iterator: Iterator[Path] = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if (
            path.is_file()
            and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            and (
                include_hidden
                or not any(part.startswith(".") for part in path.relative_to(root).parts)
            )
        ):
            yield path


def enumerate_images_with_total(
    directory: str | Path,
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    limit: int | None = None,
) -> tuple[list[Path], int]:
    """Return supported image files in stable, case-insensitive path order.

    This is intentionally read-only and does not follow a directory's file
    contents until the caller explicitly asks to load a selected file.  When a
    limit is supplied, only the lexicographically smallest paths are retained;
    the complete supported-file count is still returned.
    """

    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"图片目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"图片路径不是目录: {root}")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError("limit 必须是非负整数")

    # Keep an insertion-sorted top-N.  The worker caps limit at 50, so this
    # avoids retaining an unbounded directory listing while remaining simple
    # and deterministic across filesystem traversal orders.
    selected: list[Path] = []
    selected_keys: list[tuple[str, str]] = []
    total_count = 0
    for path in _image_paths(root, recursive=recursive, include_hidden=include_hidden):
        total_count += 1
        key = _image_sort_key(root, path)
        if limit is None:
            selected.append(path)
            selected_keys.append(key)
            continue
        index = bisect_left(selected_keys, key)
        if index >= limit and len(selected) >= limit:
            continue
        selected_keys.insert(index, key)
        selected.insert(index, path)
        if len(selected) > limit:
            selected.pop()
            selected_keys.pop()
    if limit is None:
        selected.sort(key=lambda path: _image_sort_key(root, path))
    return selected, total_count


def enumerate_images(
    directory: str | Path,
    *,
    recursive: bool = True,
    include_hidden: bool = False,
) -> list[Path]:
    """Backward-compatible unlimited image enumeration."""

    paths, _ = enumerate_images_with_total(
        directory, recursive=recursive, include_hidden=include_hidden
    )
    return paths


def validate_image_pixel_budget(size: tuple[int, int]) -> None:
    """Reject dimensions whose decoded pixel buffer exceeds the process budget."""

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸必须为正数")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"图片像素数 {width * height:,} 超过安全上限 {MAX_IMAGE_PIXELS:,}"
        )


def read_image(path: str | Path) -> PILImage:
    """Read an image with EXIF orientation applied and detached file handles.

    The returned image is RGB/RGBA-compatible and owns its pixel data; the
    source file is closed before this function returns.  EXIF metadata is not
    copied to derived crops, avoiding accidental propagation of private data.
    """

    source = Path(path).expanduser()
    try:
        with Image.open(source) as opened:
            validate_image_pixel_budget(opened.size)
            oriented = ImageOps.exif_transpose(opened)
            # copy() detaches from the file and also handles the case where
            # exif_transpose returns the original object.
            image = oriented.convert("RGBA" if "A" in oriented.getbands() else "RGB").copy()
    except (FileNotFoundError, PermissionError):
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"无法安全读取图片: {source}") from exc
    return image


@dataclass(frozen=True)
class SampledFrames:
    """Frames sampled from a (possibly animated) image file."""

    frames: list[PILImage]
    indices: list[int]
    frame_count: int
    is_animated: bool


def _detached_frame(opened: Image.Image) -> PILImage:
    """Convert one frame to RGB/RGBA and detach it from the file handle."""

    oriented = ImageOps.exif_transpose(opened)
    return oriented.convert("RGBA" if "A" in oriented.getbands() else "RGB").copy()


def sample_image_frames(path: str | Path, *, max_frames: int = 3) -> SampledFrames:
    """Read a still image or sample up to ``max_frames`` frames of an animation.

    Animated GIF/WebP files are represented by their first, middle and last
    frame, which keeps similarity comparisons cheap while still matching a
    still image against any part of the animation. Still images yield exactly
    one frame and behave identically to :func:`read_image`.
    """

    if max_frames < 1:
        raise ValueError("max_frames 必须是正整数")
    source = Path(path).expanduser()
    try:
        with Image.open(source) as opened:
            validate_image_pixel_budget(opened.size)
            frame_count = int(getattr(opened, "n_frames", 1) or 1)
            is_animated = frame_count > 1
            if is_animated:
                candidates = [0, frame_count // 2, frame_count - 1][:max_frames]
            else:
                candidates = [0]
            indices: list[int] = []
            frames: list[PILImage] = []
            for index in candidates:
                if index in indices:
                    continue
                opened.seek(index)
                frames.append(_detached_frame(opened))
                indices.append(index)
    except (FileNotFoundError, PermissionError):
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, EOFError) as exc:
        raise ValueError(f"无法安全读取图片: {source}") from exc
    return SampledFrames(
        frames=frames,
        indices=indices,
        frame_count=frame_count,
        is_animated=is_animated,
    )


def inspect_image(path: str | Path) -> ImageRecord:
    """Read header dimensions/format and EXIF orientation without decoding pixels."""

    source = Path(path).expanduser()
    try:
        with Image.open(source) as opened:
            image_format = opened.format or source.suffix.lstrip(".").upper()
            width, height = opened.size
            validate_image_pixel_budget((width, height))
            orientation = opened.getexif().get(274)
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"无法安全读取图片头: {source}") from exc
    return ImageRecord(source, source.name, image_format.upper(), width, height)


def crop_box(
    box: DetectionBox,
    image_size: tuple[int, int],
    crop_type: CropType | str,
    *,
    expansions: dict[CropType, CropExpansion] | None = None,
) -> tuple[int, int, int, int]:
    """Calculate an integer crop box, clamped to image bounds.

    Coordinates use left/top inclusive and right/bottom exclusive PIL
    semantics.  A degenerate detector box still yields a one-pixel crop when
    the image has non-zero dimensions, preventing surprising empty crops.
    """

    crop_kind = CropType(crop_type)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸必须为正数")
    normalized = box.normalized(width, height)
    expansion = (expansions or DEFAULT_CROP_EXPANSIONS)[crop_kind]
    box_width = max(0.0, normalized.x2 - normalized.x1)
    box_height = max(0.0, normalized.y2 - normalized.y1)
    left = math.floor(normalized.x1 - box_width * expansion.left)
    top = math.floor(normalized.y1 - box_height * expansion.top)
    right = math.ceil(normalized.x2 + box_width * expansion.right)
    bottom = math.ceil(normalized.y2 + box_height * expansion.bottom)
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def crop_image(
    image: PILImage,
    box: DetectionBox,
    crop_type: CropType | str,
    *,
    expansions: dict[CropType, CropExpansion] | None = None,
) -> PILImage:
    """Return a detached crop without modifying the source image."""

    coordinates = crop_box(box, image.size, crop_type, expansions=expansions)
    return image.crop(coordinates).copy()


def make_multiscale_crops(
    image: PILImage,
    box: DetectionBox,
    *,
    crop_types: Iterable[CropType | str] = (CropType.HEAD, CropType.BUST, CropType.LARGE),
    expansions: dict[CropType, CropExpansion] | None = None,
) -> dict[CropType, PILImage]:
    """Create Head/Bust/Large crops associated with one detection box."""

    return {
        CropType(crop_type): crop_image(image, box, crop_type, expansions=expansions)
        for crop_type in crop_types
    }
