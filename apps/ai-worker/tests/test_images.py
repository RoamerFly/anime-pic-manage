from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import ai_worker.images as images
from ai_worker.images import (
    MAX_IMAGE_PIXELS,
    CropType,
    DetectionBox,
    crop_box,
    enumerate_images,
    enumerate_images_with_total,
    inspect_image,
    read_image,
    validate_image_pixel_budget,
)


def test_enumerate_and_read_chinese_windows_path(tmp_path: Path) -> None:
    root = tmp_path / "测试图库" / "角色"
    root.mkdir(parents=True)
    image_path = root / "头像 中文.PNG"
    Image.new("RGB", (12, 8), "red").save(image_path)
    (root / "notes.txt").write_text("not an image", encoding="utf-8")

    paths = enumerate_images(root)
    assert paths == [image_path]
    image = read_image(image_path)
    assert image.size == (12, 8)


def test_limited_enumeration_keeps_stable_top_n_and_total_without_hidden_files(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (4, 3), "white").save(tmp_path / "z.png")
    Image.new("RGB", (4, 3), "white").save(tmp_path / "A.PNG")
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (4, 3), "white").save(nested / "b.jpg")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    Image.new("RGB", (4, 3), "white").save(hidden / "a.png")

    paths, total = enumerate_images_with_total(tmp_path, limit=2)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == ["A.PNG", "nested/b.jpg"]
    assert total == 3
    assert enumerate_images(tmp_path) == sorted(
        [tmp_path / "z.png", tmp_path / "A.PNG", nested / "b.jpg"],
        key=lambda path: path.relative_to(tmp_path).as_posix().casefold(),
    )


def test_limited_enumeration_can_include_hidden_paths(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    image_path = hidden / "a.png"
    Image.new("RGB", (4, 3), "white").save(image_path)

    paths, total = enumerate_images_with_total(tmp_path, include_hidden=True, limit=1)

    assert paths == [image_path]
    assert total == 1


def test_inspect_image_reads_header_and_corrects_exif_orientation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90° clockwise: displayed dimensions are swapped.
    Image.new("RGB", (10, 4), "white").save(image_path, exif=exif)
    monkeypatch.setattr(images, "read_image", lambda _: pytest.fail("must not decode pixels"))

    record = inspect_image(image_path)

    assert record.format == "JPEG"
    assert (record.width, record.height) == (4, 10)


def test_image_pixel_budget_rejects_oversized_dimensions_without_allocating_pixels() -> None:
    validate_image_pixel_budget((1, 1))
    width = MAX_IMAGE_PIXELS // 1000 + 1

    with pytest.raises(ValueError, match="像素数"):
        validate_image_pixel_budget((width, 1000))


def test_crop_boxes_are_clamped_and_non_empty() -> None:
    size = (100, 80)
    assert crop_box(DetectionBox(-20, -10, 10, 12), size, CropType.HEAD) == (0, 0, 12, 14)
    right_edge = crop_box(DetectionBox(95, 75, 120, 100), size, CropType.LARGE)
    assert right_edge[0] >= 0 and right_edge[1] >= 0
    assert right_edge[2] <= 100 and right_edge[3] <= 80
    assert right_edge[2] > right_edge[0] and right_edge[3] > right_edge[1]


def test_degenerate_detection_still_produces_one_pixel_crop() -> None:
    assert crop_box(DetectionBox(50, 40, 50, 40), (100, 80), CropType.HEAD) == (50, 40, 51, 41)
