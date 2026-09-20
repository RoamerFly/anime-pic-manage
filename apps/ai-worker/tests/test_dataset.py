from __future__ import annotations

from pathlib import Path

import imgutils.tagging
import pytest
from PIL import Image, ImageDraw

from ai_worker.dataset import export_lora_dataset
from ai_worker.errors import WorkerError


def make_image(
    path: Path,
    color: tuple[int, int, int],
    size: tuple[int, int] = (640, 960),
    pattern: str = "ellipse",
) -> Path:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    if pattern == "ellipse":
        draw.ellipse([40, 40, 200, 200], fill=(250, 250, 250))
    else:
        draw.rectangle([20, 20, size[0] - 20, size[1] // 2], fill=(250, 250, 250))
    image.save(path)
    return path


@pytest.fixture()
def fake_tagger(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_wd14_tags(*_args: object, **_kwargs: object):
        return (
            {"rating_explicit": 0.9},
            {"long_hair": 0.9, "smile": 0.8, "red_hair": 0.7},
            {"character_alpha": 0.95},
        )

    monkeypatch.setattr(imgutils.tagging, "get_wd14_tags", fake_get_wd14_tags)


def test_export_writes_pictures_captions_and_kohya_config(
    tmp_path: Path, fake_tagger: None
) -> None:
    source = make_image(tmp_path / "source.png", (180, 40, 40))
    output = tmp_path / "dataset"

    result = export_lora_dataset(
        {
            "identity_label": "character_alpha",
            "trigger": "character_alpha",
            "output_dir": str(output),
            "images": [
                {"path": str(source), "bbox": [0.1, 0.1, 0.6, 0.8]},
            ],
            "crop_mode": "bust",
            "resolution": 512,
            "quality_preset": "pony",
            "repeats": 10,
            "keep_tokens": 4,
            "shuffle_caption": True,
        }
    )

    assert result["kept"] == 1
    train_dir = Path(result["train_dir"])
    assert (train_dir / "001.png").is_file()
    caption = (train_dir / "001.txt").read_text(encoding="utf-8")
    # Pony quality prefix, then the trigger, then Danbooru-style tags.
    assert caption.startswith("score_9,score_8_up,score_7_up,character_alpha,")
    assert "long hair" in caption
    assert " " not in caption.split("character_alpha")[1].split(",")[0]
    assert "," in caption and ", " not in caption

    config = Path(result["dataset_toml"]).read_text(encoding="utf-8")
    assert "resolution = 512" in config
    assert "keep_tokens = 4" in config
    assert "shuffle_caption = true" in config
    assert "num_repeats = 10" in config

    manifest = Path(result["output_dir"]) / "dataset-manifest.json"
    assert manifest.is_file()


def test_duplicates_and_small_crops_are_skipped(tmp_path: Path, fake_tagger: None) -> None:
    first = make_image(tmp_path / "a.png", (60, 60, 200))
    second = make_image(tmp_path / "b.png", (60, 60, 200))
    tiny = make_image(tmp_path / "tiny.png", (30, 200, 30), size=(200, 200))

    result = export_lora_dataset(
        {
            "identity_label": "character_beta",
            "trigger": "character_beta",
            "output_dir": str(tmp_path / "out"),
            "images": [
                {"path": str(first), "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"path": str(second), "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"path": str(tiny), "bbox": [0.45, 0.45, 0.1, 0.1]},
            ],
            "crop_mode": "person",
            "resolution": 512,
        }
    )

    assert result["kept"] == 1
    assert result["skipped"] == 2
    reasons = {item["reason"] for item in result["skip_reasons"]}
    assert any("重复" in reason for reason in reasons)
    assert any("过小" in reason for reason in reasons)


def test_max_images_is_respected(tmp_path: Path, fake_tagger: None) -> None:
    images = [
        {
            "path": str(
                make_image(
                    tmp_path / f"img{i}.png",
                    (i * 40, 200 - i * 30, 60 + i * 20),
                    size=(512 + i * 64, 768 - i * 48),
                    pattern="ellipse" if i % 2 == 0 else "rect",
                )
            ),
            "bbox": None,
        }
        for i in range(5)
    ]

    result = export_lora_dataset(
        {
            "identity_label": "character_gamma",
            "trigger": "character_gamma",
            "output_dir": str(tmp_path / "limited"),
            "images": images,
            "resolution": 384,
            "max_images": 2,
        }
    )

    assert result["kept"] == 2
    assert len(list(Path(result["train_dir"]).glob("*.png"))) == 2


def test_missing_required_fields_are_rejected() -> None:
    with pytest.raises(WorkerError) as error:
        export_lora_dataset({"trigger": "x", "images": []})

    assert error.value.code == "INVALID_PAYLOAD"


def test_invalid_crop_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkerError) as error:
        export_lora_dataset(
            {
                "identity_label": "x",
                "trigger": "x",
                "output_dir": str(tmp_path),
                "crop_mode": "fullbody",
                "images": [{"path": "a.png"}],
            }
        )

    assert error.value.code == "INVALID_PAYLOAD"


def test_foreign_folder_is_never_overwritten(tmp_path: Path, fake_tagger: None) -> None:
    source = make_image(tmp_path / "source.png", (10, 120, 200))
    curated = tmp_path / "dataset" / "character_alpha"
    curated.mkdir(parents=True)
    precious = curated / "hand_written.txt"
    precious.write_text("看这里，这是我自己整理的数据", encoding="utf-8")

    with pytest.raises(WorkerError) as error:
        export_lora_dataset(
            {
                "identity_label": "character_alpha",
                "trigger": "character_alpha",
                "output_dir": str(tmp_path / "dataset"),
                "images": [{"path": str(source), "bbox": [0.0, 0.0, 1.0, 1.0]}],
                "resolution": 384,
            }
        )

    assert error.value.code == "DATASET_DIR_NOT_EMPTY"
    assert precious.read_text(encoding="utf-8") == "看这里，这是我自己整理的数据"


def test_reexport_replaces_the_previous_export(tmp_path: Path, fake_tagger: None) -> None:
    source = make_image(tmp_path / "source.png", (200, 40, 60))
    payload = {
        "identity_label": "character_beta",
        "trigger": "character_beta",
        "output_dir": str(tmp_path / "dataset"),
        "images": [{"path": str(source), "bbox": [0.0, 0.0, 1.0, 1.0]}],
        "resolution": 384,
    }

    first = export_lora_dataset(payload)
    stale = Path(first["train_dir"]) / "999.png"
    stale.write_bytes(b"leftover from an older export")

    second = export_lora_dataset(payload)

    assert second["kept"] == 1
    assert not stale.exists()
    assert sorted(item.name for item in Path(second["train_dir"]).iterdir()) == [
        "001.png",
        "001.txt",
    ]
