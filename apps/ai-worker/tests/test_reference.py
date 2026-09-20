from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_worker.errors import WorkerError
from ai_worker.reference import (
    DEFAULT_THRESHOLDS,
    ReferenceLibrary,
    build_library,
    match_library,
)


def make_image(path: Path, color: tuple[int, int, int] = (200, 60, 60)) -> Path:
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_library_round_trips_and_matches_by_cosine() -> None:
    library = ReferenceLibrary.from_value(
        {
            "backend": "embedding",
            "entries": [
                {
                    "identity_id": "character_alpha",
                    "display_name": "示例角色",
                    "vectors": [[1.0, 0.0], [0.8, 0.2]],
                }
            ],
        }
    )

    assert library.reference_count == 2
    assert library.threshold == DEFAULT_THRESHOLDS["embedding"]

    matched = match_library(library, [1.0, 0.0])
    assert matched is not None
    assert matched["matched"] is True
    assert matched["identity_id"] == "character_alpha"
    assert matched["similarity"] == 1.0
    assert matched["backend"] == "embedding"

    missed = match_library(library, [0.0, 1.0])
    assert missed is not None and missed["matched"] is False


def test_ccip_library_keeps_raw_vectors_and_its_own_threshold() -> None:
    # The CCIP distance itself runs the upstream metric model, which downloads
    # weights; that path is covered by the real end-to-end check instead. Here
    # we only pin the library contract around it.
    library = ReferenceLibrary.from_value(
        {
            "backend": "ccip",
            "threshold": 0.18,
            "entries": [{"identity_id": "x", "vectors": [[0.1, 0.2, 0.3]]}],
        }
    )

    assert library.threshold == 0.18
    assert library.entries[0].vectors == ((0.1, 0.2, 0.3),)
    assert library.reference_count == 1


def test_invalid_libraries_are_rejected() -> None:
    with pytest.raises(WorkerError):
        ReferenceLibrary.from_value({"backend": "magic"})
    with pytest.raises(WorkerError):
        ReferenceLibrary.from_value({"backend": "ccip", "entries": []})
    with pytest.raises(WorkerError):
        ReferenceLibrary.from_value({"backend": "ccip", "entries": [{"vectors": []}]})


def test_build_groups_samples_and_skips_failures(tmp_path: Path) -> None:
    good = make_image(tmp_path / "good.png")
    bad = make_image(tmp_path / "bad.png")

    def extract(sample) -> list[float]:
        if sample["path"] == str(bad):
            raise OSError("simulated decode failure")
        return [1.0, 0.0]

    result = build_library(
        [
            {"identity_id": "a", "display_name": "A", "path": str(good)},
            {"identity_id": "a", "display_name": "A", "path": str(good)},
            {"identity_id": "b", "display_name": "B", "path": str(bad)},
        ],
        backend="embedding",
        max_per_identity=1,
        extract=extract,
        built_at="2026-09-20T00:00:00Z",
    )

    assert result["identities"] == 1
    assert result["references"] == 1
    assert result["skipped"] == 1
    assert result["library"]["built_at"] == "2026-09-20T00:00:00Z"
    # Only one vector per identity is kept, so the second sample is ignored.
    assert len(result["library"]["entries"][0]["vectors"]) == 1


def test_build_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(WorkerError) as error:
        build_library(
            [{"identity_id": "a", "path": "x"}],
            backend="unknown",
            extract=lambda sample: [1.0],
        )

    assert error.value.code == "INVALID_PAYLOAD"
