"""The Worker keeps its Hugging Face cache inside the application folder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import ai_worker
from ai_worker import resolve_hf_cache_dir


def test_an_explicit_cache_folder_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "package" / "data" / "hf-cache"

    resolved = resolve_hf_cache_dir(
        {
            "ANIME_PIC_HF_CACHE": str(explicit),
            "ANIME_PIC_MODELS": str(tmp_path / "models"),
        }
    )

    assert resolved == explicit


def test_the_models_folder_places_the_cache_beside_it(tmp_path: Path) -> None:
    # The desktop exports only ANIME_PIC_MODELS in some launch paths, so the
    # folder it points at has to imply the package root.
    resolved = resolve_hf_cache_dir({"ANIME_PIC_MODELS": str(tmp_path / "models")})

    assert resolved == tmp_path / "data" / "hf-cache"


def test_a_frozen_worker_finds_its_own_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    executable = tmp_path / "app" / "runtime" / "ai-worker.exe"

    resolved = resolve_hf_cache_dir({}, str(executable))

    assert resolved == tmp_path / "data" / "hf-cache"


def test_a_source_checkout_without_context_keeps_the_default() -> None:
    # Nothing points at a package, so the Worker must not invent a location.
    assert resolve_hf_cache_dir({}) is None


def test_environment_defaults_pin_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "data" / "hf-cache"
    monkeypatch.setenv("ANIME_PIC_HF_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    # A machine-wide HF_HOME must not survive the pinning.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))

    ai_worker._apply_environment_defaults()

    assert os.environ["HF_HOME"] == str(cache)
    assert os.environ["HF_HUB_CACHE"] == str(cache / "hub")
    # Plain HTTPS stays the default for first-run downloads.
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
