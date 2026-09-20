"""Local AI worker for anime image analysis.

The package deliberately keeps model dependencies optional.  Image discovery,
crop generation, score filtering, fusion, and the JSON Lines protocol work on
the core Pillow dependency alone.
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path
from typing import Mapping as _Mapping

__version__ = "0.1.0"

from .protocol import SCHEMA_VERSION, WorkerRequest, WorkerResponse

__all__ = ["SCHEMA_VERSION", "WorkerRequest", "WorkerResponse", "__version__"]


def _apply_environment_defaults() -> None:
    """Make downloads portable before ``huggingface_hub`` reads its settings.

    Two things happen here, both of which must run before any module imports
    ``huggingface_hub`` (``imgutils`` does):

    * **Plain HTTPS for first-run downloads.** ``huggingface_hub`` picks up the
      Rust ``hf_xet`` transfer backend when it is installed, and that backend
      stalls indefinitely behind some Windows proxies (a 0-byte ``.incomplete``
      blob and no error).  Plain HTTPS is slower but always progresses, so make
      it the default for this process; exporting ``HF_HUB_DISABLE_XET=0`` keeps
      the Xet backend available to anyone who wants it.
    * **A cache inside the application folder.** A portable copy has to carry
      its tagging model, reference model and downloaded assets, so a stray
      ``%USERPROFILE%\\.cache\\huggingface`` must never be what the pipeline
      uses.  ``resolve_hf_cache_dir`` decides the folder and the environment
      variables below pin it.
    """

    _os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    cache_dir = resolve_hf_cache_dir()
    if cache_dir is not None:
        # ``HF_HUB_CACHE`` wins over ``HF_HOME`` inside huggingface_hub; both are
        # set so sibling caches (assets, datasets) stay in the same folder.
        _os.environ["HF_HOME"] = str(cache_dir)
        _os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")


def resolve_hf_cache_dir(
    env: _Mapping[str, str] | None = None,
    executable: str | None = None,
) -> _Path | None:
    """Return the Hugging Face cache this process should use.

    The desktop always exports ``ANIME_PIC_HF_CACHE`` (``<package>\\data\\hf-cache``).
    Everything else is a fallback so a Worker started by hand still writes into
    the checkout or package it lives in:

    * ``ANIME_PIC_MODELS`` points at ``<root>\\models``, so its parent is the
      application root.
    * A frozen executable under ``<root>\\app\\runtime\\`` reveals the same root.

    ``None`` means "no application folder known": a bare development run keeps
    the machine-wide cache instead of inventing a location.
    """

    lookup = _os.environ if env is None else env
    explicit = (lookup.get("ANIME_PIC_HF_CACHE") or "").strip()
    if explicit:
        return _Path(explicit).expanduser()

    models_root = (lookup.get("ANIME_PIC_MODELS") or "").strip()
    if models_root:
        return _Path(models_root).expanduser().parent / "data" / "hf-cache"

    root = _portable_root(executable or _sys.executable)
    if root is None:
        return None
    return root / "data" / "hf-cache"


def _portable_root(executable: str) -> _Path | None:
    """Locate ``<root>`` for a packaged Worker, or ``None`` for source runs."""

    if not getattr(_sys, "frozen", False):
        return None
    path = _Path(executable)
    for candidate in path.parents:
        if candidate.name.lower() == "app":
            return candidate.parent
    return None


# Must run before any module that imports ``huggingface_hub`` (imgutils does).
_apply_environment_defaults()
