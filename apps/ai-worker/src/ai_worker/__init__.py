"""Local AI worker for anime image analysis.

The package deliberately keeps model dependencies optional.  Image discovery,
crop generation, score filtering, fusion, and the JSON Lines protocol work on
the core Pillow dependency alone.
"""

import os as _os

__version__ = "0.1.0"

from .protocol import SCHEMA_VERSION, WorkerRequest, WorkerResponse

__all__ = ["SCHEMA_VERSION", "WorkerRequest", "WorkerResponse", "__version__"]


def _apply_environment_defaults() -> None:
    """Prefer plain HTTPS for first-run model downloads.

    ``huggingface_hub`` picks up the Rust ``hf_xet`` transfer backend when it is
    installed, and that backend stalls indefinitely behind some Windows proxies
    (a 0-byte ``.incomplete`` blob and no error).  Plain HTTPS is slower but
    always progresses, so make it the default for this process; exporting
    ``HF_HUB_DISABLE_XET=0`` keeps the Xet backend available to anyone who wants
    it.
    """

    _os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# Must run before any module that imports ``huggingface_hub`` (imgutils does).
_apply_environment_defaults()
