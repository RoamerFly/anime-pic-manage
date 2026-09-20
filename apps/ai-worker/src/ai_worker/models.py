"""Strict model manifest handling, ONNX adapters, and model catalog."""

import os

from .adapters import (
    Adapter,
    AnimeHeadONNXAdapter,
    AnimeTimmONNXAdapter,
    ONNXAdapter,
    bounded_onnx_intra_op_threads,
    create_adapter,
    make_onnx_session_options,
)
from .models_catalog import ModelCatalog
from .models_manifest import (
    ModelLabel,
    ModelManifest,
    PreprocessSpec,
    sha256_file,
)

__all__ = [
    "Adapter",
    "AnimeHeadONNXAdapter",
    "AnimeTimmONNXAdapter",
    "ModelCatalog",
    "ModelLabel",
    "ModelManifest",
    "ONNXAdapter",
    "PreprocessSpec",
    "bounded_onnx_intra_op_threads",
    "create_adapter",
    "make_onnx_session_options",
    "os",
    "sha256_file",
]
