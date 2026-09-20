"""Worker IPC request handlers."""

from .dataset import handle_dataset_export
from .models import (
    handle_model_cache_delete,
    handle_model_cache_prefetch,
    handle_model_cache_status,
    handle_model_delete,
    handle_model_install,
)
from .personal_model import (
    handle_personal_model_evaluate,
    handle_personal_model_train,
)
from .recognition import (
    handle_character_set_intersection,
    handle_crops,
    handle_enumerate,
    handle_export,
    handle_fusion,
    handle_recognition_embedding,
    handle_recognition_image,
    normalized_bbox,
    parse_embedding_annotations,
    parse_fusion_config,
    predict_with_embedding,
)
from .similarity import (
    handle_similarity_cluster,
    handle_similarity_features_extract,
    handle_similarity_scan,
)

__all__ = [
    "handle_character_set_intersection",
    "handle_crops",
    "handle_dataset_export",
    "handle_enumerate",
    "handle_export",
    "handle_fusion",
    "handle_model_cache_delete",
    "handle_model_cache_prefetch",
    "handle_model_cache_status",
    "handle_model_delete",
    "handle_model_install",
    "handle_personal_model_evaluate",
    "handle_personal_model_train",
    "handle_recognition_embedding",
    "handle_recognition_image",
    "handle_similarity_cluster",
    "handle_similarity_features_extract",
    "handle_similarity_scan",
    "normalized_bbox",
    "parse_embedding_annotations",
    "parse_fusion_config",
    "predict_with_embedding",
]
