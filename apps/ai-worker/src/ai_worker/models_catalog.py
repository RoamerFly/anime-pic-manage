"""Model catalog and discovery without eager weight loading."""

from __future__ import annotations

from pathlib import Path

from .adapters import Adapter, create_adapter
from .errors import ModelError, ModelManifestError, ModelNotInstalledError
from .models_manifest import ModelManifest


class ModelCatalog:
    """Discover manifests without loading model weights."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self._manifests: dict[str, ModelManifest] = {}
        self._errors: list[dict[str, object]] = []
        self.refresh()

    def refresh(self) -> None:
        self._manifests.clear()
        self._errors.clear()
        if not self.root.exists():
            return
        for metadata_path in sorted(self.root.rglob("metadata.json")):
            try:
                manifest = ModelManifest.from_file(metadata_path)
            except ModelManifestError as exc:
                self._errors.append({"path": str(metadata_path), "error": exc.as_dict()})
                continue
            if manifest.id in self._manifests:
                self._errors.append(
                    {"path": str(metadata_path), "error": {"code": "DUPLICATE_MODEL_ID"}}
                )
                continue
            self._manifests[manifest.id] = manifest

    def list_models(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for manifest in self._manifests.values():
            status = "not_installed"
            reason: dict[str, object] | None = None
            try:
                manifest.verify_sidecars()
                manifest.verify_model_file()
                manifest.validate()
                status = "installed"
            except ModelError as exc:
                reason = exc.as_dict()
            result.append(
                {
                    "id": manifest.id,
                    "kind": manifest.kind,
                    "adapter": manifest.adapter,
                    "version": manifest.version,
                    "status": status,
                    "supported_devices": list(manifest.supported_devices),
                    "error": reason,
                }
            )
        result.extend(
            {"status": "invalid", "error": item["error"], "path": item["path"]}
            for item in self._errors
        )
        return result

    def get(self, model_id: str) -> ModelManifest:
        try:
            return self._manifests[model_id]
        except KeyError as exc:
            raise ModelNotInstalledError(f"找不到模型 {model_id}") from exc

    def health(self, model_id: str) -> dict[str, object]:
        manifest = self.get(model_id)
        return create_adapter(manifest).health()
