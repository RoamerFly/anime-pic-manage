"""Model installation and removal for recognizers and detectors.

The desktop ships a small catalog of known-good models; this handler downloads
the declared files from Hugging Face into the local model directory and writes
the manifest the rest of the pipeline already understands. Nothing here needs
network access to *run* a model, only to install one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..errors import WorkerError

MAX_MODEL_FILES = 24
KIND_DIRECTORIES: dict[str, str] = {
    "character_recognizer": "recognizer",
    "head_detector": "detector",
}


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("INVALID_PAYLOAD", f"model.install 需要非空字段 {key}")
    return value.strip()


def _safe_name(value: str) -> str:
    name = Path(value).name
    if not name or name in {".", ".."} or name != value:
        raise WorkerError("INVALID_PAYLOAD", f"文件名不合法: {value}")
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def handle_model_install(
    service: Any,
    payload: Mapping[str, object],
    *,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    model_id = _required_string(payload, "id")
    kind = _required_string(payload, "kind")
    directory_name = KIND_DIRECTORIES.get(kind)
    if directory_name is None:
        raise WorkerError("INVALID_PAYLOAD", f"不支持的模型类型: {kind}")
    repo_id = _required_string(payload, "repo_id")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise WorkerError("INVALID_PAYLOAD", "model.install 需要 files 数组")
    if len(raw_files) > MAX_MODEL_FILES:
        raise WorkerError("INVALID_PAYLOAD", f"files 不能超过 {MAX_MODEL_FILES} 项")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise WorkerError("INVALID_PAYLOAD", "model.install 需要 metadata 对象")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise WorkerError(
            "DEPENDENCY_MISSING",
            "缺少 huggingface_hub，无法下载模型",
            str(exc),
        ) from exc

    target = Path(service.models.root) / directory_name / model_id
    target.mkdir(parents=True, exist_ok=True)
    installed: list[dict[str, Any]] = []
    total = len(raw_files)
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, Mapping):
            raise WorkerError("INVALID_PAYLOAD", "files 中的每一项必须是对象")
        remote = _required_string(entry, "path")
        name = _safe_name(_required_string(entry, "name"))
        if on_progress is not None:
            on_progress("downloading", index + 1, total, name)
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=remote)
        except Exception as exc:  # noqa: BLE001 - surface every download failure
            raise WorkerError(
                "MODEL_DOWNLOAD_FAILED",
                f"下载 {remote} 失败: {exc}",
                f"repo={repo_id}",
            ) from exc
        destination = target / name
        try:
            shutil.copyfile(cached, destination)
        except OSError as exc:
            raise WorkerError("MODEL_INSTALL_FAILED", f"写入 {name} 失败: {exc}") from exc
        installed.append({"name": name, "size": destination.stat().st_size})

    # The manifest records the hash of whatever was just downloaded so later
    # runs can detect a corrupted or replaced weight file.
    manifest = dict(metadata)
    model_name = _safe_name(str(manifest.get("model_file", "")))
    model_path = target / model_name
    if not model_path.is_file():
        raise WorkerError(
            "MODEL_INSTALL_FAILED",
            f"清单声明的模型文件 {model_name} 不在下载结果里",
        )
    manifest["sha256"] = _sha256(model_path)
    manifest.setdefault("id", model_id)
    manifest.setdefault("kind", kind)
    manifest.setdefault("format", "onnx")

    # Some upstream repositories (Camie) do not ship the category/threshold
    # sidecars this pipeline validates, so the catalog can declare them.
    generated = payload.get("generate")
    if generated is not None:
        if not isinstance(generated, Mapping):
            raise WorkerError("INVALID_PAYLOAD", "generate 必须是对象")
        for raw_name, content in generated.items():
            name = _safe_name(str(raw_name))
            text = (
                json.dumps(content, ensure_ascii=False, indent=2) + "\n"
                if isinstance(content, (list, Mapping))
                else str(content)
            )
            (target / name).write_text(text, encoding="utf-8")
            installed.append({"name": name, "size": (target / name).stat().st_size})

    (target / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    service.models.refresh()
    if on_progress is not None:
        on_progress("completed", total, total, "模型已安装")
    return {
        "id": model_id,
        "kind": kind,
        "directory": str(target),
        "files": installed,
        "total_bytes": sum(item["size"] for item in installed),
        "status": "installed",
    }


def handle_model_delete(service: Any, payload: Mapping[str, object]) -> dict[str, Any]:
    model_id = _required_string(payload, "id")
    manifest = service.models.get(model_id)
    directory_name = KIND_DIRECTORIES.get(manifest.kind)
    if directory_name is None:
        raise WorkerError("INVALID_PAYLOAD", f"不支持的模型类型: {manifest.kind}")
    directory = Path(service.models.root) / directory_name / model_id
    if not directory.is_dir():
        raise WorkerError("MODEL_NOT_FOUND", f"模型目录不存在: {directory}")
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        raise WorkerError("MODEL_DELETE_FAILED", f"删除模型失败: {exc}") from exc
    service.models.refresh()
    return {"id": model_id, "status": "deleted", "directory": str(directory)}
