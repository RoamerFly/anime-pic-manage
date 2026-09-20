"""Model installation and removal for recognizers, detectors and cache models.

The desktop ships a small catalog of known-good models. Two delivery paths
exist and both are handled here:

* ``model.install`` / ``model.delete`` — recognizers and detectors, downloaded
  from Hugging Face into ``models\\<kind>\\<id>\\`` together with the manifest the
  rest of the pipeline already understands.
* ``model.cache.status`` / ``model.cache.prefetch`` / ``model.cache.delete`` —
  optional models that the pipeline pulls from the application's Hugging Face
  cache on first use (the WD14 tagger, the CCIP reference model). They stay in
  the cache so ``imgutils`` keeps finding them by repo id, but the settings page
  still has to be able to report, pre-download and clear them.
* ``model.cache.adopt`` — copy repositories a machine-wide cache already holds
  into the application cache. Pinning the cache inside the package must not
  cost a fresh download of several hundred megabytes.

Nothing here needs network access to *run* a model, only to install one.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..errors import WorkerError

MAX_MODEL_FILES = 24
# The whole catalog is a handful of entries; the cap only stops a broken caller
# from asking for thousands of cache lookups in one request.
MAX_CACHE_ENTRIES = 32
KIND_DIRECTORIES: dict[str, str] = {
    "character_recognizer": "recognizer",
    "head_detector": "detector",
}


def _required_string(
    payload: Mapping[str, object], key: str, *, context: str = "model.install"
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("INVALID_PAYLOAD", f"{context} 需要非空字段 {key}")
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


# --------------------------------------------------------------------------
# Hugging Face cache models (WD14 tagger, CCIP reference backend, ...)
# --------------------------------------------------------------------------


def _huggingface_hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise WorkerError(
            "DEPENDENCY_MISSING",
            "缺少 huggingface_hub，无法管理模型下载",
            str(exc),
        ) from exc
    return huggingface_hub


def _hub_cache_dir() -> Path:
    """Resolve the shared cache the pipeline itself downloads into.

    Reading it from ``huggingface_hub`` (instead of assuming ``HF_HOME``) keeps
    the settings page honest even when the user relocated the cache.
    """

    constants = importlib.import_module("huggingface_hub.constants")
    raw = getattr(constants, "HF_HUB_CACHE", None)
    if not isinstance(raw, str) or not raw.strip():
        raise WorkerError("CACHE_UNAVAILABLE", "无法确定 Hugging Face 缓存目录")
    return Path(raw).expanduser()


def _directory_size(path: Path) -> int:
    total = 0
    try:
        entries = list(path.rglob("*"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _cache_entry_list(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Accept either ``{"entries": [...]}`` or a bare catalog entry object."""

    raw = payload.get("entries")
    if raw is None:
        return [payload]
    if not isinstance(raw, list) or not raw:
        raise WorkerError("INVALID_PAYLOAD", "model.cache 需要非空 entries 数组")
    if len(raw) > MAX_CACHE_ENTRIES:
        raise WorkerError("INVALID_PAYLOAD", f"entries 不能超过 {MAX_CACHE_ENTRIES} 项")
    for item in raw:
        if not isinstance(item, Mapping):
            raise WorkerError("INVALID_PAYLOAD", "entries 中的每一项必须是对象")
    return list(raw)


def _cache_entry_files(entry: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """Return ``(repo_id, remote path, local file name)`` for declared files."""

    default_repo = entry.get("repo_id")
    default_repo = default_repo.strip() if isinstance(default_repo, str) else ""
    raw_files = entry.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise WorkerError("INVALID_PAYLOAD", "model.cache 的每一项都需要 files 数组")
    if len(raw_files) > MAX_MODEL_FILES:
        raise WorkerError("INVALID_PAYLOAD", f"files 不能超过 {MAX_MODEL_FILES} 项")

    resolved: list[tuple[str, str, str]] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise WorkerError("INVALID_PAYLOAD", "files 中的每一项必须是对象")
        repo_id = item.get("repo_id")
        repo_id = repo_id.strip() if isinstance(repo_id, str) and repo_id.strip() else default_repo
        if not repo_id:
            raise WorkerError("INVALID_PAYLOAD", "files 缺少 repo_id")
        remote = _required_string(item, "path", context="model.cache")
        name = item.get("name")
        name = name.strip() if isinstance(name, str) and name.strip() else Path(remote).name
        resolved.append((repo_id, remote, name))
    return resolved


def _locate_cached_file(
    cache_dir: Path, repo_id: str, remote: str
) -> tuple[bool, int, str | None]:
    """Look a file up in the cache without touching the network."""

    hub = _huggingface_hub()
    try:
        located = hub.try_to_load_from_cache(
            repo_id=repo_id, filename=remote, cache_dir=str(cache_dir)
        )
    except Exception:  # noqa: BLE001 - a corrupt cache entry is simply "missing"
        return False, 0, None
    if not isinstance(located, str):
        return False, 0, None
    try:
        size = Path(located).stat().st_size
    except OSError:
        size = 0
    return True, size, located


def handle_model_cache_status(
    service: Any, payload: Mapping[str, object]
) -> dict[str, Any]:
    """Report which cache-backed catalog models are already on disk."""

    cache_dir = _hub_cache_dir()
    entries = _cache_entry_list(payload)
    report: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = _required_string(entry, "id", context="model.cache")
        files = _cache_entry_files(entry)
        described: list[dict[str, Any]] = []
        for repo_id, remote, name in files:
            present, size, located = _locate_cached_file(cache_dir, repo_id, remote)
            described.append(
                {
                    "name": name,
                    "repo_id": repo_id,
                    "remote": remote,
                    "present": present,
                    "size": size,
                    "path": located,
                }
            )
        found = sum(1 for item in described if item["present"])
        status = "installed" if found == len(described) else ("partial" if found else "missing")
        report.append(
            {
                "id": entry_id,
                "repo_ids": sorted({repo_id for repo_id, _, _ in files}),
                "status": status,
                "total_bytes": sum(item["size"] for item in described if item["present"]),
                "files": described,
            }
        )

    return {
        "models_dir": str(Path(service.models.root)),
        "cache_dir": str(cache_dir),
        "entries": report,
    }


def handle_model_cache_prefetch(
    service: Any,
    payload: Mapping[str, object],
    *,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Download cache-backed models ahead of the first use that needs them."""

    hub = _huggingface_hub()
    cache_dir = _hub_cache_dir()
    entries = _cache_entry_list(payload)
    results: list[dict[str, Any]] = []

    for entry in entries:
        entry_id = _required_string(entry, "id", context="model.cache")
        files = _cache_entry_files(entry)
        installed: list[dict[str, Any]] = []
        total = len(files)
        for index, (repo_id, remote, name) in enumerate(files):
            if on_progress is not None:
                on_progress("downloading", index + 1, total, name)
            try:
                located = hub.hf_hub_download(
                    repo_id=repo_id, filename=remote, cache_dir=str(cache_dir)
                )
            except Exception as exc:  # noqa: BLE001 - surface every download failure
                raise WorkerError(
                    "MODEL_DOWNLOAD_FAILED",
                    f"下载 {remote} 失败: {exc}",
                    f"repo={repo_id}",
                ) from exc
            try:
                size = Path(located).stat().st_size
            except OSError:
                size = 0
            installed.append({"name": name, "size": size, "path": str(located)})
        results.append(
            {
                "id": entry_id,
                "status": "installed",
                "repo_ids": sorted({repo_id for repo_id, _, _ in files}),
                "total_bytes": sum(item["size"] for item in installed),
                "files": installed,
            }
        )
        if on_progress is not None:
            on_progress("completed", total, total, f"{entry_id} 已下载")

    return {
        "cache_dir": str(cache_dir),
        "entries": results,
        "total_bytes": sum(item["total_bytes"] for item in results),
    }


def handle_model_cache_delete(
    service: Any, payload: Mapping[str, object]
) -> dict[str, Any]:
    """Drop the cached repositories of one catalog entry."""

    hub = _huggingface_hub()
    cache_dir = _hub_cache_dir()
    entry_id = _required_string(payload, "id", context="model.cache.delete")
    raw = payload.get("repo_ids")
    if not isinstance(raw, list):
        raise WorkerError("INVALID_PAYLOAD", "model.cache.delete 需要 repo_ids 数组")
    wanted = sorted({item.strip() for item in raw if isinstance(item, str) and item.strip()})
    if not wanted:
        raise WorkerError("INVALID_PAYLOAD", "model.cache.delete 需要非空 repo_ids 数组")

    folders = {
        repo_id: cache_dir / f"models--{repo_id.replace('/', '--')}" for repo_id in wanted
    }
    freed = sum(_directory_size(folder) for folder in folders.values() if folder.is_dir())

    try:
        cache_info = hub.scan_cache_dir(cache_dir=str(cache_dir))
    except Exception as exc:  # noqa: BLE001 - fall back to removing the folders
        cache_info = None
        scan_error: str | None = str(exc)
    else:
        scan_error = None

    if cache_info is not None:
        revisions = [
            revision.commit_hash
            for repo in cache_info.repos
            if repo.repo_id in wanted
            for revision in repo.revisions
        ]
        if revisions:
            try:
                cache_info.delete_revisions(*revisions).execute()
            except Exception as exc:  # noqa: BLE001 - reported, then retried below
                scan_error = str(exc)

    # Whatever the strategy left behind (detached refs, interrupted deletes or a
    # cache layout it does not understand) is removed wholesale so 删除 leaves
    # the disk in the state the user just asked for.
    removed: list[str] = []
    for repo_id, folder in folders.items():
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                raise WorkerError(
                    "MODEL_DELETE_FAILED", f"删除缓存 {repo_id} 失败: {exc}"
                ) from exc
        removed.append(repo_id)

    return {
        "id": entry_id,
        "status": "deleted",
        "cache_dir": str(cache_dir),
        "repo_ids": removed,
        "freed_bytes": freed,
        "warning": scan_error,
    }


def _cache_repository_root(path: Path) -> Path:
    """Accept either a ``HF_HOME`` folder or the ``hub`` folder inside it."""

    hub = path / "hub"
    return hub if hub.is_dir() else path


def handle_model_cache_adopt(
    service: Any,
    payload: Mapping[str, object],
    *,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Copy repositories another cache already holds into the application cache.

    The application keeps every model inside its own folder, which would
    otherwise mean re-downloading hundreds of megabytes that this machine
    already has. Adopting copies the repository folders over and leaves the
    source untouched: other tools may still be sharing it.
    """

    cache_dir = _hub_cache_dir()
    source = Path(
        _required_string(payload, "source_dir", context="model.cache.adopt")
    ).expanduser()
    raw = payload.get("repo_ids")
    if not isinstance(raw, list):
        raise WorkerError("INVALID_PAYLOAD", "model.cache.adopt 需要 repo_ids 数组")
    wanted = sorted(
        {item.strip() for item in raw if isinstance(item, str) and item.strip()}
    )
    if not wanted:
        raise WorkerError("INVALID_PAYLOAD", "model.cache.adopt 需要非空 repo_ids 数组")
    if len(wanted) > MAX_CACHE_ENTRIES:
        raise WorkerError(
            "INVALID_PAYLOAD", f"repo_ids 不能超过 {MAX_CACHE_ENTRIES} 项"
        )

    source_root = _cache_repository_root(source)
    target_root = _cache_repository_root(cache_dir)
    if not source_root.is_dir():
        raise WorkerError(
            "MODEL_ADOPT_SOURCE_MISSING", f"源缓存目录不存在: {source_root}"
        )
    if source_root == target_root:
        raise WorkerError(
            "INVALID_PAYLOAD", "源缓存与软件自己的缓存是同一个目录，无需复制"
        )

    adopted: list[dict[str, Any]] = []
    missing: list[str] = []
    total = len(wanted)
    for index, repo_id in enumerate(wanted):
        folder = f"models--{repo_id.replace('/', '--')}"
        origin = source_root / folder
        if not origin.is_dir():
            missing.append(repo_id)
            continue
        if on_progress is not None:
            on_progress("copying", index + 1, total, repo_id)
        destination = target_root / folder
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # ``dirs_exist_ok`` merges with a partially downloaded repository
            # instead of failing, and copying follows the snapshot links so the
            # result is a self-contained folder.
            shutil.copytree(origin, destination, dirs_exist_ok=True)
        except OSError as exc:
            raise WorkerError(
                "MODEL_ADOPT_FAILED", f"复制缓存 {repo_id} 失败: {exc}"
            ) from exc
        adopted.append({"repo_id": repo_id, "bytes": _directory_size(destination)})

    if not adopted:
        raise WorkerError(
            "MODEL_ADOPT_SOURCE_MISSING",
            f"源缓存里没有这些模型: {', '.join(missing)}",
        )

    return {
        "cache_dir": str(cache_dir),
        "source_dir": str(source_root),
        "adopted": adopted,
        "missing": missing,
        "total_bytes": sum(item["bytes"] for item in adopted),
    }
