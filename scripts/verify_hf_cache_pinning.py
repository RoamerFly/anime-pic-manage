"""End-to-end proof that the Worker keeps and adopts models in its own folder.

Runs the real Worker over the JSON Lines protocol (the same way the desktop
does) and checks three things:

1. ``model.cache.status`` reports the cache the desktop pinned, not the
   machine-wide one.
2. ``model.cache.adopt`` copies repositories out of an existing cache.
3. The copies are then reported as installed, without touching the network.

Usage:
    python scripts/verify_hf_cache_pinning.py [legacy_cache_dir]
    python scripts/verify_hf_cache_pinning.py --worker dist_windows_gpu/app/runtime/ai-worker.exe \
        E:/Cache/huggingface_cache/hub
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((REPO_ROOT / "resources" / "model-catalog.json").read_text("utf-8"))


def cache_entries() -> list[dict]:
    return [entry for entry in CATALOG if entry.get("delivery") == "hf_cache"]


def repo_ids() -> list[str]:
    return sorted(
        {
            file.get("repo_id") or entry["repo_id"]
            for entry in cache_entries()
            for file in entry["files"]
        }
    )


class Worker:
    """Minimal JSON Lines client for ``run_worker.py``."""

    def __init__(
        self,
        cache_dir: Path | None,
        executable: Path | None = None,
    ) -> None:
        env = dict(os.environ)
        if cache_dir is None:
            # No pinning at all: a packaged Worker has to derive the folder from
            # its own location.
            env.pop("ANIME_PIC_HF_CACHE", None)
            env.pop("ANIME_PIC_MODELS", None)
        else:
            env["ANIME_PIC_HF_CACHE"] = str(cache_dir)
            env["ANIME_PIC_MODELS"] = str(REPO_ROOT / "models")
        env["PYTHONUNBUFFERED"] = "1"
        if executable is None:
            command = [
                sys.executable,
                str(REPO_ROOT / "apps" / "ai-worker" / "run_worker.py"),
            ]
            cwd = REPO_ROOT / "apps" / "ai-worker"
        else:
            # A packaged Worker resolves its own models; the environment still
            # decides the cache, which is the behaviour under test.
            command = [str(executable)]
            cwd = executable.parent
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
        )

    def call(self, message_type: str, payload: dict | None = None) -> dict:
        request = {
            "schema_version": "1.0",
            "request_id": f"verify-{message_type}",
            "task_id": "verify",
            "message_type": message_type,
            "payload": payload or {},
        }
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker exited early: {self.process.stderr.read()[:2000]}")
        return json.loads(line)

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_cache", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--worker",
        type=Path,
        default=None,
        help="packaged ai-worker executable; defaults to the source checkout",
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help=(
            "start the Worker without ANIME_PIC_HF_CACHE so a packaged build has "
            "to derive <package>\\data\\hf-cache from its own location"
        ),
    )
    args = parser.parse_args()
    legacy = args.legacy_cache
    app_cache = Path(tempfile.mkdtemp(prefix="anime-hf-cache-")) / "data" / "hf-cache"
    expected = app_cache / "hub"
    if args.no_pin:
        if args.worker is None:
            parser.error("--no-pin only makes sense with --worker")
        package_root = args.worker.resolve().parents[2]
        expected = package_root / "data" / "hf-cache" / "hub"
    worker = Worker(None if args.no_pin else app_cache, args.worker)
    try:
        report = worker.call("model.cache.status", {"entries": cache_entries()})
        assert not report.get("error"), report
        payload = report["payload"]
        print(f"cache_dir          = {payload['cache_dir']}")
        assert Path(payload["cache_dir"]) == expected, payload["cache_dir"]
        before = {entry["id"]: entry["status"] for entry in payload["entries"]}
        print(f"before adopt       = {before}")

        if legacy is None or not legacy.is_dir():
            print("no legacy cache given; skipping the adoption step")
            return 0

        adopt = worker.call(
            "model.cache.adopt",
            {"source_dir": str(legacy), "repo_ids": repo_ids()},
        )
        assert not adopt.get("error"), adopt
        adopted = adopt["payload"]
        print(f"adopted            = {[item['repo_id'] for item in adopted['adopted']]}")
        print(f"copied bytes       = {adopted['total_bytes']}")
        assert adopted["total_bytes"] > 0

        after = worker.call("model.cache.status", {"entries": cache_entries()})
        statuses = {entry["id"]: entry["status"] for entry in after["payload"]["entries"]}
        print(f"after adopt        = {statuses}")
        assert all(status == "installed" for status in statuses.values()), statuses
        print("OK: cache pinned inside the app folder and reused from the old cache")
        return 0
    finally:
        worker.close()
        shutil.rmtree(app_cache.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
