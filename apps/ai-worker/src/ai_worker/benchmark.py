"""Benchmark command skeleton for reproducible V0.1 posture splits."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .export import export_json
from .images import enumerate_images

POSTURE_SPLITS = ("front", "side", "back", "multi")


def collect_benchmark_inventory(root: str | Path) -> dict[str, Any]:
    """Count posture split images without running a model or moving files."""

    benchmark_root = Path(root).expanduser()
    started = time.perf_counter()
    splits: dict[str, dict[str, Any]] = {}
    total = 0
    for split in POSTURE_SPLITS:
        split_path = benchmark_root / split
        if split_path.is_dir():
            paths = enumerate_images(split_path, recursive=True)
            count = len(paths)
            total += count
            splits[split] = {"directory": str(split_path), "image_count": count}
        else:
            splits[split] = {"directory": str(split_path), "image_count": 0, "missing": True}
    return {
        "benchmark_version": "0.1",
        "root": str(benchmark_root),
        "splits": splits,
        "total_images": total,
        "model": None,
        "metrics": {
            "detection_recall": None,
            "top1_accuracy": None,
            "top5_accuracy": None,
            "auto_classification_accuracy": None,
            "needs_review_rate": None,
            "unrecognized_rate": None,
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "note": "仅完成目录清点；接入已安装模型和标注后再运行推理指标。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anime image V0.1 benchmark inventory")
    parser.add_argument("root", type=Path, help="包含 front/side/back/multi 子目录的基准目录")
    parser.add_argument("--output", type=Path, help="可选 JSON 报告路径")
    args = parser.parse_args(argv)
    report = collect_benchmark_inventory(args.root)
    if args.output:
        export_json([report], args.output)
    else:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
