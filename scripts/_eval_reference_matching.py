"""Compare CCIP against base-embedding nearest neighbour for reference matching.

Positives: the target character's images (leave-one-out).
Negatives: images of other characters.
Both are evaluated on full images and on detector-derived bust crops, because
the production pipeline recognises crops rather than whole canvases.

Usage::

    uv run --project apps/ai-worker --extra model --extra detector \
        python scripts/eval_reference_matching.py \
        --positive-dir "D:\\datasets\\character_alpha" \
        --negative-dir "D:\\datasets\\character_beta" --negative-dir "D:\\datasets\\character_gamma" \
        --models models
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"D:\Agent\Project\anime-pic-manage")
sys.path.insert(0, str(REPO / "apps" / "ai-worker" / "src"))

from ai_worker.images import make_multiscale_crops  # noqa: E402
from ai_worker.images import CropType  # noqa: E402
from ai_worker.worker import WorkerService  # noqa: E402

MAX_NEGATIVES_PER_DIR = 12


def load_images(folder: Path, limit: int | None = None) -> list[Path]:
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if limit is not None and len(files) > limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    return files


def detector(service: WorkerService):
    manifest = service._select_manifest("head_detector", None)  # noqa: SLF001
    adapter = service._detector_adapter(manifest)  # noqa: SLF001
    adapter.activate()
    return adapter


def bust_crop(adapter, path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    detections = adapter.predict(image)
    if not detections:
        return image
    best = max(detections, key=lambda box: (box.x2 - box.x1) * (box.y2 - box.y1))
    crops = make_multiscale_crops(image, best)
    return crops.get(CropType.BUST, image)


def open_embedder(service: WorkerService):
    manifest = service._select_manifest("character_recognizer", None)  # noqa: SLF001
    adapter = service._recognizer_adapter(manifest)  # noqa: SLF001
    adapter.activate()
    return adapter


def embedding_features(adapter, images: list[Image.Image]) -> np.ndarray:
    vectors = []
    for image in images:
        _, embedding = adapter.predict_with_embedding(image, top_k=1)
        vectors.append(embedding)
    matrix = np.asarray(vectors, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def rate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """AUC plus the best achievable accuracy by threshold (higher = same)."""

    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    auc = (wins + 0.5 * ties) / (len(positives) * len(negatives))
    candidates = np.unique(np.concatenate([positives, negatives]))
    best = (0.0, 0.0)
    for threshold in candidates:
        accuracy = (
            (positives >= threshold).sum() + (negatives < threshold).sum()
        ) / len(scores)
        if accuracy > best[0]:
            best = (float(accuracy), float(threshold))
    return {
        "auc": float(auc),
        "accuracy": best[0],
        "threshold": best[1],
        "pos_mean": float(positives.mean()),
        "neg_mean": float(negatives.mean()),
    }


def evaluate(
    name: str,
    similarity: np.ndarray,
    n_positive: int,
) -> dict[str, float]:
    """Best similarity between each query and the reference set."""

    scores = np.empty(len(similarity))
    for index in range(len(similarity)):
        row = similarity[index].copy()
        row[index] = -np.inf  # leave-one-out for reference queries
        scores[index] = row[:n_positive].max()
    labels = np.concatenate([np.ones(n_positive), np.zeros(len(similarity) - n_positive)])
    result = rate(scores, labels)
    result["name"] = name  # type: ignore[assignment]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--negative-dir", action="append", default=[])
    parser.add_argument("--models", default=str(REPO / "models"))
    parser.add_argument("--negatives-per-dir", type=int, default=MAX_NEGATIVES_PER_DIR)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX intra-op threads per session; keeping this low keeps the desktop usable",
    )
    args = parser.parse_args()

    service = WorkerService(models_root=args.models)
    detect = detector(service)
    embedder = open_embedder(service)
    for adapter in (detect, embedder):
        setter = getattr(adapter, "set_onnx_threads", None)
        if callable(setter):
            setter(args.threads)
    print(f"ONNX 线程数上限: {args.threads}")
    from imgutils.metrics import ccip_batch_extract_features

    positives = load_images(Path(args.positive_dir))
    negatives: list[Path] = []
    for name in args.negative_dir:
        negatives.extend(load_images(Path(name), args.negatives_per_dir))
    print(f"正样本 {len(positives)} 张，对照 {len(negatives)} 张")

    for crop_mode in ("full", "bust"):
        started = time.time()
        prepared: list[Image.Image] = []
        for path in positives + negatives:
            if crop_mode == "full":
                prepared.append(Image.open(path).convert("RGB"))
            else:
                prepared.append(bust_crop(detect, path))
        print(f"[{crop_mode}] 裁剪完成 {len(prepared)} 张（{time.time() - started:.0f}s）", flush=True)

        ccip = ccip_batch_extract_features(prepared)
        from imgutils.metrics import ccip_batch_differences

        result = evaluate("ccip", -ccip_batch_differences(list(ccip)), len(positives))
        print(
            f"  CCIP       AUC={result['auc']:.3f} acc={result['accuracy']:.3f} "
            f"阈值=-{result['threshold']:.4f} 正={result['pos_mean']:.4f} 负={result['neg_mean']:.4f}",
            flush=True,
        )

        base = embedding_features(embedder, prepared)
        result = evaluate("embedding", base @ base.T, len(positives))
        print(
            f"  Embedding  AUC={result['auc']:.3f} acc={result['accuracy']:.3f} "
            f"阈值={result['threshold']:.4f} 正={result['pos_mean']:.4f} 负={result['neg_mean']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
