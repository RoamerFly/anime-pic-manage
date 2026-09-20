"""Local, offline image similarity feature extraction, perceptual hashing, and clustering.

Computes multi-scale perceptual hashes (64-bit pHash, 64-bit dHash), normalized
color histograms, and Laplacian variance clarity evaluation. All algorithms are
purely local, offline, and require no external deep learning weight downloads.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .errors import WorkerError
from .images import sample_image_frames

# Supported format priority for keeping best version (higher is better)
FORMAT_PRIORITY: dict[str, int] = {
    "png": 10,
    "webp": 8,
    "tiff": 7,
    "tif": 7,
    "bmp": 6,
    "jpeg": 5,
    "jpg": 5,
    "gif": 3,
}

# Relationship labels ordered by their vectorized relation codes.
_RELATION_CODES: tuple[str, ...] = ("exact_duplicate", "variation", "similar", "different")
_POPCOUNT_TABLE = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)
Array = NDArray[Any]

# Animated files are compared through their first, middle and last frame.
MAX_SAMPLED_FRAMES = 3


def similarity_worker_count(max_workers: int | None = None) -> int:
    """Bound the thread pool used for parallel image feature extraction."""

    if max_workers is not None:
        return max(1, min(int(max_workers), 32))
    override = os.environ.get("ANIME_PIC_SIMILARITY_WORKERS")
    if override:
        try:
            return max(1, min(int(override), 32))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 2))


def _dct_2d(pixels: Array) -> Array:
    """Compute 2D Discrete Cosine Transform (DCT-II, ortho) using pure NumPy."""
    n = pixels.shape[0]
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    matrix = np.cos((np.pi * (2 * i + 1) * k) / (2.0 * n))
    matrix[0, :] *= np.sqrt(1.0 / n)
    matrix[1:, :] *= np.sqrt(2.0 / n)
    transformed: Array = matrix @ pixels @ matrix.T
    return transformed


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Compute a 64-bit perceptual hash (pHash) using 2D DCT.

    Downsamples the image to (32x32), computes 2D DCT, extracts the low-frequency
    (8x8) quadrant, and threshold-encodes against the median coefficient.
    """
    img_size = hash_size * highfreq_factor
    # Convert to grayscale and resize smoothly
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float64)

    # 2D DCT via pure NumPy (ortho normalized, perfectly identical to scipy.fft.dct)
    dct_matrix = _dct_2d(pixels)

    # Extract top-left low frequencies (hash_size x hash_size)
    low_freq = dct_matrix[:hash_size, :hash_size]

    # Use median of the low frequencies, excluding the DC component at (0, 0)
    flat = low_freq.flatten()
    median_val = float(np.median(flat[1:]))

    # Binary bits: 1 if > median, else 0
    bits = low_freq > median_val
    # Pack 64 boolean bits into 8 bytes, then convert to 16 hex characters
    packed_bytes = np.packbits(bits.flatten())
    return bytes(packed_bytes).hex()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (dHash).

    Downsamples the image to (9x8) grayscale and compares adjacent columns.
    """
    # 9 columns by 8 rows to compare 8 adjacent differences per row
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int32)

    # Compare pixel[x] > pixel[x + 1]
    diff = pixels[:, :-1] > pixels[:, 1:]
    packed_bytes = np.packbits(diff.flatten())
    return bytes(packed_bytes).hex()


def compute_color_histogram(image: Image.Image, bins_per_channel: int = 16) -> list[float]:
    """Compute a 48-dimensional normalized RGB color histogram.

    Divides R, G, and B color channels into 16 bins each and L1-normalizes them.
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    hists: list[float] = []

    for c in range(3):
        channel_data = arr[:, :, c].flatten()
        hist, _ = np.histogram(channel_data, bins=bins_per_channel, range=(0, 256))
        total = float(np.sum(hist))
        if total > 0:
            norm_hist = hist / total
        else:
            norm_hist = np.zeros(bins_per_channel, dtype=np.float64)
        hists.extend(round(float(v), 6) for v in norm_hist)

    return hists


def compute_clarity_score(image: Image.Image, max_dim: int = 512) -> float:
    """Compute sharpness / clarity score using Variance of Laplacian.

    Larger variance means sharper edges and higher visual clarity.
    Downscales image to max_dim if larger to maintain constant fast speed.
    """
    gray = image.convert("L")
    w, h = gray.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        gray = gray.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)

    pixels = np.asarray(gray, dtype=np.float64)

    # 3x3 discrete Laplace kernel
    top = pixels[:-2, 1:-1]
    bottom = pixels[2:, 1:-1]
    left = pixels[1:-1, :-2]
    right = pixels[1:-1, 2:]
    center = pixels[1:-1, 1:-1]

    if top.size == 0:
        return 0.0

    laplacian = top + bottom + left + right - 4.0 * center
    variance = float(np.var(laplacian))
    return round(variance, 2)


def hamming_distance(hex1: str, hex2: str) -> int:
    """Compute Hamming distance between two 16-hex-character (64-bit) hashes."""
    try:
        val1 = int(hex1, 16)
        val2 = int(hex2, 16)
        return bin(val1 ^ val2).count("1")
    except (ValueError, TypeError):
        return 64


def color_similarity(hist1: Sequence[float], hist2: Sequence[float]) -> float:
    """Compute intersection similarity of two normalized 48-dim color histograms (0.0 ~ 1.0)."""
    if len(hist1) != len(hist2) or not hist1:
        return 0.0
    # Three channels, each normalized to sum to 1.0; total sum is 3.0
    intersection = sum(min(v1, v2) for v1, v2 in zip(hist1, hist2, strict=False))
    return round(float(intersection / 3.0), 4)


def extract_image_features(path: str | Path) -> dict[str, Any]:
    """Extract the similarity feature set from a still image or animation.

    Animated GIF/WebP files contribute up to three frames (first, middle, last)
    so a still image can still be matched against any part of the animation.
    The first frame is also copied to the top level, keeping older cached
    features and single-frame consumers working unchanged.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise WorkerError("FILE_NOT_FOUND", f"图片文件不存在: {file_path}")

    file_size = file_path.stat().st_size
    format_name = file_path.suffix.lstrip(".").lower() or "unknown"

    sampled = sample_image_frames(file_path, max_frames=MAX_SAMPLED_FRAMES)
    w, h = sampled.frames[0].size
    frames: list[dict[str, Any]] = []
    for frame_index, image in zip(sampled.indices, sampled.frames, strict=True):
        frames.append(
            {
                "frame_index": frame_index,
                "phash": compute_phash(image),
                "dhash": compute_dhash(image),
                "color_histogram": compute_color_histogram(image),
                "clarity_score": compute_clarity_score(image),
            }
        )
    primary = frames[0]

    return {
        "path": str(file_path),
        "file_size": file_size,
        "dimensions": [w, h],
        "format": format_name,
        "clarity_score": primary["clarity_score"],
        "phash": primary["phash"],
        "dhash": primary["dhash"],
        "color_histogram": primary["color_histogram"],
        "frame_count": sampled.frame_count,
        "is_animated": sampled.is_animated,
        "sampled_frames": list(sampled.indices),
        "frames": frames,
    }


def extract_image_features_batch(
    paths: Sequence[str | Path],
    *,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Extract features for many images, preserving the input order.

    Image decode, hashing and disk IO all run inside a bounded thread pool so
    that large folders no longer pay a strictly serial decode cost.  The
    returned feature list keeps the original order, which keeps clustering
    deterministic across runs.
    """

    resolved = [str(path) for path in paths]
    if not resolved:
        return [], []

    workers = similarity_worker_count(max_workers)
    if workers <= 1 or len(resolved) == 1:
        features: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for path in resolved:
            try:
                features.append(extract_image_features(path))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
                errors.append({"path": path, "error": str(exc)})
        return features, errors

    results: list[dict[str, Any] | None] = [None] * len(resolved)
    failures: list[tuple[int, dict[str, str]]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="similarity") as executor:
        futures = {
            executor.submit(extract_image_features, path): index
            for index, path in enumerate(resolved)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
                failures.append((index, {"path": resolved[index], "error": str(exc)}))

    failures.sort(key=lambda item: item[0])
    return [item for item in results if item is not None], [item for _, item in failures]


def _frame_features(feature: dict[str, Any]) -> list[dict[str, Any]]:
    """Frames of a feature dict, falling back to the single-frame shape."""

    frames = feature.get("frames")
    if isinstance(frames, list) and frames:
        return [frame for frame in frames if isinstance(frame, dict)]
    return [feature]


def _safe_int(value: object, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return int(value)


def _safe_frame_indices(value: object) -> list[int]:
    if not isinstance(value, list):
        return [0]
    indices = [
        int(item)
        for item in value
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    return indices or [0]


def _compare_frame_pair(
    frame1: dict[str, Any],
    frame2: dict[str, Any],
    aspect_ratio_diff: float,
) -> tuple[float, str]:
    """Composite similarity and relation for one frame pair."""

    phash_dist = hamming_distance(frame1["phash"], frame2["phash"])
    dhash_dist = hamming_distance(frame1["dhash"], frame2["dhash"])

    phash_sim = max(0.0, 1.0 - (phash_dist / 64.0))
    dhash_sim = max(0.0, 1.0 - (dhash_dist / 64.0))
    color_sim = color_similarity(frame1["color_histogram"], frame2["color_histogram"])

    composite = 0.55 * phash_sim + 0.25 * dhash_sim + 0.20 * color_sim

    # If aspect ratio is drastically different (>25% mismatch), penalize composite
    if aspect_ratio_diff > 1.25:
        composite *= 0.6

    composite = round(min(1.0, max(0.0, composite)), 4)

    if phash_dist <= 3 and dhash_dist <= 3 and color_sim >= 0.94 and aspect_ratio_diff <= 1.05:
        relation = "exact_duplicate"
    elif composite >= 0.85:
        relation = "variation"
    elif composite >= 0.70:
        relation = "similar"
    else:
        relation = "different"

    return composite, relation


def calculate_pair_similarity(f1: dict[str, Any], f2: dict[str, Any]) -> tuple[float, str]:
    """Compute composite similarity score (0.0 to 1.0) and relationship classification.

    Weights:
      0.55 * pHashSim + 0.25 * dHashSim + 0.20 * ColorSim
    With aspect ratio discrepancy penalty.

    Animated files are compared frame by frame: the best matching frame pair
    decides the score, and the most duplicate-like relation wins. Still images
    have a single frame, so their result is unchanged.
    """
    w1, h1 = f1["dimensions"]
    w2, h2 = f2["dimensions"]

    # Aspect ratio check
    aspect1 = w1 / max(1, h1)
    aspect2 = w2 / max(1, h2)
    aspect_ratio_diff = max(aspect1 / max(1e-4, aspect2), aspect2 / max(1e-4, aspect1))

    best_similarity = -1.0
    best_relation_code = len(_RELATION_CODES) - 1
    for frame1 in _frame_features(f1):
        for frame2 in _frame_features(f2):
            similarity, relation = _compare_frame_pair(
                frame1, frame2, aspect_ratio_diff
            )
            if similarity > best_similarity:
                best_similarity = similarity
            best_relation_code = min(
                best_relation_code, _RELATION_CODES.index(relation)
            )

    return round(best_similarity, 4), _RELATION_CODES[best_relation_code]


class DisjointSetUnion:
    """Disjoint Set Union (Union-Find) with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def pick_best_item(items: list[dict[str, Any]]) -> int:
    """Select index of the best image in a group based on resolution, format, and sharpness.

    Rule:
      1. Pixel count (width * height) descending;
      2. Lossless/high-fidelity format priority descending;
      3. Clarity score descending;
      4. File size descending.
    """
    def item_key(item: dict[str, Any]) -> tuple[int, int, float, int]:
        dims = item["dimensions"]
        pixels = dims[0] * dims[1]
        fmt_priority = FORMAT_PRIORITY.get(item["format"].lower(), 1)
        clarity = float(item.get("clarity_score", 0.0))
        size = int(item.get("file_size", 0))
        return (pixels, fmt_priority, clarity, size)

    best_idx = 0
    best_key = item_key(items[0])
    for i in range(1, len(items)):
        current_key = item_key(items[i])
        if current_key > best_key:
            best_key = current_key
            best_idx = i
    return best_idx


def _hashes_to_uint64(hashes: Sequence[object]) -> Array | None:
    """Pack hex hashes into a uint64 vector, or ``None`` when any hash is invalid."""

    values = np.zeros(len(hashes), dtype=np.uint64)
    for index, raw in enumerate(hashes):
        if not isinstance(raw, str):
            return None
        try:
            value = int(raw, 16)
        except ValueError:
            return None
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            return None
        values[index] = value
    return values


def _popcount(values: Array) -> Array:
    """Population count of a contiguous uint64 array through a byte lookup table."""

    byte_view = values.view(np.uint8).reshape(-1, 8)
    counts = _POPCOUNT_TABLE[byte_view].sum(axis=1, dtype=np.int32)
    return np.asarray(counts, dtype=np.int32)


def _feature_arrays(features: list[dict[str, Any]]) -> dict[str, Array] | None:
    """Build frame-level numeric arrays used by the vectorized comparison.

    Every frame of every feature becomes one row; ``frame_image`` maps each row
    back to its image so the comparison can be reduced per image pair.
    """

    try:
        dimensions = np.asarray(
            [[int(item["dimensions"][0]), int(item["dimensions"][1])] for item in features],
            dtype=np.int64,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if dimensions.shape != (len(features), 2):
        return None

    phash_values: list[object] = []
    dhash_values: list[object] = []
    histogram_rows: list[list[float]] = []
    frame_image: list[int] = []
    frame_index: list[int] = []
    for image_index, feature in enumerate(features):
        for frame in _frame_features(feature):
            try:
                histogram_rows.append([float(value) for value in frame["color_histogram"]])
            except (KeyError, IndexError, TypeError, ValueError):
                return None
            phash_values.append(frame.get("phash"))
            dhash_values.append(frame.get("dhash"))
            frame_image.append(image_index)
            frame_index.append(int(frame.get("frame_index", 0)))

    phash = _hashes_to_uint64(phash_values)
    dhash = _hashes_to_uint64(dhash_values)
    if phash is None or dhash is None:
        return None
    histograms = np.asarray(histogram_rows, dtype=np.float64)
    if histograms.ndim != 2 or histograms.shape[0] != len(frame_image) or histograms.shape[1] == 0:
        return None
    return {
        "dimensions": dimensions,
        "histograms": histograms,
        "phash": phash,
        "dhash": dhash,
        "frame_image": np.asarray(frame_image, dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
    }


def _pair_block_size(count: int, *, target_cells: int = 1_000_000) -> int:
    """Pick a row block so one comparison matrix stays around ``target_cells``."""

    return max(16, min(256, target_cells // max(1, count)))


def _pair_block_similarity(
    arrays: dict[str, Array],
    start: int,
    end: int,
) -> tuple[Array, Array, Array]:
    """Frame-level composite similarity and relation codes for rows ``[start, end)``."""

    phash = arrays["phash"]
    dhash = arrays["dhash"]
    histograms = arrays["histograms"]
    dimensions = arrays["dimensions"]
    frame_image = arrays["frame_image"]

    phash_dist = _popcount(phash[start:end, None] ^ phash[None, :]).reshape(end - start, -1)
    dhash_dist = _popcount(dhash[start:end, None] ^ dhash[None, :]).reshape(end - start, -1)
    phash_sim = np.maximum(0.0, 1.0 - (phash_dist / 64.0))
    dhash_sim = np.maximum(0.0, 1.0 - (dhash_dist / 64.0))
    color_sim = np.round(
        np.minimum(histograms[start:end, None, :], histograms[None, :, :]).sum(axis=2) / 3.0,
        4,
    )

    aspect = dimensions[:, 0] / np.maximum(1, dimensions[:, 1])
    row_aspect = aspect[frame_image[start:end]]
    column_aspect = aspect[frame_image]
    aspect_ratio_diff = np.maximum(
        row_aspect[:, None] / np.maximum(1e-4, column_aspect[None, :]),
        column_aspect[None, :] / np.maximum(1e-4, row_aspect[:, None]),
    )
    composite = 0.55 * phash_sim + 0.25 * dhash_sim + 0.20 * color_sim
    composite = np.where(aspect_ratio_diff > 1.25, composite * 0.6, composite)
    composite = np.round(np.clip(composite, 0.0, 1.0), 4)

    relations = np.where(
        (phash_dist <= 3) & (dhash_dist <= 3) & (color_sim >= 0.94) & (aspect_ratio_diff <= 1.05),
        0,
        np.where(
            composite >= 0.85,
            1,
            np.where(composite >= 0.70, 2, 3),
        ),
    )
    return composite, relations, frame_image[start:end]


def _collect_pairs_vectorized(
    arrays: dict[str, Array],
    count: int,
    dsu: DisjointSetUnion,
    pair_similarities: dict[tuple[int, int], tuple[float, str]],
    threshold: float,
) -> None:
    """Union every image pair at or above ``threshold`` using blocked comparisons.

    Frames are compared first; each image pair then keeps the best matching
    frame score and the most duplicate-like relation, mirroring
    :func:`calculate_pair_similarity`.
    """

    frame_image = arrays["frame_image"]
    frame_count = int(frame_image.shape[0])
    if frame_count == 0:
        return
    # Frames are stored image by image, so every image owns a contiguous range.
    group_offsets = np.searchsorted(frame_image, np.arange(count), side="left")

    block = _pair_block_size(frame_count)
    for start in range(0, frame_count, block):
        end = min(frame_count, start + block)
        composite, relations, row_images = _pair_block_similarity(arrays, start, end)
        # Reduce the frames of one image: best score wins, most duplicate-like
        # relation wins, exactly like the scalar implementation.
        per_image_best = np.maximum.reduceat(composite, group_offsets, axis=1)
        per_image_relation = np.minimum.reduceat(relations, group_offsets, axis=1)
        for row in range(end - start):
            index = int(row_images[row])
            if index + 1 >= count:
                break
            offset = index + 1
            row_values = per_image_best[row]
            candidates = np.nonzero(row_values[offset:] >= threshold)[0]
            if candidates.size == 0:
                continue
            row_relations = per_image_relation[row]
            for position in candidates:
                other = offset + int(position)
                dsu.union(index, other)
                pair_similarities[(index, other)] = (
                    round(float(row_values[other]), 4),
                    _RELATION_CODES[int(row_relations[other])],
                )


def cluster_features(
    features: list[dict[str, Any]],
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Group feature dicts into similarity clusters using DSU.

    Returns:
      {
        "groups": [...],
        "total_scanned": len(features),
        "duplicates_count": total duplicates count,
        "potential_space_saved": potential saved bytes,
      }
    """
    n = len(features)
    if n < 2:
        return {
            "groups": [],
            "total_scanned": n,
            "duplicates_count": 0,
            "potential_space_saved": 0,
        }

    dsu = DisjointSetUnion(n)
    pair_similarities: dict[tuple[int, int], tuple[float, str]] = {}

    arrays = _feature_arrays(features)
    if arrays is not None:
        # Vectorized exact comparison: identical scoring rules, but the O(n^2)
        # work runs inside NumPy blocks instead of Python-level nested loops.
        _collect_pairs_vectorized(arrays, n, dsu, pair_similarities, threshold)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                sim, rel = calculate_pair_similarity(features[i], features[j])
                if sim >= threshold:
                    dsu.union(i, j)
                    pair_similarities[(i, j)] = (sim, rel)

    # Collect grouped indices
    groups_map: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        root = dsu.find(i)
        groups_map[root].append(i)

    # Filter out singletons
    cluster_groups: list[dict[str, Any]] = []
    total_duplicates = 0
    potential_space_saved = 0

    group_counter = 1
    for root, indices in groups_map.items():
        if len(indices) < 2:
            continue

        # Calculate average similarity and dominant type
        sims: list[float] = []
        types_count: dict[str, int] = defaultdict(int)

        for a_idx in range(len(indices)):
            for b_idx in range(a_idx + 1, len(indices)):
                idx1 = min(indices[a_idx], indices[b_idx])
                idx2 = max(indices[a_idx], indices[b_idx])
                if (idx1, idx2) in pair_similarities:
                    sim, rel = pair_similarities[(idx1, idx2)]
                else:
                    sim, rel = calculate_pair_similarity(features[idx1], features[idx2])
                sims.append(sim)
                types_count[rel] += 1

        avg_sim = round(sum(sims) / len(sims), 4) if sims else threshold

        # Group type determination
        if types_count.get("exact_duplicate", 0) >= len(sims) * 0.7:
            group_type = "exact_duplicate"
        elif types_count.get("variation", 0) > 0 or avg_sim >= 0.85:
            group_type = "variation"
        else:
            group_type = "similar"

        # Build group items
        raw_items = [dict(features[idx]) for idx in indices]
        best_idx = pick_best_item(raw_items)

        items_payload: list[dict[str, Any]] = []
        group_saved = 0

        for idx, itm in enumerate(raw_items):
            is_rec = idx == best_idx
            w, h = itm["dimensions"]
            fmt = itm["format"].upper()

            if is_rec:
                reason = f"推荐保留: 最高画质/分辨率 ({w}x{h}, {fmt})"
                decision = "keep"
            else:
                reason = "重复/较差版本，建议归档隔离或清理"
                decision = "archive"
                group_saved += itm["file_size"]
                total_duplicates += 1

            items_payload.append({
                "path": itm["path"],
                "file_size": itm["file_size"],
                "dimensions": itm["dimensions"],
                "format": itm["format"],
                "clarity_score": itm["clarity_score"],
                "is_recommended": is_rec,
                "recommend_reason": reason,
                "decision": decision,
                "frame_count": _safe_int(itm.get("frame_count"), 1),
                "is_animated": bool(itm.get("is_animated", False)),
                "sampled_frames": _safe_frame_indices(itm.get("sampled_frames")),
            })

        potential_space_saved += group_saved

        cluster_groups.append({
            "group_id": f"group-{group_counter:03d}",
            "group_type": group_type,
            "average_similarity": avg_sim,
            "items": items_payload,
        })
        group_counter += 1

    # Sort groups: exact_duplicate first, then by duplicate count descending
    cluster_groups.sort(
        key=lambda g: (
            0 if g["group_type"] == "exact_duplicate" else (1 if g["group_type"] == "variation" else 2),
            -len(g["items"]),
        )
    )

    return {
        "groups": cluster_groups,
        "total_scanned": n,
        "duplicates_count": total_duplicates,
        "potential_space_saved": potential_space_saved,
    }
