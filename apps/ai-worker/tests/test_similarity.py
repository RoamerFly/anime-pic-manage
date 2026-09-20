from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ai_worker import similarity as similarity_module
from ai_worker.protocol import WorkerRequest
from ai_worker.similarity import (
    calculate_pair_similarity,
    cluster_features,
    color_similarity,
    compute_clarity_score,
    compute_color_histogram,
    compute_dhash,
    compute_phash,
    extract_image_features,
    extract_image_features_batch,
    hamming_distance,
    similarity_worker_count,
)
from ai_worker.worker import WorkerService


def create_test_image(color: tuple[int, int, int], pattern: str = "rect", size: tuple[int, int] = (128, 128)) -> Image.Image:
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    if pattern == "rect":
        draw.rectangle([20, 20, 80, 80], fill=(255, 255, 255))
        draw.line([0, 0, size[0], size[1]], fill=(0, 0, 0), width=3)
    elif pattern == "circle":
        draw.ellipse([20, 20, 80, 80], fill=(255, 255, 255))
        draw.line([0, size[1], size[0], 0], fill=(0, 0, 0), width=3)
    return img


def test_hash_identical_and_different(tmp_path: Path):
    img1 = create_test_image((200, 50, 50), "rect")
    img2 = create_test_image((200, 50, 50), "rect")
    img3 = create_test_image((50, 50, 200), "circle")

    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    h3 = compute_phash(img3)

    assert h1 == h2
    assert hamming_distance(h1, h2) == 0
    assert hamming_distance(h1, h3) > 5

    dh1 = compute_dhash(img1)
    dh2 = compute_dhash(img2)
    dh3 = compute_dhash(img3)
    assert dh1 == dh2
    assert hamming_distance(dh1, dh2) == 0
    assert hamming_distance(dh1, dh3) > 5


def test_color_histogram_and_clarity():
    img_red = Image.new("RGB", (100, 100), (255, 0, 0))
    img_blue = Image.new("RGB", (100, 100), (0, 0, 255))

    hist_red = compute_color_histogram(img_red)
    hist_blue = compute_color_histogram(img_blue)

    assert len(hist_red) == 48
    assert len(hist_blue) == 48

    sim_red_blue = color_similarity(hist_red, hist_blue)
    assert sim_red_blue <= 0.35

    sim_red_red = color_similarity(hist_red, hist_red)
    assert sim_red_red > 0.99

    # Sharp vs flat image clarity
    flat_img = Image.new("RGB", (100, 100), (128, 128, 128))
    sharp_img = create_test_image((100, 100, 100), "rect")
    assert compute_clarity_score(flat_img) == 0.0
    assert compute_clarity_score(sharp_img) > 0.0


def test_feature_extraction_and_pair_similarity(tmp_path: Path):
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.jpg"
    p3 = tmp_path / "img3.png"

    # img1 & img2: almost identical (p2 is jpeg saved version of p1)
    base = create_test_image((120, 80, 200), "rect", (200, 200))
    base.save(p1)
    base.save(p2, "JPEG", quality=85)

    # img3: completely different
    diff = create_test_image((10, 200, 10), "circle", (200, 200))
    diff.save(p3)

    f1 = extract_image_features(p1)
    f2 = extract_image_features(p2)
    f3 = extract_image_features(p3)

    sim_12, rel_12 = calculate_pair_similarity(f1, f2)
    assert sim_12 >= 0.90
    assert rel_12 in ("exact_duplicate", "variation")

    sim_13, rel_13 = calculate_pair_similarity(f1, f3)
    assert sim_13 < 0.70
    assert rel_13 == "different"


def test_cluster_features_and_recommendation(tmp_path: Path):
    # Create 3 images: 2 identical (different resolution/format), 1 different
    p_high = tmp_path / "high.png"
    p_low = tmp_path / "low.jpg"
    p_other = tmp_path / "other.png"

    base_high = create_test_image((100, 150, 200), "rect", (400, 400))
    base_low = base_high.resize((100, 100))
    other = create_test_image((20, 20, 20), "circle", (200, 200))

    base_high.save(p_high)
    base_low.save(p_low, "JPEG", quality=70)
    other.save(p_other)

    f_high = extract_image_features(p_high)
    f_low = extract_image_features(p_low)
    f_other = extract_image_features(p_other)

    result = cluster_features([f_high, f_low, f_other], threshold=0.80)
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert len(group["items"]) == 2

    # High res png should be recommended
    high_item = next(it for it in group["items"] if it["path"] == str(p_high))
    low_item = next(it for it in group["items"] if it["path"] == str(p_low))

    assert high_item["is_recommended"] is True
    assert high_item["decision"] == "keep"
    assert "最高" in (high_item["recommend_reason"] or "")

    assert low_item["is_recommended"] is False
    assert low_item["decision"] == "archive"


def test_batch_extraction_preserves_order_and_reports_failures(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    missing = tmp_path / "missing.png"
    create_test_image((10, 20, 30), "rect").save(first)
    create_test_image((200, 120, 60), "circle").save(second)

    features, errors = extract_image_features_batch([first, missing, second], max_workers=4)

    assert [item["path"] for item in features] == [str(first), str(second)]
    assert [item["path"] for item in errors] == [str(missing)]


def test_batch_extraction_matches_serial_results(tmp_path: Path):
    paths = []
    for index in range(6):
        path = tmp_path / f"image-{index}.png"
        create_test_image((20 * index, 60, 200 - 10 * index), "rect", (96, 96)).save(path)
        paths.append(path)

    serial, serial_errors = extract_image_features_batch(paths, max_workers=1)
    parallel, parallel_errors = extract_image_features_batch(paths, max_workers=4)

    assert serial == parallel
    assert serial_errors == parallel_errors == []


def test_similarity_worker_count_is_bounded():
    assert similarity_worker_count(0) == 1
    assert similarity_worker_count(3) == 3
    assert similarity_worker_count(999) == 32
    assert 1 <= similarity_worker_count() <= 8


def test_vectorized_clustering_matches_scalar_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    paths = []
    for index in range(8):
        path = tmp_path / f"cluster-{index}.png"
        pattern = "rect" if index % 2 == 0 else "circle"
        size = (160, 160) if index < 4 else (96, 96)
        create_test_image((90 + index, 40, 210 - index), pattern, size).save(path)
        paths.append(path)

    duplicate = tmp_path / "cluster-0-copy.png"
    Image.open(paths[0]).save(duplicate)
    paths.append(duplicate)

    features = [extract_image_features(path) for path in paths]

    with monkeypatch.context() as patch:
        patch.setattr(similarity_module, "_feature_arrays", lambda _features: None)
        scalar = cluster_features(features, threshold=0.80)

    vectorized = cluster_features(features, threshold=0.80)

    assert vectorized["total_scanned"] == scalar["total_scanned"]
    assert vectorized["duplicates_count"] == scalar["duplicates_count"]
    assert vectorized["potential_space_saved"] == scalar["potential_space_saved"]
    assert vectorized["groups"] == scalar["groups"]


def test_worker_similarity_dispatch(tmp_path: Path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    create_test_image((50, 100, 150), "rect").save(p1)
    create_test_image((50, 100, 150), "rect").save(p2)

    service = WorkerService()

    # Test extract
    req_extract = WorkerRequest.from_dict({
        "schema_version": "1.0",
        "request_id": "r1",
        "task_id": "t1",
        "message_type": "similarity.features.extract",
        "payload": {"paths": [str(p1), str(p2)]},
    })
    resp_extract = service.handle(req_extract)
    assert resp_extract.error is None
    features = resp_extract.payload["features"]
    assert len(features) == 2

    # Test cluster
    req_cluster = WorkerRequest.from_dict({
        "schema_version": "1.0",
        "request_id": "r2",
        "task_id": "t2",
        "message_type": "similarity.cluster",
        "payload": {"features": features, "threshold": 0.85},
    })
    resp_cluster = service.handle(req_cluster)
    assert resp_cluster.error is None
    assert len(resp_cluster.payload["groups"]) == 1

    # Test end-to-end scan
    req_scan = WorkerRequest.from_dict({
        "schema_version": "1.0",
        "request_id": "r3",
        "task_id": "t3",
        "message_type": "similarity.scan",
        "payload": {"directory": str(tmp_path), "threshold": 0.85},
    })
    resp_scan = service.handle(req_scan)
    assert resp_scan.error is None
    assert resp_scan.payload["duplicates_count"] >= 1


def test_similarity_extract_honors_max_workers(tmp_path: Path):
    paths = []
    for index in range(4):
        path = tmp_path / f"worker-{index}.png"
        create_test_image((30 * index, 90, 180), "rect").save(path)
        paths.append(str(path))

    service = WorkerService()
    request = WorkerRequest.from_dict({
        "schema_version": "1.0",
        "request_id": "r-workers",
        "task_id": "t-workers",
        "message_type": "similarity.features.extract",
        "payload": {"paths": paths, "max_workers": 2},
    })

    response = service.handle(request)

    assert response.error is None
    assert len(response.payload["features"]) == 4


def test_similarity_extract_rejects_invalid_max_workers(tmp_path: Path):
    path = tmp_path / "bad-workers.png"
    create_test_image((10, 10, 10), "rect").save(path)

    for invalid in (0, -3, 999):
        request = WorkerRequest.from_dict({
            "schema_version": "1.0",
            "request_id": "r-bad-workers",
            "task_id": "t-bad-workers",
            "message_type": "similarity.features.extract",
            "payload": {"paths": [str(path)], "max_workers": invalid},
        })
        response = WorkerService().handle(request)
        assert response.error is not None
        assert response.error["code"] == "INVALID_PAYLOAD"


def _write_animated_gif(path: Path, frames: list[Image.Image], duration: int = 100) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def test_animated_gif_samples_first_middle_and_last_frame(tmp_path: Path):
    frames = []
    for index in range(5):
        frame = create_test_image(
            (30 + index * 20, 60, 200 - index * 10), "rect", (160, 160)
        )
        ImageDraw.Draw(frame).rectangle(
            [10 + index * 12, 10, 40 + index * 12, 40], fill=(255, 0, 0)
        )
        frames.append(frame)
    gif_path = tmp_path / "animation.gif"
    _write_animated_gif(gif_path, frames)

    features = extract_image_features(gif_path)

    assert features["format"] == "gif"
    assert features["frame_count"] == 5
    assert features["is_animated"] is True
    assert features["sampled_frames"] == [0, 2, 4]
    assert len(features["frames"]) == 3
    # The first frame stays available at the top level for older consumers.
    assert features["phash"] == features["frames"][0]["phash"]
    assert features["color_histogram"] == features["frames"][0]["color_histogram"]


def test_still_image_keeps_single_frame_shape(tmp_path: Path):
    still = tmp_path / "still.png"
    create_test_image((10, 20, 30), "rect", (96, 96)).save(still)

    features = extract_image_features(still)

    assert features["frame_count"] == 1
    assert features["is_animated"] is False
    assert features["sampled_frames"] == [0]
    assert len(features["frames"]) == 1


def test_animation_matches_a_still_image_through_any_sampled_frame(tmp_path: Path):
    frames = []
    for index in range(5):
        frame = create_test_image((20 * index, 40, 180), "circle", (160, 160))
        frames.append(frame)
    gif_path = tmp_path / "motion.gif"
    _write_animated_gif(gif_path, frames)

    # The still image equals the last frame only.
    still = tmp_path / "last-frame.png"
    frames[-1].save(still)

    animation = extract_image_features(gif_path)
    still_features = extract_image_features(still)
    similarity, relation = calculate_pair_similarity(animation, still_features)

    assert animation["sampled_frames"][-1] == 4
    assert similarity >= 0.95
    assert relation == "exact_duplicate"


def test_vectorized_clustering_matches_scalar_path_with_animations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    base_frames = [
        create_test_image((60, 90, 200), "rect", (140, 140)),
        create_test_image((90, 60, 150), "circle", (140, 140)),
    ]
    animation = tmp_path / "cluster-anim.gif"
    _write_animated_gif(animation, base_frames)
    still = tmp_path / "cluster-still.png"
    base_frames[0].save(still)
    other = tmp_path / "cluster-other.png"
    create_test_image((10, 200, 10), "rect", (140, 140)).save(other)

    features = [
        extract_image_features(animation),
        extract_image_features(still),
        extract_image_features(other),
    ]

    with monkeypatch.context() as patch:
        patch.setattr(similarity_module, "_feature_arrays", lambda _features: None)
        scalar = cluster_features(features, threshold=0.80)

    vectorized = cluster_features(features, threshold=0.80)

    assert vectorized["groups"] == scalar["groups"]
    assert vectorized["duplicates_count"] == scalar["duplicates_count"]


def test_cluster_items_expose_animation_metadata(tmp_path: Path):
    frames = [
        create_test_image((70, 70, 200), "rect", (128, 128)),
        create_test_image((70, 200, 70), "rect", (128, 128)),
    ]
    animation = tmp_path / "meta.gif"
    _write_animated_gif(animation, frames)
    still = tmp_path / "meta-still.png"
    frames[0].save(still)

    result = cluster_features(
        [extract_image_features(animation), extract_image_features(still)],
        threshold=0.80,
    )

    items = result["groups"][0]["items"]
    animated_item = next(item for item in items if item["path"].endswith(".gif"))
    still_item = next(item for item in items if item["path"].endswith(".png"))

    assert animated_item["is_animated"] is True
    assert animated_item["frame_count"] == 2
    assert animated_item["sampled_frames"] == [0, 1]
    assert still_item["is_animated"] is False
