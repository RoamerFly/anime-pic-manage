from __future__ import annotations

from ai_worker.images import CropType
from ai_worker.recognition import DecisionStatus, FusionConfig, fuse_multiscale, top_k_scores
from ai_worker.whitelist import Character, CharacterSet


def test_top_k_filters_allowed_tags_and_sorts_ties() -> None:
    scores = [
        {"tag": "outside", "score": 0.99},
        {"tag": "character_beta", "score": 0.8},
        {"tag": "character_alpha", "score": 0.8},
    ]
    result = top_k_scores(scores, allowed_tags={"character_alpha", "character_beta"}, k=2)
    assert [item.tag for item in result] == ["character_alpha", "character_beta"]


def test_multiscale_fusion_high_confidence_and_margin() -> None:
    result = fuse_multiscale(
        {
            CropType.HEAD: [{"tag": "character_alpha", "score": 0.95}],
            CropType.BUST: [{"tag": "character_alpha", "score": 0.90}, {"tag": "x", "score": 0.2}],
            CropType.LARGE: [{"tag": "character_alpha", "score": 0.88}],
        }
    )
    assert result.status is DecisionStatus.HIGH_CONFIDENCE
    assert result.top1 is not None and result.top1.tag == "character_alpha"
    assert result.margin > 0.8
    assert result.consistency == 1.0


def test_multiscale_fusion_margin_and_conflict_need_review() -> None:
    config = FusionConfig(auto_threshold=0.6, min_margin=0.2, min_consistency=0.9)
    result = fuse_multiscale(
        {
            "head": [
                {"tag": "character_alpha", "score": 0.8},
                {"tag": "character_beta", "score": 0.7},
            ],
            "bust": [
                {"tag": "character_beta", "score": 0.8},
                {"tag": "character_alpha", "score": 0.7},
            ],
        },
        config=config,
    )
    assert result.status is DecisionStatus.NEEDS_REVIEW
    assert result.margin < 0.2
    assert result.consistency == 0.5


def test_low_score_is_unrecognized() -> None:
    result = fuse_multiscale({"head": [{"tag": "character_alpha", "score": 0.2}]})
    assert result.status is DecisionStatus.UNRECOGNIZED


def test_external_character_set_intersects_model_labels() -> None:
    character_set = CharacterSet(
        "1.0",
        "example",
        "示例",
        (Character("character_alpha", "角色甲"), Character("character_beta", "角色乙")),
    )
    supported, missing = character_set.model_intersection(["character_alpha", "other"])
    assert supported == {"character_alpha"}
    assert missing == {"character_beta"}
