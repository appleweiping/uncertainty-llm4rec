"""Unit tests for CU-GR offline feature and label construction."""

from __future__ import annotations

from llm4rec.analysis.calibrator_features import (
    build_override_topk,
    compute_offline_labels,
    extract_features,
    fallback_top_items_scores,
    metric_row,
)


def test_fallback_top_items_scores_orders_by_score():
    row = {
        "predicted_items": ["a", "b", "c"],
        "scores": [1.0, 3.0, 2.0],
        "candidate_items": ["a", "b", "c"],
        "target_item": "b",
    }
    items, scores = fallback_top_items_scores(row, k=2)
    assert items[0] == "b" and scores[0] == 3.0


def test_override_ranking_promotes_grounded():
    F = ["a", "b", "c"]
    O = build_override_topk(F, "b", {"a", "b", "c"}, k=3)
    assert O[0] == "b"


def test_labels_improve_when_target_moves_up():
    F = ["x", "y", "z"]
    O = ["y", "x", "z"]
    cand = ["x", "y", "z", "t"]
    lab = compute_offline_labels(
        target_item="y",
        candidate_items=cand,
        fallback_items_topk=F,
        override_items_topk=O,
        parse_success=True,
        grounding_success=True,
        candidate_adherent=True,
        hallucination_flag=False,
    )
    assert lab["override_improves"] == 1
    assert lab["delta_ndcg_at_10"] > 0


def test_extract_features_no_target_leak():
    ours = {
        "user_id": "1",
        "target_item": "SECRET",
        "candidate_items": ["a", "b"],
        "predicted_items": ["a"],
        "metadata": {
            "parse_success": True,
            "grounding_success": True,
            "candidate_adherent": True,
            "confidence": 0.7,
            "grounded_item_id": "a",
            "grounding_method": "exact",
            "grounding_score": 1.0,
            "is_catalog_valid": True,
            "is_hallucinated": False,
            "history_item_ids": [],
            "history_titles": [],
            "generated_title": "A title",
            "echo_risk": False,
        },
    }
    bm = {"predicted_items": ["a", "b"], "scores": [2.0, 1.0], "candidate_items": ["a", "b"], "metadata": {}}
    ctx = {
        "train_popularity": {"a": 5, "b": 1},
        "item_buckets": {"a": "head", "b": "tail"},
        "item_titles": {"a": "A title", "b": "B title"},
        "item_category": {"a": "c1", "b": "c2"},
    }
    feats = extract_features(ours, bm, None, context=ctx, train_stats={"mean_confidence_train_seed": 0.5})
    assert "SECRET" not in repr(feats)
    assert feats["llm_confidence"] == 0.7


def test_metric_row_matches_ranking_helpers():
    row = {"target_item": "t", "predicted_items": ["x", "t"], "candidate_items": ["x", "t"]}
    m = metric_row("t", ["x", "t"], ["x", "t"])
    from llm4rec.metrics.ranking import _ndcg_at_k

    assert abs(m["ndcg@10"] - _ndcg_at_k(row, 10)) < 1e-9
