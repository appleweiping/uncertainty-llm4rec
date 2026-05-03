"""Tests for CU-GR v2 candidate panel construction."""

from __future__ import annotations

from llm4rec.methods.candidate_panel import (
    build_candidate_panel,
    fallback_full_ranking_in_candidates,
    oracle_rerank_top10_metrics,
    panel_item_ids,
)


def _ctx():
    return {
        "train_popularity": {"a": 100, "b": 50, "c": 1, "d": 2, "e": 3},
        "item_buckets": {"a": "head", "b": "head", "c": "tail", "d": "tail", "e": "mid"},
        "item_titles": {x: f"T{x}" for x in "abcde"},
        "item_category": {x: f"g{x}" for x in "abcde"},
    }


def test_fallback_full_ranking_respects_candidate_set():
    bm = {
        "user_id": "u1",
        "target_item": "c",
        "candidate_items": ["a", "b", "c"],
        "predicted_items": ["a", "b", "c", "x"],
        "scores": [3.0, 2.0, 1.0, 99.0],
        "metadata": {"example_id": "e1"},
    }
    items, sc = fallback_full_ranking_in_candidates(bm)
    assert set(items) == {"a", "b", "c"}
    assert items[0] == "a"


def test_panel_contains_top5_and_respects_size():
    bm = {
        "user_id": "u1",
        "target_item": "c",
        "candidate_items": [f"i{i}" for i in range(60)],
        "predicted_items": [f"i{i}" for i in range(60)],
        "scores": [float(60 - i) for i in range(60)],
        "metadata": {"example_id": "e1"},
    }
    ctx = {
        "train_popularity": {f"i{i}": i for i in range(60)},
        "item_buckets": {f"i{i}": "mid" for i in range(60)},
        "item_titles": {f"i{i}": f"T{i}" for i in range(60)},
        "item_category": {f"i{i}": "g" for i in range(60)},
    }
    panel = build_candidate_panel(
        bm25_row=bm,
        ours_row=None,
        sequential_row=None,
        context=ctx,
        panel_size=10,
        seed=13,
    )
    assert len(panel["panel_items"]) == 10
    ids = panel_item_ids(panel)
    assert "i0" in ids and "i4" in ids


def test_oracle_gain_when_target_in_panel_but_not_top10():
    # Target at index 12 (outside top-10); panel covers rank-1 slot so oracle can move target to global index 0.
    F = [f"i{i}" for i in range(30)]
    target = "i12"
    panel = set(F[0:20])
    assert target in panel
    cand = list(F)
    m = oracle_rerank_top10_metrics(full_fallback_order=F, panel_ids=panel, target_item=target, candidate_items=cand)
    assert m["target_in_panel"] == 1.0
    assert m["recall_gain"] >= 1.0 - 1e-6
