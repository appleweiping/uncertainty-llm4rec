"""CU-GR policy construction tests."""

from __future__ import annotations

from llm4rec.methods.cu_gr import (
    build_cu_gr_prediction,
    decide_promote,
    gate_parse_grounding_adherence,
)


def test_gates_block_bad_parse():
    meta = {"parse_success": False, "grounding_success": True, "candidate_adherent": True}
    ok, _ = gate_parse_grounding_adherence(meta)
    assert ok is False


def test_decide_promote_requires_both_thresholds():
    assert decide_promote(gates_ok=True, p_improve=0.99, p_harm=0.5, tau_improve=0.5, tau_harm=0.4) is False
    assert decide_promote(gates_ok=True, p_improve=0.6, p_harm=0.05, tau_improve=0.5, tau_harm=0.1) is True


def test_build_prediction_promote_changes_order():
    ours = {
        "user_id": "u",
        "target_item": "t",
        "domain": "movies",
        "candidate_items": ["a", "b"],
        "predicted_items": ["b"],
        "metadata": {
            "parse_success": True,
            "grounding_success": True,
            "candidate_adherent": True,
            "grounded_item_id": "b",
            "fallback_method": "bm25",
        },
    }
    bm = {"predicted_items": ["a", "b"], "scores": [2.0, 1.0], "candidate_items": ["a", "b"], "metadata": {}}
    out = build_cu_gr_prediction(
        ours_row=ours,
        bm25_row=bm,
        promote=True,
        p_improve=0.8,
        p_harm=0.1,
        tau_improve=0.5,
        tau_harm=0.2,
        calibrator_model="test",
        features_version="v0",
    )
    assert out["predicted_items"][0] == "b"
    assert (out["metadata"].get("cu_gr_metadata") or {}).get("decision") == "promote_generated"
