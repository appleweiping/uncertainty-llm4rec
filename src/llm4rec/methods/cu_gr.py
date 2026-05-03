"""CU-GR: Calibrated Uncertainty-Gated Recommendation (offline replay policy).

Uses fallback top-K by default; promotes LLM grounded item to rank 1 only when
calibrated improve/harm heads allow. Does not change prompts or evaluators.
"""

from __future__ import annotations

import json
from typing import Any

from llm4rec.analysis.calibrator_features import TOP_K_LABEL, fallback_top_items_scores

EPSILON_SCORE = 1e-6


def gate_parse_grounding_adherence(meta: dict[str, Any]) -> tuple[bool, str | None]:
    """Hard gates before learned scores (inference path)."""
    if not bool(meta.get("parse_success", False)):
        return False, "parse_failed"
    if not bool(meta.get("grounding_success", False)):
        return False, "grounding_failed"
    if not bool(meta.get("candidate_adherent", False)):
        return False, "not_candidate_adherent"
    return True, None


def decide_promote(
    *,
    gates_ok: bool,
    p_improve: float,
    p_harm: float,
    tau_improve: float,
    tau_harm: float,
) -> bool:
    if not gates_ok:
        return False
    return p_improve >= tau_improve and p_harm <= tau_harm


def build_cu_gr_prediction(
    *,
    ours_row: dict[str, Any],
    bm25_row: dict[str, Any],
    promote: bool,
    p_improve: float,
    p_harm: float,
    tau_improve: float,
    tau_harm: float,
    calibrator_model: str,
    features_version: str,
) -> dict[str, Any]:
    """Return a full prediction record mirroring experiment schema for metric code."""
    meta_in = dict(ours_row.get("metadata") or {})
    fb_items, fb_scores = fallback_top_items_scores(bm25_row, k=TOP_K_LABEL)
    g_id = meta_in.get("grounded_item_id")
    g_str = str(g_id) if g_id not in (None, "") else ""
    candidates = list(ours_row.get("candidate_items") or [])
    gates_ok, gate_reason = gate_parse_grounding_adherence(meta_in)
    decision = "keep_fallback"
    predicted = list(fb_items)
    scores = list(fb_scores)
    can_promote_item = bool(g_str and g_str in set(candidates))

    if promote and can_promote_item:
        decision = "promote_generated"
        rest = [x for x in fb_items if x != g_str]
        predicted = [g_str] + rest
        predicted = predicted[:TOP_K_LABEL]
        max_s = max(fb_scores) if fb_scores else 1.0
        g_score = float(max_s) + EPSILON_SCORE
        score_by_item = {str(a): float(b) for a, b in zip(fb_items, fb_scores, strict=False)}
        scores = []
        for item in predicted:
            if item == g_str:
                scores.append(g_score)
            else:
                scores.append(float(score_by_item.get(item, 0.0)))

    cu_meta = {
        "cu_gr": True,
        "decision": decision,
        "p_improve": float(p_improve),
        "p_harm": float(p_harm),
        "tau_improve": float(tau_improve),
        "tau_harm": float(tau_harm),
        "features_version": features_version,
        "calibrator_model": calibrator_model,
        "calibrator_train_seed": 13,
        "calibrator_validation_seed": 21,
        "candidate_adherent": bool(meta_in.get("candidate_adherent", False)),
        "grounding_success": bool(meta_in.get("grounding_success", False)),
        "fallback_method": str(meta_in.get("fallback_method") or "bm25"),
        "override_item_id": g_str,
        "override_reason": (
            []
            if decision == "promote_generated"
            else ([gate_reason] if (not gates_ok and gate_reason) else ["threshold_or_gates"])
        ),
        "gate_blocked": not gates_ok,
    }
    out_meta = dict(meta_in)
    out_meta["cu_gr_metadata"] = cu_meta
    return {
        "user_id": str(ours_row.get("user_id", "")),
        "target_item": str(ours_row.get("target_item", "")),
        "candidate_items": candidates,
        "predicted_items": predicted,
        "scores": scores,
        "method": "cu_gr",
        "domain": str(ours_row.get("domain", "movies")),
        "raw_output": None,
        "metadata": out_meta,
    }


def summarize_override_outcomes(
    rows: list[dict[str, Any]],
    *,
    delta_ndcg: list[float],
) -> dict[str, Any]:
    """Rows are CU-GR predictions; delta_ndcg aligned by index from offline labels."""
    beneficial = harmful = neutral = accepted = 0
    for i, row in enumerate(rows):
        m = (row.get("metadata") or {}).get("cu_gr_metadata") or {}
        if m.get("decision") != "promote_generated":
            continue
        accepted += 1
        d = float(delta_ndcg[i]) if i < len(delta_ndcg) else 0.0
        if d > 0:
            beneficial += 1
        elif d < 0:
            harmful += 1
        else:
            neutral += 1
    n = max(len(rows), 1)
    return {
        "accepted_override_count": accepted,
        "beneficial_override_count": beneficial,
        "harmful_override_count": harmful,
        "neutral_override_count": neutral,
        "harmful_override_rate": harmful / n,
    }


def dump_json(path: str, payload: dict[str, Any]) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
