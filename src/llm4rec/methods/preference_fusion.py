"""Offline preference fusion helpers for CU-GR v2 (after LLM signals exist)."""

from __future__ import annotations

from typing import Any

from llm4rec.metrics.ranking import dedupe


def merge_panel_order_into_full_ranking(
    full_fallback_order: list[str],
    panel_ids: list[str],
    fused_panel_order: list[str],
) -> list[str]:
    """Replace positions of panel items with fused_panel_order; preserve non-panel positions."""
    panel_set = set(panel_ids)
    positions = sorted(i for i, v in enumerate(full_fallback_order) if v in panel_set)
    if len(positions) != len(fused_panel_order):
        fused_panel_order = list(fused_panel_order)[: len(positions)]
    out = list(full_fallback_order)
    for j, pos in enumerate(positions):
        if j < len(fused_panel_order):
            out[pos] = fused_panel_order[j]
    return out


def fused_top_k(
    full_fallback_order: list[str],
    panel_ids: list[str],
    fused_panel_order: list[str],
    *,
    k: int = 10,
) -> list[str]:
    merged = merge_panel_order_into_full_ranking(full_fallback_order, panel_ids, fused_panel_order)
    return dedupe(merged)[:k]


def linear_fusion_scores(
    *,
    panel_item_ids: list[str],
    normalized_fallback: dict[str, float],
    normalized_llm: dict[str, float],
    alpha: float = 0.65,
    beta: float = 0.35,
) -> list[tuple[str, float]]:
    """Weighted linear fusion over panel items (diagnostic)."""
    out: list[tuple[str, float]] = []
    for iid in panel_item_ids:
        fb = float(normalized_fallback.get(iid, 0.0))
        ll = float(normalized_llm.get(iid, 0.0))
        out.append((iid, alpha * fb + beta * ll))
    out.sort(key=lambda x: -x[1])
    return [x[0] for x in out]
