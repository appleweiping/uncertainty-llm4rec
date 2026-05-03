"""Prompt templates for CU-GR v2 listwise / pairwise candidate preference (no target leakage)."""

from __future__ import annotations

import json
from typing import Any


def _local_labels(n: int) -> list[str]:
    if n > 26:
        return [f"L{i}" for i in range(n)]
    return [chr(ord("A") + i) for i in range(n)]


def build_listwise_preference_prompt(
    *,
    history_titles: list[str],
    panel: dict[str, Any],
    show_fallback_rank: bool = False,
) -> tuple[str, dict[str, str]]:
    """Return (prompt_text, label_to_item_id)."""
    items = panel.get("panel_items") or []
    labels = _local_labels(len(items))
    label_map = {labels[i]: str(items[i]["item_id"]) for i in range(len(items))}
    lines = []
    for lab, p in zip(labels, items, strict=False):
        extra = ""
        if show_fallback_rank:
            extra = f" fallback_rank={int(p.get('fallback_rank') or 0)}"
        lines.append(
            f"- {lab}: title={json.dumps(p.get('title',''), ensure_ascii=False)}; "
            f"genre={json.dumps(p.get('genre',''), ensure_ascii=False)}{extra}"
        )
    hist = "\n".join(f"- {json.dumps(t, ensure_ascii=False)}" for t in (history_titles or [])[:40])
    schema = """{
  "ranking": [
    {"label": "C", "score": 0.91, "confidence": 0.72, "reason": "short text"}
  ],
  "uncertainty": {
    "most_uncertain_labels": ["H", "K"],
    "overall_confidence": 0.63
  }
}"""
    prompt = (
        "You are helping with movie recommendation using ONLY the labeled candidates below.\n"
        "Do not invent items outside the labels. Output valid JSON only.\n\n"
        "User history (recent titles):\n"
        f"{hist if hist else '(none)'}\n\n"
        "Candidate panel (anonymous labels):\n"
        + "\n".join(lines)
        + "\n\n"
        "Task: rank and score each anonymous label for predicted relevance to this user's NEXT movie watch, "
        "using only the titles/genres shown (scores in [0,1], higher = more likely next watch). "
        "Provide per-label confidence in [0,1] and a very short reason (at most 12 words).\n"
        "Return JSON matching this schema (example shape):\n"
        f"{schema}\n"
    )
    return prompt, label_map


def build_pairwise_preference_prompt(
    *,
    history_titles: list[str],
    pairs: list[tuple[str, str, str, str]],
) -> tuple[str, dict[str, str]]:
    """pairs: (left_label, right_label, left_item_id, right_item_id)."""
    label_map: dict[str, str] = {}
    for a, b, ia, ib in pairs:
        label_map[a] = ia
        label_map[b] = ib
    hist = "\n".join(f"- {json.dumps(t, ensure_ascii=False)}" for t in (history_titles or [])[:40])
    plines = [f"- Compare {a} vs {b}: titles only in JSON pairs section." for a, b, _, _ in pairs]
    schema = """{
  "pairs": [
    {"left": "A", "right": "B", "winner": "A", "confidence": 0.67, "reason": "..."}
  ],
  "overall_confidence": 0.62
}"""
    prompt = (
        "You compare anonymous movie candidates for the same user.\n"
        "Pick a winner per pair using only the provided titles/genres in the JSON block.\n"
        "Output valid JSON only.\n\n"
        f"User history:\n{hist if hist else '(none)'}\n\n"
        + "\n".join(plines)
        + "\n\nSchema:\n"
        f"{schema}\n"
    )
    return prompt, label_map


def default_pairwise_pairs_from_panel(panel: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Build a small fixed set of pairs using local labels (A=rank1, etc.)."""
    items = panel.get("panel_items") or []
    if len(items) < 2:
        return []
    labs = _local_labels(len(items))
    out: list[tuple[str, str, str, str]] = []

    def add_pair(i: int, j: int) -> None:
        if i < len(labs) and j < len(labs):
            out.append((labs[i], labs[j], str(items[i]["item_id"]), str(items[j]["item_id"])))

    add_pair(0, min(9, len(items) - 1))
    add_pair(0, min(4, len(items) - 1))
    if len(items) > 5:
        add_pair(4, min(9, len(items) - 1))
    return out[:8]
