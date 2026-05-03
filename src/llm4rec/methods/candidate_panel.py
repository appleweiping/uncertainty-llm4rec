"""Local candidate panel construction for CU-GR v2 (offline, no API)."""

from __future__ import annotations

import hashlib
from typing import Any

from llm4rec.metrics.ranking import _ndcg_at_k, _recall_at_k, dedupe


def _stable_tie_key(example_id: str, seed: int, item_id: str) -> tuple[int, str]:
    h = hashlib.sha256(f"{seed}:{example_id}:{item_id}".encode()).hexdigest()
    return (int(h[:8], 16), item_id)


def fallback_full_ranking_in_candidates(bm25_row: dict[str, Any]) -> tuple[list[str], list[float]]:
    """BM25 scores over catalog; keep only items in this row's candidate set, order by score desc."""
    cand = set(str(x) for x in (bm25_row.get("candidate_items") or []))
    items = [str(x) for x in (bm25_row.get("predicted_items") or [])]
    scores = [float(x) for x in (bm25_row.get("scores") or [])]
    if len(scores) != len(items):
        scores = [float(i) for i in range(len(items), 0, -1)]
    pairs = [(it, sc) for it, sc in zip(items, scores, strict=False) if it in cand]
    pairs.sort(key=lambda x: (-x[1], _stable_tie_key("", 0, x[0])))
    out_items: list[str] = []
    out_scores: list[float] = []
    seen: set[str] = set()
    for it, sc in pairs:
        if it in seen:
            continue
        seen.add(it)
        out_items.append(it)
        out_scores.append(sc)
    return out_items, out_scores


def sequential_ranking_in_candidates(sequential_row: dict[str, Any] | None) -> list[str]:
    if not sequential_row:
        return []
    cand = set(str(x) for x in (sequential_row.get("candidate_items") or []))
    items = [str(x) for x in (sequential_row.get("predicted_items") or [])]
    scores = [float(x) for x in (sequential_row.get("scores") or [])]
    if len(scores) != len(items):
        scores = [float(i) for i in range(len(items), 0, -1)]
    pairs = [(it, sc) for it, sc in zip(items, scores, strict=False) if it in cand]
    pairs.sort(key=lambda x: (-x[1], _stable_tie_key("", 0, x[0])))
    return dedupe([p[0] for p in pairs])


def _item_at_fallback_rank(ranked_items: list[str], rank_1based: int | None) -> str | None:
    if not rank_1based or rank_1based < 1 or rank_1based > len(ranked_items):
        return None
    return ranked_items[rank_1based - 1]


def _pick_head_contrast(
    candidate_ids: list[str],
    chosen: set[str],
    train_popularity: dict[str, int],
    *,
    example_id: str,
    seed: int,
) -> tuple[str | None, str]:
    """Highest train popularity among candidates not yet chosen."""
    pool = [c for c in candidate_ids if c not in chosen]
    if not pool:
        return None, "head_contrast"
    pool.sort(key=lambda i: (-int(train_popularity.get(i, 0)), _stable_tie_key(example_id, seed, i)))
    return pool[0], "head_contrast"


def _pick_tail_contrast(
    candidate_ids: list[str],
    chosen: set[str],
    train_popularity: dict[str, int],
    item_buckets: dict[str, str],
    *,
    example_id: str,
    seed: int,
) -> tuple[str | None, str]:
    """Prefer tail-bucket item with low popularity; else globally lowest pop."""
    pool = [c for c in candidate_ids if c not in chosen]
    if not pool:
        return None, "tail_contrast"
    tails = [c for c in pool if item_buckets.get(c) == "tail"]
    use = tails if tails else pool
    use.sort(key=lambda i: (int(train_popularity.get(i, 0)), _stable_tie_key(example_id, seed, i)))
    return use[0], "tail_contrast"


def build_candidate_panel(
    *,
    bm25_row: dict[str, Any],
    ours_row: dict[str, Any] | None,
    sequential_row: dict[str, Any] | None,
    context: dict[str, Any],
    panel_size: int,
    seed: int,
) -> dict[str, Any]:
    """Construct deterministic local panel (all items in candidate set)."""
    example_id = str((bm25_row.get("metadata") or {}).get("example_id") or "")
    user_id = str(bm25_row.get("user_id") or "")
    candidate_ids = [str(x) for x in (bm25_row.get("candidate_items") or [])]
    cand_set = set(candidate_ids)
    ranked, scores = fallback_full_ranking_in_candidates(bm25_row)
    score_by_item = {ranked[i]: scores[i] for i in range(len(ranked))}

    train_pop: dict[str, int] = context.get("train_popularity") or {}
    item_buckets: dict[str, str] = context.get("item_buckets") or {}
    item_titles: dict[str, str] = context.get("item_titles") or {}
    item_genres: dict[str, str] = context.get("item_category") or context.get("item_genres") or {}

    seq_ranked = sequential_ranking_in_candidates(sequential_row)

    grounded_id: str | None = None
    if ours_row:
        meta = ours_row.get("metadata") or {}
        gid = meta.get("grounded_item_id")
        adherent = bool(meta.get("candidate_adherent", False))
        parse_ok = bool(meta.get("parse_success", False))
        ground_ok = bool(meta.get("grounding_success", False))
        if gid and adherent and parse_ok and ground_ok and str(gid) in cand_set:
            grounded_id = str(gid)

    # Ordered (priority, source_tag, item_id or None placeholder rank)
    slots: list[tuple[int, str, str | None, int | None]] = []
    for r in (1, 2, 3, 4, 5):
        slots.append((0, "fallback_top", None, r))
    for r in (10, 20, 50):
        slots.append((1, "fallback_mid", None, r))
    slots.append((2, "head_contrast", "SPECIAL_HEAD", None))
    slots.append((3, "tail_contrast", "SPECIAL_TAIL", None))
    slots.append((4, "llm_generated", grounded_id, None))
    slots.append((5, "sequential_top1", None, 1))
    slots.append((6, "sequential_top5", None, 5))

    chosen: set[str] = set()
    ordered_items: list[tuple[str, list[str]]] = []

    def add_item(item_id: str | None, sources: list[str]) -> None:
        if not item_id or item_id not in cand_set or item_id in chosen:
            return
        chosen.add(item_id)
        ordered_items.append((item_id, sources))

    for prio, tag, special, fb_rank in sorted(slots, key=lambda x: x[0]):
        if len(ordered_items) >= panel_size:
            break
        if special == "SPECIAL_HEAD":
            hid, _ = _pick_head_contrast(candidate_ids, chosen, train_pop, example_id=example_id, seed=seed)
            add_item(hid, [tag])
        elif special == "SPECIAL_TAIL":
            tid, _ = _pick_tail_contrast(candidate_ids, chosen, train_pop, item_buckets, example_id=example_id, seed=seed)
            add_item(tid, [tag])
        elif tag == "llm_generated":
            add_item(grounded_id, [tag] if grounded_id else [])
        elif tag.startswith("sequential"):
            if not seq_ranked:
                continue
            r = fb_rank or 1
            sid = seq_ranked[r - 1] if r <= len(seq_ranked) else None
            add_item(sid, [tag])
        else:
            it = _item_at_fallback_rank(ranked, fb_rank)
            add_item(it, [tag])

    # Truncate to panel_size (already) but ensure exact size by padding from next fallback ranks not in panel
    if len(ordered_items) < panel_size:
        for it in ranked:
            if len(ordered_items) >= panel_size:
                break
            add_item(it, ["fallback_fill"])

    ordered_items = ordered_items[:panel_size]

    panel_items: list[dict[str, Any]] = []
    for item_id, sources in ordered_items:
        fr = ranked.index(item_id) + 1 if item_id in ranked else 999
        panel_items.append(
            {
                "item_id": item_id,
                "title": str(item_titles.get(item_id, "")),
                "genre": str(item_genres.get(item_id, "")),
                "fallback_rank": fr,
                "fallback_score": float(score_by_item.get(item_id, 0.0)),
                "popularity_bucket": str(item_buckets.get(item_id, "unknown")),
                "source": sources,
            }
        )

    return {
        "user_id": user_id,
        "example_id": example_id,
        "panel_items": panel_items,
        "metadata": {
            "candidate_size": len(candidate_ids),
            "panel_size": len(panel_items),
            "seed": seed,
            "max_panel_size_requested": panel_size,
        },
    }


def panel_item_ids(panel: dict[str, Any]) -> set[str]:
    return {str(p["item_id"]) for p in panel.get("panel_items") or []}


def oracle_rerank_top10_metrics(
    *,
    full_fallback_order: list[str],
    panel_ids: set[str],
    target_item: str,
    candidate_items: list[str],
) -> dict[str, float]:
    """Place target at earliest index occupied by any panel item; recompute top-10 metrics vs baseline."""
    F = list(full_fallback_order)
    if not F:
        return {
            "fallback_ndcg_at_10": 0.0,
            "oracle_ndcg_at_10": 0.0,
            "ndcg_gain": 0.0,
            "fallback_recall_at_10": 0.0,
            "oracle_recall_at_10": 0.0,
            "recall_gain": 0.0,
            "target_in_panel": 0.0,
        }
    base = {"target_item": str(target_item), "predicted_items": F[:10], "candidate_items": list(candidate_items)}
    fb_ndcg = _ndcg_at_k(base, 10)
    fb_rec = _recall_at_k(base, 10)
    tip = 1.0 if target_item in panel_ids else 0.0
    if target_item not in panel_ids:
        return {
            "fallback_ndcg_at_10": float(fb_ndcg),
            "oracle_ndcg_at_10": float(fb_ndcg),
            "ndcg_gain": 0.0,
            "fallback_recall_at_10": float(fb_rec),
            "oracle_recall_at_10": float(fb_rec),
            "recall_gain": 0.0,
            "target_in_panel": tip,
        }
    positions = sorted(i for i, v in enumerate(F) if v in panel_ids)
    if not positions:
        return {
            "fallback_ndcg_at_10": float(fb_ndcg),
            "oracle_ndcg_at_10": float(fb_ndcg),
            "ndcg_gain": 0.0,
            "fallback_recall_at_10": float(fb_rec),
            "oracle_recall_at_10": float(fb_rec),
            "recall_gain": 0.0,
            "target_in_panel": tip,
        }
    segment = [F[i] for i in positions]
    new_seg = [target_item] + [x for x in segment if x != target_item]
    f2 = list(F)
    for j, pos in enumerate(positions):
        f2[pos] = new_seg[j]
    oracle = {"target_item": str(target_item), "predicted_items": f2[:10], "candidate_items": list(candidate_items)}
    o_ndcg = _ndcg_at_k(oracle, 10)
    o_rec = _recall_at_k(oracle, 10)
    return {
        "fallback_ndcg_at_10": float(fb_ndcg),
        "oracle_ndcg_at_10": float(o_ndcg),
        "ndcg_gain": float(o_ndcg - fb_ndcg),
        "fallback_recall_at_10": float(fb_rec),
        "oracle_recall_at_10": float(o_rec),
        "recall_gain": float(o_rec - fb_rec),
        "target_in_panel": tip,
    }


def normalized_fallback_scores_in_panel(panel: dict[str, Any]) -> dict[str, float]:
    """Min-max normalize fallback_score over panel items (avoid div0)."""
    items = panel.get("panel_items") or []
    scores = [float(p.get("fallback_score") or 0.0) for p in items]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0
    out: dict[str, float] = {}
    for p in items:
        sid = str(p["item_id"])
        s = float(p.get("fallback_score") or 0.0)
        out[sid] = (s - lo) / span if span else 0.0
    return out
