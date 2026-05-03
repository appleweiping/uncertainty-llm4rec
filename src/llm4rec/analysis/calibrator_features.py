"""Offline feature extraction for CU-GR (Calibrated Uncertainty-Gated Recommendation).

Reads only existing prediction JSONL rows and train-only dataset context.
Target / label-derived quantities must never appear in the feature vector.
"""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any

from llm4rec.metrics.long_tail import assign_popularity_buckets
from llm4rec.metrics.novelty import train_item_popularity
from llm4rec.metrics.ranking import _mrr_at_k, _ndcg_at_k, _recall_at_k, dedupe

R3_RUN_PREFIX = "r3_movielens_1m_real_llm_full_candidate500"
TOP_K_LABEL = 10


def example_id_key(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    eid = meta.get("example_id")
    if eid is not None and str(eid):
        return str(eid)
    return f"{row.get('user_id','')}::{row.get('target_item','')}"


def _token_set(text: str) -> set[str]:
    return {t for t in re.split(r"[^\w]+", text.lower()) if len(t) > 1}


def token_jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def fallback_top_items_scores(
    bm25_row: dict[str, Any],
    *,
    k: int = TOP_K_LABEL,
) -> tuple[list[str], list[float]]:
    """BM25 run: predicted_items and scores are aligned catalog-wide ordering; take top-k by score."""
    items = list(bm25_row.get("predicted_items") or [])
    scores = list(bm25_row.get("scores") or [])
    if len(scores) != len(items):
        scores = list(range(len(items), 0, -1))
    pairs = sorted(zip(items, scores, strict=False), key=lambda x: float(x[1]), reverse=True)
    out_items: list[str] = []
    out_scores: list[float] = []
    seen: set[str] = set()
    for item, score in pairs:
        if item in seen:
            continue
        seen.add(item)
        out_items.append(str(item))
        out_scores.append(float(score))
        if len(out_items) >= k:
            break
    return out_items, out_scores


def rank_in_list(items: list[str], item_id: str | None) -> int | None:
    if not item_id:
        return None
    for i, x in enumerate(items):
        if x == item_id:
            return i + 1
    return None


def build_override_topk(
    fallback_items: list[str],
    grounded_id: str | None,
    candidate_ids: set[str],
    *,
    k: int = TOP_K_LABEL,
) -> list[str]:
    """Hypothesis O: promote grounded item to rank 1 if valid and in candidates; else fallback list."""
    if not grounded_id or grounded_id not in candidate_ids:
        return list(fallback_items[:k])
    F = dedupe(fallback_items)
    g = str(grounded_id)
    rest = [x for x in F if x != g]
    merged = [g] + rest
    return merged[:k]


def metric_row(target: str, predicted: list[str], candidates: list[str]) -> dict[str, Any]:
    base = {
        "target_item": str(target),
        "predicted_items": list(predicted),
        "candidate_items": list(candidates),
    }
    return {
        "recall@10": _recall_at_k(base, TOP_K_LABEL),
        "ndcg@10": _ndcg_at_k(base, TOP_K_LABEL),
        "mrr@10": _mrr_at_k(base, TOP_K_LABEL),
    }


def compute_offline_labels(
    *,
    target_item: str,
    candidate_items: list[str],
    fallback_items_topk: list[str],
    override_items_topk: list[str],
    parse_success: bool,
    grounding_success: bool,
    candidate_adherent: bool,
    hallucination_flag: bool,
) -> dict[str, Any]:
    cand_set = set(candidate_items)
    m_f = metric_row(target_item, fallback_items_topk, candidate_items)
    m_o = metric_row(target_item, override_items_topk, candidate_items)
    delta_r = float(m_o["recall@10"] - m_f["recall@10"])
    delta_n = float(m_o["ndcg@10"] - m_f["ndcg@10"])
    delta_m = float(m_o["mrr@10"] - m_f["mrr@10"])
    override_improves = int(delta_n > 0.0)
    override_hurts = int(delta_n < 0.0)
    override_neutral = int(delta_n == 0.0)
    safe = int(
        delta_n >= 0.0
        and bool(parse_success)
        and bool(grounding_success)
        and bool(candidate_adherent)
        and (not hallucination_flag)
    )
    return {
        "delta_recall_at_10": delta_r,
        "delta_ndcg_at_10": delta_n,
        "delta_mrr_at_10": delta_m,
        "override_improves": override_improves,
        "override_hurts": override_hurts,
        "override_neutral": override_neutral,
        "safe_override": safe,
        "recall_fallback_at_10": m_f["recall@10"],
        "ndcg_fallback_at_10": m_f["ndcg@10"],
        "mrr_fallback_at_10": m_f["mrr@10"],
        "recall_override_at_10": m_o["recall@10"],
        "ndcg_override_at_10": m_o["ndcg@10"],
        "mrr_override_at_10": m_o["mrr@10"],
    }


def _grounding_one_hot(method: str | None) -> dict[str, float]:
    m = (method or "failed").lower()
    keys = ("gr_exact", "gr_normalized", "gr_fuzzy", "gr_failed", "gr_other")
    out = {k: 0.0 for k in keys}
    if "exact" in m:
        out["gr_exact"] = 1.0
    elif "normal" in m:
        out["gr_normalized"] = 1.0
    elif "fuzzy" in m or "token" in m:
        out["gr_fuzzy"] = 1.0
    elif m in ("", "none", "failed", "fail"):
        out["gr_failed"] = 1.0
    else:
        out["gr_other"] = 1.0
    return out


def _pop_bucket_one_hot(bucket: str | None) -> dict[str, float]:
    b = (bucket or "unknown").lower()
    return {
        "pop_bucket_head": 1.0 if b == "head" else 0.0,
        "pop_bucket_mid": 1.0 if b == "mid" else 0.0,
        "pop_bucket_tail": 1.0 if b == "tail" else 0.0,
        "pop_bucket_unknown": 1.0 if b not in ("head", "mid", "tail") else 0.0,
    }


def train_seed_statistics(ours_rows: list[dict[str, Any]]) -> dict[str, float]:
    confs = []
    for row in ours_rows:
        meta = row.get("metadata") or {}
        c = meta.get("confidence")
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            confs.append(float(c))
    mean_c = sum(confs) / len(confs) if confs else 0.0
    return {"mean_confidence_train_seed": mean_c, "n_confidence_train": float(len(confs))}


def extract_features(
    ours_row: dict[str, Any],
    bm25_row: dict[str, Any],
    sequential_row: dict[str, Any] | None,
    *,
    context: dict[str, Any],
    train_stats: dict[str, float],
) -> dict[str, Any]:
    """Return flat numeric features only (no target / no label columns)."""
    meta = dict(ours_row.get("metadata") or {})
    candidates = list(ours_row.get("candidate_items") or [])
    cand_set = set(candidates)
    fb_items, fb_scores = fallback_top_items_scores(bm25_row, k=TOP_K_LABEL)
    g_id = meta.get("grounded_item_id")
    g_str = str(g_id) if g_id not in (None, "") else ""
    parse_success = bool(meta.get("parse_success", False))
    grounding_success = bool(meta.get("grounding_success", False))
    candidate_adherent = bool(meta.get("candidate_adherent", False))
    hallucination_flag = bool(meta.get("is_hallucinated", False))
    confidence = meta.get("confidence")
    llm_conf = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.0
    conf_missing = 1.0 if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) else 0.0
    cn_conf = meta.get("candidate_normalized_confidence")
    cn_conf_v = float(cn_conf) if isinstance(cn_conf, (int, float)) else 0.0
    cn_missing = 1.0 if not isinstance(cn_conf, (int, float)) else 0.0

    mean_tr = float(train_stats.get("mean_confidence_train_seed", 0.0))
    conf_minus_mean = llm_conf - mean_tr
    conf_bucket = min(9, int(llm_conf * 10)) if conf_missing < 0.5 else -1

    gr_score = meta.get("grounding_score")
    grounding_score = float(gr_score) if isinstance(gr_score, (int, float)) else 0.0
    gh = _grounding_one_hot(str(meta.get("grounding_method") or ""))

    in_catalog = 1.0 if bool(meta.get("is_catalog_valid", False)) else 0.0
    in_cand = 1.0 if g_str and g_str in cand_set else 0.0
    adherent = 1.0 if candidate_adherent else 0.0

    bm_rank = rank_in_list(fb_items, g_str) or 999
    bm_score_g = 0.0
    if g_str and bm25_row.get("predicted_items") and bm25_row.get("scores"):
        items_full = list(bm25_row["predicted_items"])
        scores_full = list(bm25_row["scores"])
        score_by_item = {str(a): float(b) for a, b in zip(items_full, scores_full, strict=False)}
        bm_score_g = float(score_by_item.get(g_str, 0.0))
    top1_score = float(fb_scores[0]) if fb_scores else 0.0
    margin_top1_minus_g = top1_score - bm_score_g
    top10_has_g = 1.0 if g_str and g_str in set(fb_items[:TOP_K_LABEL]) else 0.0

    train_pop: dict[str, int] = context.get("train_popularity") or {}
    item_buckets: dict[str, str] = context.get("item_buckets") or {}
    item_titles: dict[str, str] = context.get("item_titles") or {}

    gen_pop = int(train_pop.get(g_str, 0)) if g_str else 0
    gen_pop_log = math.log1p(gen_pop)
    fb_top1 = fb_items[0] if fb_items else ""
    fb_top1_pop = int(train_pop.get(fb_top1, 0)) if fb_top1 else 0
    pop_gap = float(gen_pop - fb_top1_pop)
    gen_bucket = item_buckets.get(g_str, "unknown")
    pb = _pop_bucket_one_hot(gen_bucket)
    gen_novelty = 1.0 / math.log1p(gen_pop + 2.0) if gen_pop >= 0 else 0.0
    tail_flag = 1.0 if gen_bucket == "tail" else 0.0

    hist_ids = list(meta.get("history_item_ids") or [])
    hist_titles_list = list(meta.get("history_titles") or [])
    hist_len = float(len(hist_ids))
    gen_title = str(meta.get("generated_title") or "")
    hist_blob = " ".join(str(t) for t in hist_titles_list)
    hist_overlap = token_jaccard(gen_title, hist_blob)
    cat_ids = [str(i) for i in hist_ids if i is not None]
    item_cat = context.get("item_category") or {}
    gen_cat = item_cat.get(g_str, "")
    hist_cats = {item_cat.get(i, "") for i in cat_ids}
    hist_cats.discard("")
    cat_overlap = 1.0 if gen_cat and gen_cat in hist_cats else 0.0
    hist_sim = meta.get("history_similarity")
    hist_sim_v = float(hist_sim) if isinstance(hist_sim, (int, float)) else 0.0
    seen_hist = 1.0 if g_str and g_str in set(cat_ids) else 0.0
    echo = 1.0 if bool(meta.get("echo_risk", False)) else 0.0

    raw_out = ours_row.get("raw_output")
    raw_len = float(len(raw_out)) if isinstance(raw_out, str) else 0.0
    pred_empty = 1.0 if not (ours_row.get("predicted_items") or []) else 0.0
    pred_dedup = dedupe(list(ours_row.get("predicted_items") or []))
    n_viol = sum(1 for x in pred_dedup if x not in cand_set)
    hall_flag = 1.0 if hallucination_flag else 0.0
    val_flag = 1.0 if pred_dedup and n_viol == 0 else 0.0
    title_tokens = len(_token_set(gen_title))
    exact_title_match = 0.0
    best_fuzzy = 0.0
    if gen_title:
        for cid in candidates[:200]:
            t = item_titles.get(str(cid), "")
            if t and t.strip().lower() == gen_title.strip().lower():
                exact_title_match = 1.0
                break
        for cid in candidates:
            t = item_titles.get(str(cid), "")
            if t:
                best_fuzzy = max(best_fuzzy, SequenceMatcher(None, gen_title.lower(), t.lower()).ratio())

    seq_rank = 999.0
    seq_score_g = 0.0
    seq_top10_has_g = 0.0
    if sequential_row and g_str:
        s_items = list(sequential_row.get("predicted_items") or [])
        s_scores = list(sequential_row.get("scores") or [])
        if len(s_scores) == len(s_items) and s_items:
            order = sorted(zip(s_items, s_scores, strict=False), key=lambda x: float(x[1]), reverse=True)
            top10 = [str(x[0]) for x in order[:TOP_K_LABEL]]
            r = rank_in_list(top10, g_str)
            if r is not None:
                seq_top10_has_g = 1.0
            full_rank = rank_in_list([str(x[0]) for x in order], g_str)
            if full_rank is not None:
                seq_rank = float(full_rank)
            smap = {str(a): float(b) for a, b in zip(s_items, s_scores, strict=False)}
            seq_score_g = float(smap.get(g_str, 0.0))

    features: dict[str, Any] = {
        "llm_confidence": llm_conf,
        "yes_no_confidence": 0.0,
        "candidate_normalized_confidence": cn_conf_v,
        "confidence_missing": conf_missing,
        "confidence_bucket": float(conf_bucket),
        "confidence_minus_mean_confidence_by_method": conf_minus_mean,
        "confidence_rank_among_candidates": 0.0,
        "parse_success": 1.0 if parse_success else 0.0,
        "grounding_success": 1.0 if grounding_success else 0.0,
        "grounding_score": grounding_score,
        "generated_item_in_catalog": in_catalog,
        "generated_item_in_candidates": in_cand,
        "candidate_adherence": adherent,
        "grounded_item_rank_in_fallback": float(min(bm_rank, 999)),
        "grounded_item_is_fallback_top1": 1.0 if bm_rank == 1 else 0.0,
        "grounded_item_is_in_fallback_top10": 1.0 if bm_rank <= TOP_K_LABEL else 0.0,
        "bm25_rank_of_grounded_item": float(min(bm_rank, 999)),
        "bm25_score_of_grounded_item": bm_score_g,
        "fallback_top1_score": top1_score,
        "fallback_score_margin_top1_minus_grounded": margin_top1_minus_g,
        "fallback_top10_contains_generated": top10_has_g,
        "sequential_rank_of_grounded_item": seq_rank,
        "sequential_score_of_grounded_item": seq_score_g,
        "sequential_top10_contains_generated": seq_top10_has_g,
        "generated_item_popularity": float(gen_pop),
        "generated_item_popularity_log": gen_pop_log,
        "fallback_top1_popularity": float(fb_top1_pop),
        "popularity_gap_generated_minus_fallback_top1": pop_gap,
        "generated_item_novelty": gen_novelty,
        "generated_item_tail_flag": tail_flag,
        "user_history_length": hist_len,
        "history_title_overlap_with_generated": hist_overlap,
        "history_category_overlap_with_generated": cat_overlap,
        "history_similarity": hist_sim_v,
        "history_category_entropy": 0.0,
        "generated_item_seen_in_history": seen_hist,
        "echo_risk": echo,
        "raw_output_length": raw_len,
        "parsed_recommendation_empty": pred_empty,
        "number_of_candidate_violations": float(n_viol),
        "hallucination_flag": hall_flag,
        "validity_flag": val_flag,
        "generated_title_token_count": float(title_tokens),
        "generated_title_exact_candidate_match": exact_title_match,
        "generated_title_fuzzy_candidate_match_score": best_fuzzy,
        "sample_consistency_agreement": 0.0,
        "sample_consistency_entropy": 0.0,
        "sample_consistency_unique_grounded_items": 0.0,
        "self_consistency_missing": 1.0,
        "candidate_normalized_confidence_missing": cn_missing,
    }
    features.update(gh)
    features.update(pb)
    return features


def build_dataset_context(processed_dir: str | None = None) -> dict[str, Any]:
    """Train-only popularity and item text (same spirit as ours_error_decomposition)."""
    from pathlib import Path

    import csv

    processed_path = Path(processed_dir or "data/processed/movielens_1m/r2_full_single_dataset")
    items_path = processed_path / "items.csv"
    examples_path = processed_path / "examples.jsonl"
    items: list[dict[str, str]] = []
    if items_path.exists():
        with items_path.open("r", encoding="utf-8", newline="") as handle:
            items = list(csv.DictReader(handle))
    train_examples: list[dict[str, Any]] = []
    if examples_path.exists():
        with examples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if str(row.get("split")) == "train":
                        train_examples.append(row)
    catalog_items = [str(r.get("item_id")) for r in items if r.get("item_id")]
    popularity = train_item_popularity(train_examples)
    item_buckets = assign_popularity_buckets(popularity, catalog_items=catalog_items)
    item_titles = {str(r.get("item_id")): str(r.get("title", "")) for r in items if r.get("item_id")}
    item_category = {str(r.get("item_id")): str(r.get("category", "") or "") for r in items if r.get("item_id")}
    return {
        "train_popularity": popularity,
        "item_buckets": item_buckets,
        "item_titles": item_titles,
        "item_category": item_category,
        "catalog_items": catalog_items,
    }
