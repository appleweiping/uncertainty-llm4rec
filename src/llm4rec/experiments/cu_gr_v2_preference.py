"""CU-GR v2: collect listwise DeepSeek preference signals over local panels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

from llm4rec.analysis.calibrator_features import build_dataset_context, example_id_key
from llm4rec.analysis.ours_error_decomposition import method_run_dir, read_jsonl
from llm4rec.experiments.config import load_config
from llm4rec.llm.base import LLMRequest
from llm4rec.methods.candidate_panel import (
    build_candidate_panel,
    fallback_full_ranking_in_candidates,
    normalized_fallback_scores_in_panel,
    oracle_rerank_top10_metrics,
    panel_item_ids,
)
from llm4rec.methods.preference_fusion import fused_top_k
from llm4rec.metrics.ranking import _ndcg_at_k, dedupe, ranking_metrics
from llm4rec.prompts.preference_parser import parse_listwise_response
from llm4rec.prompts.preference_templates import build_listwise_preference_prompt


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {example_id_key(r): r for r in rows}


def compact_bm25_row_for_signal(bm_row: dict[str, Any]) -> dict[str, Any]:
    """Minimal BM25 row persisted with each preference signal for offline replay."""
    return {
        "user_id": str(bm_row.get("user_id", "")),
        "target_item": str(bm_row.get("target_item", "")),
        "candidate_items": list(bm_row.get("candidate_items") or []),
        "domain": str(bm_row.get("domain", "movies")),
        "metadata": dict(bm_row.get("metadata") or {}),
        "predicted_items": list(bm_row.get("predicted_items") or []),
        "scores": bm_row.get("scores"),
        "method": "bm25",
    }


def _limit_examples(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset = dict(config.get("dataset") or {})
    safety = dict(config.get("safety") or {})
    subset = dict(config.get("subset") or {})
    caps: list[int] = []
    for v in (
        dataset.get("subset_size"),
        safety.get("subset_size"),
        safety.get("max_examples"),
        subset.get("max_examples"),
    ):
        if v not in (None, ""):
            caps.append(int(v))
    return rows[: min(caps)] if caps else rows


def cfg_raw_path(config: dict[str, Any], seed: int) -> Path:
    raw = dict((config.get("llm") or {}).get("raw_outputs") or {})
    path = raw.get("path")
    if path:
        return Path(str(path))
    return Path("outputs/runs") / f"r3_v2_movielens_preference_signal_subgate_seed{seed}_raw_llm_outputs.jsonl"


def _resolve_repo_path(ROOT: Path, p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p)


def _append_raw_global(path: Path, rec: dict[str, Any]) -> None:
    line = {
        "example_id": rec.get("example_id"),
        "raw_output": rec.get("raw_output"),
        "parse_success": rec.get("parse_success"),
        "cache_hit": rec.get("cache_hit"),
        "token_usage": rec.get("token_usage"),
        "latency_seconds": rec.get("latency_seconds"),
        "prompt_hash": rec.get("prompt_hash"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def _min_max_norm(vals: dict[str, float]) -> dict[str, float]:
    if not vals:
        return {}
    xs = list(vals.values())
    lo, hi = min(xs), max(xs)
    span = hi - lo if hi > lo else 1.0
    return {k: float((float(vals[k]) - lo) / span) for k in vals}


def _pop_penalty(ids: list[str], train_pop: dict[str, int]) -> dict[str, float]:
    vals = {i: math.log1p(int(train_pop.get(i, 0))) for i in ids}
    return _min_max_norm(vals)


def _fallback_ordinal_in_panel_bm25(panel: dict[str, Any], target_item: str) -> float:
    """1-based index of target in panel ordering by BM25 candidate rank (among |panel| slots)."""
    rows = sorted(
        list(panel.get("panel_items") or []),
        key=lambda p: (int(p.get("fallback_rank") or 999999), str(p.get("item_id"))),
    )
    ids = [str(p.get("item_id")) for p in rows]
    if target_item not in ids:
        return 999.0
    return float(ids.index(target_item) + 1)


def _ranking_feat(r: dict[str, Any]) -> tuple[float, float, float]:
    """target bm25-derived ordinal within locally ordered panel, llm rank (1=min best), llm_top1_hit."""
    target = str(r.get("target_item", ""))
    panel = r.get("panel") or {}

    tgt_fb = _fallback_ordinal_in_panel_bm25(panel, target)

    tgt_llm_rank = 999.0
    top1_tgt = False
    ranked = sorted(
        list(r.get("parsed_ranking") or []),
        key=lambda z: (-float(z.get("score") or 0), str(z.get("label", ""))),
    )
    if r.get("parse_success") and ranked:
        for i, rr in enumerate(ranked):
            if str(rr.get("item_id")) == target:
                tgt_llm_rank = float(i + 1)
                break
        top1_tgt = str(ranked[0].get("item_id")) == target
    return float(tgt_fb), tgt_llm_rank, 1.0 if top1_tgt else 0.0


def _fusion_top(
    r: dict[str, Any],
    params: dict[str, float],
    train_pop: dict[str, int],
) -> list[str]:
    fb, _ = fallback_full_ranking_in_candidates(r["bm_row"])
    if not r.get("parse_success"):
        return fb[:10]
    panel = r.get("panel") or {}
    panel_ordered = list(r["panel_ordered_item_ids"])
    nfb = normalized_fallback_scores_in_panel(panel)
    ranking_by_item: dict[str, dict[str, float]] = {}
    for rr in r.get("parsed_ranking") or []:
        iid = str(rr.get("item_id", ""))
        ranking_by_item[iid] = {
            "score": float(rr.get("score") or 0.0),
            "confidence": float(rr.get("confidence") or 0.0),
        }
    scores_llm = {i: ranking_by_item.get(i, {}).get("score", 0.0) for i in panel_ordered}
    nll = _min_max_norm(scores_llm)
    pens = _pop_penalty(panel_ordered, train_pop)
    confs = {i: ranking_by_item.get(i, {}).get("confidence", 0.0) for i in panel_ordered}

    fused_order = sorted(
        panel_ordered,
        key=lambda iid: (
            -(
                float(params["alpha"]) * float(nfb.get(iid, 0.0))
                + float(params["beta"]) * float(nll.get(iid, 0.0))
                + float(params["gamma"]) * float(confs.get(iid, 0.0))
                - float(params["lambda"]) * float(pens.get(iid, 0.0))
            ),
            iid,
        ),
    )
    return fused_top_k(fb, panel_ordered, fused_order, k=10)


def _safe_panel_order_llm_gate(
    r: dict[str, Any],
    *,
    margin_th: float,
    conf_th: float,
) -> list[str]:
    panel_ordered = list(r["panel_ordered_item_ids"])
    if not r.get("parse_success"):
        return panel_ordered
    ranking_by_item: dict[str, dict[str, float]] = {}
    for rr in r.get("parsed_ranking") or []:
        ranking_by_item[str(rr.get("item_id", ""))] = {
            "score": float(rr.get("score") or 0.0),
            "confidence": float(rr.get("confidence") or 0.0),
        }

    def sc(iid: str) -> float:
        return float(ranking_by_item.get(iid, {}).get("score", 0.0))

    def cnf(iid: str) -> float:
        return float(ranking_by_item.get(iid, {}).get("confidence", 0.0))

    by_llm = sorted(panel_ordered, key=lambda i: (-sc(i), i))
    if len(by_llm) < 2:
        return by_llm if by_llm else panel_ordered
    top, second = by_llm[0], by_llm[1]
    if sc(top) - sc(second) < margin_th or cnf(top) < conf_th:
        return panel_ordered
    return by_llm


def _safe_llm_panel_top_k(
    r: dict[str, Any],
    margin_th: float,
    conf_th: float,
    *,
    k: int = 10,
) -> list[str]:
    fb, _ = fallback_full_ranking_in_candidates(r["bm_row"])
    panel_ordered = list(r["panel_ordered_item_ids"])
    gated = _safe_panel_order_llm_gate(r, margin_th=margin_th, conf_th=conf_th)
    return fused_top_k(fb, panel_ordered, gated, k=k)


def _llm_panel_top(r: dict[str, Any]) -> list[str]:
    fb, _ = fallback_full_ranking_in_candidates(r["bm_row"])
    if not r.get("parse_success"):
        return fb[:10]
    panel_ordered = list(r["panel_ordered_item_ids"])
    ranking_by_item: dict[str, dict[str, float]] = {}
    for rr in r.get("parsed_ranking") or []:
        iid = str(rr.get("item_id", ""))
        ranking_by_item[iid] = {"score": float(rr.get("score") or 0.0)}
    if not ranking_by_item:
        return fb[:10]
    llm_ord = sorted(panel_ordered, key=lambda i: (-ranking_by_item.get(i, {}).get("score", 0.0), i))
    return fused_top_k(fb, panel_ordered, llm_ord, k=10)


def _fallback_top(r: dict[str, Any]) -> list[str]:
    fb, _ = fallback_full_ranking_in_candidates(r["bm_row"])
    return fb[:10]


def _prediction_row_for_policy(bm_row: dict[str, Any], predicted: list[str], policy: str, *, parse_ok: bool) -> dict[str, Any]:
    return {
        "user_id": str(bm_row.get("user_id", "")),
        "target_item": str(bm_row.get("target_item", "")),
        "candidate_items": list(bm_row.get("candidate_items") or []),
        "predicted_items": dedupe(predicted)[:10],
        "scores": None,
        "method": f"cu_gr_v2::{policy}",
        "domain": str(bm_row.get("domain", "movies")),
        "metadata": {
            "ours_method": False,
            "parse_success": parse_ok,
            "grounding_success": True,
            "grounded_item_id": "",
            "uncertainty_decision": "fallback",
            "confidence": 0.0,
            "cu_gr_v2_policy": policy,
            "fallback_method": "bm25",
            "echo_risk": False,
            "popularity_bucket": "unknown",
            "history_similarity": 0.0,
            "ablation_variant": "full",
            "disabled_components": [],
            "candidate_adherent": True,
            "prompt_template_id": "cu_gr_v2",
            "prompt_hash": "na",
            "is_catalog_valid": True,
            "is_hallucinated": False,
        },
        "raw_output": None,
    }


def _mean_ndcg(subset: list[dict[str, Any]], top_fn: Callable[[dict[str, Any]], list[str]]) -> float:
    if not subset:
        return 0.0
    s = 0.0
    for r in subset:
        pred = top_fn(r)
        row = {"target_item": str(r["target_item"]), "predicted_items": pred, "candidate_items": list(r["bm_row"].get("candidate_items") or [])}
        s += float(_ndcg_at_k(row, 10))
    return s / len(subset)


def _rows_for_ranking_metrics(pack: list[dict[str, Any]], fn: Callable[[dict[str, Any]], list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "target_item": str(r["target_item"]),
            "predicted_items": dedupe(fn(r))[:10],
            "candidate_items": list(r["bm_row"].get("candidate_items") or []),
        }
        for r in pack
    ]


def _evaluate_aggregate(rows_out: list[dict[str, Any]], *, train_pop: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    n = len(rows_out)
    if n == 0:
        return {}, [], {}, {}

    grids: list[dict[str, float]] = []
    alpha_s = [0.5, 0.7, 0.9, 1.0]
    beta_s = [0.1, 0.3, 0.5, 0.7]
    gamma_s = [0.0, 0.1, 0.2]
    lamb_s = [0.0, 0.05, 0.1]
    for a in alpha_s:
        for b in beta_s:
            for g in gamma_s:
                for lam in lamb_s:
                    grids.append({"alpha": a, "beta": b, "gamma": g, "lambda": lam})

    n_train = min(160, max(1, int(n * 0.8)))
    train_pack = [{"idx": i, **rows_out[i]} for i in range(n_train)]

    best_grid = dict(grids[0])
    best_nd = _mean_ndcg(train_pack, lambda r: _fusion_top(r, best_grid, train_pop))
    for params in grids[1:]:
        p_copy = dict(params)
        m = _mean_ndcg(train_pack, lambda r, p=p_copy: _fusion_top(r, p, train_pop))
        if m > best_nd:
            best_nd = m
            best_grid = dict(params)

    full_pack = [{"idx": i, **rows_out[i]} for i in range(n)]

    safe_margins = [0.0, 0.03, 0.05, 0.08, 0.1]
    safe_confs = [0.5, 0.55, 0.6, 0.65, 0.7]
    best_safe: tuple[float, float, float] | None = None
    swap_rows_diag: list[dict[str, Any]] = []

    def _unsafe_metrics(margin: float, conf: float) -> dict[str, Any]:
        benef = harmful = neut = 0
        for r in full_pack:
            tg = str(r["target_item"])
            bf = _fallback_top(r)
            sw = _safe_llm_panel_top_k(r, margin, conf, k=10)
            nf = float(_ndcg_at_k({"target_item": tg, "predicted_items": bf, "candidate_items": list(r["bm_row"].get("candidate_items") or [])}, 10))
            nc = float(_ndcg_at_k({"target_item": tg, "predicted_items": sw, "candidate_items": list(r["bm_row"].get("candidate_items") or [])}, 10))
            if nc > nf + 1e-14:
                benef += 1
            elif nc < nf - 1e-14:
                harmful += 1
            else:
                neut += 1
        mean_fb = ranking_metrics(_rows_for_ranking_metrics(full_pack, _fallback_top), top_k=[10])
        mean_sw = ranking_metrics(_rows_for_ranking_metrics(full_pack, lambda rr, m=margin, c=conf: _safe_llm_panel_top_k(rr, m, c, k=10)), top_k=[10])
        return {
            "margin": margin,
            "confidence_min": conf,
            "beneficial_swaps": benef,
            "harmful_swaps": harmful,
            "neutral_swaps": neut,
            "harmful_swap_rate": harmful / max(n, 1),
            "delta_ndcg_10_vs_fallback": round(float(mean_sw.get("ndcg@10") or 0.0) - float(mean_fb.get("ndcg@10") or 0.0), 8),
            "mean_ndcg_safe": round(float(mean_sw.get("ndcg@10") or 0.0), 8),
            "mean_ndcg_fallback": round(float(mean_fb.get("ndcg@10") or 0.0), 8),
        }

    for margin in safe_margins:
        for conf in safe_confs:
            diag = _unsafe_metrics(margin, conf)
            swap_rows_diag.append(diag)
            nd = float(diag["delta_ndcg_10_vs_fallback"])
            hr = float(diag["harmful_swap_rate"])
            if hr <= 0.08 and (best_safe is None or nd > best_safe[2]):
                best_safe = (margin, conf, nd)

    if best_safe is None and swap_rows_diag:
        diag0 = swap_rows_diag[0]
        best_safe = (
            float(diag0["margin"]),
            float(diag0["confidence_min"]),
            float(diag0["delta_ndcg_10_vs_fallback"]),
        )
    elif best_safe is None:
        best_safe = (safe_margins[0], safe_confs[0], 0.0)

    best_margin, best_confidence, best_delta_chk = best_safe
    fused_fn = lambda r, bg=best_grid: _fusion_top(r, bg, train_pop)

    agg: dict[str, Any] = {
        "mean_ndcg_fallback_full": round(_mean_ndcg(full_pack, _fallback_top), 8),
        "mean_ndcg_llm_listwise_full": round(_mean_ndcg(full_pack, _llm_panel_top), 8),
        "mean_ndcg_best_fusion_full": round(_mean_ndcg(full_pack, fused_fn), 8),
        "mean_ndcg_fallback_train_subset": round(_mean_ndcg(train_pack, _fallback_top), 8),
        "mean_ndcg_llm_train_subset": round(_mean_ndcg(train_pack, _llm_panel_top), 8),
        "mean_ndcg_best_fusion_train_subset": round(best_nd, 8),
        "best_fusion_params_train": best_grid,
        "safe_swap_best_margin_diag": best_margin,
        "safe_swap_best_confidence_diag": best_confidence,
        "safe_swap_best_delta_ndcg_floor": round(float(best_delta_chk), 8),
    }

    m_fb = ranking_metrics(_rows_for_ranking_metrics(full_pack, _fallback_top), top_k=[10])
    m_llm = ranking_metrics(_rows_for_ranking_metrics(full_pack, _llm_panel_top), top_k=[10])
    m_fu = ranking_metrics(_rows_for_ranking_metrics(full_pack, fused_fn), top_k=[10])
    m_safe = ranking_metrics(
        _rows_for_ranking_metrics(full_pack, lambda rr, bm=best_margin, bc=best_confidence: _safe_llm_panel_top_k(rr, bm, bc, k=10)),
        top_k=[10],
    )

    for src_name, blob in (
        ("fallback", m_fb),
        ("llm_listwise_panel", m_llm),
        ("fusion_train_best", m_fu),
        ("safe_llm_gate", m_safe),
    ):
        for k, v in blob.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, float):
                agg[f"{src_name}_{k}"] = round(float(v), 8)
            elif isinstance(v, int):
                agg[f"{src_name}_{k}"] = v

    agg["fusion_delta_ndcg10_vs_fallback"] = round(float(m_fu.get("ndcg@10") or 0.0) - float(m_fb.get("ndcg@10") or 0.0), 8)
    agg["llm_delta_ndcg10_vs_fallback"] = round(float(m_llm.get("ndcg@10") or 0.0) - float(m_fb.get("ndcg@10") or 0.0), 8)

    benef = harmful = neut = 0
    tgt_in_panel_total = llm_avg_rank_when_in_panel = fb_avg_rank_when_in_panel = llm_top1_hits = 0
    denom_panel = denom_parse = 0

    for r in full_pack:
        tg = str(r["target_item"])
        in_panel = float(tg in panel_item_ids(r.get("panel") or {}))

        tg_fb, tg_llm_rank, llm_hit1 = _ranking_feat(r)
        tgt_in_panel_total += in_panel

        bf = _fallback_top(r)
        fx = fused_fn(r)
        nf = float(_ndcg_at_k({"target_item": tg, "predicted_items": bf, "candidate_items": list(r["bm_row"].get("candidate_items") or [])}, 10))
        nc = float(_ndcg_at_k({"target_item": tg, "predicted_items": fx, "candidate_items": list(r["bm_row"].get("candidate_items") or [])}, 10))
        if nc > nf + 1e-14:
            benef += 1
        elif nc < nf - 1e-14:
            harmful += 1
        else:
            neut += 1

        if tg in panel_item_ids(r.get("panel") or {}) and r.get("parse_success"):
            denom_panel += 1
            llm_avg_rank_when_in_panel += tg_llm_rank
            fb_avg_rank_when_in_panel += _fallback_ordinal_in_panel_bm25(r.get("panel") or {}, tg)
            llm_top1_hits += llm_hit1
        if r.get("parse_success"):
            denom_parse += 1

    agg["target_in_panel_rate"] = round(tgt_in_panel_total / max(n, 1), 6)
    agg["beneficial_ndcg_swaps_vs_fallback_fusion"] = benef
    agg["harmful_ndcg_swaps_vs_fallback_fusion"] = harmful
    agg["neutral_ndcg_swaps_vs_fallback_fusion"] = neut
    agg["harmful_swap_rate"] = round(harmful / max(n, 1), 6)
    agg["mean_llm_target_rank_when_target_in_panel"] = (
        round(llm_avg_rank_when_in_panel / max(denom_panel, 1), 6) if denom_panel else float("nan")
    )
    agg["mean_fallback_target_rank_within_panel_when_in_panel"] = (
        round(fb_avg_rank_when_in_panel / max(denom_panel, 1), 6) if denom_panel else float("nan")
    )
    agg["llm_panel_top1_hit_target_rate_when_in_panel"] = (
        round(llm_top1_hits / max(denom_panel, 1), 6) if denom_panel else float("nan")
    )

    llm_vs_fb_rank_delta = agg["mean_fallback_target_rank_within_panel_when_in_panel"] - agg["mean_llm_target_rank_when_target_in_panel"]

    diagnostics = {"swap_analysis_rows": swap_rows_diag, "llm_mean_rank_minus_fallback_mean_rank_in_panel_neg_is_better_for_llm": llm_vs_fb_rank_delta}

    primary_preds = [
        _prediction_row_for_policy(r["bm_row"], fused_fn(r), "fusion_train_best", parse_ok=bool(r.get("parse_success")))
        for r in full_pack
    ]
    model_payload = {"best_fusion_params": best_grid, "train_subset_fraction": round(n_train / n, 4), "n_examples": n}
    return agg, primary_preds, model_payload, diagnostics


FUSION_FIXED_PARAMS = {"alpha": 0.7, "beta": 0.3, "gamma": 0.1, "lambda": 0.05}
FUSION_GRID = [
    {"alpha": a, "beta": b, "gamma": g, "lambda": lam}
    for a in [0.5, 0.7, 0.9, 1.0]
    for b in [0.1, 0.3, 0.5, 0.7]
    for g in [0.0, 0.1, 0.2]
    for lam in [0.0, 0.05, 0.1]
]
SAFE_MARGIN_GRID = [0.0, 0.03, 0.05, 0.08, 0.1]
SAFE_CONFIDENCE_GRID = [0.5, 0.55, 0.6, 0.65, 0.7]


def _pack(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"idx": i, **row} for i, row in enumerate(rows)]


def _metric_rows(pack: list[dict[str, Any]], top_fn: Callable[[dict[str, Any]], list[str]]) -> list[dict[str, Any]]:
    return _rows_for_ranking_metrics(pack, top_fn)


def _policy_metrics(pack: list[dict[str, Any]], top_fn: Callable[[dict[str, Any]], list[str]]) -> dict[str, Any]:
    return ranking_metrics(_metric_rows(pack, top_fn), top_k=[10])


def _ndcg_delta_and_swaps(
    pack: list[dict[str, Any]],
    top_fn: Callable[[dict[str, Any]], list[str]],
) -> dict[str, Any]:
    beneficial = harmful = neutral = changed = 0
    delta_sum = 0.0
    for row in pack:
        target = str(row["target_item"])
        cand = list(row["bm_row"].get("candidate_items") or [])
        base = _fallback_top(row)
        pred = dedupe(top_fn(row))[:10]
        if pred != dedupe(base)[:10]:
            changed += 1
        base_nd = float(_ndcg_at_k({"target_item": target, "predicted_items": base, "candidate_items": cand}, 10))
        pred_nd = float(_ndcg_at_k({"target_item": target, "predicted_items": pred, "candidate_items": cand}, 10))
        delta = pred_nd - base_nd
        delta_sum += delta
        if delta > 1e-14:
            beneficial += 1
        elif delta < -1e-14:
            harmful += 1
        else:
            neutral += 1
    n = max(len(pack), 1)
    return {
        "beneficial_swaps": beneficial,
        "harmful_swaps": harmful,
        "neutral_swaps": neutral,
        "harmful_swap_rate": harmful / n,
        "avg_ndcg_delta_per_example": delta_sum / n,
        "top10_changed_rate": changed / n,
    }


def _select_fusion_weights(
    *,
    train_pack: list[dict[str, Any]],
    val_pack: list[dict[str, Any]],
    train_pop: dict[str, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for params in FUSION_GRID:
        fn = lambda r, p=dict(params): _fusion_top(r, p, train_pop)
        train_m = _policy_metrics(train_pack, fn) if train_pack else {}
        val_m = _policy_metrics(val_pack, fn) if val_pack else {}
        swaps = _ndcg_delta_and_swaps(val_pack, fn) if val_pack else {}
        rows.append(
            {
                **params,
                "train_ndcg@10": float(train_m.get("ndcg@10") or 0.0),
                "validation_ndcg@10": float(val_m.get("ndcg@10") or 0.0),
                "validation_harmful_swap_rate": float(swaps.get("harmful_swap_rate") or 0.0),
                "constraint_met": float(swaps.get("harmful_swap_rate") or 0.0) <= 0.05,
            }
        )
    eligible = [row for row in rows if row["constraint_met"]]
    pool = eligible if eligible else rows
    best = max(
        pool,
        key=lambda row: (
            float(row["validation_ndcg@10"]),
            -float(row["validation_harmful_swap_rate"]),
            -float(row["alpha"]),
            -float(row["beta"]),
            -float(row["gamma"]),
            -float(row["lambda"]),
        ),
    )
    return {"selected": best, "grid_rows": rows, "constraint_satisfied": bool(eligible)}


def _select_safe_thresholds(*, val_pack: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for margin in SAFE_MARGIN_GRID:
        for conf in SAFE_CONFIDENCE_GRID:
            fn = lambda r, m=margin, c=conf: _safe_llm_panel_top_k(r, m, c, k=10)
            met = _policy_metrics(val_pack, fn) if val_pack else {}
            swaps = _ndcg_delta_and_swaps(val_pack, fn) if val_pack else {}
            rows.append(
                {
                    "margin": margin,
                    "confidence_min": conf,
                    "validation_ndcg@10": float(met.get("ndcg@10") or 0.0),
                    "validation_harmful_swap_rate": float(swaps.get("harmful_swap_rate") or 0.0),
                    "constraint_met": float(swaps.get("harmful_swap_rate") or 0.0) <= 0.05,
                }
            )
    eligible = [row for row in rows if row["constraint_met"]]
    pool = eligible if eligible else rows
    best = max(
        pool,
        key=lambda row: (
            float(row["validation_ndcg@10"]),
            -float(row["validation_harmful_swap_rate"]),
            -float(row["margin"]),
            -float(row["confidence_min"]),
        ),
    )
    return {"selected": best, "grid_rows": rows, "constraint_satisfied": bool(eligible)}


def _panel_stats(pack: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(pack), 1)
    in_panel = 0
    fb_rank_sum = 0.0
    llm_rank_sum = 0.0
    llm_top1 = 0
    denom_rank = 0
    oracle_ndcg_sum = 0.0
    for row in pack:
        target = str(row["target_item"])
        panel_ids = panel_item_ids(row.get("panel") or {})
        fb_order, _ = fallback_full_ranking_in_candidates(row["bm_row"])
        oracle = oracle_rerank_top10_metrics(
            full_fallback_order=fb_order,
            panel_ids=panel_ids,
            target_item=target,
            candidate_items=list(row["bm_row"].get("candidate_items") or []),
        )
        oracle_ndcg_sum += float(oracle.get("oracle_ndcg_at_10") or 0.0)
        if target not in panel_ids:
            continue
        in_panel += 1
        if row.get("parse_success"):
            fb_rank, llm_rank, top1 = _ranking_feat(row)
            fb_rank_sum += fb_rank
            llm_rank_sum += llm_rank
            llm_top1 += int(top1 > 0)
            denom_rank += 1
    return {
        "target_in_panel_rate": in_panel / n,
        "fallback_target_rank_in_panel": fb_rank_sum / denom_rank if denom_rank else "",
        "llm_target_rank_in_panel": llm_rank_sum / denom_rank if denom_rank else "",
        "llm_top1_panel_hit_rate": llm_top1 / denom_rank if denom_rank else "",
        "panel_oracle_ndcg_upper_bound": oracle_ndcg_sum / n,
    }


def _parser_stats(pack: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(pack), 1)
    parse_ok = sum(1 for row in pack if row.get("parse_success"))
    invalid = sum(1 for row in pack if int(row.get("invalid_label_incident") or 0) > 0)
    dup = sum(1 for row in pack if int(row.get("duplicate_label_incident") or 0) > 0)
    partial = 0
    confidence_values: list[float] = []
    top1_conf: list[float] = []
    top1_correct: list[float] = []
    for row in pack:
        if not row.get("parse_success"):
            continue
        ranking = list(row.get("parsed_ranking") or [])
        if len(ranking) < int(row.get("panel_size") or 0):
            partial += 1
        for rr in ranking:
            confidence_values.append(float(rr.get("confidence") or 0.0))
        if ranking:
            ordered = sorted(ranking, key=lambda rr: (-float(rr.get("score") or 0.0), str(rr.get("label") or "")))
            top = ordered[0]
            top1_conf.append(float(top.get("confidence") or 0.0))
            top1_correct.append(1.0 if str(top.get("item_id")) == str(row.get("target_item")) else 0.0)
    mean_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    top1_acc = sum(top1_correct) / len(top1_correct) if top1_correct else ""
    top1_conf_mean = sum(top1_conf) / len(top1_conf) if top1_conf else ""
    return {
        "parse_success_count": parse_ok,
        "parser_success_rate": parse_ok / n,
        "invalid_label_rate": invalid / n,
        "duplicate_label_rate": dup / n,
        "partial_ranking_rate": partial / n,
        "confidence_mean": mean_conf,
        "top1_confidence_mean": top1_conf_mean,
        "top1_panel_correct_rate": top1_acc,
    }


def _reference_metrics_for_seed(
    *,
    ROOT: Path,
    seed: int,
    method: str,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    path = method_run_dir(ROOT / "outputs" / "runs", method, seed) / "predictions.jsonl"
    if not path.exists():
        return None
    ref_by = index_rows(read_jsonl(path))
    rows = []
    for source in source_rows:
        rid = example_id_key(source["bm_row"])
        ref = ref_by.get(rid)
        if not ref:
            continue
        rows.append(
            {
                "target_item": str(ref.get("target_item")),
                "predicted_items": list(ref.get("predicted_items") or []),
                "candidate_items": list(ref.get("candidate_items") or []),
            }
        )
    if not rows:
        return None
    return ranking_metrics(rows, top_k=[10])


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_full_seed_outputs(
    *,
    ROOT: Path,
    run_name: str,
    rows_by_seed: dict[int, list[dict[str, Any]]],
    provider_summaries_by_seed: dict[int, dict[str, Any]],
    train_pop: dict[str, int],
) -> dict[str, Any]:
    out_dir = ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    packs_by_seed = {seed: _pack(rows) for seed, rows in rows_by_seed.items()}
    all_pack = [row for seed in sorted(packs_by_seed) for row in packs_by_seed[seed]]
    train_pack = packs_by_seed.get(13, [])
    val_pack = packs_by_seed.get(21, [])
    test_pack = packs_by_seed.get(42, [])

    fusion_choice = _select_fusion_weights(train_pack=train_pack, val_pack=val_pack, train_pop=train_pop)
    selected = dict(fusion_choice["selected"])
    selected_params = {k: float(selected[k]) for k in ("alpha", "beta", "gamma", "lambda")}
    safe_choice = _select_safe_thresholds(val_pack=val_pack)
    safe_selected = dict(safe_choice["selected"])
    safe_margin = float(safe_selected["margin"])
    safe_conf = float(safe_selected["confidence_min"])

    policy_top_fns: dict[str, Callable[[dict[str, Any]], list[str]]] = {
        "fallback_only": _fallback_top,
        "llm_listwise_panel": _llm_panel_top,
        "fusion_fixed_grid": lambda r: _fusion_top(r, FUSION_FIXED_PARAMS, train_pop),
        "fusion_train_best": lambda r: _fusion_top(r, selected_params, train_pop),
        "safe_fusion": lambda r: _safe_llm_panel_top_k(r, safe_margin, safe_conf, k=10),
    }

    by_seed_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    parser_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for seed, pack in sorted(packs_by_seed.items()):
        parser_rows.append({"seed": seed, **_parser_stats(pack)})
        panel_rows.append({"seed": seed, **_panel_stats(pack)})
        for policy, fn in policy_top_fns.items():
            metrics = _policy_metrics(pack, fn)
            swaps = _ndcg_delta_and_swaps(pack, fn)
            by_seed_rows.append(
                {
                    "seed": seed,
                    "method": policy,
                    "Recall@10": metrics.get("recall@10"),
                    "NDCG@10": metrics.get("ndcg@10"),
                    "MRR@10": metrics.get("mrr@10"),
                    "HitRate@10": metrics.get("hit_rate@10"),
                    "evaluated_count": metrics.get("evaluated_count"),
                }
            )
            swap_rows.append({"seed": seed, "method": policy, **swaps})

        for ref_method, label in (
            ("bm25", "BM25_reference"),
            ("sequential_markov", "sequential_markov_reference"),
            ("popularity", "popularity_reference"),
            ("ours_uncertainty_guided_real", "R3_Ours_v1_reference"),
        ):
            ref_metrics = _reference_metrics_for_seed(ROOT=ROOT, seed=seed, method=ref_method, source_rows=pack)
            if ref_metrics is None:
                continue
            by_seed_rows.append(
                {
                    "seed": seed,
                    "method": label,
                    "Recall@10": ref_metrics.get("recall@10"),
                    "NDCG@10": ref_metrics.get("ndcg@10"),
                    "MRR@10": ref_metrics.get("mrr@10"),
                    "HitRate@10": ref_metrics.get("hit_rate@10"),
                    "evaluated_count": ref_metrics.get("evaluated_count"),
                }
            )

    parser_rows.append({"seed": "aggregate", **_parser_stats(all_pack)})
    panel_rows.append({"seed": "aggregate", **_panel_stats(all_pack)})
    for policy, fn in policy_top_fns.items():
        metrics = _policy_metrics(all_pack, fn)
        swaps = _ndcg_delta_and_swaps(all_pack, fn)
        main_rows.append(
            {
                "seed": "aggregate",
                "method": policy,
                "Recall@10": metrics.get("recall@10"),
                "NDCG@10": metrics.get("ndcg@10"),
                "MRR@10": metrics.get("mrr@10"),
                "HitRate@10": metrics.get("hit_rate@10"),
                "delta_NDCG@10_vs_fallback": float(metrics.get("ndcg@10") or 0.0)
                - float(_policy_metrics(all_pack, _fallback_top).get("ndcg@10") or 0.0),
                "evaluated_count": metrics.get("evaluated_count"),
            }
        )
        swap_rows.append({"seed": "aggregate", "method": policy, **swaps})

    seed42_metrics = {policy: _policy_metrics(test_pack, fn) for policy, fn in policy_top_fns.items()} if test_pack else {}
    seed42_swaps = _ndcg_delta_and_swaps(test_pack, policy_top_fns["fusion_train_best"]) if test_pack else {}
    for policy, metrics in seed42_metrics.items():
        main_rows.append(
            {
                "seed": 42,
                "method": policy,
                "Recall@10": metrics.get("recall@10"),
                "NDCG@10": metrics.get("ndcg@10"),
                "MRR@10": metrics.get("mrr@10"),
                "HitRate@10": metrics.get("hit_rate@10"),
                "delta_NDCG@10_vs_fallback": float(metrics.get("ndcg@10") or 0.0)
                - float(seed42_metrics.get("fallback_only", {}).get("ndcg@10") or 0.0),
                "evaluated_count": metrics.get("evaluated_count"),
            }
        )

    val_metrics_selected = _policy_metrics(val_pack, policy_top_fns["fusion_train_best"]) if val_pack else {}
    test_metrics_selected = seed42_metrics.get("fusion_train_best", {})
    r3_v1_seed42 = _reference_metrics_for_seed(ROOT=ROOT, seed=42, method="ours_uncertainty_guided_real", source_rows=test_pack) if test_pack else None
    weights_rows = [
        {
            "policy": "fusion_train_best",
            "selected_on": "seed21_validation",
            "trained_on": "seed13",
            "tested_on": "seed42",
            **selected_params,
            "train_seed13_NDCG@10": selected.get("train_ndcg@10"),
            "validation_seed21_NDCG@10": val_metrics_selected.get("ndcg@10"),
            "validation_seed21_harmful_swap_rate": selected.get("validation_harmful_swap_rate"),
            "test_seed42_NDCG@10": test_metrics_selected.get("ndcg@10"),
            "test_seed42_delta_vs_fallback_NDCG@10": float(test_metrics_selected.get("ndcg@10") or 0.0)
            - float(seed42_metrics.get("fallback_only", {}).get("ndcg@10") or 0.0),
            "test_seed42_delta_vs_llm_listwise_NDCG@10": float(test_metrics_selected.get("ndcg@10") or 0.0)
            - float(seed42_metrics.get("llm_listwise_panel", {}).get("ndcg@10") or 0.0),
            "test_seed42_delta_vs_R3_Ours_v1_NDCG@10": (
                float(test_metrics_selected.get("ndcg@10") or 0.0) - float(r3_v1_seed42.get("ndcg@10") or 0.0)
                if r3_v1_seed42
                else ""
            ),
            "grid_constraint_satisfied": fusion_choice["constraint_satisfied"],
        },
        {
            "policy": "safe_fusion",
            "selected_on": "seed21_validation",
            "trained_on": "",
            "tested_on": "seed42",
            "alpha": "",
            "beta": "",
            "gamma": "",
            "lambda": "",
            "margin": safe_margin,
            "confidence_min": safe_conf,
            "validation_seed21_NDCG@10": _policy_metrics(val_pack, policy_top_fns["safe_fusion"]).get("ndcg@10") if val_pack else "",
            "validation_seed21_harmful_swap_rate": safe_selected.get("validation_harmful_swap_rate"),
            "test_seed42_NDCG@10": seed42_metrics.get("safe_fusion", {}).get("ndcg@10"),
            "grid_constraint_satisfied": safe_choice["constraint_satisfied"],
        },
    ]

    for row in test_pack:
        target = str(row["target_item"])
        cand = list(row["bm_row"].get("candidate_items") or [])
        base = _fallback_top(row)
        pred = policy_top_fns["fusion_train_best"](row)
        base_nd = float(_ndcg_at_k({"target_item": target, "predicted_items": base, "candidate_items": cand}, 10))
        pred_nd = float(_ndcg_at_k({"target_item": target, "predicted_items": pred, "candidate_items": cand}, 10))
        if row.get("parse_success") and pred_nd >= base_nd - 1e-14:
            continue
        failure_rows.append(
            {
                "seed": 42,
                "example_id": row.get("example_id"),
                "user_id": row.get("user_id"),
                "target_item": target,
                "parse_success": row.get("parse_success"),
                "parse_error": row.get("parse_error") or "",
                "fallback_top10": json.dumps(base, ensure_ascii=False),
                "fusion_top10": json.dumps(pred, ensure_ascii=False),
                "fallback_ndcg@10": base_nd,
                "fusion_ndcg@10": pred_nd,
                "delta_ndcg@10": pred_nd - base_nd,
                "raw_output_excerpt": str(row.get("raw_output") or "")[:500],
            }
        )
    failure_rows.sort(key=lambda row: float(row.get("delta_ndcg@10") or 0.0))

    cost_rows: list[dict[str, Any]] = []
    total_requests = total_cache_hits = total_tokens = 0
    total_cost = 0.0
    p50s: list[float] = []
    p95s: list[float] = []
    for seed, summary in sorted(provider_summaries_by_seed.items()):
        total_requests += int(summary.get("real_request_count") or 0)
        total_cache_hits += int(summary.get("cache_hit_count") or 0)
        total_tokens += int(summary.get("total_tokens") or 0)
        total_cost += float(summary.get("effective_cost_usd") or summary.get("estimated_cost") or 0.0)
        p50 = float(summary.get("latency_p50_seconds") or summary.get("latency_p50") or 0.0)
        p95 = float(summary.get("latency_p95_seconds") or summary.get("latency_p95") or 0.0)
        p50s.append(p50)
        p95s.append(p95)
        cost_rows.append(
            {
                "seed": seed,
                "live_requests": summary.get("real_request_count", 0),
                "cache_hits": summary.get("cache_hit_count", 0),
                "total_tokens": summary.get("total_tokens", 0),
                "effective_cost_usd": summary.get("effective_cost_usd", summary.get("estimated_cost", 0.0)),
                "p50_latency_seconds": p50,
                "p95_latency_seconds": p95,
                "retry_count": summary.get("retry_count", 0),
                "timeout_count": summary.get("timeout_count", 0),
                "rate_limit_429_count": summary.get("rate_limit_429_count", 0),
            }
        )
    cost_rows.append(
        {
            "seed": "aggregate",
            "live_requests": total_requests,
            "cache_hits": total_cache_hits,
            "total_tokens": total_tokens,
            "effective_cost_usd": total_cost,
            "p50_latency_seconds": sum(p50s) / len(p50s) if p50s else 0.0,
            "p95_latency_seconds": sum(p95s) / len(p95s) if p95s else 0.0,
            "retry_count": 0,
            "timeout_count": 0,
            "rate_limit_429_count": 0,
        }
    )

    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_main.csv", main_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_by_seed.csv", by_seed_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_fusion_weights.csv", weights_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_swap_analysis.csv", swap_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_parser_stats.csv", parser_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_panel_coverage.csv", panel_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_cost_latency.csv", cost_rows)
    _write_csv_rows(out_dir / "cu_gr_v2_full_seed_failure_cases.csv", failure_rows[:100])

    heldout_delta = float(test_metrics_selected.get("ndcg@10") or 0.0) - float(seed42_metrics.get("fallback_only", {}).get("ndcg@10") or 0.0)
    success = bool(
        test_pack
        and heldout_delta > 0.01
        and float(seed42_swaps.get("harmful_swap_rate") or 1.0) <= 0.05
        and float(_parser_stats(test_pack).get("parser_success_rate") or 0.0) >= 0.95
        and float(_panel_stats(test_pack).get("target_in_panel_rate") or 0.0) > 0.0
    )
    verdict = "PASS" if success else "MAJOR FIXES REQUIRED"
    interpretation = "method story revived" if success else "preference signal promising but not validated"
    if seed42_metrics and float(seed42_metrics.get("llm_listwise_panel", {}).get("ndcg@10") or 0.0) <= float(seed42_metrics.get("fallback_only", {}).get("ndcg@10") or 0.0):
        interpretation = "observation-first remains primary"

    report = ROOT / "docs" / "cu_gr_v2_full_seed_report.md"
    report.write_text(
        "\n".join(
            [
                "# CU-GR v2 Full Seed Report",
                "",
                f"Run name: `{run_name}`",
                "",
                "## Verdict",
                "",
                f"{verdict}. Held-out seed42 fusion_train_best NDCG@10 delta vs fallback = {heldout_delta:.8f}.",
                "",
                "## Provider",
                "",
                "DeepSeek OpenAI-compatible API, model `deepseek-v4-flash`; cache/resume enabled. API key value was not inspected or written.",
                "",
                "## Dataset / Protocol",
                "",
                "MovieLens R3 protocol, processed directory `data/processed/movielens_1m/r2_full_single_dataset`, sampled candidate_size=500, include_target=true, panel_size=15, subset_size=200 per seed.",
                "",
                "## Fusion Weights",
                "",
                f"Selected alpha={selected_params['alpha']}, beta={selected_params['beta']}, gamma={selected_params['gamma']}, lambda={selected_params['lambda']} on seed21 validation. Safe fusion thresholds: margin={safe_margin}, confidence_min={safe_conf}.",
                "",
                "## Held-out Seed42",
                "",
                f"fallback_only NDCG@10={float(seed42_metrics.get('fallback_only', {}).get('ndcg@10') or 0.0):.8f}; "
                f"llm_listwise_panel NDCG@10={float(seed42_metrics.get('llm_listwise_panel', {}).get('ndcg@10') or 0.0):.8f}; "
                f"fusion_train_best NDCG@10={float(test_metrics_selected.get('ndcg@10') or 0.0):.8f}; "
                f"harmful_swap_rate={float(seed42_swaps.get('harmful_swap_rate') or 0.0):.6f}.",
                "",
                "## Interpretation",
                "",
                interpretation + ".",
                "",
                "## Artifacts",
                "",
                "- `outputs/tables/cu_gr_v2_full_seed_main.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_by_seed.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_fusion_weights.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_swap_analysis.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_parser_stats.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_panel_coverage.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_cost_latency.csv`",
                "- `outputs/tables/cu_gr_v2_full_seed_failure_cases.csv`",
                "",
                "## Next Recommended Action",
                "",
                "Run CU-GR v2 on a second dataset/domain." if success else "Refine fusion model.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_out = ROOT / "outputs" / "models" / "cu_gr_v2_preference_fusion"
    model_out.mkdir(parents=True, exist_ok=True)
    (model_out / "model.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "selected_fusion_params": selected_params,
                "selected_on": "seed21_validation",
                "trained_on": "seed13",
                "tested_on": "seed42",
                "safe_fusion": {"margin": safe_margin, "confidence_min": safe_conf},
                "success_criteria_pass": success,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "verdict": verdict,
        "success_criteria_pass": success,
        "selected_fusion_params": selected_params,
        "safe_fusion": {"margin": safe_margin, "confidence_min": safe_conf},
        "heldout_seed42_delta_ndcg": heldout_delta,
        "report": str(report),
    }


def _write_metric_csv(path: Path, flat: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in flat.items() if isinstance(v, (str, int, float, bool)) or v is None}
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _write_latency_csv(path: Path, *, seed: int, summary: dict[str, Any], source_run: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "seed": seed,
        "source_run_dir": source_run,
        "real_requests": summary.get("real_request_count", ""),
        "cache_hits": summary.get("cache_hit_count", ""),
        "cache_hit_rate": summary.get("cache_hit_rate", ""),
        "prompt_tokens": summary.get("prompt_tokens", ""),
        "completion_tokens": summary.get("completion_tokens", ""),
        "total_tokens": summary.get("total_tokens", ""),
        "estimated_cost_usd": summary.get("estimated_cost") or summary.get("effective_cost_usd", ""),
        "latency_p50_seconds": summary.get("latency_p50_seconds", summary.get("latency_p50")),
        "latency_p95_seconds": summary.get("latency_p95_seconds", summary.get("latency_p95")),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def _write_preference_tables(
    *,
    ROOT: Path,
    seed: int,
    run_dir: Path,
    rows_packed: list[dict[str, Any]],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    summary: dict[str, Any],
    model_payload: dict[str, Any],
) -> dict[str, str]:
    out_dir = ROOT / Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    fusion_path = out_dir / "cu_gr_v2_fusion_results.csv"
    vs_path = out_dir / "cu_gr_v2_vs_fallback.csv"
    m_fb_nd = float(metrics.get("fallback_ndcg@10") or 0.0)
    m_llm_nd = float(metrics.get("llm_listwise_panel_ndcg@10") or 0.0)
    m_fu_nd = float(metrics.get("fusion_train_best_ndcg@10") or 0.0)
    m_safe_nd = float(metrics.get("safe_llm_gate_ndcg@10") or 0.0)

    fuse_rows = [
        {"policy": "fallback_only", "recall_at_10": metrics.get("fallback_recall@10"), "ndcg_at_10": m_fb_nd, "mrr_at_10": metrics.get("fallback_mrr@10"), "hit_rate_at_10": metrics.get("fallback_hit_rate@10")},
        {"policy": "llm_listwise_panel", "recall_at_10": metrics.get("llm_listwise_panel_recall@10"), "ndcg_at_10": m_llm_nd, "mrr_at_10": metrics.get("llm_listwise_panel_mrr@10"), "hit_rate_at_10": metrics.get("llm_listwise_panel_hit_rate@10")},
        {"policy": "fusion_train_best", "recall_at_10": metrics.get("fusion_train_best_recall@10"), "ndcg_at_10": m_fu_nd, "mrr_at_10": metrics.get("fusion_train_best_mrr@10"), "hit_rate_at_10": metrics.get("fusion_train_best_hit_rate@10")},
        {"policy": "safe_llm_gate_diag", "recall_at_10": metrics.get("safe_llm_gate_recall@10"), "ndcg_at_10": m_safe_nd, "mrr_at_10": metrics.get("safe_llm_gate_mrr@10"), "hit_rate_at_10": metrics.get("safe_llm_gate_hit_rate@10")},
    ]
    written[str(fusion_path.relative_to(ROOT))] = "ok"
    with fusion_path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fuse_rows[0].keys()))
        w.writeheader()
        for row in fuse_rows:
            w.writerow(row)

    vs_rows = [
        {"metric": "ndcg_at_10", "llm_minus_fallback": metrics.get("llm_delta_ndcg10_vs_fallback"), "fusion_minus_fallback": metrics.get("fusion_delta_ndcg10_vs_fallback")},
        {"metric": "recall_at_10", "llm_minus_fallback": _minus(metrics.get("llm_listwise_panel_recall@10"), metrics.get("fallback_recall@10")), "fusion_minus_fallback": _minus(metrics.get("fusion_train_best_recall@10"), metrics.get("fallback_recall@10"))},
    ]
    with vs_path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["metric", "llm_minus_fallback", "fusion_minus_fallback"])
        w.writeheader()
        for row in vs_rows:
            w.writerow(row)
    written[str(vs_path.relative_to(ROOT))] = "ok"

    swap_path = out_dir / "cu_gr_v2_swap_analysis.csv"
    with swap_path.open("w", encoding="utf-8", newline="") as h:
        fields = ["margin", "confidence_min", "beneficial_swaps", "harmful_swaps", "neutral_swaps", "harmful_swap_rate", "delta_ndcg_10_vs_fallback"]
        sw = diagnostics.get("swap_analysis_rows") or []
        ww = csv.DictWriter(h, fieldnames=fields)
        ww.writeheader()
        for row in sw:
            ww.writerow({k: row.get(k, "") for k in fields})

    fi_path = out_dir / "cu_gr_v2_feature_importance.csv"
    params = dict(model_payload.get("best_fusion_params") or {})
    fi_rows = [{"feature": k, "weight": v, "role": "fusion_linear_grid_best"} for k, v in sorted(params.items())]
    if not fi_rows:
        fi_rows = [{"feature": "", "weight": "", "role": "none"}]
    with fi_path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(fi_rows[0].keys()))
        w.writeheader()
        for row in fi_rows:
            w.writerow(row)

    cost_json = ROOT / Path("outputs/tables/cu_gr_v2_cost_latency.json")
    with cost_json.open("w", encoding="utf-8") as h:
        json.dump({"seed": seed, "run_dir": str(run_dir), **summary}, h, indent=2)
    _write_latency_csv(out_dir / "cu_gr_v2_cost_latency.csv", seed=seed, summary=summary, source_run=str(run_dir))

    return written


def _minus(a: Any, b: Any) -> str:
    try:
        if a is None or b is None:
            return ""
        return str(round(float(a) - float(b), 8))
    except (TypeError, ValueError):
        return ""


def _build_preference_dataset_csv(path: Path, rows_packed: list[dict[str, Any]], train_pop: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id",
        "run_seed",
        "panel_size",
        "item_id",
        "label_relevance",
        "fallback_rank",
        "fallback_score",
        "normalized_fallback_score",
        "llm_listwise_score",
        "llm_listwise_rank",
        "llm_confidence_for_candidate",
        "pairwise_win_count",
        "pairwise_loss_count",
        "pairwise_avg_confidence",
        "popularity_bucket",
        "item_popularity_log",
        "history_similarity",
        "category_overlap",
        "is_tail",
        "source_flags",
        "status",
    ]
    out_rows: list[dict[str, Any]] = []
    for r in rows_packed:
        eid = str(r.get("example_id", ""))
        seed = int(r.get("run_seed", 0))
        panel = r.get("panel") or {}
        psize = int(r.get("panel_size", 0))
        parse_ok = bool(r.get("parse_success"))
        score_by_item: dict[str, float] = {}
        conf_by_item: dict[str, float] = {}
        for rr in r.get("parsed_ranking") or []:
            score_by_item[str(rr.get("item_id"))] = float(rr.get("score") or 0.0)
            conf_by_item[str(rr.get("item_id"))] = float(rr.get("confidence") or 0.0)
        ranked_ids = sorted(score_by_item.keys(), key=lambda i: (-score_by_item.get(i, 0.0), i))
        rank_map = {iid: idx + 1 for idx, iid in enumerate(ranked_ids)}

        for p in panel.get("panel_items") or []:
            iid = str(p.get("item_id", ""))
            pop = int(train_pop.get(iid, 0))
            out_rows.append(
                {
                    "example_id": eid,
                    "run_seed": seed,
                    "panel_size": psize,
                    "item_id": iid,
                    "label_relevance": score_by_item.get(iid, ""),
                    "fallback_rank": p.get("fallback_rank", ""),
                    "fallback_score": p.get("fallback_score", ""),
                    "normalized_fallback_score": normalized_fallback_scores_in_panel(panel).get(iid, ""),
                    "llm_listwise_score": score_by_item.get(iid, "") if parse_ok else "",
                    "llm_listwise_rank": rank_map.get(iid, "") if parse_ok else "",
                    "llm_confidence_for_candidate": conf_by_item.get(iid, "") if parse_ok else "",
                    "pairwise_win_count": "",
                    "pairwise_loss_count": "",
                    "pairwise_avg_confidence": "",
                    "popularity_bucket": "",
                    "item_popularity_log": round(math.log1p(pop), 6) if pop >= 0 else "",
                    "history_similarity": "",
                    "category_overlap": "",
                    "is_tail": "",
                    "source_flags": json.dumps(p.get("source_flags") or [], ensure_ascii=False),
                    "status": "ok" if parse_ok else "parse_failed",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)


def _write_parser_stats(path: Path, *, n_examples: int, parse_ok: int, invalid_unknown: int, duplicate: int, other_fail: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate_ok = parse_ok / max(n_examples, 1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=[
                "n_examples",
                "parse_success_count",
                "parser_success_rate",
                "unknown_label_incidents",
                "duplicate_label_incidents",
                "other_parse_failure_incidents",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "n_examples": n_examples,
                "parse_success_count": parse_ok,
                "parser_success_rate": round(rate_ok, 8),
                "unknown_label_incidents": invalid_unknown,
                "duplicate_label_incidents": duplicate,
                "other_parse_failure_incidents": other_fail,
            }
        )


def build_dataset_csv_from_signals(
    *,
    ROOT: Path,
    signals_path: str | Path,
    processed_dir: str,
    dataset_csv: str | Path | None = None,
    parser_stats_csv: str | Path | None = None,
) -> int:
    """Write preference dataset CSV + parser stats from preference_signals.jsonl (no fusion recompute)."""
    sp = Path(signals_path)
    if not sp.is_absolute():
        sp = ROOT / sp
    ds = Path(dataset_csv or "outputs/tables/cu_gr_v2_preference_dataset.csv")
    if not ds.is_absolute():
        ds = ROOT / ds
    pst = Path(parser_stats_csv or "outputs/tables/cu_gr_v2_preference_parser_stats.csv")
    if not pst.is_absolute():
        pst = ROOT / pst

    lines = []
    with sp.open("r", encoding="utf-8") as handle:
        for ln in handle:
            if ln.strip():
                lines.append(json.loads(ln))
    packed = rebuild_packed_rows_from_signals(lines)
    ctx = build_dataset_context(processed_dir)
    train_pop = dict(ctx.get("train_popularity") or {})
    _build_preference_dataset_csv(ds, packed, train_pop)
    po = sum(1 for ln in lines if ln.get("parse_success"))
    unk = sum(1 for ln in lines if not ln.get("parse_success") and "unknown_label" in str(ln.get("parse_error") or ""))
    dup = sum(1 for ln in lines if not ln.get("parse_success") and "duplicate_label" in str(ln.get("parse_error") or ""))
    oth = len(lines) - po - unk - dup
    _write_parser_stats(pst, n_examples=len(lines), parse_ok=po, invalid_unknown=unk, duplicate=dup, other_fail=max(0, oth))
    return len(lines)


def build_dataset_csv_from_signal_paths(
    *,
    ROOT: Path,
    signals_paths: Sequence[str | Path],
    processed_dir: str,
    dataset_csv: str | Path | None = None,
    parser_stats_csv: str | Path | None = None,
) -> int:
    lines: list[dict[str, Any]] = []
    for raw_path in signals_paths:
        sp = Path(raw_path)
        if not sp.is_absolute():
            sp = ROOT / sp
        with sp.open("r", encoding="utf-8") as handle:
            for ln in handle:
                if ln.strip():
                    lines.append(json.loads(ln))
    ds = Path(dataset_csv or "outputs/tables/cu_gr_v2_preference_dataset.csv")
    if not ds.is_absolute():
        ds = ROOT / ds
    pst = Path(parser_stats_csv or "outputs/tables/cu_gr_v2_preference_parser_stats.csv")
    if not pst.is_absolute():
        pst = ROOT / pst
    packed = rebuild_packed_rows_from_signals(lines)
    ctx = build_dataset_context(processed_dir)
    train_pop = dict(ctx.get("train_popularity") or {})
    _build_preference_dataset_csv(ds, packed, train_pop)
    po = sum(1 for ln in lines if ln.get("parse_success"))
    unk = sum(1 for ln in lines if not ln.get("parse_success") and "unknown_label" in str(ln.get("parse_error") or ""))
    dup = sum(1 for ln in lines if not ln.get("parse_success") and "duplicate_label" in str(ln.get("parse_error") or ""))
    oth = len(lines) - po - unk - dup
    _write_parser_stats(pst, n_examples=len(lines), parse_ok=po, invalid_unknown=unk, duplicate=dup, other_fail=max(0, oth))
    return len(lines)


def _worker_build_request(
    args: tuple[Any, ...],
) -> tuple[int, LLMRequest, dict[str, Any], dict[str, Any], dict[str, str], str]:
    idx, bm_row, seed, panel_size, ctx, ours_by, seq_by, show_fallback_rank, llm_temperature, llm_blk, max_tokens = args
    eid = example_id_key(bm_row)

    panel = build_candidate_panel(
        bm25_row=bm_row,
        ours_row=ours_by.get(eid),
        sequential_row=seq_by.get(eid),
        context=ctx,
        panel_size=panel_size,
        seed=seed,
    )
    ours_row = ours_by.get(eid) or {}
    meta_raw = ours_row.get("metadata")
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    ht = meta.get("history_titles") or []
    history_titles = list(ht) if isinstance(ht, (list, tuple)) else []
    prompt, label_map = build_listwise_preference_prompt(
        history_titles=history_titles,
        panel=panel,
        show_fallback_rank=show_fallback_rank,
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    rid = LLMRequest(
        prompt=prompt,
        metadata={"prompt_template_id": "cu_gr_v2.listwise.v1", "prompt_hash": prompt_hash},
        temperature=llm_temperature,
        top_p=float(llm_blk.get("top_p") or 1.0),
        max_tokens=max_tokens,
        seed=int(seed),
    )
    return idx, rid, bm_row, panel, label_map, prompt_hash


def packed_row_from_signal_record(line: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild in-memory packed row from a persisted signal; re-parses raw_output with current parser."""
    bm25 = line.get("bm25_snapshot")
    if not isinstance(bm25, dict):
        return None
    label_map = dict(line.get("label_map") or {})
    lm = {str(k): str(v) for k, v in label_map.items()}
    parsed = parse_listwise_response(str(line.get("raw_output") or ""), label_to_item_id=lm)
    out = dict(line)
    out["bm_row"] = bm25
    out["parse_success"] = bool(parsed["ok"])
    out["parse_error"] = parsed.get("error")
    out["parsed_ranking"] = list(parsed.get("ranking") or []) if parsed["ok"] else []
    unc = parsed.get("uncertainty")
    out["uncertainty_meta"] = unc if isinstance(unc, dict) else {}
    return out


def rebuild_packed_rows_from_signals(
    signals_lines: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hydrate packed rows from preference_signals.jsonl using fresh JSON parse rules."""
    out: list[dict[str, Any]] = []
    for line in signals_lines:
        packed = packed_row_from_signal_record(line)
        if packed is not None:
            out.append(packed)
    return out


def export_tables_after_signals(
    *,
    ROOT: Path,
    signals_path: Path,
    processed_dir: str,
    run_dir_hint: Path | None = None,
) -> dict[str, Any]:
    """Offline: rebuild preference CSV/model summaries from preference_signals.jsonl."""
    signals_path = ROOT / signals_path if not signals_path.is_absolute() else signals_path
    lines = []
    with signals_path.open("r", encoding="utf-8") as handle:
        for ln in handle:
            if ln.strip():
                lines.append(json.loads(ln))
    packed = rebuild_packed_rows_from_signals(lines)
    ctx = build_dataset_context(processed_dir)
    train_pop = dict(ctx.get("train_popularity") or {})
    agg, _preds, model_payload, diagnostics = _evaluate_aggregate(packed, train_pop=train_pop)
    seed = int(lines[0].get("run_seed") or agg.get("seed") or 13) if lines else 13
    summary_stub: dict[str, Any] = {
        "real_request_count": "n/a_offline_replay",
        "cache_hit_count": "",
        "cache_hit_rate": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "estimated_cost": "",
        "latency_p50_seconds": "",
        "latency_p95_seconds": "",
    }
    merged_metrics = dict(agg)
    merged_metrics["subset_size"] = len(lines)
    merged_metrics["seed"] = seed
    merged_metrics.update(diagnostics)

    rd = Path(run_dir_hint or signals_path.parent)
    _tables = _write_preference_tables(
        ROOT=ROOT,
        seed=seed,
        run_dir=rd,
        rows_packed=packed,
        metrics=merged_metrics,
        diagnostics=diagnostics,
        summary=summary_stub,
        model_payload=model_payload,
    )
    ds_path = ROOT / "outputs/tables/cu_gr_v2_preference_dataset.csv"
    _build_preference_dataset_csv(ds_path, packed, train_pop)
    stats_path = ROOT / "outputs/tables/cu_gr_v2_preference_parser_stats.csv"
    po = sum(1 for ln in lines if ln.get("parse_success"))
    unk = sum(1 for ln in lines if not ln.get("parse_success") and "unknown_label" in str(ln.get("parse_error") or ""))
    dup = sum(1 for ln in lines if not ln.get("parse_success") and "duplicate_label" in str(ln.get("parse_error") or ""))
    oth = len(lines) - po - unk - dup
    _write_parser_stats(stats_path, n_examples=len(lines), parse_ok=po, invalid_unknown=unk, duplicate=dup, other_fail=max(0, oth))
    model_out = ROOT / Path("outputs/models/cu_gr_v2_preference_fusion")
    model_out.mkdir(parents=True, exist_ok=True)
    model_out.joinpath("model.json").write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    log_path = ROOT / "outputs/tables/cu_gr_v2_offline_export.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"signals": str(signals_path), "tables_written": _tables}, indent=2))

    return merged_metrics


def run_cu_gr_v2_preference_subgate(config_path: str | Path) -> dict[str, Any]:
    ROOT = Path(__file__).resolve().parents[3]
    path = Path(config_path)
    config = load_config(path)

    seeds = config.get("seeds") if isinstance(config.get("seeds"), list) else [config.get("seed", 13)]
    seeds = [int(s) for s in seeds]

    pref = dict(config.get("preference") or {})
    cu2 = dict(config.get("cu_gr_v2") or {})
    panel_size = int(pref.get("panel_size") or cu2.get("panel_size") or config.get("safety", {}).get("panel_size") or 15)
    show_fallback_rank = bool(pref.get("show_fallback_rank") or cu2.get("show_fallback_rank", False))

    llm_blk = dict(config.get("llm") or {})
    max_tokens = int(llm_blk.get("max_tokens") or 2048)
    llm_temperature = float(llm_blk.get("temperature") or 0.0)

    safety = dict(config.get("safety") or {})
    req_limits = dict(llm_blk.get("request_limits") or {})
    concurrency = max(1, int(safety.get("concurrency") or req_limits.get("max_concurrency") or 16))

    output_block = dict(config.get("output") or {})
    out_base = Path(str(output_block.get("output_dir") or config.get("output_dir") or "outputs/runs"))
    run_name = str(output_block.get("run_name") or config.get("run_name") or "r3_v2_movielens_preference_signal_subgate")

    proc_dir = str((config.get("dataset") or {}).get("processed_dir") or "data/processed/movielens_1m/r2_full_single_dataset")
    ctx = build_dataset_context(proc_dir)
    train_pop_ctx = dict(ctx.get("train_popularity") or {})

    from llm4rec.experiments import runner as exp_runner

    manifests: list[dict[str, Any]] = []
    full_seed_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    provider_summaries_by_seed: dict[int, dict[str, Any]] = {}

    runs_dir = ROOT / "outputs" / "runs"
    for seed in seeds:
        run_dir = Path(out_base) / f"{run_name}_seed{seed}"
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        shutil.copy(path, run_dir / "resolved_config.yaml")
        with Path(run_dir / "environment.json").open("w", encoding="utf-8") as handle:
            json.dump({"experiment_kind": "cu_gr_v2_preference_subgate", "processed_dir": proc_dir}, handle, indent=2)

        bm_file = method_run_dir(runs_dir, "bm25", seed) / "predictions.jsonl"
        if not bm_file.exists():
            raise FileNotFoundError(f"missing bm25 artifact: {bm_file}")
        ours_file = method_run_dir(runs_dir, "ours_uncertainty_guided_real", seed) / "predictions.jsonl"
        seq_file = method_run_dir(runs_dir, "sequential_markov", seed) / "predictions.jsonl"
        bm_all = read_jsonl(bm_file)
        bm_rows = _limit_examples(config, bm_all)
        ours_by = index_rows(read_jsonl(ours_file)) if ours_file.exists() else {}
        seq_by = index_rows(read_jsonl(seq_file)) if seq_file.exists() else {}

        controlled_llm = exp_runner._build_llm_provider(config, run_dir=run_dir)

        signals_path = Path(run_dir) / "preference_signals.jsonl"
        raw_path = _resolve_repo_path(ROOT, cfg_raw_path(config, seed))

        parse_ok_count = dup_label_incidents = invalid_label_incidents = other_parse_incidents = 0
        rows_packed: list[dict[str, Any]] = []

        indexed_inputs: list[tuple[int, dict[str, Any]]] = [(i, row) for i, row in enumerate(bm_rows)]
        staged: list[Any] = []
        for idx, bm_row in indexed_inputs:
            t = _worker_build_request(
                (
                    idx,
                    bm_row,
                    seed,
                    panel_size,
                    ctx,
                    ours_by,
                    seq_by,
                    show_fallback_rank,
                    llm_temperature,
                    llm_blk,
                    max_tokens,
                )
            )
            staged.append(t)

        by_idx: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

        def _finalize(parsed: dict[str, Any], resp: Any, bm_row: dict[str, Any], panel: dict[str, Any], label_map: dict[str, str], prompt_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
            err = str(parsed.get("error") or "")
            dup_inc = "duplicate_label" in err if not parsed["ok"] else False
            panel_ordered_item_ids = [str(p["item_id"]) for p in panel["panel_items"]]
            bm25_snapshot = compact_bm25_row_for_signal(bm_row)
            eid_inner = example_id_key(bm_row)
            rec = {
                "example_id": eid_inner,
                "user_id": str(bm_row.get("user_id", "")),
                "run_seed": seed,
                "target_item": str(bm_row.get("target_item", "")),
                "panel_size": panel_size,
                "panel": panel,
                "label_map": label_map,
                "prompt_template_id": "cu_gr_v2.listwise.v1",
                "prompt_hash": prompt_hash,
                "raw_output": resp.text,
                "parse_success": parsed["ok"],
                "parse_error": parsed.get("error"),
                "duplicate_label_incident": 1 if dup_inc else 0,
                "invalid_label_incident": 1 if (not parsed["ok"] and "unknown_label" in err) else 0,
                "parsed_ranking": parsed.get("ranking") if parsed["ok"] else [],
                "uncertainty_meta": parsed.get("uncertainty") or {},
                "panel_ordered_item_ids": panel_ordered_item_ids,
                "cache_hit": bool(resp.cache_hit),
                "latency_seconds": float(resp.latency_seconds or 0.0),
                "token_usage": dict(resp.usage or {}),
                "cache_key": str((resp.metadata or {}).get("cache_key") or ""),
                "model": str(resp.model or ""),
                "provider": str(resp.provider or ""),
                "bm25_snapshot": bm25_snapshot,
            }
            packed_inner = dict(rec)
            packed_inner["bm_row"] = bm_row
            return rec, packed_inner

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futs = {}
            for tup in staged:
                idx_i, rq, bm_row_i, panel_i, lmap_i, phash_i = tup[0], tup[1], tup[2], tup[3], tup[4], tup[5]
                futs[executor.submit(controlled_llm.generate, rq)] = (idx_i, bm_row_i, panel_i, lmap_i, phash_i)
            for fu in as_completed(futs):
                idx_i, bm_row_i, panel_i, lmap_i, phash_i = futs[fu]
                resp = fu.result()
                parsed = parse_listwise_response(resp.text, label_to_item_id=lmap_i)
                rec_fin, pk = _finalize(parsed, resp, bm_row_i, panel_i, lmap_i, phash_i)
                by_idx[idx_i] = (rec_fin, pk)

        for _idx_ord in range(len(bm_rows)):
            rec_ord, _ = by_idx[_idx_ord]
            if rec_ord["parse_success"]:
                parse_ok_count += 1
                continue
            err = str(rec_ord.get("parse_error") or "")
            if "duplicate_label" in err:
                dup_label_incidents += 1
            elif "unknown_label" in err:
                invalid_label_incidents += 1
            else:
                other_parse_incidents += 1

        with signals_path.open("w", encoding="utf-8") as sig_handle:
            for idx in range(len(bm_rows)):
                rec_fin, pk = by_idx[idx]
                sig_handle.write(json.dumps(rec_fin, ensure_ascii=False) + "\n")
                rows_packed.append(pk)
                _append_raw_global(raw_path, rec_fin)

        if hasattr(controlled_llm, "summary"):
            summary = controlled_llm.summary()
            summary_path = run_dir / "artifacts" / "llm_provider_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        else:
            summary = {}
        provider_summaries_by_seed[seed] = dict(summary)

        n_sig = len(rows_packed)
        parse_rate = parse_ok_count / max(n_sig, 1)
        full_seed_rows_by_seed[seed] = list(rows_packed)

        agg, primary_preds, model_payload, diagnostics = _evaluate_aggregate(rows_packed, train_pop=train_pop_ctx)

        preds_path = run_dir / "predictions.jsonl"
        with preds_path.open("w", encoding="utf-8") as handle:
            for prow in primary_preds:
                handle.write(json.dumps(prow, ensure_ascii=False) + "\n")

        metrics = {
            "subset_size": n_sig,
            "seed": seed,
            "panel_size": panel_size,
            "parser_success_rate": parse_rate,
            "parse_success_count": parse_ok_count,
            "invalid_label_incidents": invalid_label_incidents,
            "duplicate_label_incidents": dup_label_incidents,
            "other_parse_failure_incidents": other_parse_incidents,
            "invalid_label_rate_per_example": invalid_label_incidents / max(n_sig, 1),
            "duplicate_label_rate_per_example": dup_label_incidents / max(n_sig, 1),
            "other_parse_failure_rate_per_example": other_parse_incidents / max(n_sig, 1),
            **agg,
            "provider_summary": summary,
            "preference_diagnostics": diagnostics,
        }
        with Path(run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        merged_flat = dict(metrics)

        ds_path_tables = ROOT / "outputs/tables/cu_gr_v2_preference_dataset.csv"
        _build_preference_dataset_csv(ds_path_tables, rows_packed, train_pop_ctx)
        stats_path_tables = ROOT / "outputs/tables/cu_gr_v2_preference_parser_stats.csv"
        _write_parser_stats(
            stats_path_tables,
            n_examples=n_sig,
            parse_ok=parse_ok_count,
            invalid_unknown=invalid_label_incidents,
            duplicate=dup_label_incidents,
            other_fail=other_parse_incidents,
        )
        _write_preference_tables(
            ROOT=ROOT,
            seed=seed,
            run_dir=run_dir,
            rows_packed=rows_packed,
            metrics=merged_flat,
            diagnostics=diagnostics,
            summary=dict(summary),
            model_payload=model_payload,
        )

        logs_path = run_dir / "logs.txt"
        with logs_path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"cu_gr_v2_preference_subgate seed={seed} n={n_sig} parse_rate={parse_rate:.6f} "
                f"invalid_unknown={invalid_label_incidents} duplicate={dup_label_incidents} other_fail={other_parse_incidents}\n"
            )

        _write_metric_csv(run_dir / "metrics.csv", {k: v for k, v in merged_flat.items() if isinstance(v, (str, int, float, bool))})

        write_json_fallback = ROOT / Path("outputs/tables/cu_gr_v2_cost_latency.json")
        write_json_fallback.parent.mkdir(parents=True, exist_ok=True)
        write_json_fallback.write_text(json.dumps({"seed": seed, **summary}, indent=2), encoding="utf-8")

        model_out = ROOT / Path("outputs/models/cu_gr_v2_preference_fusion")
        model_out.mkdir(parents=True, exist_ok=True)
        model_out.joinpath("model.json").write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

        cost_latency_stub = {"note": "use_run_artifacts_when_present", "run_dir": str(run_dir)}
        write_json(run_dir / "cost_latency.json", {**cost_latency_stub, **summary})

        manifests.append({"run_dir": str(run_dir), "metrics_summary": agg, "n_examples": n_sig})
    full_seed_outputs: dict[str, Any] = {}
    if len(seeds) > 1:
        full_seed_outputs = _write_full_seed_outputs(
            ROOT=ROOT,
            run_name=run_name,
            rows_by_seed=full_seed_rows_by_seed,
            provider_summaries_by_seed=provider_summaries_by_seed,
            train_pop=train_pop_ctx,
        )
    return {"experiment_kind": "cu_gr_v2_preference_subgate", "seeds": seeds, "runs": manifests, "full_seed_outputs": full_seed_outputs}


def write_json(p: Path, obj: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def run_cu_gr_v2_collect_only(config_path: str | Path) -> dict[str, Any]:
    return run_cu_gr_v2_preference_subgate(config_path)
