#!/usr/bin/env python3
"""Offline CU-GR v2 panel feasibility: coverage, oracle rerank upper bound (no API)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm4rec.analysis.calibrator_features import build_dataset_context, example_id_key
from llm4rec.analysis.ours_error_decomposition import method_run_dir, read_jsonl
from llm4rec.methods.candidate_panel import (
    build_candidate_panel,
    fallback_full_ranking_in_candidates,
    oracle_rerank_top10_metrics,
    panel_item_ids,
)
def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {example_id_key(r): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--panel-sizes", type=str, default="10,15,20")
    parser.add_argument("--seeds", type=str, default="13,21,42")
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args()

    panel_sizes = [int(x.strip()) for x in args.panel_sizes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    ctx = build_dataset_context(str(args.processed_dir) if args.processed_dir else None)

    coverage_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    vs_rows: list[dict[str, Any]] = []

    for seed in seeds:
        bm_path = method_run_dir(args.runs, "bm25", seed) / "predictions.jsonl"
        if not bm_path.exists():
            raise SystemExit(f"missing {bm_path}")
        bm_rows = read_jsonl(bm_path)
        ours_path = method_run_dir(args.runs, "ours_uncertainty_guided_real", seed) / "predictions.jsonl"
        ours_rows = read_jsonl(ours_path) if ours_path.exists() else []
        seq_path = method_run_dir(args.runs, "sequential_markov", seed) / "predictions.jsonl"
        seq_rows = read_jsonl(seq_path) if seq_path.exists() else []
        ours_by = index_rows(ours_rows)
        seq_by = index_rows(seq_rows)

        for psz in panel_sizes:
            tip = fb_hit = 0
            sum_fb_ndcg = sum_or_ndcg = sum_gain_ndcg = sum_gain_rec = 0.0
            n_pos_ndcg = n_pos_rec = n_beneficial = 0
            n_harm_ndcg = 0
            n = 0
            for bm in bm_rows:
                eid = example_id_key(bm)
                panel = build_candidate_panel(
                    bm25_row=bm,
                    ours_row=ours_by.get(eid),
                    sequential_row=seq_by.get(eid),
                    context=ctx,
                    panel_size=psz,
                    seed=seed,
                )
                pids = panel_item_ids(panel)
                ranked, _ = fallback_full_ranking_in_candidates(bm)
                tgt = str(bm.get("target_item") or "")
                cand = [str(x) for x in (bm.get("candidate_items") or [])]
                m = oracle_rerank_top10_metrics(
                    full_fallback_order=ranked,
                    panel_ids=pids,
                    target_item=tgt,
                    candidate_items=cand,
                )
                n += 1
                tip += int(m["target_in_panel"] >= 0.5)
                fb_hit += int(m["fallback_recall_at_10"] >= 0.5)
                sum_fb_ndcg += m["fallback_ndcg_at_10"]
                sum_or_ndcg += m["oracle_ndcg_at_10"]
                sum_gain_ndcg += m["ndcg_gain"]
                sum_gain_rec += m["recall_gain"]
                if m["ndcg_gain"] > 1e-12:
                    n_pos_ndcg += 1
                if m["recall_gain"] > 1e-12:
                    n_pos_rec += 1
                if m["ndcg_gain"] > 1e-12 or m["recall_gain"] > 1e-12:
                    n_beneficial += 1
                if m["ndcg_gain"] < -1e-12:
                    n_harm_ndcg += 1

            rate_tip = tip / max(n, 1)
            rate_fb = fb_hit / max(n, 1)
            coverage_rows.append(
                {
                    "run_seed": seed,
                    "panel_size": psz,
                    "n_examples": n,
                    "target_in_panel_rate": round(rate_tip, 6),
                    "fallback_hit_at_10_rate": round(rate_fb, 6),
                    "panel_coverage_minus_fallback_hit": round(rate_tip - rate_fb, 6),
                    "mean_fallback_ndcg_at_10": round(sum_fb_ndcg / max(n, 1), 8),
                    "mean_oracle_ndcg_at_10": round(sum_or_ndcg / max(n, 1), 8),
                    "mean_oracle_ndcg_gain_vs_fallback": round(sum_gain_ndcg / max(n, 1), 8),
                    "mean_oracle_recall_gain_vs_fallback": round(sum_gain_rec / max(n, 1), 8),
                    "n_strictly_positive_ndcg_gain": n_pos_ndcg,
                    "n_strictly_positive_recall_gain": n_pos_rec,
                    "n_beneficial_swap_opportunities": n_beneficial,
                    "n_strictly_negative_ndcg_gain": n_harm_ndcg,
                }
            )
            swap_rows.append(
                {
                    "run_seed": seed,
                    "panel_size": psz,
                    "n_beneficial_oracle_swaps": n_beneficial,
                    "n_positive_ndcg_gain": n_pos_ndcg,
                    "n_positive_recall_gain": n_pos_rec,
                    "n_negative_ndcg_gain": n_harm_ndcg,
                }
            )
            vs_rows.append(
                {
                    "run_seed": seed,
                    "panel_size": psz,
                    "mean_fallback_ndcg_at_10": round(sum_fb_ndcg / max(n, 1), 8),
                    "mean_oracle_ndcg_at_10": round(sum_or_ndcg / max(n, 1), 8),
                    "mean_delta_ndcg_oracle_minus_fallback": round(sum_gain_ndcg / max(n, 1), 8),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    cov_path = args.output / "cu_gr_v2_panel_coverage.csv"
    with cov_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(coverage_rows[0].keys()))
        w.writeheader()
        for row in coverage_rows:
            w.writerow(row)

    swap_path = args.output / "cu_gr_v2_swap_analysis.csv"
    with swap_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(swap_rows[0].keys()))
        w.writeheader()
        for row in swap_rows:
            w.writerow(row)

    vs_path = args.output / "cu_gr_v2_vs_fallback.csv"
    with vs_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(vs_rows[0].keys()))
        w.writeheader()
        for row in vs_rows:
            w.writerow(row)

    print(json.dumps({"panel_coverage": str(cov_path), "swap_analysis": str(swap_path), "vs_fallback": str(vs_path)}, indent=2))


if __name__ == "__main__":
    main()
