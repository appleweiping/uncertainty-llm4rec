#!/usr/bin/env python3
"""Replay CU-GR policy on held-out seed using trained calibrator (no API)."""

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

import numpy as np

from llm4rec.analysis.calibrator_features import example_id_key
from llm4rec.analysis.ours_error_decomposition import method_run_dir, read_jsonl
from llm4rec.methods.cu_gr import (
    build_cu_gr_prediction,
    decide_promote,
    gate_parse_grounding_adherence,
    summarize_override_outcomes,
)
from llm4rec.methods.override_calibrator import load_bundle, predict_heads
from llm4rec.metrics.ranking import ranking_metrics
from llm4rec.metrics.validity import validity_metrics

EXCLUDE_FEATURES = {
    "example_id",
    "user_id",
    "target_item",
    "run_seed",
    "ours_method",
    "candidate_items_json",
    "fallback_top10_json",
    "grounded_item_id",
    "delta_recall_at_10",
    "delta_ndcg_at_10",
    "delta_mrr_at_10",
    "override_improves",
    "override_hurts",
    "override_neutral",
    "safe_override",
    "recall_fallback_at_10",
    "ndcg_fallback_at_10",
    "mrr_fallback_at_10",
    "recall_override_at_10",
    "ndcg_override_at_10",
    "mrr_override_at_10",
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            v = row.get(name, 0)
            try:
                X[i, j] = float(v) if v not in ("", None) else 0.0
            except (TypeError, ValueError):
                X[i, j] = 0.0
    return X


def index_by_example(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {example_id_key(r): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/tables/cu_gr_calibrator_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/cu_gr_calibrator"))
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--ours-method", type=str, default="ours_uncertainty_guided_real")
    args = parser.parse_args()

    meta = json.loads((args.model / "metadata.json").read_text(encoding="utf-8"))
    feature_names = list(meta["feature_names"])
    sel = meta.get("threshold_selection") or {}
    degenerate = isinstance(sel, dict) and sel.get("status") == "No reliable override region found"
    if degenerate:
        tau_i, tau_h = 10.0, 0.0
    else:
        tau_i = float(sel.get("tau_improve", 10.0))
        tau_h = float(sel.get("tau_harm", 0.0))

    bundle = load_bundle(args.model)
    rows_all = load_dataset(args.input)
    test_rows = [r for r in rows_all if int(r["run_seed"]) == args.test_seed]
    if not test_rows:
        raise SystemExit("no test rows")

    X_test = feature_matrix(test_rows, feature_names)
    p_imp, p_harm = predict_heads(bundle, X_test)

    ours_by_ex = index_by_example(
        read_jsonl(method_run_dir(args.runs, args.ours_method, args.test_seed) / "predictions.jsonl")
    )
    bm_by_ex = index_by_example(read_jsonl(method_run_dir(args.runs, "bm25", args.test_seed) / "predictions.jsonl"))

    preds: list[dict[str, Any]] = []
    deltas_ndcg: list[float] = []
    errors: list[dict[str, Any]] = []
    for i, row in enumerate(test_rows):
        eid = str(row["example_id"])
        ours = ours_by_ex.get(eid)
        bm = bm_by_ex.get(eid)
        if ours is None or bm is None:
            raise RuntimeError(f"missing predictions for example_id={eid}")
        meta_o = dict(ours.get("metadata") or {})
        gates_ok, _ = gate_parse_grounding_adherence(meta_o)
        prom = decide_promote(
            gates_ok=gates_ok,
            p_improve=float(p_imp[i]),
            p_harm=float(p_harm[i]),
            tau_improve=tau_i,
            tau_harm=tau_h,
        )
        pr = build_cu_gr_prediction(
            ours_row=ours,
            bm25_row=bm,
            promote=prom,
            p_improve=float(p_imp[i]),
            p_harm=float(p_harm[i]),
            tau_improve=tau_i,
            tau_harm=tau_h,
            calibrator_model=f"{bundle.improve_kind}|{bundle.harm_kind}",
            features_version="cu_gr_v1",
        )
        preds.append(pr)
        deltas_ndcg.append(float(row["delta_ndcg_at_10"]))
        if prom and float(row["delta_ndcg_at_10"]) < 0:
            errors.append(
                {
                    "example_id": eid,
                    "delta_ndcg_at_10": row["delta_ndcg_at_10"],
                    "p_improve": float(p_imp[i]),
                    "p_harm": float(p_harm[i]),
                    "grounded_item_id": row.get("grounded_item_id", ""),
                }
            )

    top_k = [10]
    rank_m = ranking_metrics(preds, top_k=top_k)
    val_m = validity_metrics(preds)
    ov = summarize_override_outcomes(preds, delta_ndcg=deltas_ndcg)

    args.output.mkdir(parents=True, exist_ok=True)

    empty_extras = {
        "accepted_override_count": "",
        "beneficial_override_count": "",
        "harmful_override_count": "",
        "neutral_override_count": "",
        "harmful_override_rate": "",
        "degenerate_fallback_only": "",
        "tau_improve": "",
        "tau_harm": "",
    }

    def method_metrics(method: str) -> dict[str, Any]:
        path = method_run_dir(args.runs, method, args.test_seed) / "predictions.jsonl"
        if not path.exists():
            return {
                "method": method,
                "missing": True,
                "recall@10": None,
                "ndcg@10": None,
                "mrr@10": None,
                "hit_rate@10": None,
                "validity_rate": None,
                "hallucination_rate": None,
                **empty_extras,
            }
        mrows = read_jsonl(path)
        rm = ranking_metrics(mrows, top_k=top_k)
        vm = validity_metrics(mrows)
        return {
            "method": method,
            "missing": False,
            "recall@10": rm.get("recall@10", 0.0),
            "ndcg@10": rm.get("ndcg@10", 0.0),
            "mrr@10": rm.get("mrr@10", 0.0),
            "hit_rate@10": rm.get("hit_rate@10", 0.0),
            "validity_rate": vm.get("validity_rate", 0.0),
            "hallucination_rate": vm.get("hallucination_rate", 0.0),
            **empty_extras,
        }

    compare_methods = [
        "cu_gr",
        "ours_fallback_only",
        args.ours_method,
        "bm25",
        "popularity",
        "sequential_markov",
        "llm_generative_real",
    ]
    results: list[dict[str, Any]] = []
    cu_row = {
        "method": "cu_gr",
        "missing": False,
        "recall@10": rank_m.get("recall@10", 0.0),
        "ndcg@10": rank_m.get("ndcg@10", 0.0),
        "mrr@10": rank_m.get("mrr@10", 0.0),
        "hit_rate@10": rank_m.get("hit_rate@10", 0.0),
        "validity_rate": val_m.get("validity_rate", 0.0),
        "hallucination_rate": val_m.get("hallucination_rate", 0.0),
        "accepted_override_count": ov["accepted_override_count"],
        "beneficial_override_count": ov["beneficial_override_count"],
        "harmful_override_count": ov["harmful_override_count"],
        "neutral_override_count": ov["neutral_override_count"],
        "harmful_override_rate": ov["harmful_override_rate"],
        "degenerate_fallback_only": int(degenerate),
        "tau_improve": tau_i,
        "tau_harm": tau_h,
    }
    results.append(cu_row)
    for m in compare_methods[1:]:
        results.append(method_metrics(m))

    cons_dir = (
        args.runs
        / f"r3b_movielens_1m_conservative_gate_cache_replay_ours_conservative_uncertainty_gate_seed{args.test_seed}"
    )
    cons_path = cons_dir / "predictions.jsonl"
    if cons_path.is_file():
        mrows = read_jsonl(cons_path)
        rm = ranking_metrics(mrows, top_k=top_k)
        vm = validity_metrics(mrows)
        results.append(
            {
                "method": "ours_conservative_uncertainty_gate",
                "missing": False,
                "recall@10": rm.get("recall@10", 0.0),
                "ndcg@10": rm.get("ndcg@10", 0.0),
                "mrr@10": rm.get("mrr@10", 0.0),
                "hit_rate@10": rm.get("hit_rate@10", 0.0),
                "validity_rate": vm.get("validity_rate", 0.0),
                "hallucination_rate": vm.get("hallucination_rate", 0.0),
                **empty_extras,
            }
        )

    with (args.output / "cu_gr_policy_results.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        w.writeheader()
        for row in results:
            w.writerow(row)

    fb = next((r for r in results if r["method"] == "ours_fallback_only" and not r.get("missing")), None)
    cu = results[0]
    vs_rows = [
        {
            "metric": "ndcg@10",
            "cu_gr": cu.get("ndcg@10"),
            "fallback_only": fb.get("ndcg@10") if fb else None,
            "delta_cu_minus_fallback": (cu.get("ndcg@10") - fb.get("ndcg@10")) if fb else None,
        },
        {
            "metric": "recall@10",
            "cu_gr": cu.get("recall@10"),
            "fallback_only": fb.get("recall@10") if fb else None,
            "delta_cu_minus_fallback": (cu.get("recall@10") - fb.get("recall@10")) if fb else None,
        },
    ]
    with (args.output / "cu_gr_vs_fallback.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(vs_rows[0].keys()))
        w.writeheader()
        for row in vs_rows:
            w.writerow(row)

    with (args.output / "cu_gr_error_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["example_id", "delta_ndcg_at_10", "p_improve", "p_harm", "grounded_item_id"])
        w.writeheader()
        for row in errors[:200]:
            w.writerow(row)

    print(json.dumps({"policy_results": str(args.output / "cu_gr_policy_results.csv"), "n_errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
