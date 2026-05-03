#!/usr/bin/env python3
"""Train CU-GR improve/harm calibrators and sweep thresholds on validation seed."""

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

from llm4rec.analysis.calibrator_features import TOP_K_LABEL, build_override_topk, metric_row
from llm4rec.methods.cu_gr import decide_promote
from llm4rec.methods.override_calibrator import predict_heads, save_bundle, train_bundle

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


def infer_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(k for k in rows[0].keys() if k not in EXCLUDE_FEATURES)


def rows_to_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            v = row.get(name, 0)
            try:
                X[i, j] = float(v) if v not in ("", None) else 0.0
            except (TypeError, ValueError):
                X[i, j] = 0.0
    return X


def gates_ok_from_row(row: dict[str, Any]) -> bool:
    return (
        float(row.get("parse_success", 0) or 0) >= 0.5
        and float(row.get("grounding_success", 0) or 0) >= 0.5
        and float(row.get("candidate_adherence", 0) or 0) >= 0.5
    )


def predicted_list_for_decision(row: dict[str, Any], promote: bool) -> list[str]:
    cand = json.loads(row["candidate_items_json"])
    fb = json.loads(row["fallback_top10_json"])
    g = str(row.get("grounded_item_id") or "")
    cand_set = set(cand)
    if promote and g and g in cand_set:
        return build_override_topk(fb, g, cand_set, k=TOP_K_LABEL)
    return list(fb)


def evaluate_thresholds(
    rows_val: list[dict[str, Any]],
    p_improve: np.ndarray,
    p_harm: np.ndarray,
    *,
    harm_tolerance: float,
    min_accepts: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    tau_improve_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tau_harm_grid = [0.05, 0.1, 0.2, 0.3, 0.4]
    sweep_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    n = len(rows_val)
    for ti in tau_improve_grid:
        for th in tau_harm_grid:
            ndcgs: list[float] = []
            accepts = 0
            harmful = 0
            beneficial = 0
            neutral = 0
            for i, row in enumerate(rows_val):
                g_ok = gates_ok_from_row(row)
                prom = decide_promote(
                    gates_ok=g_ok,
                    p_improve=float(p_improve[i]),
                    p_harm=float(p_harm[i]),
                    tau_improve=ti,
                    tau_harm=th,
                )
                pred = predicted_list_for_decision(row, prom)
                m = metric_row(str(row["target_item"]), pred, json.loads(row["candidate_items_json"]))
                ndcgs.append(float(m["ndcg@10"]))
                if prom:
                    accepts += 1
                    d = float(row["delta_ndcg_at_10"])
                    if d < 0:
                        harmful += 1
                    elif d > 0:
                        beneficial += 1
                    else:
                        neutral += 1
            mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
            harm_rate = harmful / max(n, 1)
            sweep_rows.append(
                {
                    "tau_improve": ti,
                    "tau_harm": th,
                    "mean_ndcg_at_10": mean_ndcg,
                    "accepted_override_count": accepts,
                    "beneficial_override_count": beneficial,
                    "harmful_override_count": harmful,
                    "neutral_override_count": neutral,
                    "harmful_override_rate": harm_rate,
                    "feasible": int(accepts >= min_accepts and harm_rate <= harm_tolerance),
                }
            )
            if accepts < min_accepts or harm_rate > harm_tolerance:
                continue
            if best is None or mean_ndcg > float(best["mean_ndcg_at_10"]):
                best = {
                    "tau_improve": ti,
                    "tau_harm": th,
                    "mean_ndcg_at_10": mean_ndcg,
                    "accepted_override_count": accepts,
                    "harmful_override_rate": harm_rate,
                }
    return best, sweep_rows


def _estimator_coefs(model: Any) -> np.ndarray | None:
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0].estimator
        lr = inner.named_steps.get("lr") if hasattr(inner, "named_steps") else None
        if lr is not None and hasattr(lr, "coef_"):
            return np.asarray(lr.coef_).ravel()
        if hasattr(inner, "coef_"):
            return np.asarray(inner.coef_).ravel()
    if hasattr(model, "named_steps") and "lr" in model.named_steps:
        lr = model.named_steps["lr"]
        if hasattr(lr, "coef_"):
            return np.asarray(lr.coef_).ravel()
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()
    return None


def feature_importance_rows(bundle: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head, model, kind in (
        ("improve", bundle.improve, bundle.improve_kind),
        ("harm", bundle.harm, bundle.harm_kind),
    ):
        coefs = _estimator_coefs(model)
        if coefs is None or len(coefs) != len(feature_names):
            for name in feature_names:
                rows.append({"head": head, "feature": name, "importance": 0.0, "model_kind": kind})
            continue
        for name, c in zip(feature_names, coefs, strict=False):
            rows.append({"head": head, "feature": name, "importance": float(c), "model_kind": kind})
    rows.sort(key=lambda r: abs(r["importance"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/tables/cu_gr_calibrator_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/cu_gr_calibrator"))
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--train-seed", type=int, default=13)
    parser.add_argument("--valid-seed", type=int, default=21)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--harm-tolerance", type=float, default=0.01)
    parser.add_argument("--min-accepts", type=int, default=10)
    args = parser.parse_args()

    rows = load_dataset(args.input)
    if not rows:
        raise SystemExit("empty dataset")

    feature_names = infer_feature_names(rows)
    train_rows = [r for r in rows if int(r["run_seed"]) == args.train_seed]
    val_rows = [r for r in rows if int(r["run_seed"]) == args.valid_seed]
    if not train_rows or not val_rows:
        raise SystemExit("missing train or validation rows for requested seeds")

    X_train = rows_to_matrix(train_rows, feature_names)
    X_val = rows_to_matrix(val_rows, feature_names)
    y_imp_train = np.array([float(r["override_improves"]) for r in train_rows])
    y_harm_train = np.array([float(r["override_hurts"]) for r in train_rows])
    y_imp_val = np.array([float(r["override_improves"]) for r in val_rows])
    y_harm_val = np.array([float(r["override_hurts"]) for r in val_rows])

    bundle = train_bundle(
        X_train,
        y_imp_train,
        y_harm_train,
        X_val,
        y_imp_val,
        y_harm_val,
        feature_names,
    )
    p_imp_val, p_harm_val = predict_heads(bundle, X_val)
    best, sweep = evaluate_thresholds(
        val_rows,
        p_imp_val,
        p_harm_val,
        harm_tolerance=args.harm_tolerance,
        min_accepts=args.min_accepts,
    )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.tables_dir / "cu_gr_threshold_sweep.csv"
    if sweep:
        with sweep_path.open("w", encoding="utf-8", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            for row in sweep:
                w.writerow(row)
    else:
        sweep_path.write_text("", encoding="utf-8")

    imp_rows = feature_importance_rows(bundle, feature_names)
    with (args.tables_dir / "cu_gr_feature_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["head", "feature", "importance", "model_kind"])
        w.writeheader()
        for row in imp_rows:
            w.writerow(row)

    meta_out = {
        "feature_names": feature_names,
        "class_balance": bundle.class_balance,
        "metrics": bundle.metrics,
        "threshold_selection": best
        if best
        else {"status": "No reliable override region found", "reason": "constraints_not_satisfied"},
        "harm_tolerance": args.harm_tolerance,
        "min_accepts": args.min_accepts,
        "train_seed": args.train_seed,
        "valid_seed": args.valid_seed,
        "test_seed": args.test_seed,
    }
    save_bundle(args.output, bundle)
    meta_path = args.output / "metadata.json"
    blob = json.loads(meta_path.read_text(encoding="utf-8"))
    blob.update(meta_out)
    meta_path.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"model_dir": str(args.output), "threshold": best}, indent=2))


if __name__ == "__main__":
    main()
