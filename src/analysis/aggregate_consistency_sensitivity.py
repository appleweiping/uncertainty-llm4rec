from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_EXPERIMENTS = [
    "beauty_deepseek",
    "beauty_deepseek_sc_t03",
    "beauty_deepseek_sc_t06",
    "beauty_deepseek_sc_t09",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--exp_names", nargs="*", default=None)
    parser.add_argument("--base_exp", type=str, default="beauty_deepseek")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_exp_config(exp_name: str) -> Path:
    path = Path("configs") / "exp" / f"{exp_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config for sensitivity aggregation: {path}")
    return path


def resolve_model_config(exp_cfg: dict[str, Any]) -> Path:
    model_config = exp_cfg.get("model_config")
    if not model_config:
        raise ValueError("Experiment config is missing model_config.")
    path = Path(str(model_config))
    if not path.exists():
        raise FileNotFoundError(f"Missing model config: {path}")
    return path


def extract_temperature(exp_name: str) -> float:
    exp_cfg = load_yaml(resolve_exp_config(exp_name))
    model_cfg = load_yaml(resolve_model_config(exp_cfg))
    generation = model_cfg.get("generation", {}) or {}
    return float(generation.get("temperature", 0.0))


def load_self_consistency_summary(output_root: Path, exp_name: str) -> dict[str, Any]:
    path = output_root / exp_name / "self_consistency" / "test_self_consistency.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing self-consistency file: {path}")

    df = pd.read_json(path, lines=True)
    return {
        "num_samples": int(len(df)),
        "avg_yes_ratio": float(df["yes_ratio"].mean()),
        "avg_vote_entropy": float(df["vote_entropy"].mean()),
        "avg_vote_variance": float(df["vote_variance"].mean()),
        "avg_mean_confidence": float(df["mean_confidence"].mean()),
        "avg_confidence_variance": float(df["confidence_variance"].mean()),
        "avg_consistency_confidence": float(df["consistency_confidence"].mean()),
        "avg_consistency_uncertainty": float(df["consistency_uncertainty"].mean()),
        "nonzero_entropy_fraction": float((df["vote_entropy"] > 0).mean()),
        "high_entropy_fraction": float((df["vote_entropy"] >= 0.5).mean()),
    }


def load_estimator_rows(output_root: Path, exp_name: str) -> dict[str, dict[str, Any]]:
    path = output_root / exp_name / "tables" / "estimator_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing estimator comparison file: {path}")

    df = pd.read_csv(path)
    rows: dict[str, dict[str, Any]] = {}
    for estimator_name in ["verbalized_calibrated", "consistency", "fused"]:
        sub = df[df["estimator"].astype(str).str.lower() == estimator_name.lower()].copy()
        if sub.empty:
            continue
        rows[estimator_name] = sub.iloc[0].to_dict()
    return rows


def build_row(output_root: Path, exp_name: str, base_exp: str) -> dict[str, Any]:
    sc_summary = load_self_consistency_summary(output_root, exp_name)
    estimator_rows = load_estimator_rows(output_root, exp_name)
    base_rows = load_estimator_rows(output_root, base_exp)

    row: dict[str, Any] = {
        "exp_name": exp_name,
        "temperature": extract_temperature(exp_name),
    }
    row.update(sc_summary)

    for estimator_name, values in estimator_rows.items():
        prefix = estimator_name
        row[f"{prefix}_ece"] = values.get("calib_ece")
        row[f"{prefix}_brier_score"] = values.get("calib_brier_score")
        row[f"{prefix}_auroc"] = values.get("calib_auroc")
        row[f"{prefix}_rerank_ndcg_at_10"] = values.get("rerank_NDCG@10")
        row[f"{prefix}_rerank_mrr_at_10"] = values.get("rerank_MRR@10")

    if exp_name != base_exp:
        for estimator_name in ["consistency", "fused"]:
            current = estimator_rows.get(estimator_name, {})
            baseline = base_rows.get(estimator_name, {})
            if current and baseline:
                try:
                    row[f"{estimator_name}_ece_delta_vs_t00"] = float(current.get("calib_ece")) - float(
                        baseline.get("calib_ece")
                    )
                    row[f"{estimator_name}_brier_delta_vs_t00"] = float(
                        current.get("calib_brier_score")
                    ) - float(baseline.get("calib_brier_score"))
                except Exception:
                    pass

    return row


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    exp_names = args.exp_names or DEFAULT_EXPERIMENTS

    rows = [build_row(output_root, exp_name, args.base_exp) for exp_name in exp_names]
    df = pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)

    out_path = output_root / "summary" / "beauty_consistency_sensitivity.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved Beauty consistency sensitivity summary to: {out_path}")


if __name__ == "__main__":
    main()
