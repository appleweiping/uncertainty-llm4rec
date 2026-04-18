from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.analysis.confidence_correctness import (
    compute_confidence_bins_accuracy,
    compute_confidence_correctness_summary,
    prepare_prediction_dataframe,
)
from src.analysis.plotting import (
    plot_confidence_histogram,
    plot_popularity_avg_confidence,
    plot_reliability_diagram,
)
from src.analysis.popularity_bias import compute_popularity_group_stats
from src.baselines.score_proxies import (
    build_proxy_pointwise_df,
    build_proxy_rows,
    build_ranked_dataframe,
    compute_proxy_bin_rows,
    compute_proxy_popularity_rows,
    dump_minimal_example_json,
    infer_input_format,
    load_baseline_input,
)
from src.eval.calibration_metrics import compute_calibration_metrics, get_reliability_dataframe
from src.eval.ranking_metrics import compute_ranking_metrics
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_dict(summary: dict[str, Any], path: str | Path) -> None:
    save_table(pd.DataFrame([summary]), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional baseline confidence config path.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Baseline score input path (jsonl/csv).")
    parser.add_argument("--input_format", type=str, default=None, help="auto | score_rows | rank_rows | grouped_scores")
    parser.add_argument("--output_root", type=str, default=None, help="Output root directory.")
    parser.add_argument("--k", type=int, default=None, help="Top-k cutoff for NH metrics.")
    parser.add_argument("--n_bins", type=int, default=None, help="Number of proxy reliability bins.")
    parser.add_argument("--high_conf_threshold", type=float, default=None, help="High confidence threshold.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed.")
    parser.add_argument(
        "--dump_example",
        type=str,
        default=None,
        help="Optional path for dumping a minimal baseline score_rows JSONL example, then exit.",
    )
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    return {
        "exp_name": args.exp_name if args.exp_name is not None else cfg.get("exp_name", "baseline_confidence"),
        "input_path": args.input_path if args.input_path is not None else cfg.get("input_path"),
        "input_format": args.input_format if args.input_format is not None else cfg.get("input_format", "auto"),
        "output_root": args.output_root if args.output_root is not None else cfg.get("output_root", "outputs"),
        "k": args.k if args.k is not None else cfg.get("k", 10),
        "n_bins": args.n_bins if args.n_bins is not None else cfg.get("n_bins", 10),
        "high_conf_threshold": (
            args.high_conf_threshold
            if args.high_conf_threshold is not None
            else cfg.get("high_conf_threshold", 0.8)
        ),
        "seed": args.seed if args.seed is not None else cfg.get("seed"),
        "dump_example": args.dump_example,
    }


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    if cfg["dump_example"]:
        dump_minimal_example_json(cfg["dump_example"])
        print(f"Minimal baseline example dumped to: {cfg['dump_example']}")
        return

    if not cfg["input_path"]:
        raise ValueError("input_path is required unless --dump_example is used.")

    if cfg["seed"] is not None:
        set_global_seed(int(cfg["seed"]))

    paths = ensure_exp_dirs(cfg["exp_name"], output_root=cfg["output_root"])
    raw_df = load_baseline_input(cfg["input_path"])
    input_format = infer_input_format(raw_df, explicit_input_format=cfg["input_format"])
    ranked_df = build_ranked_dataframe(raw_df, input_format=input_format)
    proxy_df = build_proxy_rows(ranked_df)
    pointwise_proxy_df = build_proxy_pointwise_df(proxy_df)
    prepared_proxy_df = prepare_prediction_dataframe(pointwise_proxy_df)

    print(f"[{cfg['exp_name']}] Loaded baseline input rows: {len(raw_df)}")
    print(f"[{cfg['exp_name']}] Resolved baseline input format: {input_format}")
    print(f"[{cfg['exp_name']}] Ranked rows: {len(ranked_df)}")
    print(f"[{cfg['exp_name']}] Proxy users: {len(proxy_df)}")

    ranking_metrics = compute_ranking_metrics(ranked_df, k=int(cfg["k"]))
    save_summary_dict(ranking_metrics, paths.tables_dir / "baseline_ranking_metrics.csv")
    save_table(ranked_df, paths.tables_dir / "baseline_ranking_rows.csv")

    calibration_metrics = compute_calibration_metrics(
        prepared_proxy_df,
        confidence_col="confidence",
        target_col="is_correct",
        n_bins=int(cfg["n_bins"]),
    )

    confidence_summary = compute_confidence_correctness_summary(
        prepared_proxy_df,
        high_conf_threshold=float(cfg["high_conf_threshold"]),
    )
    proxy_bins_accuracy = compute_confidence_bins_accuracy(
        prepared_proxy_df,
        n_bins=int(cfg["n_bins"]),
    )
    proxy_reliability = get_reliability_dataframe(
        prepared_proxy_df["label"].to_numpy(),
        prepared_proxy_df["confidence"].to_numpy(),
        n_bins=int(cfg["n_bins"]),
    )
    proxy_popularity = compute_popularity_group_stats(
        prepared_proxy_df,
        high_conf_threshold=float(cfg["high_conf_threshold"]),
    )

    extra_summary = {
        "num_users": int(len(proxy_df)),
        "avg_top1_score": float(proxy_df["top1_score"].mean()) if proxy_df["top1_score"].notna().any() else float("nan"),
        "avg_top2_score": float(proxy_df["top2_score"].mean()) if proxy_df["top2_score"].notna().any() else float("nan"),
        "avg_score_margin": float(proxy_df["score_margin"].mean()) if proxy_df["score_margin"].notna().any() else float("nan"),
        "avg_proxy_confidence": float(proxy_df["proxy_confidence"].mean()) if proxy_df["proxy_confidence"].notna().any() else float("nan"),
        "avg_score_entropy": float(proxy_df["score_entropy"].mean()) if proxy_df["score_entropy"].notna().any() else float("nan"),
        "avg_score_sharpness": float(proxy_df["score_sharpness"].mean()) if proxy_df["score_sharpness"].notna().any() else float("nan"),
        "top1_accuracy": float(proxy_df["top1_label"].mean()) if len(proxy_df) else float("nan"),
    }
    proxy_summary = {**ranking_metrics, **calibration_metrics, **confidence_summary, **extra_summary}

    proxy_bin_rows = compute_proxy_bin_rows(proxy_df, n_bins=int(cfg["n_bins"]))
    proxy_popularity_rows = compute_proxy_popularity_rows(
        proxy_df,
        high_conf_threshold=float(cfg["high_conf_threshold"]),
    )

    save_summary_dict(proxy_summary, paths.tables_dir / "baseline_proxy_summary.csv")
    save_table(proxy_df, paths.tables_dir / "baseline_proxy_rows.csv")
    save_table(proxy_bins_accuracy, paths.tables_dir / "baseline_proxy_bins_accuracy.csv")
    save_table(proxy_reliability, paths.tables_dir / "baseline_proxy_reliability_bins.csv")
    save_table(proxy_popularity, paths.tables_dir / "baseline_proxy_popularity_stats.csv")
    save_table(proxy_bin_rows, paths.tables_dir / "baseline_margin_bins.csv")
    save_table(proxy_popularity_rows, paths.tables_dir / "baseline_proxy_grouped_rows.csv")

    plot_confidence_histogram(
        prepared_proxy_df,
        paths.figures_dir / "baseline_proxy_confidence_histogram.png",
    )
    plot_reliability_diagram(
        proxy_reliability,
        paths.figures_dir / "baseline_proxy_reliability_diagram.png",
    )
    plot_popularity_avg_confidence(
        proxy_popularity,
        paths.figures_dir / "baseline_proxy_popularity_avg_confidence.png",
    )

    print(f"[{cfg['exp_name']}] Saved NH metrics to: {paths.tables_dir / 'baseline_ranking_metrics.csv'}")
    print(f"[{cfg['exp_name']}] Saved proxy summary to: {paths.tables_dir / 'baseline_proxy_summary.csv'}")


if __name__ == "__main__":
    main()
