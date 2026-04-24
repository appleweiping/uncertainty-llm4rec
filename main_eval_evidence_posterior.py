from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.eval.calibration_metrics import compute_calibration_metrics, get_reliability_dataframe
from src.uncertainty.evidence_features import build_evidence_feature_frame
from src.utils.paths import ensure_exp_dirs


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.mean())


def _safe_count(condition: pd.Series) -> int:
    return int(condition.fillna(False).sum())


def _with_confidence(df: pd.DataFrame, confidence_col: str) -> pd.DataFrame:
    out = df.copy()
    out["confidence"] = pd.to_numeric(out[confidence_col], errors="coerce").clip(0.0, 1.0)
    return out


def plot_reliability(raw_rel: pd.DataFrame, repaired_rel: pd.DataFrame, output_path: str | Path) -> None:
    _ensure_parent(output_path)
    raw_plot = raw_rel.dropna(subset=["avg_confidence", "accuracy"])
    repaired_plot = repaired_rel.dropna(subset=["avg_confidence", "accuracy"])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    plt.plot(raw_plot["avg_confidence"], raw_plot["accuracy"], marker="o", label="raw")
    plt.plot(repaired_plot["avg_confidence"], repaired_plot["accuracy"], marker="o", label="repaired")
    plt.xlabel("Average confidence")
    plt.ylabel("Accuracy")
    plt.title("Evidence Posterior Reliability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confidence_histogram(df: pd.DataFrame, output_path: str | Path) -> None:
    _ensure_parent(output_path)
    plt.figure(figsize=(8, 5))
    plt.hist(df["raw_confidence"], bins=10, alpha=0.55, label="raw_confidence")
    plt.hist(df["repaired_confidence"], bins=10, alpha=0.55, label="repaired_confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Raw vs Repaired Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_raw_vs_repaired_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    _ensure_parent(output_path)
    colors = df["is_correct"].map({1: "tab:green", 0: "tab:red"}).fillna("tab:gray")
    plt.figure(figsize=(6, 6))
    plt.scatter(df["raw_confidence"], df["repaired_confidence"], c=colors, alpha=0.75)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Raw confidence")
    plt.ylabel("Repaired confidence")
    plt.title("Raw vs Repaired Confidence")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_signal_vs_error(
    df: pd.DataFrame,
    signal_col: str,
    output_path: str | Path,
    title: str,
) -> None:
    _ensure_parent(output_path)
    plot_df = df.dropna(subset=[signal_col, "is_correct"]).copy()
    if plot_df.empty:
        return
    plot_df["is_error"] = 1 - plot_df["is_correct"].astype(int)
    colors = plot_df["is_error"].map({0: "tab:green", 1: "tab:red"})
    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df[signal_col], plot_df["is_error"], c=colors, alpha=0.75)
    plt.xlabel(signal_col)
    plt.ylabel("Error indicator")
    plt.yticks([0, 1], ["correct", "wrong"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_signal_bin_table(df: pd.DataFrame, signal_col: str, n_bins: int = 5) -> pd.DataFrame:
    if signal_col not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=[signal_col, "is_correct"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["bin"] = pd.cut(work[signal_col], bins=n_bins, include_lowest=True)
    grouped = work.groupby("bin", observed=False)
    rows = []
    for interval, group in grouped:
        rows.append(
            {
                "signal": signal_col,
                "bin": str(interval),
                "count": int(len(group)),
                "avg_signal": float(group[signal_col].mean()),
                "accuracy": float(group["is_correct"].mean()),
                "error_rate": float(1.0 - group["is_correct"].mean()),
                "avg_raw_confidence": float(group["raw_confidence"].mean()),
                "avg_repaired_confidence": float(group["repaired_confidence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def diagnostics_summary(df: pd.DataFrame, high_conf_threshold: float) -> pd.DataFrame:
    raw_high = df["raw_confidence"] >= high_conf_threshold
    repaired_high = df["repaired_confidence"] >= high_conf_threshold
    wrong = df["is_correct"].astype(int) == 0

    rows: list[dict[str, Any]] = [
        {
            "num_samples": int(len(df)),
            "parse_success_rate": float(df["parse_success"].mean()) if "parse_success" in df.columns else float("nan"),
            "parse_failed_count": _safe_count(~df["parse_success"].astype(bool)) if "parse_success" in df.columns else 0,
            "raw_confidence_mean": _safe_mean(df, "raw_confidence"),
            "raw_confidence_min": float(df["raw_confidence"].min()),
            "raw_confidence_max": float(df["raw_confidence"].max()),
            "repaired_confidence_mean": _safe_mean(df, "repaired_confidence"),
            "repaired_confidence_min": float(df["repaired_confidence"].min()),
            "repaired_confidence_max": float(df["repaired_confidence"].max()),
            "abs_evidence_margin_mean": _safe_mean(df, "abs_evidence_margin"),
            "ambiguity_mean": _safe_mean(df, "ambiguity"),
            "missing_information_mean": _safe_mean(df, "missing_information"),
            "high_conf_threshold": float(high_conf_threshold),
            "raw_high_conf_count": _safe_count(raw_high),
            "raw_high_conf_wrong_count": _safe_count(raw_high & wrong),
            "repaired_high_conf_count": _safe_count(repaired_high),
            "repaired_high_conf_wrong_count": _safe_count(repaired_high & wrong),
        }
    ]
    return pd.DataFrame(rows)


def high_conf_wrong_cases(df: pd.DataFrame, threshold: float, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work["raw_over_repaired_gap"] = work["raw_confidence"] - work["repaired_confidence"]
    wrong = work["is_correct"].astype(int) == 0
    high_raw_wrong = work[(work["raw_confidence"] >= threshold) & wrong].copy()
    if high_raw_wrong.empty:
        high_raw_wrong = work[wrong].copy()
    columns = [
        "user_id",
        "candidate_item_id",
        "label",
        "recommend",
        "raw_confidence",
        "repaired_confidence",
        "raw_over_repaired_gap",
        "positive_evidence",
        "negative_evidence",
        "abs_evidence_margin",
        "ambiguity",
        "missing_information",
        "reason",
        "raw_response",
    ]
    available = [column for column in columns if column in high_raw_wrong.columns]
    return (
        high_raw_wrong.sort_values(
            ["raw_confidence", "raw_over_repaired_gap"],
            ascending=[False, False],
        )
        .head(top_n)[available]
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Evidence experiment name.")
    parser.add_argument("--output_root", type=str, default="output-repaired", help="Output root.")
    parser.add_argument("--raw_test_path", type=str, default=None, help="Optional raw test predictions path.")
    parser.add_argument("--repaired_test_path", type=str, default=None, help="Optional repaired test jsonl path.")
    parser.add_argument("--high_conf_threshold", type=float, default=0.8, help="High confidence threshold.")
    parser.add_argument("--top_n_cases", type=int, default=30, help="Number of case rows to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    raw_path = Path(args.raw_test_path) if args.raw_test_path else paths.predictions_dir / "test_raw.jsonl"
    repaired_path = (
        Path(args.repaired_test_path)
        if args.repaired_test_path
        else paths.calibrated_dir / "evidence_posterior_test.jsonl"
    )
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw test predictions not found: {raw_path}")
    if not repaired_path.exists():
        raise FileNotFoundError(f"Repaired test predictions not found: {repaired_path}")

    raw_df = build_evidence_feature_frame(load_jsonl(raw_path))
    repaired_df = build_evidence_feature_frame(load_jsonl(repaired_path))
    if "repaired_confidence" not in repaired_df.columns:
        raise ValueError("Repaired dataframe must contain `repaired_confidence`.")

    # Keep repaired output as the canonical table, while preserving raw fields for direct diagnosis.
    eval_df = repaired_df.copy()
    eval_df["raw_confidence"] = pd.to_numeric(eval_df["raw_confidence"], errors="coerce").clip(0.0, 1.0)
    eval_df["repaired_confidence"] = pd.to_numeric(eval_df["repaired_confidence"], errors="coerce").clip(0.0, 1.0)

    raw_eval = _with_confidence(raw_df, "raw_confidence")
    repaired_eval = _with_confidence(eval_df, "repaired_confidence")
    raw_metrics = compute_calibration_metrics(raw_eval, confidence_col="confidence")
    repaired_metrics = compute_calibration_metrics(repaired_eval, confidence_col="confidence")
    metrics_df = pd.DataFrame(
        [
            {"variant": "raw_confidence", **raw_metrics},
            {"variant": "repaired_confidence", **repaired_metrics},
        ]
    )
    save_table(metrics_df, paths.tables_dir / "evidence_posterior_diagnostic_metrics.csv")

    raw_rel = get_reliability_dataframe(
        raw_eval["is_correct"].to_numpy(),
        raw_eval["confidence"].to_numpy(),
        n_bins=10,
    )
    repaired_rel = get_reliability_dataframe(
        repaired_eval["is_correct"].to_numpy(),
        repaired_eval["confidence"].to_numpy(),
        n_bins=10,
    )
    save_table(raw_rel, paths.tables_dir / "reliability_raw_confidence.csv")
    save_table(repaired_rel, paths.tables_dir / "reliability_repaired_confidence.csv")

    summary_df = diagnostics_summary(eval_df, high_conf_threshold=args.high_conf_threshold)
    save_table(summary_df, paths.tables_dir / "evidence_posterior_diagnostics_summary.csv")

    bin_tables = [
        build_signal_bin_table(eval_df, "abs_evidence_margin"),
        build_signal_bin_table(eval_df, "ambiguity"),
        build_signal_bin_table(eval_df, "missing_information"),
    ]
    save_table(pd.concat(bin_tables, ignore_index=True), paths.tables_dir / "evidence_signal_bin_diagnostics.csv")

    cases_df = high_conf_wrong_cases(
        eval_df,
        threshold=args.high_conf_threshold,
        top_n=args.top_n_cases,
    )
    save_table(cases_df, paths.tables_dir / "top_high_conf_wrong_cases.csv")

    plot_reliability(raw_rel, repaired_rel, paths.figures_dir / "evidence_posterior_reliability.png")
    plot_confidence_histogram(eval_df, paths.figures_dir / "raw_vs_repaired_confidence_histogram.png")
    plot_raw_vs_repaired_scatter(eval_df, paths.figures_dir / "raw_vs_repaired_confidence_scatter.png")
    plot_signal_vs_error(
        eval_df,
        "abs_evidence_margin",
        paths.figures_dir / "evidence_margin_vs_error.png",
        "Evidence Margin vs Error",
    )
    plot_signal_vs_error(
        eval_df,
        "ambiguity",
        paths.figures_dir / "ambiguity_vs_error.png",
        "Ambiguity vs Error",
    )
    plot_signal_vs_error(
        eval_df,
        "missing_information",
        paths.figures_dir / "missing_information_vs_error.png",
        "Missing Information vs Error",
    )

    print(f"[{args.exp_name}] Evidence posterior diagnosis done.")
    print(f"Tables saved to:  {paths.tables_dir}")
    print(f"Figures saved to: {paths.figures_dir}")
    print(
        f"Raw ECE={raw_metrics['ece']:.4f}; repaired ECE={repaired_metrics['ece']:.4f}; "
        f"raw AUROC={raw_metrics['auroc']:.4f}; repaired AUROC={repaired_metrics['auroc']:.4f}"
    )


if __name__ == "__main__":
    main()
