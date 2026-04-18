from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_ROOT = Path("outputs")
SUMMARY_PATH = OUTPUT_ROOT / "summary" / "week5_day4_multitask_eval_summary.csv"


def load_single_row_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Expected non-empty csv: {path}")
    return df.iloc[0]


def extract_pointwise_head_exposure(path: Path) -> float:
    exposure_df = pd.read_csv(path)
    if exposure_df.empty or "target_popularity_group" not in exposure_df.columns:
        return float("nan")

    head_rows = exposure_df[
        exposure_df["target_popularity_group"].astype(str).str.strip().str.lower() == "head"
    ]
    if head_rows.empty or "high_conf_fraction" not in head_rows.columns:
        return float("nan")
    return float(head_rows.iloc[0]["high_conf_fraction"])


def extract_prediction_stats(path: Path) -> tuple[float, float]:
    prediction_df = pd.read_json(path, lines=True)
    parse_success_rate = float(prediction_df["parse_success"].fillna(False).astype(bool).mean()) if "parse_success" in prediction_df.columns else float("nan")
    avg_latency = float(pd.to_numeric(prediction_df.get("latency"), errors="coerce").mean()) if "latency" in prediction_df.columns else float("nan")
    return parse_success_rate, avg_latency


def main() -> None:
    pointwise_metrics_path = OUTPUT_ROOT / "beauty_qwen_pointwise" / "tables" / "diagnostic_metrics.csv"
    pointwise_exposure_path = OUTPUT_ROOT / "beauty_qwen_pointwise" / "tables" / "high_confidence_exposure.csv"
    pointwise_predictions_path = OUTPUT_ROOT / "beauty_qwen_pointwise" / "predictions" / "test_raw.jsonl"
    ranking_metrics_path = OUTPUT_ROOT / "beauty_qwen_rank" / "tables" / "ranking_metrics.csv"
    pairwise_metrics_path = OUTPUT_ROOT / "beauty_qwen_pairwise" / "tables" / "pairwise_metrics.csv"

    pointwise_row = load_single_row_csv(pointwise_metrics_path)
    ranking_row = load_single_row_csv(ranking_metrics_path)
    pairwise_row = load_single_row_csv(pairwise_metrics_path)
    pointwise_parse_success_rate, pointwise_avg_latency = extract_prediction_stats(pointwise_predictions_path)

    rows = [
        {
            "task": "pointwise_yesno",
            "sample_count": pointwise_row.get("num_samples"),
            "HR@10": pd.NA,
            "NDCG@10": pd.NA,
            "MRR": pd.NA,
            "pairwise_accuracy": pd.NA,
            "ECE": pointwise_row.get("ece"),
            "Brier": pointwise_row.get("brier_score"),
            "coverage": pd.NA,
            "head_exposure": extract_pointwise_head_exposure(pointwise_exposure_path),
            "longtail_coverage": pd.NA,
            "parse_success_rate": pointwise_parse_success_rate,
            "avg_latency": pointwise_avg_latency,
            "source_path": str(pointwise_metrics_path),
        },
        {
            "task": "candidate_ranking",
            "sample_count": ranking_row.get("sample_count"),
            "HR@10": ranking_row.get("HR@10"),
            "NDCG@10": ranking_row.get("NDCG@10"),
            "MRR": ranking_row.get("MRR"),
            "pairwise_accuracy": pd.NA,
            "ECE": pd.NA,
            "Brier": pd.NA,
            "coverage": ranking_row.get("coverage@10"),
            "head_exposure": ranking_row.get("head_exposure_ratio@10"),
            "longtail_coverage": ranking_row.get("longtail_coverage@10"),
            "parse_success_rate": ranking_row.get("parse_success_rate"),
            "avg_latency": ranking_row.get("avg_latency"),
            "source_path": str(ranking_metrics_path),
        },
        {
            "task": "pairwise_preference",
            "sample_count": pairwise_row.get("sample_count"),
            "HR@10": pd.NA,
            "NDCG@10": pd.NA,
            "MRR": pd.NA,
            "pairwise_accuracy": pairwise_row.get("pairwise_accuracy"),
            "ECE": pd.NA,
            "Brier": pd.NA,
            "coverage": pd.NA,
            "head_exposure": pd.NA,
            "longtail_coverage": pd.NA,
            "parse_success_rate": pairwise_row.get("parse_success_rate"),
            "avg_latency": pairwise_row.get("avg_latency"),
            "source_path": str(pairwise_metrics_path),
        },
    ]

    summary_df = pd.DataFrame(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved multitask summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
