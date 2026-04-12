from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    return parser.parse_args()


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary file: {path}")
    return pd.read_csv(path)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_beauty_main_results(final_df: pd.DataFrame) -> pd.DataFrame:
    beauty_df = final_df[final_df["domain"].astype(str).str.lower() == "beauty"].copy()
    columns = [
        "model",
        "lambda",
        "diagnostic_accuracy",
        "diagnostic_ece",
        "diagnostic_brier_score",
        "calibration_test_ece_after",
        "calibration_test_brier_score_after",
        "rerank_hr_at_10",
        "rerank_ndcg_at_10",
        "rerank_mrr_at_10",
        "rerank_head_exposure_ratio_at_10",
        "rerank_long_tail_coverage_at_10",
    ]
    return beauty_df[[column for column in columns if column in beauty_df.columns]].sort_values(
        ["model", "lambda"]
    )


def build_beauty_estimator_brief(estimator_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "estimator",
        "lambda",
        "fusion_alpha",
        "num_eval_samples",
        "calibration_ece",
        "calibration_brier_score",
        "calibration_auroc",
        "rerank_ndcg_at_10",
        "rerank_mrr_at_10",
        "rerank_head_exposure_ratio_at_10",
        "rerank_long_tail_coverage_at_10",
    ]
    return estimator_df[[column for column in columns if column in estimator_df.columns]].sort_values(
        ["model", "estimator", "lambda"]
    )


def build_beauty_robustness_curve_brief(curve_df: pd.DataFrame) -> pd.DataFrame:
    beauty_df = curve_df[curve_df["domain"].astype(str).str.lower() == "beauty"].copy()
    columns = [
        "model",
        "clean_exp",
        "noisy_exp",
        "noise_level",
        "rerank_HR@10_drop",
        "rerank_NDCG@10_drop",
        "rerank_MRR@10_drop",
        "calibration_ece_after_drop",
        "calibration_brier_score_after_drop",
        "confidence_wrong_high_conf_fraction_drop",
    ]
    return beauty_df[[column for column in columns if column in beauty_df.columns]].sort_values(
        ["model", "noise_level"]
    )


def build_beauty_reproducibility_brief(repro_df: pd.DataFrame) -> pd.DataFrame:
    beauty_df = repro_df[repro_df["domain"].astype(str).str.lower() == "beauty"].copy()
    columns = [
        "setting",
        "model",
        "run_a",
        "run_b",
        "diagnostic_ece_abs_diff",
        "calibration_test_ece_after_abs_diff",
        "calibration_test_brier_score_after_abs_diff",
        "rerank_hr_at_10_abs_diff",
        "rerank_ndcg_at_10_abs_diff",
        "rerank_mrr_at_10_abs_diff",
        "max_abs_diff",
        "mean_abs_diff",
    ]
    return beauty_df[[column for column in columns if column in beauty_df.columns]].sort_values(
        ["model", "setting"]
    )


def build_beauty_consistency_sensitivity_brief(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "exp_name",
        "temperature",
        "num_samples",
        "avg_vote_entropy",
        "nonzero_entropy_fraction",
        "high_entropy_fraction",
        "avg_consistency_confidence",
        "avg_consistency_uncertainty",
        "consistency_ece",
        "consistency_brier_score",
        "fused_ece",
        "fused_brier_score",
        "consistency_ece_delta_vs_t00",
        "fused_ece_delta_vs_t00",
    ]
    return sensitivity_df[
        [column for column in columns if column in sensitivity_df.columns]
    ].sort_values(["temperature"])


def build_beauty_fused_alpha_brief(alpha_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "estimator",
        "fusion_alpha",
        "num_eval_samples",
        "calibration_ece",
        "calibration_brier_score",
        "calibration_auroc",
        "rerank_ndcg_at_10",
        "rerank_mrr_at_10",
    ]
    sort_cols = [col for col in ["estimator", "fusion_alpha"] if col in alpha_df.columns]
    return alpha_df[[column for column in columns if column in alpha_df.columns]].sort_values(sort_cols)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    summary_dir = output_root / "summary"

    final_df = load_required_csv(summary_dir / "final_results.csv")
    estimator_df = load_required_csv(summary_dir / "beauty_estimator_results.csv")
    robustness_curve_df = load_required_csv(summary_dir / "robustness_curve_results.csv")
    reproducibility_df = load_required_csv(summary_dir / "reproducibility_delta.csv")
    sensitivity_df = load_optional_csv(summary_dir / "beauty_consistency_sensitivity.csv")
    fused_alpha_df = load_optional_csv(summary_dir / "beauty_fused_alpha_ablation.csv")

    beauty_main_results = build_beauty_main_results(final_df)
    beauty_estimator_brief = build_beauty_estimator_brief(estimator_df)
    beauty_robustness_curve_brief = build_beauty_robustness_curve_brief(robustness_curve_df)
    beauty_reproducibility_brief = build_beauty_reproducibility_brief(reproducibility_df)

    beauty_main_results.to_csv(summary_dir / "beauty_main_results.csv", index=False)
    beauty_estimator_brief.to_csv(summary_dir / "beauty_estimator_brief.csv", index=False)
    beauty_robustness_curve_brief.to_csv(
        summary_dir / "beauty_robustness_curve_brief.csv", index=False
    )
    beauty_reproducibility_brief.to_csv(
        summary_dir / "beauty_reproducibility_brief.csv", index=False
    )
    if sensitivity_df is not None and not sensitivity_df.empty:
        build_beauty_consistency_sensitivity_brief(sensitivity_df).to_csv(
            summary_dir / "beauty_consistency_sensitivity_brief.csv", index=False
        )
    if fused_alpha_df is not None and not fused_alpha_df.empty:
        build_beauty_fused_alpha_brief(fused_alpha_df).to_csv(
            summary_dir / "beauty_fused_alpha_brief.csv", index=False
        )

    print("Saved Beauty paper-facing tables:")
    print(f"- {summary_dir / 'beauty_main_results.csv'}")
    print(f"- {summary_dir / 'beauty_estimator_brief.csv'}")
    print(f"- {summary_dir / 'beauty_robustness_curve_brief.csv'}")
    print(f"- {summary_dir / 'beauty_reproducibility_brief.csv'}")
    if sensitivity_df is not None and not sensitivity_df.empty:
        print(f"- {summary_dir / 'beauty_consistency_sensitivity_brief.csv'}")
    if fused_alpha_df is not None and not fused_alpha_df.empty:
        print(f"- {summary_dir / 'beauty_fused_alpha_brief.csv'}")


if __name__ == "__main__":
    main()
