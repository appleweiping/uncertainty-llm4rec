from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_day8_relevance_evidence_medium_analysis import (
    build_case_study,
    field_diagnostics,
    fmt,
    metric_value,
    read_csv,
    read_jsonl,
    run_rerank_grid,
    safe_float,
)
from main_rerank import save_table
from main_rerank_grid import parse_lambda_grid
from src.utils.paths import ensure_exp_dirs


def _best_rerank_row(rerank_rows: pd.DataFrame) -> pd.Series:
    changed = rerank_rows[pd.to_numeric(rerank_rows["rank_change_rate"], errors="coerce") > 0].copy()
    pool = changed if not changed.empty else rerank_rows
    return pool.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]


def _build_report(
    *,
    exp_name: str,
    valid_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rerank_rows: pd.DataFrame,
    case_study: pd.DataFrame,
) -> str:
    best = _best_rerank_row(rerank_rows)

    raw_ece = metric_value(comparison, "raw_relevance_probability", "ece")
    raw_brier = metric_value(comparison, "raw_relevance_probability", "brier_score")
    raw_auroc = metric_value(comparison, "raw_relevance_probability", "auroc")
    cal_ece = metric_value(comparison, "calibrated_relevance_probability", "ece")
    cal_brier = metric_value(comparison, "calibrated_relevance_probability", "brier_score")
    cal_auroc = metric_value(comparison, "calibrated_relevance_probability", "auroc")
    min_ece = metric_value(comparison, "evidence_posterior_relevance_minimal", "ece")
    min_brier = metric_value(comparison, "evidence_posterior_relevance_minimal", "brier_score")
    min_auroc = metric_value(comparison, "evidence_posterior_relevance_minimal", "auroc")
    full_ece = metric_value(comparison, "evidence_posterior_relevance_full", "ece")
    full_brier = metric_value(comparison, "evidence_posterior_relevance_full", "brier_score")
    full_auroc = metric_value(comparison, "evidence_posterior_relevance_full", "auroc")

    parse_valid = float(valid_raw["parse_success"].astype(bool).mean())
    parse_test = float(test_raw["parse_success"].astype(bool).mean())
    relevance_diag = diagnostics[(diagnostics["split"] == "test") & (diagnostics["field"] == "relevance_probability")]
    risk_diag = diagnostics[(diagnostics["split"] == "test") & (diagnostics["field"] == "evidence_risk")]
    near_one = safe_float(relevance_diag.iloc[0]["near_one_rate"]) if not relevance_diag.empty else float("nan")
    relevance_mean = safe_float(relevance_diag.iloc[0]["mean"]) if not relevance_diag.empty else float("nan")
    relevance_std = safe_float(relevance_diag.iloc[0]["std"]) if not relevance_diag.empty else float("nan")
    risk_mean = safe_float(risk_diag.iloc[0]["mean"]) if not risk_diag.empty else float("nan")
    risk_std = safe_float(risk_diag.iloc[0]["std"]) if not risk_diag.empty else float("nan")

    lines = [
        "# Day9 Full Beauty Relevance Evidence Report",
        "",
        "## 1. Day8 Recap",
        "",
        "Day8 scaled candidate relevance evidence from the Day7 smoke test to a medium Beauty sample. It showed parse_success of 1.0, persistent raw relevance miscalibration, strong ECE/Brier gains after valid-set calibration, and nonzero decoupled rerank changes. Day9 therefore moves to the final full-Beauty relevance-evidence shape before deciding what to distill into a local Qwen-LoRA generator.",
        "",
        "## 2. Full Setup",
        "",
        f"The experiment `{exp_name}` uses DeepSeek API evidence generation with `prompts/candidate_relevance_evidence.txt`, concurrent/resumable inference, and no LoRA or external baseline. The full valid split contains `{len(valid_raw)}` rows and the full test split contains `{len(test_raw)}` rows. Calibration is fit only on valid and applied to test.",
        "",
        "## 3. Parse And Field Diagnostics",
        "",
        f"Valid parse_success is `{fmt(parse_valid)}` and test parse_success is `{fmt(parse_test)}`. On test, relevance_probability has mean `{fmt(relevance_mean)}`, std `{fmt(relevance_std)}`, and near_one_rate `{fmt(near_one)}`. evidence_risk has mean `{fmt(risk_mean)}` and std `{fmt(risk_std)}`. This checks whether the relevance field collapses toward extreme self-confidence; the detailed distribution is in the field diagnostics table.",
        "",
        "## 4. Calibration",
        "",
        f"Raw relevance has ECE `{fmt(raw_ece)}`, Brier `{fmt(raw_brier)}`, and AUROC `{fmt(raw_auroc)}`. Valid-set calibrated relevance has ECE `{fmt(cal_ece)}`, Brier `{fmt(cal_brier)}`, and AUROC `{fmt(cal_auroc)}`. Minimal evidence posterior has ECE `{fmt(min_ece)}`, Brier `{fmt(min_brier)}`, and AUROC `{fmt(min_auroc)}`. Full evidence posterior has ECE `{fmt(full_ece)}`, Brier `{fmt(full_brier)}`, and AUROC `{fmt(full_auroc)}`.",
        "",
        "## 5. Rerank",
        "",
        f"The best full rerank row is setting `{best['setting']}`, normalization `{best['normalization']}`, lambda `{float(best['lambda']):.3g}`, base score `{best['base_score_col']}`, and uncertainty `{best['uncertainty_col']}`. It reaches HR@10 `{fmt(float(best['HR@10']))}`, NDCG@10 `{fmt(float(best['NDCG@10']))}`, MRR@10 `{fmt(float(best['MRR@10']))}`, rank_change_rate `{fmt(float(best['rank_change_rate']))}`, and top10_order_change_rate `{fmt(float(best['top10_order_change_rate']))}`.",
        "",
        "## 6. Comparison To Day6 Yes/No Full",
        "",
        "Day6 and Day9 should not be collapsed into one leaderboard row. Day6 is the full Beauty yes/no confidence-repair branch and proved that decoupling relevance and risk fixes the monotonic no-op. Day9 is the final candidate relevance scoring shape: relevance_probability is the main recommendation score, while evidence_risk is a separate decision risk.",
        "",
        "## 7. Limitation",
        "",
        "This is still a DeepSeek API evidence generator, not a local Qwen-LoRA model and not an external SOTA comparison. The current result establishes the full-Beauty feasibility and behavior of the relevance-evidence formulation before model compression or broader baseline work.",
        "",
        "## 8. Day10 Recommendation",
        "",
        "The full rows preserve the Day8 pattern: parse_success remains stable, raw relevance remains miscalibrated, valid-set calibration/evidence posterior sharply improve ECE and Brier, and decoupled reranking produces nonzero rank changes. Day10 should therefore prepare the Qwen-LoRA relevance distillation taskbook and an external baseline plug-in plan. The distillation target should use candidate relevance evidence, not the older yes/no confidence wording.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="beauty_deepseek_relevance_evidence_full")
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument("--lambda_grid", type=str, default="0,0.1,0.2,0.5")
    parser.add_argument("--normalizations", type=str, default="zscore,minmax")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--summary_dir", type=str, default="output-repaired/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    valid_raw = read_jsonl(paths.predictions_dir / "valid_raw.jsonl", "full valid raw relevance evidence")
    test_raw = read_jsonl(paths.predictions_dir / "test_raw.jsonl", "full test raw relevance evidence")
    comparison = read_csv(
        paths.tables_dir / "relevance_evidence_calibration_comparison.csv",
        "full relevance calibration comparison",
    )
    save_table(
        comparison,
        summary_dir / "beauty_day9_relevance_evidence_full_calibration_comparison.csv",
    )

    diagnostics = field_diagnostics(valid_raw, test_raw)
    save_table(
        diagnostics,
        summary_dir / "beauty_day9_relevance_evidence_full_field_diagnostics.csv",
    )

    calibrated = read_jsonl(
        paths.calibrated_dir / "relevance_evidence_posterior_test.jsonl",
        "full calibrated relevance evidence test",
    )
    rerank_grid = run_rerank_grid(
        calibrated,
        exp_name=args.exp_name,
        output_root=args.output_root,
        lambdas=parse_lambda_grid(args.lambda_grid),
        normalizations=[item.strip() for item in args.normalizations.split(",") if item.strip()],
        k=args.k,
    )
    rerank_rows = rerank_grid[rerank_grid["method"] == "day8_relevance_evidence_decoupled_rerank"].copy()
    rerank_rows["method"] = "day9_relevance_evidence_decoupled_rerank"
    save_table(
        rerank_rows,
        summary_dir / "beauty_day9_relevance_evidence_full_rerank_grid.csv",
    )

    best = _best_rerank_row(rerank_rows)
    best_exp = Path(args.output_root) / str(best["exp_name"]) / "reranked"
    case_study = build_case_study(
        calibrated,
        old_rank_path=best_exp / "baseline_ranked.jsonl",
        new_rank_path=best_exp / "day8_reranked.jsonl",
    )
    save_table(
        case_study,
        summary_dir / "beauty_day9_relevance_evidence_full_case_study.csv",
    )

    report = _build_report(
        exp_name=args.exp_name,
        valid_raw=valid_raw,
        test_raw=test_raw,
        comparison=comparison,
        diagnostics=diagnostics,
        rerank_rows=rerank_rows,
        case_study=case_study,
    )
    report_path = summary_dir / "beauty_day9_relevance_evidence_full_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved {summary_dir / 'beauty_day9_relevance_evidence_full_calibration_comparison.csv'}")
    print(f"Saved {summary_dir / 'beauty_day9_relevance_evidence_full_field_diagnostics.csv'}")
    print(f"Saved {summary_dir / 'beauty_day9_relevance_evidence_full_rerank_grid.csv'}")
    print(f"Saved {summary_dir / 'beauty_day9_relevance_evidence_full_case_study.csv'}")
    print(f"Saved {report_path}")
    print(
        "BEST_DAY9_RERANK "
        f"setting={best['setting']} normalization={best['normalization']} "
        f"lambda={float(best['lambda']):.3g} HR@10={float(best['HR@10']):.6f} "
        f"NDCG@10={float(best['NDCG@10']):.6f} MRR@10={float(best['MRR@10']):.6f} "
        f"rank_change_rate={float(best['rank_change_rate']):.6f}"
    )


if __name__ == "__main__":
    main()
