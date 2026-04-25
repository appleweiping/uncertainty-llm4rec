from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from main_rerank import build_result_row, load_jsonl, save_jsonl, save_table
from main_rerank_grid import lambda_slug, parse_lambda_grid, usable_rerank_rows, validate_input_columns
from src.analysis.rerank_diagnostics import add_user_normalized_column, compute_rank_change_diagnostics
from src.eval.bias_metrics import compute_topk_exposure_distribution
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rank_by_rerank_score
from src.utils.paths import ensure_exp_dirs


FIELD_COLUMNS = [
    "relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "evidence_risk",
]

RERANK_SETTINGS = {
    "R-B": {
        "base_score_col": "relevance_probability",
        "uncertainty_col": "evidence_risk",
    },
    "R-C": {
        "base_score_col": "calibrated_relevance_probability",
        "uncertainty_col": "evidence_risk",
    },
}

CASE_COLUMNS = [
    "case_type",
    "user_id",
    "candidate_item_id",
    "label",
    "relevance_probability",
    "calibrated_relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "evidence_risk",
    "old_rank",
    "new_rank",
    "rank_delta",
    "candidate_title_or_text",
    "reason",
]


def read_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def read_jsonl(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_json(path, lines=True)


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def field_diagnostics(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in [("valid", valid_df), ("test", test_df)]:
        for field in FIELD_COLUMNS:
            values = pd.to_numeric(df[field], errors="coerce").dropna()
            row = {
                "split": split,
                "field": field,
                "count": int(len(values)),
                "mean": float(values.mean()) if len(values) else float("nan"),
                "std": float(values.std()) if len(values) else float("nan"),
                "min": float(values.min()) if len(values) else float("nan"),
                "q05": float(values.quantile(0.05)) if len(values) else float("nan"),
                "q25": float(values.quantile(0.25)) if len(values) else float("nan"),
                "q50": float(values.quantile(0.50)) if len(values) else float("nan"),
                "q75": float(values.quantile(0.75)) if len(values) else float("nan"),
                "q95": float(values.quantile(0.95)) if len(values) else float("nan"),
                "max": float(values.max()) if len(values) else float("nan"),
                "near_zero_rate": float((values <= 0.05).mean()) if len(values) else float("nan"),
                "near_one_rate": float((values >= 0.95).mean()) if len(values) else float("nan"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_setting(df: pd.DataFrame, base_score_col: str, uncertainty_col: str, normalization: str) -> pd.DataFrame:
    out = add_user_normalized_column(
        df,
        base_score_col,
        "normalized_base_score",
        method=normalization,
    )
    out = add_user_normalized_column(
        out,
        uncertainty_col,
        "normalized_uncertainty",
        method=normalization,
    )
    return out


def run_rerank_setting(
    *,
    df: pd.DataFrame,
    exp_name: str,
    output_root: str,
    setting: str,
    lambda_penalty: float,
    normalization: str,
    base_score_col: str,
    uncertainty_col: str,
    k: int,
) -> pd.DataFrame:
    paths = ensure_exp_dirs(exp_name, output_root)
    prepared = prepare_setting(
        df,
        base_score_col=base_score_col,
        uncertainty_col=uncertainty_col,
        normalization=normalization,
    )

    baseline_df = add_baseline_score(prepared, score_col="normalized_base_score", output_col="baseline_score")
    baseline_ranked = rank_by_score(
        baseline_df,
        user_col="user_id",
        score_col="baseline_score",
        rank_col="rank",
    )
    save_jsonl(baseline_ranked, paths.reranked_dir / "baseline_ranked.jsonl")

    rerank_scored = prepared.copy()
    rerank_scored["final_score"] = (
        rerank_scored["normalized_base_score"].astype(float)
        - float(lambda_penalty) * rerank_scored["normalized_uncertainty"].astype(float)
    )
    rerank_ranked = rank_by_rerank_score(
        rerank_scored,
        user_col="user_id",
        score_col="final_score",
        rank_col="rank",
    )
    save_jsonl(rerank_ranked, paths.reranked_dir / "day8_reranked.jsonl")

    baseline_row = build_result_row(f"baseline_{setting}", baseline_ranked, k=k)
    rerank_row = build_result_row(
        "day8_relevance_evidence_decoupled_rerank",
        rerank_ranked,
        k=k,
        lambda_penalty=lambda_penalty,
    )
    result = pd.concat([baseline_row, rerank_row], ignore_index=True)
    diagnostics = compute_rank_change_diagnostics(
        baseline_ranked,
        rerank_ranked,
        base_score_col=base_score_col,
        uncertainty_col=uncertainty_col,
        lambda_penalty=lambda_penalty,
        setting=setting,
        normalization=normalization,
        k=k,
    )
    for key, value in diagnostics.items():
        result[key] = value
    result["setting"] = setting
    result["normalization"] = normalization
    result["base_score_col"] = base_score_col
    result["uncertainty_col"] = uncertainty_col
    result["exp_name"] = exp_name
    save_table(result, paths.tables_dir / "rerank_results.csv")
    save_table(result, paths.reranked_dir / "rerank_results.csv")

    baseline_dist = compute_topk_exposure_distribution(baseline_ranked, k=k)
    baseline_dist["method"] = f"baseline_{setting}"
    baseline_dist["lambda_penalty"] = float(lambda_penalty)
    rerank_dist = compute_topk_exposure_distribution(rerank_ranked, k=k)
    rerank_dist["method"] = "day8_relevance_evidence_decoupled_rerank"
    rerank_dist["lambda_penalty"] = float(lambda_penalty)
    save_table(
        pd.concat([baseline_dist, rerank_dist], ignore_index=True),
        paths.tables_dir / "topk_exposure_distribution.csv",
    )
    return result


def run_rerank_grid(
    df: pd.DataFrame,
    *,
    exp_name: str,
    output_root: str,
    lambdas: list[float],
    normalizations: list[str],
    k: int,
) -> pd.DataFrame:
    required = ["user_id", "candidate_item_id", "label", "target_popularity_group", "evidence_risk"]
    for spec in RERANK_SETTINGS.values():
        required.append(spec["base_score_col"])
    validate_input_columns(df, sorted(set(required)))

    all_results: list[pd.DataFrame] = []
    for setting, spec in RERANK_SETTINGS.items():
        usable = usable_rerank_rows(
            df,
            score_column=spec["base_score_col"],
            uncertainty_column=spec["uncertainty_col"],
        )
        if usable.empty:
            continue
        for normalization in normalizations:
            for lambda_penalty in lambdas:
                run_name = f"{exp_name}_day8_{setting.lower()}_{normalization}_{lambda_slug(lambda_penalty)}"
                result = run_rerank_setting(
                    df=usable,
                    exp_name=run_name,
                    output_root=output_root,
                    setting=setting,
                    lambda_penalty=lambda_penalty,
                    normalization=normalization,
                    base_score_col=spec["base_score_col"],
                    uncertainty_col=spec["uncertainty_col"],
                    k=k,
                )
                result["base_exp_name"] = exp_name
                result["usable_rows"] = int(len(usable))
                result["total_rows"] = int(len(df))
                all_results.append(result)
    return pd.concat(all_results, ignore_index=True)


def extract_candidate_title(prompt: Any, max_len: int = 180) -> str:
    text = str(prompt or "")
    match = re.search(r"Title:\s*(.*?)\nDescription:", text, re.S)
    if not match:
        match = re.search(r"candidate item:\s*(.*?)(?:\n\n|$)", text, re.S | re.I)
    title = " ".join((match.group(1) if match else text[:max_len]).split())
    return title[:max_len]


def select_case_rows(df: pd.DataFrame, case_type: str, limit: int = 10) -> pd.DataFrame:
    work = df.copy()
    if case_type == "promoted":
        work = work[work["rank_delta"] > 0].sort_values(["rank_delta", "evidence_risk"], ascending=[False, True])
    elif case_type == "demoted":
        work = work[work["rank_delta"] < 0].sort_values(["rank_delta", "evidence_risk"], ascending=[True, False])
    elif case_type == "high_risk_demoted":
        threshold = work["evidence_risk"].quantile(0.75)
        work = work[(work["old_rank"] <= 3) & (work["rank_delta"] < 0) & (work["evidence_risk"] >= threshold)]
        work = work.sort_values(["evidence_risk", "rank_delta"], ascending=[False, True])
    else:
        raise ValueError(f"Unknown case_type: {case_type}")
    out = work.head(limit).copy()
    out["case_type"] = case_type
    return out


def build_case_study(
    calibrated_df: pd.DataFrame,
    old_rank_path: Path,
    new_rank_path: Path,
) -> pd.DataFrame:
    old_ranked = read_jsonl(old_rank_path, "baseline ranked")[["user_id", "candidate_item_id", "rank"]].rename(
        columns={"rank": "old_rank"}
    )
    new_ranked = read_jsonl(new_rank_path, "best reranked")[["user_id", "candidate_item_id", "rank"]].rename(
        columns={"rank": "new_rank"}
    )
    merged = calibrated_df.merge(old_ranked, on=["user_id", "candidate_item_id"], how="inner").merge(
        new_ranked,
        on=["user_id", "candidate_item_id"],
        how="inner",
    )
    merged["rank_delta"] = merged["old_rank"] - merged["new_rank"]
    merged["candidate_title_or_text"] = merged["prompt"].apply(extract_candidate_title)
    cases = pd.concat(
        [
            select_case_rows(merged, "promoted"),
            select_case_rows(merged, "demoted"),
            select_case_rows(merged, "high_risk_demoted"),
        ],
        ignore_index=True,
    )
    return cases[[column for column in CASE_COLUMNS if column in cases.columns]].copy()


def metric_value(comparison: pd.DataFrame, variant: str, metric: str, split: str = "test") -> float:
    row = comparison[(comparison["split"] == split) & (comparison["variant"] == variant)]
    if row.empty or metric not in row.columns:
        return float("nan")
    return safe_float(row.iloc[0][metric])


def fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def build_report(
    *,
    exp_name: str,
    valid_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rerank_grid: pd.DataFrame,
    case_study: pd.DataFrame,
) -> str:
    test_rows = rerank_grid[rerank_grid["method"] == "day8_relevance_evidence_decoupled_rerank"].copy()
    best = test_rows.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]
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
    relevance_mean = safe_float(relevance_diag.iloc[0]["mean"]) if not relevance_diag.empty else float("nan")
    relevance_std = safe_float(relevance_diag.iloc[0]["std"]) if not relevance_diag.empty else float("nan")
    risk_mean = safe_float(risk_diag.iloc[0]["mean"]) if not risk_diag.empty else float("nan")
    risk_std = safe_float(risk_diag.iloc[0]["std"]) if not risk_diag.empty else float("nan")
    high_risk_cases = case_study[case_study["case_type"] == "high_risk_demoted"]
    avg_high_risk_delta = safe_float(high_risk_cases["rank_delta"].mean()) if not high_risk_cases.empty else float("nan")

    lines = [
        "# Day8 Relevance Evidence Medium Report",
        "",
        "## 1. Day7 Recap",
        "",
        "Day7 verified that the candidate relevance evidence schema, parser, valid-set calibration, and decoupled rerank smoke test can run on Beauty 100. Day8 scales this path to a medium Beauty sample without moving to full data, LoRA, or external baselines.",
        "",
        "## 2. Day8 Setup",
        "",
        f"The experiment `{exp_name}` uses `prompts/candidate_relevance_evidence.txt` with 1000 valid rows and 1000 test rows. The main score remains `relevance_probability`, not `raw_confidence`. Calibration produces `calibrated_relevance_probability`, and uncertainty is represented by `relevance_uncertainty` or the decoupled `evidence_risk` used for reranking.",
        "",
        "## 3. Parse And Field Diagnostics",
        "",
        f"Valid parse_success is `{fmt(parse_valid)}` and test parse_success is `{fmt(parse_test)}`. On the test split, relevance_probability has mean `{fmt(relevance_mean)}` and std `{fmt(relevance_std)}`, while evidence_risk has mean `{fmt(risk_mean)}` and std `{fmt(risk_std)}`. The field diagnostics table records mean/std/min/max/quantiles and near-extreme rates for every evidence field.",
        "",
        "## 4. Calibration Result",
        "",
        f"Raw relevance_probability has ECE `{fmt(raw_ece)}`, Brier `{fmt(raw_brier)}`, and AUROC `{fmt(raw_auroc)}`. Valid-set calibrated relevance has ECE `{fmt(cal_ece)}`, Brier `{fmt(cal_brier)}`, and AUROC `{fmt(cal_auroc)}`. Minimal evidence posterior has ECE `{fmt(min_ece)}`, Brier `{fmt(min_brier)}`, and AUROC `{fmt(min_auroc)}`. Full evidence posterior has ECE `{fmt(full_ece)}`, Brier `{fmt(full_brier)}`, and AUROC `{fmt(full_auroc)}`.",
        "",
        "## 5. Rerank Result",
        "",
        f"The best medium-sample rerank row is setting `{best['setting']}` with `{best['normalization']}` normalization and lambda `{float(best['lambda']):.3g}`. It reaches NDCG@10 `{fmt(float(best['NDCG@10']))}`, MRR@10 `{fmt(float(best['MRR@10']))}`, rank_change_rate `{fmt(float(best['rank_change_rate']))}`, and top10_order_change_rate `{fmt(float(best['top10_order_change_rate']))}`. Day8 does not compare directly against Day6 because Day6 is the yes/no full-Beauty branch and Day8 is the relevance medium branch.",
        "",
        "## 6. Case Study",
        "",
        f"The case study exports promoted, demoted, and high-risk-demoted rows for the best nonzero-change setting. In the high-risk-demoted group, average rank_delta is `{fmt(avg_high_risk_delta)}`; negative values indicate that high-risk candidates move down under the decoupled evidence-risk penalty.",
        "",
        "## 7. Decision For Day9",
        "",
        "If the calibration rows show stable ECE/Brier improvement and rerank rows show nonzero rank changes, Day9 can expand relevance evidence to full Beauty or prepare the Qwen-LoRA relevance generator. If calibration is unstable or fields collapse toward extremes, the next step should be prompt/schema/feature repair before full scaling.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="beauty_deepseek_relevance_evidence_1000")
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument("--lambda_grid", type=str, default="0,0.1,0.2,0.5")
    parser.add_argument("--normalizations", type=str, default="zscore,minmax")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="output-repaired/summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    valid_raw = read_jsonl(paths.predictions_dir / "valid_raw.jsonl", "valid raw relevance evidence")
    test_raw = read_jsonl(paths.predictions_dir / "test_raw.jsonl", "test raw relevance evidence")
    comparison = read_csv(
        paths.tables_dir / "relevance_evidence_calibration_comparison.csv",
        "relevance calibration comparison",
    )

    save_table(
        comparison,
        summary_dir / "beauty_day8_relevance_evidence_calibration_comparison.csv",
    )

    diagnostics = field_diagnostics(valid_raw, test_raw)
    save_table(
        diagnostics,
        summary_dir / "beauty_day8_relevance_evidence_field_diagnostics.csv",
    )

    calibrated_path = paths.calibrated_dir / "relevance_evidence_posterior_test.jsonl"
    calibrated = read_jsonl(calibrated_path, "calibrated relevance evidence test")
    lambdas = parse_lambda_grid(args.lambda_grid)
    normalizations = [item.strip() for item in args.normalizations.split(",") if item.strip()]
    rerank_grid = run_rerank_grid(
        calibrated,
        exp_name=args.exp_name,
        output_root=args.output_root,
        lambdas=lambdas,
        normalizations=normalizations,
        k=args.k,
    )
    rerank_rows = rerank_grid[rerank_grid["method"] == "day8_relevance_evidence_decoupled_rerank"].copy()
    save_table(
        rerank_rows,
        summary_dir / "beauty_day8_relevance_evidence_rerank_grid.csv",
    )

    nonzero = rerank_rows[rerank_rows["rank_change_rate"].astype(float) > 0].copy()
    best_pool = nonzero if not nonzero.empty else rerank_rows
    best = best_pool.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]
    best_exp = Path(args.output_root) / str(best["exp_name"]) / "reranked"
    case_study = build_case_study(
        calibrated,
        old_rank_path=best_exp / "baseline_ranked.jsonl",
        new_rank_path=best_exp / "day8_reranked.jsonl",
    )
    save_table(
        case_study,
        summary_dir / "beauty_day8_relevance_evidence_case_study.csv",
    )

    report = build_report(
        exp_name=args.exp_name,
        valid_raw=valid_raw,
        test_raw=test_raw,
        comparison=comparison,
        diagnostics=diagnostics,
        rerank_grid=rerank_rows,
        case_study=case_study,
    )
    report_path = summary_dir / "beauty_day8_relevance_evidence_medium_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved {summary_dir / 'beauty_day8_relevance_evidence_calibration_comparison.csv'}")
    print(f"Saved {summary_dir / 'beauty_day8_relevance_evidence_field_diagnostics.csv'}")
    print(f"Saved {summary_dir / 'beauty_day8_relevance_evidence_rerank_grid.csv'}")
    print(f"Saved {summary_dir / 'beauty_day8_relevance_evidence_case_study.csv'}")
    print(f"Saved {report_path}")
    print(
        "BEST_DAY8_RERANK "
        f"setting={best['setting']} normalization={best['normalization']} "
        f"lambda={float(best['lambda']):.3g} NDCG@10={float(best['NDCG@10']):.6f} "
        f"MRR@10={float(best['MRR@10']):.6f} rank_change_rate={float(best['rank_change_rate']):.6f}"
    )


if __name__ == "__main__":
    main()
