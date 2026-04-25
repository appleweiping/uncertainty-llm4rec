from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def safe_metric(df: pd.DataFrame, split: str, variant: str, metric: str) -> float:
    row = df[(df["split"] == split) & (df["variant"] == variant)]
    if row.empty or metric not in row.columns:
        return float("nan")
    return float(row.iloc[0][metric])


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
    rerank: pd.DataFrame,
) -> str:
    valid_parse = float(valid_raw["parse_success"].astype(bool).mean()) if "parse_success" in valid_raw.columns else float("nan")
    test_parse = float(test_raw["parse_success"].astype(bool).mean()) if "parse_success" in test_raw.columns else float("nan")
    test_relevance_mean = float(pd.to_numeric(test_raw["relevance_probability"], errors="coerce").mean())
    test_risk_mean = float(pd.to_numeric(test_raw["evidence_risk"], errors="coerce").mean())

    raw_ece = safe_metric(comparison, "test", "raw_relevance_probability", "ece")
    raw_brier = safe_metric(comparison, "test", "raw_relevance_probability", "brier_score")
    raw_auroc = safe_metric(comparison, "test", "raw_relevance_probability", "auroc")
    cal_ece = safe_metric(comparison, "test", "calibrated_relevance_probability", "ece")
    cal_brier = safe_metric(comparison, "test", "calibrated_relevance_probability", "brier_score")
    cal_auroc = safe_metric(comparison, "test", "calibrated_relevance_probability", "auroc")
    min_ece = safe_metric(comparison, "test", "evidence_posterior_relevance_minimal", "ece")
    min_brier = safe_metric(comparison, "test", "evidence_posterior_relevance_minimal", "brier_score")
    min_auroc = safe_metric(comparison, "test", "evidence_posterior_relevance_minimal", "auroc")
    full_ece = safe_metric(comparison, "test", "evidence_posterior_relevance_full", "ece")
    full_brier = safe_metric(comparison, "test", "evidence_posterior_relevance_full", "brier_score")
    full_auroc = safe_metric(comparison, "test", "evidence_posterior_relevance_full", "auroc")

    rerank_rows = rerank[rerank["method"] == "relevance_evidence_decoupled_rerank"].copy()
    best = rerank_rows.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0] if not rerank_rows.empty else None

    lines = [
        "# Day7 Candidate Relevance Evidence Report",
        "",
        "## 1. Motivation",
        "",
        "The previous yes/no evidence posterior branch is a controlled confidence-repair benchmark. Day7 migrates the same evidence-posterior idea into a more recommendation-native candidate relevance scoring setting, where the primary model output is `relevance_probability` rather than a self-reported yes/no confidence.",
        "",
        "## 2. Schema",
        "",
        "`relevance_probability` estimates whether the candidate matches the user history. It is not named or treated as `raw_confidence`. Evidence fields remain explicit: `positive_evidence`, `negative_evidence`, `ambiguity`, and `missing_information`. The derived decision risk is `evidence_risk = (1 - abs_evidence_margin + ambiguity + missing_information) / 3`.",
        "",
        "## 3. Parser And Inference",
        "",
        f"The Beauty DeepSeek relevance-evidence smoke test uses `{exp_name}` with 100 valid rows and 100 test rows. Valid parse_success is `{fmt(valid_parse)}` and test parse_success is `{fmt(test_parse)}`. On the test split, average relevance_probability is `{fmt(test_relevance_mean)}` and average evidence_risk is `{fmt(test_risk_mean)}`.",
        "",
        "## 4. Calibration Smoke Test",
        "",
        f"On the test split, raw relevance has ECE `{fmt(raw_ece)}`, Brier `{fmt(raw_brier)}`, and AUROC `{fmt(raw_auroc)}`. Valid-set calibrated relevance has ECE `{fmt(cal_ece)}`, Brier `{fmt(cal_brier)}`, and AUROC `{fmt(cal_auroc)}`. Minimal evidence posterior relevance has ECE `{fmt(min_ece)}`, Brier `{fmt(min_brier)}`, and AUROC `{fmt(min_auroc)}`. Full evidence posterior relevance has ECE `{fmt(full_ece)}`, Brier `{fmt(full_brier)}`, and AUROC `{fmt(full_auroc)}`.",
        "",
        "## 5. Rerank Smoke Test",
        "",
    ]
    if best is not None:
        lines.append(
            f"The decoupled rerank smoke test runs lambda in {{0, 0.1, 0.2}} with `base_score = relevance_probability` and `uncertainty = evidence_risk`. The best row uses lambda `{float(best['lambda_penalty']):.3g}` with NDCG@10 `{fmt(float(best['NDCG@10']))}`, MRR@10 `{fmt(float(best['MRR@10']))}`, rank_change_rate `{fmt(float(best['rank_change_rate']))}`, and top10_order_change_rate `{fmt(float(best['top10_order_change_rate']))}`."
        )
    else:
        lines.append("The rerank smoke test did not produce a usable decoupled row.")
    lines.extend(
        [
            "",
            "## 6. Limitation",
            "",
            "This is a Beauty 100 schema smoke test, not a full-domain conclusion. Its purpose is to verify that candidate relevance scoring, evidence posterior calibration, and decoupled relevance-risk reranking can run end-to-end without confusing relevance probability with confidence.",
            "",
            "## 7. Next Step",
            "",
            "If the parser and calibration remain stable, Day8 should expand candidate relevance evidence to a larger Beauty sample or full Beauty. If parse_success or calibration is unstable, the next step should be prompt/parser repair before scaling.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="beauty_deepseek_relevance_evidence_100")
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument(
        "--rerank_summary_path",
        type=str,
        default="output-repaired/summary/beauty_day7_relevance_evidence_rerank_smoke.csv",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="output-repaired/summary/beauty_day7_candidate_relevance_evidence_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root) / args.exp_name
    valid_raw = read_jsonl(root / "predictions" / "valid_raw.jsonl", "valid raw relevance evidence")
    test_raw = read_jsonl(root / "predictions" / "test_raw.jsonl", "test raw relevance evidence")
    comparison = read_csv(
        root / "tables" / "relevance_evidence_calibration_comparison.csv",
        "relevance calibration comparison",
    )
    rerank = read_csv(args.rerank_summary_path, "relevance rerank smoke")

    report = build_report(
        exp_name=args.exp_name,
        valid_raw=valid_raw,
        test_raw=test_raw,
        comparison=comparison,
        rerank=rerank,
    )
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved Day7 report to: {report_path}")


if __name__ == "__main__":
    main()
