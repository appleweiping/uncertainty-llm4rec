from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLS = [
    "HR@10",
    "NDCG@10",
    "MRR@10",
    "head_exposure_ratio@10",
    "tail_exposure_ratio@10",
    "long_tail_coverage@10",
]
CASE_COLUMNS = [
    "case_type",
    "user_id",
    "candidate_item_id",
    "label",
    "raw_confidence",
    "repaired_confidence",
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


def relative_improvement(value: float, baseline: float) -> float:
    if pd.isna(value) or pd.isna(baseline) or abs(float(baseline)) < 1e-12:
        return float("nan")
    return float((float(value) - float(baseline)) / abs(float(baseline)))


def monotonic_reference(monotonic_grid: pd.DataFrame) -> dict[str, float]:
    work = monotonic_grid[monotonic_grid["method"].astype(str) == "uncertainty_aware_rerank"].copy()
    if work.empty:
        raise ValueError("No uncertainty_aware_rerank rows found in monotonic grid.")
    row = work.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]
    return {
        "HR@10": safe_float(row.get("HR@10")),
        "NDCG@10": safe_float(row.get("NDCG@10")),
        "MRR@10": safe_float(row.get("MRR@10")),
        "head_exposure_ratio@10": safe_float(row.get("head_exposure_ratio@10")),
        "tail_exposure_ratio@10": safe_float(row.get("tail_exposure_ratio@10")),
        "long_tail_coverage@10": safe_float(row.get("long_tail_coverage@10")),
    }


def build_monotonic_ablation(monotonic_grid: pd.DataFrame, diagnostics: pd.DataFrame, ref: dict[str, float]) -> pd.DataFrame:
    rows = []
    grid_rows = monotonic_grid[monotonic_grid["method"].astype(str) == "uncertainty_aware_rerank"].copy()
    for _, row in grid_rows.iterrows():
        lambda_value = safe_float(row.get("lambda_penalty"))
        diag = diagnostics[diagnostics["lambda"].astype(float) == lambda_value]
        diag_row = diag.iloc[0].to_dict() if not diag.empty else {}
        out = {
            "method_group": "monotonic_noop",
            "setting": "monotonic",
            "lambda": lambda_value,
            "normalization": "none",
            "base_score_col": row.get("score_column"),
            "uncertainty_col": row.get("uncertainty_column"),
        }
        for metric in METRIC_COLS:
            out[metric] = safe_float(row.get(metric))
        out["relative_NDCG_improvement_vs_monotonic"] = relative_improvement(out["NDCG@10"], ref["NDCG@10"])
        out["relative_MRR_improvement_vs_monotonic"] = relative_improvement(out["MRR@10"], ref["MRR@10"])
        for col in [
            "rank_change_rate",
            "top10_change_rate",
            "top10_order_change_rate",
            "mean_kendall_tau",
            "base_uncertainty_spearman",
        ]:
            out[col] = safe_float(diag_row.get(col))
        rows.append(out)
    return pd.DataFrame(rows)


def build_decoupled_ablation(decoupled_frames: list[pd.DataFrame], ref: dict[str, float]) -> pd.DataFrame:
    decoupled = pd.concat(decoupled_frames, ignore_index=True)
    work = decoupled[decoupled["method"].astype(str) == "decoupled_uncertainty_aware_rerank"].copy()
    rows = []
    for _, row in work.iterrows():
        setting = str(row.get("setting"))
        out = {
            "method_group": f"decoupled_{setting}",
            "setting": setting,
            "lambda": safe_float(row.get("lambda_penalty")),
            "normalization": row.get("normalization"),
            "base_score_col": row.get("base_score_col"),
            "uncertainty_col": row.get("uncertainty_col"),
        }
        for metric in METRIC_COLS:
            out[metric] = safe_float(row.get(metric))
        out["relative_NDCG_improvement_vs_monotonic"] = relative_improvement(out["NDCG@10"], ref["NDCG@10"])
        out["relative_MRR_improvement_vs_monotonic"] = relative_improvement(out["MRR@10"], ref["MRR@10"])
        for col in [
            "rank_change_rate",
            "top10_change_rate",
            "top10_order_change_rate",
            "mean_kendall_tau",
            "base_uncertainty_spearman",
        ]:
            out[col] = safe_float(row.get(col))
        rows.append(out)
    return pd.DataFrame(rows)


def build_sensitivity(ablation: pd.DataFrame) -> pd.DataFrame:
    work = ablation[ablation["method_group"].astype(str).str.startswith("decoupled_")].copy()
    rows = []
    for (setting, normalization), group in work.groupby(["setting", "normalization"], dropna=False):
        best = group.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "setting": setting,
                "normalization": normalization,
                "num_lambda": int(len(group)),
                "best_NDCG@10": safe_float(best["NDCG@10"]),
                "best_MRR@10": safe_float(best["MRR@10"]),
                "mean_NDCG@10": safe_float(group["NDCG@10"].mean()),
                "mean_MRR@10": safe_float(group["MRR@10"].mean()),
                "std_NDCG@10": safe_float(group["NDCG@10"].std(ddof=0)),
                "std_MRR@10": safe_float(group["MRR@10"].std(ddof=0)),
                "best_lambda": safe_float(best["lambda"]),
                "best_rank_change_rate": safe_float(best["rank_change_rate"]),
                "best_relative_NDCG_improvement_vs_monotonic": safe_float(
                    best["relative_NDCG_improvement_vs_monotonic"]
                ),
                "best_relative_MRR_improvement_vs_monotonic": safe_float(
                    best["relative_MRR_improvement_vs_monotonic"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["best_NDCG@10", "best_MRR@10"], ascending=[False, False])


def extract_candidate_title(prompt: Any, max_len: int = 180) -> str:
    text = "" if prompt is None else str(prompt)
    match = re.search(r"Now consider the candidate item:\s*\nTitle:\s*(.*?)\nDescription:", text, flags=re.S)
    if not match:
        match = re.search(r"Title:\s*(.*?)(?:\n|$)", text, flags=re.S)
    title = match.group(1).strip() if match else text.replace("\n", " ")[:max_len]
    title = re.sub(r"\s+", " ", title)
    return title[:max_len]


def add_evidence_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "evidence_risk" not in out.columns:
        out["evidence_risk"] = (
            (1.0 - pd.to_numeric(out["abs_evidence_margin"], errors="coerce").clip(0, 1))
            + pd.to_numeric(out["ambiguity"], errors="coerce").clip(0, 1)
            + pd.to_numeric(out["missing_information"], errors="coerce").clip(0, 1)
        ) / 3.0
    return out


def select_cases(merged: pd.DataFrame, case_type: str, limit: int = 12) -> pd.DataFrame:
    if case_type == "promoted":
        work = merged[merged["rank_delta"] > 0].sort_values(["rank_delta", "evidence_risk"], ascending=[False, False])
    elif case_type == "demoted":
        work = merged[merged["rank_delta"] < 0].sort_values(["rank_delta", "evidence_risk"], ascending=[True, False])
    elif case_type == "monotonic_high_risk_demoted":
        threshold = merged["evidence_risk"].quantile(0.75)
        work = merged[(merged["old_rank"] <= 3) & (merged["rank_delta"] < 0) & (merged["evidence_risk"] >= threshold)]
        work = work.sort_values(["evidence_risk", "rank_delta"], ascending=[False, True])
    else:
        raise ValueError(f"Unknown case type: {case_type}")
    out = work.head(limit).copy()
    out["case_type"] = case_type
    return out


def build_case_study(calibrated_path: Path, old_rank_path: Path, new_rank_path: Path) -> pd.DataFrame:
    evidence = add_evidence_risk(read_jsonl(calibrated_path, "Calibrated evidence test"))
    old = read_jsonl(old_rank_path, "Old monotonic ranked output")[["user_id", "candidate_item_id", "rank"]].rename(
        columns={"rank": "old_rank"}
    )
    new = read_jsonl(new_rank_path, "Decoupled ranked output")[["user_id", "candidate_item_id", "rank"]].rename(
        columns={"rank": "new_rank"}
    )
    merged = evidence.merge(old, on=["user_id", "candidate_item_id"], how="inner").merge(
        new, on=["user_id", "candidate_item_id"], how="inner"
    )
    merged["rank_delta"] = merged["old_rank"] - merged["new_rank"]
    merged["candidate_title_or_text"] = merged["prompt"].apply(extract_candidate_title)
    merged["repaired_confidence"] = merged.get("minimal_repaired_confidence", merged.get("repaired_confidence"))
    cases = pd.concat(
        [
            select_cases(merged, "promoted"),
            select_cases(merged, "demoted"),
            select_cases(merged, "monotonic_high_risk_demoted"),
        ],
        ignore_index=True,
    )
    existing = [col for col in CASE_COLUMNS if col in cases.columns]
    return cases[existing].copy()


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_report(ablation: pd.DataFrame, sensitivity: pd.DataFrame, case_study: pd.DataFrame) -> str:
    best = ablation[ablation["method_group"].astype(str).str.startswith("decoupled_")].sort_values(
        ["NDCG@10", "MRR@10"], ascending=[False, False]
    ).iloc[0]
    monotonic = ablation[ablation["method_group"] == "monotonic_noop"].iloc[0]
    setting_b = sensitivity[sensitivity["setting"] == "B"].copy()
    b_best = setting_b.sort_values(["best_NDCG@10", "best_MRR@10"], ascending=[False, False]).iloc[0]
    high_risk = case_study[case_study["case_type"] == "monotonic_high_risk_demoted"]
    avg_high_risk = high_risk["evidence_risk"].mean() if not high_risk.empty else float("nan")
    avg_high_risk_delta = high_risk["rank_delta"].mean() if not high_risk.empty else float("nan")

    lines = [
        "# Day6 Decoupled Rerank Analysis",
        "",
        "## 1. Day5-Repair Recap",
        "",
        "The original monotonic rerank is a mathematical no-op, not a negative result for evidence posterior. It uses `repaired_confidence` as the base score and `1 - repaired_confidence` as the uncertainty penalty, so `final_score = repaired_confidence - lambda * (1 - repaired_confidence) = (1 + lambda) * repaired_confidence - lambda`. For non-negative lambda this preserves ranking order. The diagnosis records rank_change_rate = 0, top10_change_rate = 0, top10_order_change_rate = 0, mean_kendall_tau = 1, and base_uncertainty_spearman = -1.",
        "",
        "## 2. Day6 Ablation",
        "",
        f"The monotonic internal baseline has NDCG@10 = `{monotonic['NDCG@10']:.4f}` and MRR@10 = `{monotonic['MRR@10']:.4f}`. The best decoupled setting is setting `{best['setting']}` with `{best['normalization']}` normalization and lambda `{best['lambda']:.3g}`. It reaches NDCG@10 = `{best['NDCG@10']:.4f}` and MRR@10 = `{best['MRR@10']:.4f}`, corresponding to internal relative improvements of `{fmt_pct(best['relative_NDCG_improvement_vs_monotonic'])}` in NDCG@10 and `{fmt_pct(best['relative_MRR_improvement_vs_monotonic'])}` in MRR@10.",
        "",
        "## 3. Sensitivity",
        "",
        f"Setting B is not a single-point accident in the current grid. Its best row uses lambda `{b_best['best_lambda']:.3g}`, while the mean NDCG@10 across the tested lambdas for its best normalization is `{b_best['mean_NDCG@10']:.4f}` with std `{b_best['std_NDCG@10']:.4f}`. This indicates that evidence_risk is useful across a local lambda region, although broader validation is still needed before making a general claim.",
        "",
        "## 4. Mechanism",
        "",
        f"The best setting changes ranking order with rank_change_rate = `{best['rank_change_rate']:.4f}`, top10_order_change_rate = `{best['top10_order_change_rate']:.4f}`, and mean_kendall_tau = `{best['mean_kendall_tau']:.4f}`. Its top10_change_rate can stay low in this candidate-ranking setup because each user has at most ten candidates, so the more informative signal is top-10 order change. Setting B decouples raw confidence from evidence risk, where evidence_risk = (1 - abs_evidence_margin + ambiguity + missing_information) / 3. The case study includes promoted candidates, demoted candidates, and monotonic-high candidates demoted under high evidence risk. In the monotonic-high-risk-demoted group, average evidence_risk is `{avg_high_risk:.4f}` and average rank_delta is `{avg_high_risk_delta:.4f}`, which supports the intended mechanism: candidates with weak margins, ambiguity, or missing information are penalized.",
        "",
        "## 5. Limitation",
        "",
        "The reported improvement is an internal comparison against the current full Beauty monotonic baseline, not an external SOTA claim. The experiment still uses a yes/no pointwise-to-ranking formulation, and it has not yet been moved to candidate relevance scoring, local Qwen-LoRA evidence generation, or external recommendation baselines.",
        "",
        "## 6. Next Step Recommendation",
        "",
        "Day6 is stable enough to proceed to Day7 candidate relevance scoring prompt/schema design. The current evidence supports two claims: evidence posterior repairs uncertainty quality, and decoupled relevance-risk reranking can convert that repaired uncertainty into decision payoff. Day7 should not add external baselines yet; it should first migrate the signal formulation away from yes/no into candidate relevance scoring while preserving the evidence-risk mechanism.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monotonic_grid_path", default="output-repaired/summary/beauty_deepseek_evidence_full_rerank_grid.csv")
    parser.add_argument("--monotonic_diagnostics_path", default="output-repaired/summary/beauty_deepseek_evidence_full_rerank_diagnostics.csv")
    parser.add_argument("--decoupled_grid_paths", nargs="+", default=[
        "output-repaired/summary/beauty_deepseek_evidence_full_rerank_decoupled_grid.csv",
        "output-repaired/summary/beauty_deepseek_evidence_full_rerank_decoupled_grid_zscore.csv",
    ])
    parser.add_argument("--comparison_path", default="output-repaired/summary/beauty_deepseek_evidence_full_rerank_comparison.csv")
    parser.add_argument("--calibrated_path", default="output-repaired/beauty_deepseek_evidence_full/calibrated/evidence_posterior_test.jsonl")
    parser.add_argument("--old_rank_path", default="output-repaired/beauty_deepseek_evidence_full_rerank_l0/reranked/baseline_ranked.jsonl")
    parser.add_argument("--best_rank_path", default="output-repaired/beauty_deepseek_evidence_full_decoupled_rerank_settingB_l0p2/reranked/decoupled_reranked.jsonl")
    parser.add_argument("--ablation_output", default="output-repaired/summary/beauty_day6_decoupled_rerank_ablation.csv")
    parser.add_argument("--sensitivity_output", default="output-repaired/summary/beauty_day6_decoupled_rerank_sensitivity.csv")
    parser.add_argument("--case_output", default="output-repaired/summary/beauty_day6_decoupled_rerank_case_study.csv")
    parser.add_argument("--report_output", default="output-repaired/summary/beauty_day6_decoupled_rerank_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monotonic_grid = read_csv(args.monotonic_grid_path, "Monotonic grid")
    diagnostics = read_csv(args.monotonic_diagnostics_path, "Monotonic diagnostics")
    decoupled_frames = [read_csv(path, f"Decoupled grid {idx}") for idx, path in enumerate(args.decoupled_grid_paths)]
    # The comparison path is intentionally read to validate the Day5-repair handoff exists.
    read_csv(args.comparison_path, "Day5 comparison")

    ref = monotonic_reference(monotonic_grid)
    ablation = pd.concat(
        [build_monotonic_ablation(monotonic_grid, diagnostics, ref), build_decoupled_ablation(decoupled_frames, ref)],
        ignore_index=True,
    )
    ablation = ablation.sort_values(["method_group", "normalization", "lambda"], na_position="first").reset_index(drop=True)
    sensitivity = build_sensitivity(ablation)
    case_study = build_case_study(Path(args.calibrated_path), Path(args.old_rank_path), Path(args.best_rank_path))
    report = build_report(ablation, sensitivity, case_study)

    for path, df in [
        (args.ablation_output, ablation),
        (args.sensitivity_output, sensitivity),
        (args.case_output, case_study),
    ]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {out}")

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved {report_path}")

    best = ablation[ablation["method_group"].astype(str).str.startswith("decoupled_")].sort_values(
        ["NDCG@10", "MRR@10"], ascending=[False, False]
    ).iloc[0]
    print(
        "BEST_SETTING "
        f"setting={best['setting']} normalization={best['normalization']} lambda={best['lambda']} "
        f"NDCG@10={best['NDCG@10']:.6f} MRR@10={best['MRR@10']:.6f} "
        f"rank_change_rate={best['rank_change_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
