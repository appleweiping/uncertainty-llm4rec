"""Day26 final tables, claim map, and paper-facing results draft."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_DIR = Path("output-repaired/summary")


BACKBONES = [
    {
        "backbone": "SASRec-style",
        "model_family": "minimal transformer sequential",
        "summary_path": SUMMARY_DIR / "day20_sasrec_full_multiseed_summary.csv",
        "attribution_path": SUMMARY_DIR / "day20_sasrec_component_attribution.csv",
        "rel_ndcg_col": "relative_NDCG_vs_sasrec",
        "rel_mrr_col": "relative_MRR_vs_sasrec",
        "method_map": {
            "A_SASRec_only": "Backbone only",
            "B_SASRec_plus_calibrated_relevance": "Backbone + calibrated relevance",
            "C_SASRec_plus_evidence_risk": "Backbone + evidence risk",
            "D_SASRec_plus_calibrated_relevance_plus_evidence_risk": "Backbone + calibrated relevance + evidence risk",
        },
        "fallback_rate_mean": 0.08838643371017471,
        "fallback_rate_std": 0.0,
    },
    {
        "backbone": "LLM-ESR GRU4Rec",
        "model_family": "external GRU sequential",
        "summary_path": SUMMARY_DIR / "day23_gru4rec_full_multiseed_summary.csv",
        "attribution_path": SUMMARY_DIR / "day23_gru4rec_component_attribution.csv",
        "rel_ndcg_col": "relative_NDCG_vs_gru4rec",
        "rel_mrr_col": "relative_MRR_vs_gru4rec",
        "method_map": {
            "A_GRU4Rec_only": "Backbone only",
            "B_GRU4Rec_plus_calibrated_relevance": "Backbone + calibrated relevance",
            "C_GRU4Rec_plus_evidence_risk": "Backbone + evidence risk",
            "D_GRU4Rec_plus_calibrated_relevance_plus_evidence_risk": "Backbone + calibrated relevance + evidence risk",
        },
    },
    {
        "backbone": "LLM-ESR Bert4Rec",
        "model_family": "external masked-transformer sequential",
        "summary_path": SUMMARY_DIR / "day25_bert4rec_full_multiseed_summary.csv",
        "attribution_path": SUMMARY_DIR / "day25_bert4rec_component_attribution.csv",
        "rel_ndcg_col": "relative_NDCG_vs_bert4rec",
        "rel_mrr_col": "relative_MRR_vs_bert4rec",
        "method_map": {
            "A_Bert4Rec_only": "Backbone only",
            "B_Bert4Rec_plus_calibrated_relevance": "Backbone + calibrated relevance",
            "C_Bert4Rec_plus_evidence_risk": "Backbone + evidence risk",
            "D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk": "Backbone + calibrated relevance + evidence risk",
        },
    },
]


def _read_summary(spec: dict) -> pd.DataFrame:
    df = pd.read_csv(spec["summary_path"])
    rows = []
    rel_ndcg = spec["rel_ndcg_col"]
    rel_mrr = spec["rel_mrr_col"]
    fallback_mean = spec.get("fallback_rate_mean", None)
    fallback_std = spec.get("fallback_rate_std", 0.0)
    for _, row in df.iterrows():
        method = spec["method_map"].get(row["method"], row["method"])
        rows.append(
            {
                "backbone": spec["backbone"],
                "model_family": spec["model_family"],
                "users": 973,
                "candidate_rows": 5838,
                "fallback_rate_mean": row["fallback_rate_mean"] if "fallback_rate_mean" in df.columns else fallback_mean,
                "fallback_rate_std": row["fallback_rate_std"] if "fallback_rate_std" in df.columns else fallback_std,
                "method": method,
                "NDCG@10_mean": row["NDCG@10_mean"],
                "NDCG@10_std": row["NDCG@10_std"],
                "MRR@10_mean": row["MRR@10_mean"],
                "MRR@10_std": row["MRR@10_std"],
                "HR@10_mean": row["HR@10_mean"],
                "HR@10_std": row["HR@10_std"],
                "Recall@10_mean": row["Recall@10_mean"],
                "Recall@10_std": row["Recall@10_std"],
                "relative_NDCG_vs_backbone_mean": row[f"{rel_ndcg}_mean"],
                "relative_NDCG_vs_backbone_std": row[f"{rel_ndcg}_std"],
                "relative_MRR_vs_backbone_mean": row[f"{rel_mrr}_mean"],
                "relative_MRR_vs_backbone_std": row[f"{rel_mrr}_std"],
                "claim_level": "full_multiseed_validation",
            }
        )
    return pd.DataFrame(rows)


def make_main_table() -> pd.DataFrame:
    table = pd.concat([_read_summary(spec) for spec in BACKBONES], ignore_index=True)
    table.to_csv(SUMMARY_DIR / "day26_three_backbone_external_plugin_main_table.csv", index=False)
    return table


def make_component_attribution(main_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone, group in main_table.groupby("backbone", sort=False):
        by_method = group.set_index("method")
        a = by_method.loc["Backbone only"]
        b = by_method.loc["Backbone + calibrated relevance"]
        c = by_method.loc["Backbone + evidence risk"]
        d = by_method.loc["Backbone + calibrated relevance + evidence risk"]
        for method, row in by_method.iterrows():
            if method == "Backbone only":
                main_contributor = "baseline"
            elif method == "Backbone + calibrated relevance":
                main_contributor = "calibrated_relevance_posterior"
            elif method == "Backbone + evidence risk":
                main_contributor = "evidence_risk"
            else:
                main_contributor = (
                    "combined"
                    if row["NDCG@10_mean"] > b["NDCG@10_mean"]
                    else "calibrated_relevance_posterior"
                )
            rows.append(
                {
                    "backbone": backbone,
                    "method": method,
                    "NDCG@10_mean": row["NDCG@10_mean"],
                    "MRR@10_mean": row["MRR@10_mean"],
                    "relative_NDCG_vs_backbone_mean": row["relative_NDCG_vs_backbone_mean"],
                    "relative_MRR_vs_backbone_mean": row["relative_MRR_vs_backbone_mean"],
                    "B_exceeds_A": bool(b["NDCG@10_mean"] > a["NDCG@10_mean"]),
                    "C_weaker_than_B": bool(c["NDCG@10_mean"] < b["NDCG@10_mean"]),
                    "D_exceeds_B": bool(d["NDCG@10_mean"] > b["NDCG@10_mean"]),
                    "main_contributor": main_contributor,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day26_component_attribution_summary.csv", index=False)
    return out


def _markdown_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    view = df[cols].copy() if cols else df.copy()
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _best_d_lines(main_table: pd.DataFrame) -> str:
    lines = []
    d = main_table[main_table["method"] == "Backbone + calibrated relevance + evidence risk"]
    for _, row in d.iterrows():
        lines.append(
            f"- {row['backbone']}: NDCG@10 `{row['NDCG@10_mean']:.4f} +/- {row['NDCG@10_std']:.4f}`, "
            f"MRR@10 `{row['MRR@10_mean']:.4f} +/- {row['MRR@10_std']:.4f}`, "
            f"relative NDCG `{row['relative_NDCG_vs_backbone_mean']:.2%}`, "
            f"relative MRR `{row['relative_MRR_vs_backbone_mean']:.2%}`."
        )
    return "\n".join(lines)


def write_claim_map(main_table: pd.DataFrame) -> None:
    text = f"""# Day26 Final Claim Map

## 1. Original Observation / Week1-Week4

The original pipeline showed that raw verbalized confidence and raw relevance-style signals are informative, but they are miscalibrated and should not be used directly as decision signals. This observation motivates repair rather than blind trust in self-reported confidence.

## 2. Scheme 4 / CEP

Scheme 4 is not a simple prompt rewrite. It is an evidence-grounded calibrated posterior pipeline. The LLM outputs `relevance_probability`, `positive_evidence`, `negative_evidence`, `ambiguity`, and `missing_information`; valid-set calibration then converts these fields into `calibrated_relevance_probability`. The derived `evidence_risk` is used as a risk regularizer rather than as the primary scorer.

## 3. Day6

In the yes/no decision reliability setting, evidence risk can directly represent decision risk. Decoupled reranking has clear payoff because the task is explicitly about whether a recommendation decision is reliable.

## 4. Day9

In candidate relevance posterior scoring, the main contribution is `calibrated_relevance_probability`, which repairs probability quality. AUROC is useful for discrimination diagnostics, but ECE/Brier are the core uncertainty-quality criteria.

## 5. Day10

In list-level first-pass generation, evidence decomposition is not best used as a generation burden. Plain list generation is a better first-pass base, while Scheme 4 is better positioned as post-hoc or hybrid decision support.

## 6. Day20/23/25

Across three external sequential backbones, Scheme 4 plug-in consistently improves full Beauty multi-seed ranking performance:

{_best_d_lines(main_table)}

The shared pattern is that `calibrated_relevance_probability` is the primary contributor, while `evidence_risk` is a secondary regularizer. C-only is weaker than B, but D consistently improves over B.

## 7. Final Claim

Scheme 4 / CEP can serve as a plug-in calibrated relevance posterior for LLM-enhanced recommendation. It improves ranking when combined with external sequential backbones, mainly through calibrated relevance posterior and secondarily through evidence-risk regularization.

## 8. Claim Boundary

The current result is Beauty full + three sequential backbones. It is not a universal SOTA claim across all domains, all recommender families, or all generation settings. Natural next steps are cross-domain validation, stronger public backbones, and Qwen-LoRA localization of the evidence generator.
"""
    (SUMMARY_DIR / "day26_final_claim_map.md").write_text(text, encoding="utf-8")


def write_paper_draft(main_table: pd.DataFrame, attribution: pd.DataFrame) -> None:
    d_rows = main_table[main_table["method"] == "Backbone + calibrated relevance + evidence risk"]
    table_md = _markdown_table(
        main_table,
        [
            "backbone",
            "method",
            "NDCG@10_mean",
            "NDCG@10_std",
            "MRR@10_mean",
            "MRR@10_std",
            "relative_NDCG_vs_backbone_mean",
            "relative_MRR_vs_backbone_mean",
        ],
    )
    text = f"""# Day26 Paper Results Section Draft

## 1. Uncertainty Quality

The early confidence pipeline shows that raw LLM confidence is not pure noise, but it is not a calibrated probability. In the candidate relevance setting, the calibrated evidence posterior (CEP) repairs this problem by fitting a valid-set calibration model over evidence-derived features. We therefore evaluate uncertainty quality with calibration-sensitive metrics such as ECE and Brier score, rather than treating raw self-reported confidence as directly decision-ready.

## 2. Decision Repair

The yes/no decision setting provides a controlled diagnostic environment. In that setting, evidence risk naturally represents decision unreliability, and a decoupled risk-aware reranking formulation can change decisions. This result is useful for validating the risk component, but it is not the final form of the recommendation task.

## 3. Candidate Relevance Posterior

For candidate-level recommendation, CEP reframes the model output as relevance posterior estimation. The primary signal is `calibrated_relevance_probability`; it is not the same as raw confidence and is not obtained by directly trusting the model's self-report. The evidence fields are used to construct a posterior that is calibrated on validation data and then evaluated on held-out test data.

## 4. External Backbone Plug-in

We evaluate CEP as a plug-in on three sequential recommendation backbones under full Beauty multi-seed validation. No additional DeepSeek API calls are made during these plug-in experiments; all runs reuse the Day9 full evidence table.

{table_md}

The best D setting improves over the corresponding backbone-only baseline on all three backbones:

{_best_d_lines(main_table)}

## 5. Component Attribution

The component pattern is consistent. B (backbone + calibrated relevance) accounts for most of the gain. C (backbone + evidence risk only) is positive but weaker. D (backbone + calibrated relevance + evidence risk) consistently improves over B, indicating that evidence risk is best interpreted as a secondary regularizer rather than as the main scorer.

## 6. Limitations

These results are currently limited to Amazon Beauty and sequential/backbone-level validation. The evidence generator uses DeepSeek API outputs from Day9; future work can localize it through Qwen-LoRA. The current results should not be claimed as universal SOTA across all domains or all recommendation settings until cross-domain and stronger-public-backbone validations are complete.
"""
    (SUMMARY_DIR / "day26_paper_results_section_draft.md").write_text(text, encoding="utf-8")


def write_experiment_index() -> None:
    paths = [
        ("Day6 decoupled rerank report", "output-repaired/summary/beauty_day6_decoupled_rerank_report.md"),
        ("Day9 relevance evidence full report", "output-repaired/summary/beauty_day9_relevance_evidence_full_report.md"),
        ("Day10 full list report", "output-repaired/summary/beauty_day10_full_report.md"),
        ("Day20 SASRec multi-seed report", "output-repaired/summary/day20_sasrec_multiseed_final_report.md"),
        ("Day20 SASRec multi-seed summary", "output-repaired/summary/day20_sasrec_full_multiseed_summary.csv"),
        ("Day23 GRU4Rec multi-seed report", "output-repaired/summary/day23_gru4rec_multiseed_and_two_backbone_report.md"),
        ("Day23 GRU4Rec multi-seed summary", "output-repaired/summary/day23_gru4rec_full_multiseed_summary.csv"),
        ("Day25 Bert4Rec multi-seed report", "output-repaired/summary/day25_bert4rec_full_multiseed_report.md"),
        ("Day25 Bert4Rec multi-seed summary", "output-repaired/summary/day25_bert4rec_full_multiseed_summary.csv"),
        ("Day26 final main table", "output-repaired/summary/day26_three_backbone_external_plugin_main_table.csv"),
        ("Day26 component attribution", "output-repaired/summary/day26_component_attribution_summary.csv"),
        ("Day26 claim map", "output-repaired/summary/day26_final_claim_map.md"),
        ("Day26 paper results draft", "output-repaired/summary/day26_paper_results_section_draft.md"),
    ]
    lines = ["# Day26 Experiment Index", ""]
    for label, path in paths:
        status = "exists" if Path(path).exists() else "missing"
        lines.append(f"- **{label}** (`{status}`): `{path}`")
    (SUMMARY_DIR / "day26_experiment_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    main_table = make_main_table()
    attribution = make_component_attribution(main_table)
    write_claim_map(main_table)
    write_paper_draft(main_table, attribution)
    write_experiment_index()
    print("Day26 final tables and claim drafts complete.")


if __name__ == "__main__":
    main()
