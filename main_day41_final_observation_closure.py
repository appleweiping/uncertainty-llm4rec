"""Day41 final observation-stage closure tables and narrative drafts.

No experiments are run here. The script only aggregates existing summary
artifacts into final tables, claim maps, and paper-section drafts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_DIR = Path("output-repaired/summary")


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if math.isnan(val):
        return "NA"
    return f"{val:.{digits}f}"


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def _write_experiment_index() -> None:
    rows = [
        {
            "stage": "Week1-Week4 / raw confidence observation",
            "key_files": "output-repaired/summary/day29b_beauty_multimodel_raw_confidence_diagnostics.csv; output-repaired/summary/day29b_observation_raw_llm_confidence_miscalibration_report.md",
            "main_claim": "Raw verbalized confidence/relevance signals are informative but miscalibrated across multiple LLMs; they should not be used directly as probabilities.",
            "claim_level": "primary_observation",
            "limitations": "Observation is strongest on Beauty diagnostics; raw confidence is not evaluated as a final recommender by itself.",
        },
        {
            "stage": "Day6 decision confidence repair",
            "key_files": "output-repaired/summary/day26_final_claim_map.md; prior Day6 report referenced in day26_experiment_index.md",
            "main_claim": "Evidence-risk can strongly support yes/no decision reliability reranking.",
            "claim_level": "primary_method",
            "limitations": "Decision reliability is a different setting from candidate relevance posterior ranking.",
        },
        {
            "stage": "Day9 full Beauty candidate relevance posterior",
            "key_files": "output-repaired/summary/day29b_beauty_relevance_probability_diagnostics.csv; output-repaired/summary/day26_final_claim_map.md",
            "main_claim": "Raw candidate relevance probability is miscalibrated; valid-set calibration produces a useful calibrated relevance posterior.",
            "claim_level": "primary_method",
            "limitations": "Calibration improves probability quality; AUROC is not expected to transform dramatically.",
        },
        {
            "stage": "Day10 list-level first-pass boundary",
            "key_files": "output-repaired/summary/day26_final_claim_map.md; output-repaired/summary/day26_paper_results_section_draft.md",
            "main_claim": "Evidence decomposition is better as post-hoc/hybrid decision support than as a heavy first-pass generation burden.",
            "claim_level": "boundary_analysis",
            "limitations": "Boundary is about first-pass list generation, not CEP plug-in reranking.",
        },
        {
            "stage": "Day20/23/25 Beauty full three-backbone multi-seed",
            "key_files": "output-repaired/summary/day26_three_backbone_external_plugin_main_table_metric_repaired.csv; output-repaired/summary/day26_component_attribution_summary_metric_repaired.csv",
            "main_claim": "CEP plug-in improves NDCG/MRR across SASRec-style, GRU4Rec, and Bert4Rec full Beauty multi-seed.",
            "claim_level": "primary_performance",
            "limitations": "Beauty full is primary evidence but not universal SOTA across all domains/backbones.",
        },
        {
            "stage": "Day29b observation consolidation",
            "key_files": "output-repaired/summary/day29b_beauty_multimodel_calibration_effect.csv; output-repaired/summary/day29b_paper_motivation_snippet.md",
            "main_claim": "Multi-model diagnostics support the motivation that raw LLM confidence is informative but unreliable without calibration.",
            "claim_level": "primary_observation",
            "limitations": "Aggregates existing outputs; does not add new model inference.",
        },
        {
            "stage": "Day29c backbone score calibration diagnostic",
            "key_files": "output-repaired/summary/day29c_backbone_score_calibration_diagnostics.csv; output-repaired/summary/day29c_backbone_score_miscalibration_report.md",
            "main_claim": "Backbone scores are useful ranking logits but are not calibrated probabilities.",
            "claim_level": "diagnostic_only",
            "limitations": "Diagnostic does not claim backbone methods fail; it separates ranking ability from uncertainty estimation.",
        },
        {
            "stage": "Day30 CEP robustness",
            "key_files": "output-repaired/summary/day30_cep_robustness_metrics.csv; output-repaired/summary/day30_sasrec_cep_robustness_grid.csv; output-repaired/summary/day30_cep_backbone_robustness_report.md",
            "main_claim": "CEP and SASRec+CEP remain useful under controlled input perturbations; D degradation is bounded in the 500-user robustness setting.",
            "claim_level": "robustness_support",
            "limitations": "Beauty 500-user robustness first run, primarily SASRec.",
        },
        {
            "stage": "Day31/37/39 small-domain CEP calibration",
            "key_files": "output-repaired/summary/day31_movies_medium5_calibration_comparison.csv; output-repaired/summary/day37_movies_small_calibration_comparison.csv; output-repaired/summary/day39_books_electronics_small_calibration_comparison.csv",
            "main_claim": "Movies/books/electronics small results support cross-domain calibration consistency and directionality.",
            "claim_level": "cross_domain_sanity",
            "limitations": "Small-domain candidate pool is 6, so HR@10 is trivial; backbone fallback caveats apply.",
        },
        {
            "stage": "Day38/40 small-domain fallback sensitivity",
            "key_files": "output-repaired/summary/day40_small_domains_fallback_sensitivity_summary.csv; output-repaired/summary/day40_books_electronics_fallback_sensitivity_report.md",
            "main_claim": "Small-domain gains are not explained by fallback flag alone, but many are best interpreted as fallback/cold compensation or sample-limited directionality.",
            "claim_level": "cross_domain_sanity",
            "limitations": "Do not describe small-domain results as fully healthy ID-backbone benchmarks.",
        },
        {
            "stage": "Day34/35 regular-medium cold-start/content-carrier route",
            "key_files": "output-repaired/summary/day34_movies_cold_content_carrier_report.md; output-repaired/summary/day35_cross_domain_route_decision_report.md",
            "main_claim": "Regular medium domains reveal real cold-start issues for ID-only sequential backbones and motivate content-aware/cold-aware carriers.",
            "claim_level": "boundary_analysis",
            "limitations": "Content carrier is diagnostic/cold-aware, not a SOTA recommender claim.",
        },
    ]
    lines = ["# Day41 Final Experiment Index", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['stage']}",
                "",
                f"- key_files: `{row['key_files']}`",
                f"- main_claim: {row['main_claim']}",
                f"- claim_level: `{row['claim_level']}`",
                f"- limitations: {row['limitations']}",
                "",
            ]
        )
    (SUMMARY_DIR / "day41_final_experiment_index.md").write_text("\n".join(lines), encoding="utf-8")


def _beauty_rows() -> list[dict[str, Any]]:
    df = _read_csv(SUMMARY_DIR / "day26_three_backbone_external_plugin_main_table_metric_repaired.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "section": "Beauty full primary performance",
                "domain": "beauty",
                "dataset_type": "beauty_full",
                "backbone": r["backbone"],
                "method": r["method"],
                "NDCG@10": r.get("NDCG@10_mean", math.nan),
                "MRR": r.get("MRR_mean", math.nan),
                "HR@1": r.get("HR@1_mean", math.nan),
                "HR@3": r.get("HR@3_mean", math.nan),
                "NDCG@3": r.get("NDCG@3_mean", math.nan),
                "NDCG@5": r.get("NDCG@5_mean", math.nan),
                "relative_NDCG": r.get("relative_NDCG@10_vs_backbone_mean", math.nan),
                "relative_MRR": r.get("relative_MRR_vs_backbone_mean", math.nan),
                "fallback_rate": r.get("fallback_rate_mean", math.nan),
                "raw_ECE": math.nan,
                "calibrated_ECE": math.nan,
                "fallback_health_status": "primary_full_multiseed",
                "clean_ECE": math.nan,
                "noisy_ECE": math.nan,
                "clean_NDCG": math.nan,
                "noisy_NDCG": math.nan,
                "max_NDCG_drop": math.nan,
                "max_MRR_drop": math.nan,
                "positive_cold_rate": math.nan,
                "negative_cold_rate": math.nan,
                "content_carrier_result": "",
                "hr10_trivial_flag": r.get("hr10_trivial_flag", True),
                "claim_level": "primary_full_multiseed",
            }
        )
    return rows


def _small_rows() -> list[dict[str, Any]]:
    rows = []
    cal_movies = _read_csv(SUMMARY_DIR / "day37_movies_small_calibration_comparison.csv")
    cal_be = _read_csv(SUMMARY_DIR / "day39_books_electronics_small_calibration_comparison.csv")
    cal = pd.concat([cal_movies, cal_be], ignore_index=True)
    fallback = _read_csv(SUMMARY_DIR / "day39_books_electronics_small_fallback_summary.csv")
    movies_diag_rows = []
    for b in ["sasrec", "gru4rec", "bert4rec"]:
        p = SUMMARY_DIR / f"day37_movies_small_{b}_plugin_diagnostics.csv"
        if p.exists():
            d = pd.read_csv(p).iloc[0].to_dict()
            movies_diag_rows.append(
                {
                    "domain": "movies_small",
                    "backbone": b,
                    "fallback_rate": d["fallback_rate"],
                    "positive_fallback_rate": d["fallback_rate_positive"],
                    "negative_fallback_rate": d["fallback_rate_negative"],
                    "health_status": "fallback_heavy" if d["fallback_rate"] >= 0.5 or d["fallback_rate_positive"] >= 0.5 else "caution",
                }
            )
    fallback = pd.concat([fallback, pd.DataFrame(movies_diag_rows)], ignore_index=True)

    for domain in ["movies_small", "books_small", "electronics_small"]:
        test = cal[(cal["domain"] == domain) & (cal["split"] == "test")]
        if test.empty:
            continue
        raw = test[test["score_type"] == "raw_relevance_probability"].iloc[0]
        calibrated = test[test["score_type"] == "calibrated_relevance_probability"].iloc[0]
        for backbone in ["sasrec", "gru4rec", "bert4rec"]:
            prefix = "day37_movies_small" if domain == "movies_small" else f"day39_{domain}"
            diag_path = SUMMARY_DIR / f"{prefix}_{backbone}_plugin_diagnostics.csv"
            if not diag_path.exists():
                continue
            d = pd.read_csv(diag_path).iloc[0]
            f = fallback[(fallback["domain"] == domain) & (fallback["backbone"] == backbone)]
            health = f["health_status"].iloc[0] if not f.empty and "health_status" in f.columns else "fallback_heavy"
            claim = "small_cross_domain_sanity" if health == "healthy" else "fallback_heavy_caution"
            rows.append(
                {
                    "section": "Small-domain cross-domain sanity",
                    "domain": domain,
                    "dataset_type": domain,
                    "backbone": backbone,
                    "method": d["best_method"],
                    "NDCG@10": d["best_NDCG@10"],
                    "MRR": d["best_MRR"],
                    "HR@1": d["best_HR@1"],
                    "HR@3": d["best_HR@3"],
                    "NDCG@3": math.nan,
                    "NDCG@5": math.nan,
                    "relative_NDCG": d["best_relative_NDCG_vs_backbone"],
                    "relative_MRR": d["best_relative_MRR_vs_backbone"],
                    "fallback_rate": d["fallback_rate"],
                    "raw_ECE": raw["ECE"],
                    "calibrated_ECE": calibrated["ECE"],
                    "fallback_health_status": health,
                    "clean_ECE": math.nan,
                    "noisy_ECE": math.nan,
                    "clean_NDCG": math.nan,
                    "noisy_NDCG": math.nan,
                    "max_NDCG_drop": math.nan,
                    "max_MRR_drop": math.nan,
                    "positive_cold_rate": math.nan,
                    "negative_cold_rate": math.nan,
                    "content_carrier_result": "",
                    "hr10_trivial_flag": True,
                    "claim_level": claim,
                }
            )
    return rows


def _robustness_rows() -> list[dict[str, Any]]:
    df = _read_csv(SUMMARY_DIR / "day30_robustness_degradation_summary.csv")
    if df.empty:
        return []
    d_rows = df[df["method"].str.startswith("D_", na=False)]
    b_rows = df[df["method"].str.startswith("B_", na=False)]
    all_rows = pd.concat([d_rows, b_rows], ignore_index=True)
    row = {
        "section": "Robustness",
        "domain": "beauty",
        "dataset_type": "beauty_robustness_500",
        "backbone": "SASRec-style",
        "method": "SASRec + CEP under noisy evidence",
        "NDCG@10": math.nan,
        "MRR": math.nan,
        "HR@1": math.nan,
        "HR@3": math.nan,
        "NDCG@3": math.nan,
        "NDCG@5": math.nan,
        "relative_NDCG": math.nan,
        "relative_MRR": math.nan,
        "fallback_rate": math.nan,
        "raw_ECE": math.nan,
        "calibrated_ECE": math.nan,
        "fallback_health_status": "robustness_support",
        "clean_ECE": float(all_rows["clean_ECE"].mean()),
        "noisy_ECE": float(all_rows["noisy_ECE"].mean()),
        "clean_NDCG": float(d_rows["clean_NDCG"].mean()) if not d_rows.empty else math.nan,
        "noisy_NDCG": float(d_rows["noisy_NDCG"].mean()) if not d_rows.empty else math.nan,
        "max_NDCG_drop": float(all_rows["NDCG_drop"].max()),
        "max_MRR_drop": float(all_rows["MRR_drop"].max()),
        "positive_cold_rate": math.nan,
        "negative_cold_rate": math.nan,
        "content_carrier_result": "",
        "hr10_trivial_flag": True,
        "claim_level": "robustness_support",
    }
    return [row]


def _regular_medium_rows() -> list[dict[str, Any]]:
    rows = []
    movies = _read_csv(SUMMARY_DIR / "movies_medium_5neg_cold_rate_diagnostics.csv")
    content = _read_csv(SUMMARY_DIR / "day35_movies_content_carrier_attribution.csv")
    if not movies.empty:
        base = movies[(movies["split"] == "test") & (movies["vocab_definition"] == "train_backbone_vocab")]
        if not base.empty:
            r = base.iloc[0]
            content_result = ""
            if not content.empty:
                best = content.sort_values("D_plus_both_NDCG", ascending=False).iloc[0]
                content_result = f"{best['backbone_name']} D NDCG={_fmt(best['D_plus_both_NDCG'])}"
            rows.append(
                {
                    "section": "Regular medium cold-start route",
                    "domain": "movies_regular_medium",
                    "dataset_type": "regular_medium_cold_style",
                    "backbone": "content carrier / ID-backbone diagnostic",
                    "method": "cold-rate + content-carrier route",
                    "NDCG@10": math.nan,
                    "MRR": math.nan,
                    "HR@1": math.nan,
                    "HR@3": math.nan,
                    "NDCG@3": math.nan,
                    "NDCG@5": math.nan,
                    "relative_NDCG": math.nan,
                    "relative_MRR": math.nan,
                    "fallback_rate": math.nan,
                    "raw_ECE": math.nan,
                    "calibrated_ECE": math.nan,
                    "fallback_health_status": "cold_start_diagnostic",
                    "clean_ECE": math.nan,
                    "noisy_ECE": math.nan,
                    "clean_NDCG": math.nan,
                    "noisy_NDCG": math.nan,
                    "max_NDCG_drop": math.nan,
                    "max_MRR_drop": math.nan,
                    "positive_cold_rate": r["positive_cold_rate"],
                    "negative_cold_rate": r["negative_cold_rate"],
                    "content_carrier_result": content_result,
                    "hr10_trivial_flag": True,
                    "claim_level": "cold_start_diagnostic",
                }
            )
    return rows


def _write_main_table() -> pd.DataFrame:
    rows = _beauty_rows() + _small_rows() + _robustness_rows() + _regular_medium_rows()
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day41_final_main_results_table.csv", index=False)
    return out


def _write_component_attribution() -> None:
    beauty = _read_csv(SUMMARY_DIR / "day26_component_attribution_summary_metric_repaired.csv")
    rows = []
    if not beauty.empty and {"backbone", "method", "NDCG@10_mean"}.issubset(beauty.columns):
        method_to_col = {
            "Backbone only": "A_NDCG@10",
            "Backbone + calibrated relevance": "B_NDCG@10",
            "Backbone + evidence risk": "C_NDCG@10",
            "Backbone + calibrated relevance + evidence risk": "D_NDCG@10",
        }
        for backbone, group in beauty.groupby("backbone"):
            row = {
                "domain": "beauty",
                "backbone": backbone,
                "A_NDCG@10": math.nan,
                "B_NDCG@10": math.nan,
                "C_NDCG@10": math.nan,
                "D_NDCG@10": math.nan,
                "main_contributor": "calibrated_relevance_posterior",
                "risk_role": "secondary_regularizer",
                "claim_level": "primary_full_multiseed",
            }
            for method, out_col in method_to_col.items():
                match = group[group["method"] == method]
                if not match.empty:
                    row[out_col] = _to_float(match.iloc[0].get("NDCG@10_mean"))
            rows.append(row)
    # Add compact small-domain attribution from diagnostics.
    for domain in ["movies_small", "books_small", "electronics_small"]:
        for backbone in ["sasrec", "gru4rec", "bert4rec"]:
            prefix = "day37_movies_small" if domain == "movies_small" else f"day39_{domain}"
            p = SUMMARY_DIR / f"{prefix}_{backbone}_plugin_diagnostics.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p).iloc[0]
            rows.append(
                {
                    "domain": domain,
                    "backbone": backbone,
                    "A_NDCG@10": d["backbone_NDCG@10"],
                    "B_NDCG@10": d["B_NDCG@10"],
                    "C_NDCG@10": d["C_NDCG@10"],
                    "D_NDCG@10": d["D_NDCG@10"],
                    "main_contributor": "calibrated_relevance_posterior",
                    "risk_role": "secondary_regularizer_when_D_exceeds_B",
                    "claim_level": "small_cross_domain_sanity_with_fallback_caveat",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day41_final_component_attribution.csv", index=False)
    lines = [
        "# Day41 Final Component Attribution",
        "",
        "## Summary",
        "",
        "Across the observation-stage experiments, the primary contributor is `calibrated_relevance_probability`. Evidence risk is useful, but its best-supported role is a secondary regularizer rather than a standalone scorer.",
        "",
        "## Findings",
        "",
        "1. `calibrated_relevance_probability` is the main contributor in the external backbone plug-in setting.",
        "2. `evidence_risk` alone is usually weaker than calibrated relevance in candidate ranking.",
        "3. `D = calibrated relevance + evidence risk` often exceeds B, especially in the full Beauty backbone results, supporting evidence risk as a secondary regularizer.",
        "4. Day6 yes/no decision reliability is the setting where evidence risk is strongest as a direct decision-risk signal.",
        "5. Day9/relevance/backbone plug-in should be described as calibrated posterior first, risk regularization second.",
        "",
        "## Boundary",
        "",
        "Do not describe evidence risk as the main scorer for candidate-level recommendation. Do not use small-domain fallback-heavy gains as fully healthy external-backbone proof.",
    ]
    (SUMMARY_DIR / "day41_final_component_attribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_claim_map() -> None:
    text = """# Day41 Final Claim Map

## 1. Starting Observation

LLM recommendation confidence and relevance signals are informative but miscalibrated. Multi-model Beauty diagnostics show this is not a single-model accident: raw verbalized confidence and raw relevance probability carry signal, but ECE/Brier/high-confidence error indicate they are unreliable as direct probabilities.

## 2. Method

CEP / Scheme 4 is an evidence-grounded calibrated posterior, not a prompt rewrite. It asks for relevance and evidence fields, then uses valid-set calibration to produce `calibrated_relevance_probability` and derives `evidence_risk` from ambiguity, missing information, and evidence margin.

## 3. Task-Specific Formulation

Day6 treats evidence risk as a decision-reliability signal for yes/no reranking. Day9 treats candidate-level output as relevance posterior calibration. Day10 establishes that evidence decomposition is not ideal as a heavy first-pass list-generation burden. External backbone experiments use CEP as a plug-in calibrated posterior with secondary risk regularization.

## 4. Main Performance Evidence

The main performance evidence is Beauty full + three sequential backbones + multi-seed: SASRec-style, GRU4Rec, and Bert4Rec. In this setting, CEP improves NDCG/MRR consistently, with calibrated relevance as the main contributor and evidence risk as a secondary regularizer.

## 5. Robustness Evidence

Day30 shows CEP does not collapse under controlled noisy input on a Beauty 500-user subset. Noisy D remains close to clean CEP, and observed NDCG/MRR drops are bounded in this first robustness run.

## 6. Cross-Domain Evidence

Small-domain Movies/Books/Electronics support calibration consistency and directionality, but backbone fallback is non-trivial. These results are cross-domain sanity / continuity evidence, not fully healthy external-backbone proof. Regular medium analysis reveals realistic cold-start issues and motivates content-aware/cold-aware carriers.

## 7. Boundary

Do not claim universal SOTA. Do not claim evidence risk is the main scorer. Do not use HR@10 as primary evidence when the candidate pool has six items. Do not describe fallback-heavy small-domain results as fully healthy backbone benchmarks.

## 8. Next Phase

Two reasonable next directions are: (1) Qwen-LoRA / local evidence-generator framework, if the goal is system ownership and cost reduction; or (2) stronger content-aware/cold-aware cross-domain backbone, if the goal is extending beyond Beauty into regular medium domains.
"""
    (SUMMARY_DIR / "day41_final_claim_map.md").write_text(text, encoding="utf-8")


def _write_paper_draft() -> None:
    text = """# Day41 Paper Results Section Final Draft

## RQ1: Are Raw LLM Confidence/Relevance Signals Reliable?

Our starting question is whether raw LLM confidence or relevance probability can be directly used as a recommendation decision signal. The answer is no. Across the observation diagnostics, raw confidence and raw relevance are informative, but they are substantially miscalibrated. This is visible in ECE/Brier and high-confidence error behavior. Therefore, raw LLM confidence should not be interpreted as calibrated probability.

## RQ2: Does CEP Improve Calibration?

CEP improves probability quality by converting evidence-grounded relevance outputs into a calibrated relevance posterior. On Beauty and small-domain replications, calibrated relevance consistently reduces ECE/Brier relative to raw relevance probability. AUROC does not need to improve dramatically because calibration primarily fixes probability scale rather than creating a new ranker.

## RQ3: Can CEP Improve External Recommender Backbones?

On Beauty full multi-seed experiments, CEP improves three sequential backbones: SASRec-style, GRU4Rec, and Bert4Rec. The strongest evidence is the full Beauty three-backbone multi-seed table. Component attribution shows that calibrated relevance posterior provides the primary gain, while evidence risk works as a secondary regularizer. We do not present CEP as replacing recommender backbones; instead, the backbone supplies ranking ability and CEP supplies calibrated posterior/risk information.

## RQ4: Is CEP Robust Under Input Perturbations?

The Day30 controlled robustness experiment perturbs user history and candidate text on a 500-user Beauty subset. CEP degrades modestly rather than collapsing, and the combined D setting remains close to clean CEP performance. This supports robustness, but it remains a first-run robustness setting rather than a full-domain robustness claim.

## RQ5: Does The Trend Generalize Beyond Beauty?

Small-domain Movies/Books/Electronics replicate calibration consistency and directional plug-in behavior, but the ID-backbone fallback caveat is important. Fallback sensitivity shows that gains are not explained by fallback flags alone, yet many small-domain gains are best interpreted as fallback/cold compensation or sample-limited directionality. Thus, small domains support cross-domain sanity/continuity, while Beauty full remains the primary performance evidence. Regular medium analysis further shows that realistic cross-domain settings may require content-aware/cold-aware backbones.

## Metrics Boundary

Several experiments use six candidates per user. In those settings, HR@10 is trivial and should not be used as claim-supporting evidence. Primary metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.
"""
    (SUMMARY_DIR / "day41_paper_results_section_final_draft.md").write_text(text, encoding="utf-8")


def _write_next_phase() -> None:
    text = """# Day41 Next Phase Recommendation

## Option A: Cross-Domain Continuation

Regular medium domains are more realistic, but they exposed cold-start issues for ID-only sequential backbones. Continuing this path requires a content-aware or cold-aware backbone rather than directly reusing SASRec/GRU4Rec/Bert4Rec. Good next steps include a stronger text/content carrier, a limited Movies/Books/Electronics medium_20neg run, or a public content-aware recommender that can export candidate-level scores.

## Option B: Qwen-LoRA Framework

Qwen-LoRA is not just API replacement. It requires designing a local evidence generator framework: training data, schema alignment, losses for relevance/evidence fields, calibration evaluation, and downstream plug-in validation. This direction improves system ownership and cost control but is a new method-development phase.

## Recommendation By Goal

If the goal is paper-mainline closure, first write and polish the paper around the current evidence: Beauty full three-backbone multi-seed as primary performance, CEP calibration as core method, robustness as support, and small-domain results as sanity/continuity.

If the goal is continuing system development, prioritize Qwen-LoRA after freezing the current claims.

If the goal is stronger cross-domain evidence, prioritize a content-aware/cold-aware backbone for regular medium domains rather than forcing ID-only sequential backbones into cold candidate pools.
"""
    (SUMMARY_DIR / "day41_next_phase_recommendation.md").write_text(text, encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    _write_experiment_index()
    _write_main_table()
    _write_component_attribution()
    _write_claim_map()
    _write_paper_draft()
    _write_next_phase()
    print("Day41 final observation-stage closure generated.")


if __name__ == "__main__":
    main()
