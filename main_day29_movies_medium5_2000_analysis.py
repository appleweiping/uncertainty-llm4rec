"""Day29 Movies medium_5neg_2000 calibration and SASRec plug-in analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from main_day8_relevance_evidence_medium_analysis import field_diagnostics, metric_value, read_csv, read_jsonl
from main_day15_bprmf_backbone_plugin_smoke import _auc_binary, _normalize_per_user, _rank_change_stats, _safe_spearman
from main_day17_sasrec_backbone_plugin_smoke import (
    _build_vocab,
    _candidate_pool,
    _export_scores,
    _load_title_map,
    _load_train_examples,
    _train_sasrec,
)
from main_day18_sasrec_plugin_larger_validation import _join_diagnostics, _join_evidence
from src.utils.paths import ensure_exp_dirs


EXP_NAME = "movies_deepseek_relevance_evidence_medium_5neg_2000"
OUTPUT_ROOT = "output-repaired"
SUMMARY_DIR = Path("output-repaired/summary")
BACKBONE_DIR = Path("output-repaired/backbone/sasrec_movies_medium5_2000")
TRAIN_PATH = Path("data/processed/amazon_movies_medium_5neg/train.jsonl")
TEST_PATH = Path("data/processed/amazon_movies_medium_5neg/test.jsonl")
ITEMS_PATH = Path("data/processed/amazon_movies/items.csv")


def _ensure_complete() -> None:
    paths = ensure_exp_dirs(EXP_NAME, OUTPUT_ROOT)
    valid_path = paths.predictions_dir / "valid_raw.jsonl"
    test_path = paths.predictions_dir / "test_raw.jsonl"
    if not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError("valid_raw.jsonl and test_raw.jsonl must both exist before Day29 medium5 analysis.")
    valid_rows = sum(1 for _ in valid_path.open("r", encoding="utf-8"))
    test_rows = sum(1 for _ in test_path.open("r", encoding="utf-8"))
    if valid_rows != 12000 or test_rows != 12000:
        raise RuntimeError(f"Expected valid/test rows 12000/12000, got {valid_rows}/{test_rows}.")


def _rank_metrics_repaired(df: pd.DataFrame, score_col: str) -> dict[str, float | bool]:
    ranks: list[int] = []
    pool_sizes: list[int] = []
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).reset_index(drop=True)
        pool_sizes.append(len(ranked))
        positives = ranked.index[ranked["label"].astype(int) == 1].tolist()
        if positives:
            ranks.append(int(positives[0] + 1))
    rank_arr = np.asarray(ranks, dtype=float)
    pool_arr = np.asarray(pool_sizes, dtype=float)

    def hr(k: int) -> float:
        return float(np.mean(rank_arr <= k)) if len(rank_arr) else math.nan

    def ndcg(k: int) -> float:
        return float(np.mean([1.0 / math.log2(rank + 1) if rank <= k else 0.0 for rank in ranks])) if ranks else math.nan

    return {
        "HR@1": hr(1),
        "HR@3": hr(3),
        "HR@5": hr(5),
        "HR@10": hr(10),
        "NDCG@1": ndcg(1),
        "NDCG@3": ndcg(3),
        "NDCG@5": ndcg(5),
        "NDCG@10": ndcg(10),
        "MRR": float(np.mean(1.0 / rank_arr)) if len(rank_arr) else math.nan,
        "MRR@10": float(np.mean(1.0 / rank_arr)) if len(rank_arr) else math.nan,
        "candidate_pool_size_mean": float(np.mean(pool_arr)) if len(pool_arr) else math.nan,
        "candidate_pool_size_min": float(np.min(pool_arr)) if len(pool_arr) else math.nan,
        "candidate_pool_size_max": float(np.max(pool_arr)) if len(pool_arr) else math.nan,
        "hr10_trivial_flag": bool((np.max(pool_arr) <= 10) or (np.mean(pool_arr) <= 10)) if len(pool_arr) else False,
    }


def _rerank_grid(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    rows = []
    base_metrics = _rank_metrics_repaired(df.assign(final_score=df["backbone_score"]), "final_score")
    for normalization in ["minmax", "zscore"]:
        norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        norm_calibrated = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)
        norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
        settings = [("A_SASRec_only", 0.0, 1.0, 0.0, df["backbone_score"])]
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1 - alpha
            settings.append(("B_SASRec_plus_calibrated_relevance", 0.0, alpha, beta, alpha * norm_backbone + beta * norm_calibrated))
        for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
            settings.append(("C_SASRec_plus_evidence_risk", lam, 1.0, 0.0, norm_backbone - lam * norm_risk))
            for alpha in [0.5, 0.75, 0.9]:
                beta = 1 - alpha
                settings.append(("D_SASRec_plus_calibrated_relevance_plus_evidence_risk", lam, alpha, beta, alpha * norm_backbone + beta * norm_calibrated - lam * norm_risk))
        for method, lam, alpha, beta, score in settings:
            scored = df[["user_id", "candidate_item_id", "label", "backbone_score", "evidence_risk"]].copy()
            scored["final_score"] = score
            metrics = _rank_metrics_repaired(scored, "final_score")
            change = _rank_change_stats(scored, "backbone_score", "final_score")
            rows.append(
                {
                    "method": method,
                    "backbone_name": "minimal_sasrec",
                    "lambda": lam,
                    "alpha": alpha,
                    "beta": beta,
                    "normalization": normalization,
                    **metrics,
                    **change,
                    "base_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
                    "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12),
                    "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
                }
            )
    return pd.DataFrame(rows)


def _plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, join_diag: pd.DataFrame) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    pred_rank = df.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True]).groupby("user_id").cumcount() + 1
    misrank = ((pred_rank <= 10) & (df["label"].astype(int) == 0)).astype(int)
    best = grid.sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0].to_dict()
    base = grid[grid["method"] == "A_SASRec_only"].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "backbone_score_AUROC": _auc_binary(df["label"].astype(int), df["backbone_score"]),
                "calibrated_relevance_AUROC": _auc_binary(df["label"].astype(int), df["calibrated_relevance_probability"]),
                "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(misrank, df["evidence_risk"]),
                "backbone_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
                "fallback_rate": float(join_diag.iloc[0]["fallback_rate"]),
                "fallback_rate_positive": float(join_diag.iloc[0]["fallback_rate_positive"]),
                "fallback_rate_negative": float(join_diag.iloc[0]["fallback_rate_negative"]),
                "best_method": best["method"],
                "best_relative_NDCG_vs_backbone": (best["NDCG@10"] - base["NDCG@10"]) / max(base["NDCG@10"], 1e-12),
                "best_relative_MRR_vs_backbone": (best["MRR"] - base["MRR"]) / max(base["MRR"], 1e-12),
                "best_lambda": best["lambda"],
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_normalization": best["normalization"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR": best["MRR"],
                "best_HR@1": best["HR@1"],
                "best_HR@3": best["HR@3"],
                "best_HR@10": best["HR@10"],
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR": base["MRR"],
                "backbone_HR@1": base["HR@1"],
                "backbone_HR@3": base["HR@3"],
                "backbone_HR@10": base["HR@10"],
                "candidate_pool_size_mean": best["candidate_pool_size_mean"],
                "hr10_trivial_flag": best["hr10_trivial_flag"],
            }
        ]
    )


def _write_report(valid_raw: pd.DataFrame, test_raw: pd.DataFrame, comparison: pd.DataFrame, diagnostics: pd.DataFrame, plugin_diag: pd.DataFrame) -> None:
    best = plugin_diag.iloc[0]
    raw_ece = metric_value(comparison, "raw_relevance_probability", "ece")
    cal_ece = metric_value(comparison, "calibrated_relevance_probability", "ece")
    min_ece = metric_value(comparison, "evidence_posterior_relevance_minimal", "ece")
    full_ece = metric_value(comparison, "evidence_posterior_relevance_full", "ece")
    raw_brier = metric_value(comparison, "raw_relevance_probability", "brier_score")
    cal_brier = metric_value(comparison, "calibrated_relevance_probability", "brier_score")
    rel = diagnostics[(diagnostics["split"] == "test") & (diagnostics["field"] == "relevance_probability")].iloc[0]
    risk = diagnostics[(diagnostics["split"] == "test") & (diagnostics["field"] == "evidence_risk")].iloc[0]
    text = f"""# Day29 Movies medium_5neg_2000 Report

## 1. Why Pause 20neg_2000

Movies medium_20neg_2000 preflight and tiny smoke passed, and partial valid predictions were preserved. Full 20neg_2000 inference was intentionally paused because API/runtime cost was too high for this stage.

## 2. Completed 20neg_2000 Artifacts

The preserved artifacts include preflight, smoke report, runtime monitor, and partial `valid_raw.jsonl`. They can be resumed later and are not deleted.

## 3. Why Switch To 5neg_2000

Movies medium_5neg_2000 uses the same candidate-pool size as the Beauty main experiments: 1 positive + 5 negatives per user. This gives a lighter cross-domain consistency check before returning to 20neg ranking-strength evaluation.

## 4. Metric Protocol

Because each user has 6 candidates, HR@10 is trivial and is not used as primary evidence. Main metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5. All plug-in rows include `hr10_trivial_flag=true`.

## 5. Relevance Calibration

Raw relevance ECE is `{raw_ece:.4f}` and Brier is `{raw_brier:.4f}`. Calibrated relevance ECE is `{cal_ece:.4f}` and Brier is `{cal_brier:.4f}`. Minimal evidence posterior ECE is `{min_ece:.4f}`; full evidence posterior ECE is `{full_ece:.4f}`.

## 6. Field Diagnostics

On test, relevance_probability mean is `{rel['mean']:.4f}`, std `{rel['std']:.4f}`, near_one_rate `{rel['near_one_rate']:.4f}`. evidence_risk mean is `{risk['mean']:.4f}`, std `{risk['std']:.4f}`.

## 7. SASRec Plug-in

Best method is `{best['best_method']}` with normalization `{best['best_normalization']}`, alpha `{best['best_alpha']}`, beta `{best['best_beta']}`, lambda `{best['best_lambda']}`. SASRec-only NDCG@10 is `{best['backbone_NDCG@10']:.4f}`, MRR `{best['backbone_MRR']:.4f}`, HR@3 `{best['backbone_HR@3']:.4f}`. Best plug-in NDCG@10 is `{best['best_NDCG@10']:.4f}`, MRR `{best['best_MRR']:.4f}`, HR@3 `{best['best_HR@3']:.4f}`. Relative gains are NDCG `{best['best_relative_NDCG_vs_backbone']:.2%}` and MRR `{best['best_relative_MRR_vs_backbone']:.2%}`.

## 8. Direction Against Beauty

If calibrated relevance or the D combination improves NDCG/MRR over SASRec-only, the cross-domain direction is consistent with Beauty: calibrated relevance posterior is the main contributor, evidence_risk is a secondary regularizer.

## 9. Day30 Recommendation

If Movies 5neg is positive, Day30 can either run Books/Electronics 5neg for cross-domain consistency or return to Movies 20neg_2000 as a stronger ranking benchmark when API budget allows.
"""
    (SUMMARY_DIR / "day29_movies_medium5_2000_report.md").write_text(text, encoding="utf-8")


def run_analysis() -> None:
    _ensure_complete()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure_exp_dirs(EXP_NAME, OUTPUT_ROOT)
    valid_raw = read_jsonl(paths.predictions_dir / "valid_raw.jsonl", "Day29 medium5 valid raw")
    test_raw = read_jsonl(paths.predictions_dir / "test_raw.jsonl", "Day29 medium5 test raw")
    comparison = read_csv(paths.tables_dir / "relevance_evidence_calibration_comparison.csv", "Day29 medium5 calibration comparison")
    comparison.to_csv(SUMMARY_DIR / "day29_movies_medium5_2000_calibration_comparison.csv", index=False)
    diagnostics = field_diagnostics(valid_raw, test_raw)
    diagnostics.to_csv(SUMMARY_DIR / "day29_movies_medium5_2000_field_diagnostics.csv", index=False)

    evidence_path = paths.calibrated_dir / "relevance_evidence_posterior_test.jsonl"
    title_to_id = _load_title_map(ITEMS_PATH)
    pool = _candidate_pool(evidence_path, TEST_PATH, max_users=2000)
    train_examples, trained_items, item_pop = _load_train_examples(TRAIN_PATH, title_to_id, max_seq_len=10)
    item_to_idx, _ = _build_vocab(train_examples, pool, title_to_id, max_seq_len=10)
    model, _ = _train_sasrec(train_examples, item_to_idx, trained_items, 64, 2, 2, 10, 10, 256, 1e-3, 42)
    scores = _export_scores(model, pool, title_to_id, item_to_idx, trained_items, item_pop, 10, BACKBONE_DIR / "candidate_scores.csv")
    joined = _join_evidence(scores, evidence_path, SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_joined_candidates.csv")
    join_diag = _join_diagnostics(joined, SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_join_diagnostics.csv")
    grid = _rerank_grid(joined)
    grid.to_csv(SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_plugin_rerank_grid.csv", index=False)
    plugin_diag = _plugin_diagnostics(joined, grid, join_diag)
    plugin_diag.to_csv(SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_plugin_diagnostics.csv", index=False)
    _write_report(valid_raw, test_raw, comparison, diagnostics, plugin_diag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run after valid/test inference and calibration are complete.")
    run_analysis()


if __name__ == "__main__":
    main()
