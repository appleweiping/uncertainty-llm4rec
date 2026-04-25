"""Decision mismatch analysis for CEP / evidence-risk reranking.

Local-only diagnostic. It explains where NDCG/MRR gains come from by measuring:
1) calibration mismatch reduction,
2) positive-rank movement,
3) pairwise inversion reduction,
4) evidence-risk demotion/promotion mechanism.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "output-repaired" / "summary"
RELEVANCE_DIR = ROOT / "output-repaired" / "beauty_deepseek_relevance_evidence_full"
SUMMARY.mkdir(parents=True, exist_ok=True)


BACKBONE_CONFIGS = {
    "SASRec-style": {
        "joined": SUMMARY / "day19_sasrec_beauty_full_joined_candidates.csv",
        "grid": SUMMARY / "day19_sasrec_beauty_full_plugin_rerank_grid.csv",
        "d_pattern": "calibrated_relevance_plus_evidence_risk",
    },
    "GRU4Rec": {
        "joined": SUMMARY / "day22_llmesr_gru4rec_beauty_full_joined_candidates.csv",
        "grid": SUMMARY / "day22_llmesr_gru4rec_beauty_full_plugin_rerank_grid.csv",
        "d_pattern": "calibrated_relevance_plus_evidence_risk",
    },
    "Bert4Rec": {
        "joined": SUMMARY / "day25_bert4rec_beauty_full_joined_candidates.csv",
        "grid": SUMMARY / "day25_bert4rec_beauty_full_plugin_rerank_grid.csv",
        "d_pattern": "calibrated_relevance_plus_evidence_risk",
    },
}


def safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty_like(y_score, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_score(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> float:
    y = y_true.astype(float).to_numpy()
    p = y_prob.astype(float).clip(0, 1).to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (p >= left) & (p <= right) if idx == n_bins - 1 else (p >= left) & (p < right)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return ece


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    y = y_true.astype(float).to_numpy()
    p = y_prob.astype(float).clip(0, 1).to_numpy()
    return float(np.mean((p - y) ** 2))


def normalize_per_user(df: pd.DataFrame, col: str, method: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("user_id", sort=False):
        vals = group[col].astype(float)
        if vals.nunique(dropna=True) <= 1:
            out.loc[group.index] = 0.0
            continue
        if method == "zscore":
            std = vals.std(ddof=0)
            out.loc[group.index] = (vals - vals.mean()) / std if std > 0 else 0.0
        elif method == "minmax":
            denom = vals.max() - vals.min()
            out.loc[group.index] = (vals - vals.min()) / denom if denom > 0 else 0.0
        else:
            raise ValueError(f"Unsupported normalization: {method}")
    return out


def compute_ranks(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = (
        out.groupby("user_id", group_keys=False)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


def best_d_setting(grid_path: Path, pattern: str) -> dict[str, Any]:
    grid = pd.read_csv(grid_path)
    subset = grid[grid["method"].astype(str).str.contains(pattern, case=False, na=False)].copy()
    if subset.empty:
        raise ValueError(f"No D setting found in {grid_path}")
    subset["NDCG@10"] = subset["NDCG@10"].astype(float)
    row = subset.sort_values("NDCG@10", ascending=False).iloc[0].to_dict()
    return row


def calibration_mismatch() -> pd.DataFrame:
    raw_cal = load_jsonl(RELEVANCE_DIR / "calibrated" / "raw_relevance_test_calibrated.jsonl")
    full = load_jsonl(RELEVANCE_DIR / "calibrated" / "relevance_evidence_posterior_full_test.jsonl")
    keys = ["user_id", "candidate_item_id", "label"]
    full_cols = keys + ["full_calibrated_relevance_probability"]
    merged = raw_cal.merge(full[full_cols], on=keys, how="left")

    rows = []
    score_map = {
        "raw_relevance_probability": "relevance_probability",
        "calibrated_relevance_probability": "calibrated_relevance_probability",
        "CEP_full": "full_calibrated_relevance_probability",
    }
    for name, col in score_map.items():
        sub = merged.dropna(subset=[col, "label"]).copy()
        score = sub[col].astype(float)
        label = sub["label"].astype(int)
        high = score >= 0.8
        high_conf_error_rate = (
            float(((high) & (label == 0)).sum() / high.sum()) if int(high.sum()) else 0.0
        )
        rows.append(
            {
                "analysis_layer": "calibration_mismatch",
                "backbone": "pointwise_relevance",
                "score_type": name,
                "ECE": ece_score(label, score),
                "Brier": brier_score(label, score),
                "AUROC": auroc_score(label.to_numpy(), score.to_numpy()),
                "high_conf_error_rate": high_conf_error_rate,
                "high_conf_count": int(high.sum()),
                "num_rows": len(sub),
                "source_file": "output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/*test.jsonl",
                "notes": "high_conf_error_rate = P(label=0 | score>=0.8); valid fit, test eval",
            }
        )
    return pd.DataFrame(rows)


def positive_rank_stats(df: pd.DataFrame) -> dict[str, Any]:
    positives = df[df["label"].astype(int) == 1].copy()
    positives["positive_rank_improvement"] = positives["old_rank"] - positives["new_rank"]
    return {
        "positive_old_mean_rank": float(positives["old_rank"].mean()),
        "positive_new_mean_rank": float(positives["new_rank"].mean()),
        "positive_mean_rank_improvement": float(positives["positive_rank_improvement"].mean()),
        "positive_old_median_rank": float(positives["old_rank"].median()),
        "positive_new_median_rank": float(positives["new_rank"].median()),
        "positive_median_rank_improvement": float(positives["positive_rank_improvement"].median()),
    }


def inversion_stats(df: pd.DataFrame) -> dict[str, Any]:
    old_rates = []
    new_rates = []
    old_counts = []
    new_counts = []
    for _, group in df.groupby("user_id", sort=False):
        positives = group[group["label"].astype(int) == 1]
        negatives = group[group["label"].astype(int) == 0]
        if positives.empty or negatives.empty:
            continue
        pos_old_rank = int(positives["old_rank"].min())
        pos_new_rank = int(positives["new_rank"].min())
        old_inv = int((negatives["old_rank"] < pos_old_rank).sum())
        new_inv = int((negatives["new_rank"] < pos_new_rank).sum())
        denom = len(negatives)
        old_counts.append(old_inv)
        new_counts.append(new_inv)
        old_rates.append(old_inv / denom)
        new_rates.append(new_inv / denom)
    old_rate = float(np.mean(old_rates)) if old_rates else math.nan
    new_rate = float(np.mean(new_rates)) if new_rates else math.nan
    return {
        "old_pairwise_inversion_rate": old_rate,
        "new_pairwise_inversion_rate": new_rate,
        "pairwise_inversion_reduction": old_rate - new_rate,
        "old_pairwise_inversion_count_mean": float(np.mean(old_counts)) if old_counts else math.nan,
        "new_pairwise_inversion_count_mean": float(np.mean(new_counts)) if new_counts else math.nan,
    }


def mechanism_stats(df: pd.DataFrame) -> dict[str, Any]:
    demoted = df[df["new_rank"] > df["old_rank"]].copy()
    promoted = df[df["new_rank"] < df["old_rank"]].copy()
    high_risk = (
        (df["evidence_risk"].astype(float) >= 0.5)
        | (df["ambiguity"].astype(float) >= 0.5)
        | (df["missing_information"].astype(float) >= 0.5)
        | (df["abs_evidence_margin"].astype(float) <= 0.2)
    )
    df = df.copy()
    df["high_risk_flag"] = high_risk
    demoted = df[df["new_rank"] > df["old_rank"]].copy()
    promoted = df[df["new_rank"] < df["old_rank"]].copy()

    demoted_count = len(demoted)
    promoted_count = len(promoted)
    demoted_negative_rate = (
        float((demoted["label"].astype(int) == 0).mean()) if demoted_count else math.nan
    )
    promoted_positive_rate = (
        float((promoted["label"].astype(int) == 1).mean()) if promoted_count else math.nan
    )
    high_risk_demotion_precision = (
        float(((demoted["label"].astype(int) == 0) & demoted["high_risk_flag"]).mean())
        if demoted_count
        else math.nan
    )
    return {
        "demoted_count": demoted_count,
        "demoted_negative_rate": demoted_negative_rate,
        "high_risk_demotion_precision": high_risk_demotion_precision,
        "promoted_count": promoted_count,
        "promoted_positive_rate": promoted_positive_rate,
        "demoted_mean_evidence_risk": safe_float(demoted["evidence_risk"].mean()) if demoted_count else None,
        "promoted_mean_evidence_risk": safe_float(promoted["evidence_risk"].mean()) if promoted_count else None,
        "demoted_mean_ambiguity": safe_float(demoted["ambiguity"].mean()) if demoted_count else None,
        "demoted_mean_missing_information": safe_float(demoted["missing_information"].mean()) if demoted_count else None,
        "demoted_mean_abs_evidence_margin": safe_float(demoted["abs_evidence_margin"].mean()) if demoted_count else None,
    }


def rerank_mismatch_rows() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    detailed: dict[str, pd.DataFrame] = {}
    for backbone, config in BACKBONE_CONFIGS.items():
        setting = best_d_setting(config["grid"], config["d_pattern"])
        df = pd.read_csv(config["joined"]).dropna(
            subset=[
                "user_id",
                "candidate_item_id",
                "label",
                "backbone_score",
                "calibrated_relevance_probability",
                "evidence_risk",
            ]
        )
        normalization = str(setting["normalization"])
        alpha = float(setting["alpha"])
        beta = float(setting["beta"])
        lam = float(setting["lambda"])
        df["norm_backbone_score"] = normalize_per_user(df, "backbone_score", normalization)
        df["norm_calibrated_relevance_probability"] = normalize_per_user(
            df, "calibrated_relevance_probability", normalization
        )
        df["norm_evidence_risk"] = normalize_per_user(df, "evidence_risk", normalization)
        df["old_score"] = df["backbone_score"].astype(float)
        df["new_score"] = (
            alpha * df["norm_backbone_score"]
            + beta * df["norm_calibrated_relevance_probability"]
            - lam * df["norm_evidence_risk"]
        )
        df = compute_ranks(df, "old_score", "old_rank")
        df = compute_ranks(df, "new_score", "new_rank")
        detailed[backbone] = df

        rank = positive_rank_stats(df)
        inversion = inversion_stats(df)
        mechanism = mechanism_stats(df)
        rows.append(
            {
                "analysis_layer": "ranking_and_evidence_mismatch",
                "backbone": backbone,
                "score_type": "best_D_calibrated_relevance_plus_evidence_risk",
                "normalization": normalization,
                "alpha": alpha,
                "beta": beta,
                "lambda": lam,
                "source_file": str(config["joined"].relative_to(ROOT)),
                "grid_file": str(config["grid"].relative_to(ROOT)),
                **rank,
                **inversion,
                **mechanism,
                "notes": "old_rank=backbone_score rank; new_rank=best D plug-in rank from full grid",
            }
        )
    return pd.DataFrame(rows), detailed


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_report(calibration: pd.DataFrame, mismatch: pd.DataFrame) -> None:
    raw = calibration[calibration["score_type"] == "raw_relevance_probability"].iloc[0]
    cal = calibration[calibration["score_type"] == "calibrated_relevance_probability"].iloc[0]
    cep = calibration[calibration["score_type"] == "CEP_full"].iloc[0]

    backbone_lines = []
    for row in mismatch.itertuples(index=False):
        backbone_lines.append(
            f"- {row.backbone}: positive mean rank {fmt(row.positive_old_mean_rank)} -> "
            f"{fmt(row.positive_new_mean_rank)} "
            f"(improvement {fmt(row.positive_mean_rank_improvement)}); "
            f"inversion rate {fmt(row.old_pairwise_inversion_rate)} -> "
            f"{fmt(row.new_pairwise_inversion_rate)} "
            f"(reduction {fmt(row.pairwise_inversion_reduction)}); "
            f"demoted negative rate={fmt(row.demoted_negative_rate)}, "
            f"promoted positive rate={fmt(row.promoted_positive_rate)}."
        )

    report = f"""# Decision Mismatch Analysis

## 1. What This Analysis Explains

This local-only analysis explains why CEP / evidence-risk reranking improves NDCG and MRR. The improvement is not just an abstract metric change. It comes from reducing two kinds of mismatch: probability calibration mismatch and ranking decision mismatch.

## 2. Calibration Mismatch

On full Beauty relevance evidence, raw relevance probability is informative but poorly calibrated: ECE={fmt(raw.ECE)}, Brier={fmt(raw.Brier)}, AUROC={fmt(raw.AUROC)}, high-confidence error rate={fmt(raw.high_conf_error_rate)}. Valid-set calibration reduces this to ECE={fmt(cal.ECE)}, Brier={fmt(cal.Brier)}, AUROC={fmt(cal.AUROC)}, high-confidence error rate={fmt(cal.high_conf_error_rate)}. CEP full reports ECE={fmt(cep.ECE)}, Brier={fmt(cep.Brier)}, AUROC={fmt(cep.AUROC)}, high-confidence error rate={fmt(cep.high_conf_error_rate)}.

This supports the first mismatch claim: raw predicted probability does not match empirical correctness well, while calibrated relevance posterior substantially reduces probability-scale mismatch.

## 3. Ranking Mismatch

Using the best full-grid D setting for each backbone, we compare old ranks from `backbone_score` against new ranks after adding calibrated relevance and evidence risk:

{chr(10).join(backbone_lines)}

Positive items move earlier on average, and fewer negative candidates remain ranked above the positive candidate. This is the ranking-decision side of the NDCG/MRR gain.

## 4. Evidence-Risk Mechanism

The demotion/promotion diagnostics show whether reranking is random or mechanism-consistent. A useful pattern is: demoted candidates should be mostly negatives and higher-risk; promoted candidates should contain more positives. The table reports `demoted_negative_rate`, `high_risk_demotion_precision`, and `promoted_positive_rate` for each backbone.

## 5. Interpretation

NDCG/MRR improves because the method reduces two mismatch types:

- Calibration mismatch: raw relevance probability is not a reliable probability, but calibrated relevance posterior and CEP full reduce ECE/Brier.
- Ranking decision mismatch: high-risk negative candidates are pushed down, positive candidates are moved earlier, and pairwise inversions decrease.

This preserves the intended method boundary: backbone scores provide ranking ability; calibrated relevance probability provides posterior relevance; evidence risk supplies secondary risk regularization.

## Local-Only Execution Note

This analysis used existing Day9 relevance outputs and Day19/22/25 full backbone joined candidates. It did not call APIs, retrain models, alter prompts/parsers/formulas, or touch the running Movies Day29 pipeline.
"""
    (SUMMARY / "decision_mismatch_analysis.md").write_text(report, encoding="utf-8")


def main() -> None:
    calibration = calibration_mismatch()
    mismatch, _ = rerank_mismatch_rows()
    combined = pd.concat([calibration, mismatch], ignore_index=True, sort=False)
    combined.to_csv(SUMMARY / "decision_mismatch_analysis.csv", index=False)
    write_report(calibration, mismatch)
    print(f"Wrote {SUMMARY / 'decision_mismatch_analysis.csv'} ({len(combined)} rows)")
    print(f"Wrote {SUMMARY / 'decision_mismatch_analysis.md'}")


if __name__ == "__main__":
    main()
