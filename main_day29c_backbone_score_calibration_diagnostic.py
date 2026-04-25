"""Day29c local-only backbone score calibration diagnostic.

Backbone scores are ranking logits/sequence scores, not calibrated
probabilities. This script checks what happens if they are naively mapped to
probability-like scores. It does not call APIs, train models, or change the
Day29 Movies pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "output-repaired" / "summary"
BACKBONE = ROOT / "output-repaired" / "backbone"
SUMMARY.mkdir(parents=True, exist_ok=True)


BACKBONE_SOURCES = {
    "SASRec-style": BACKBONE / "sasrec_beauty_full" / "candidate_scores.csv",
    "GRU4Rec": BACKBONE / "llmesr_gru4rec_beauty_full" / "candidate_scores.csv",
    "Bert4Rec": BACKBONE / "llmesr_bert4rec_beauty_full" / "candidate_scores.csv",
}


def safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sigmoid(x: pd.Series) -> pd.Series:
    clipped = x.astype(float).clip(-50, 50)
    return 1.0 / (1.0 + np.exp(-clipped))


def minmax(series: pd.Series) -> tuple[pd.Series, int]:
    values = series.astype(float)
    min_v = values.min()
    max_v = values.max()
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v <= min_v:
        return pd.Series(0.5, index=series.index), len(series)
    return (values - min_v) / (max_v - min_v), 0


def per_user_minmax(df: pd.DataFrame, score_col: str) -> tuple[pd.Series, int]:
    output = pd.Series(index=df.index, dtype=float)
    fallback_rows = 0
    for _, group in df.groupby("user_id", sort=False):
        scaled, fallback = minmax(group[score_col])
        output.loc[group.index] = scaled
        fallback_rows += fallback
    return output, fallback_rows


def per_user_softmax(df: pd.DataFrame, score_col: str) -> tuple[pd.Series, int]:
    output = pd.Series(index=df.index, dtype=float)
    fallback_rows = 0
    for _, group in df.groupby("user_id", sort=False):
        scores = group[score_col].astype(float)
        if scores.nunique(dropna=True) <= 1:
            output.loc[group.index] = 1.0 / max(len(group), 1)
            fallback_rows += len(group)
            continue
        shifted = scores - scores.max()
        exp = np.exp(shifted.clip(-50, 50))
        denom = exp.sum()
        if denom <= 0 or not np.isfinite(denom):
            output.loc[group.index] = 1.0 / max(len(group), 1)
            fallback_rows += len(group)
        else:
            output.loc[group.index] = exp / denom
    return output, fallback_rows


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
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def ece_score(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> float:
    y = y_true.astype(float).to_numpy()
    p = y_prob.astype(float).clip(0, 1).to_numpy()
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not mask.any():
            continue
        ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return ece


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    y = y_true.astype(float).to_numpy()
    p = y_prob.astype(float).clip(0, 1).to_numpy()
    return float(np.mean((p - y) ** 2))


def dcg(labels: list[int]) -> float:
    return float(sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(labels)))


def ranking_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float | bool]:
    per_user: list[dict[str, float]] = []
    pool_sizes: list[int] = []
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values(score_col, ascending=False)
        labels = ranked["label"].astype(int).tolist()
        pool_sizes.append(len(labels))
        positive_ranks = [idx + 1 for idx, label in enumerate(labels) if label == 1]
        first_positive = min(positive_ranks) if positive_ranks else math.inf

        def hr(k: int) -> float:
            return float(first_positive <= k)

        def ndcg(k: int) -> float:
            top = labels[:k]
            ideal = sorted(labels, reverse=True)[:k]
            ideal_dcg = dcg(ideal)
            if ideal_dcg == 0:
                return 0.0
            return dcg(top) / ideal_dcg

        per_user.append(
            {
                "HR@1": hr(1),
                "HR@3": hr(3),
                "HR@10": hr(10),
                "NDCG@3": ndcg(3),
                "NDCG@5": ndcg(5),
                "NDCG@10": ndcg(10),
                "MRR": 0.0 if first_positive is math.inf else 1.0 / first_positive,
            }
        )
    metric_df = pd.DataFrame(per_user)
    pool = pd.Series(pool_sizes, dtype=float)
    return {
        "NDCG@10": float(metric_df["NDCG@10"].mean()),
        "MRR": float(metric_df["MRR"].mean()),
        "HR@1": float(metric_df["HR@1"].mean()),
        "HR@3": float(metric_df["HR@3"].mean()),
        "HR@10": float(metric_df["HR@10"].mean()),
        "NDCG@3": float(metric_df["NDCG@3"].mean()),
        "NDCG@5": float(metric_df["NDCG@5"].mean()),
        "candidate_pool_size_mean": float(pool.mean()),
        "candidate_pool_size_min": float(pool.min()),
        "candidate_pool_size_max": float(pool.max()),
        "hr10_trivial_flag": bool(pool.max() <= 10 or pool.mean() <= 10),
        "num_users": int(len(pool_sizes)),
    }


def collect_inventory() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    inventory: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for backbone, path in BACKBONE_SOURCES.items():
        rel = str(path.relative_to(ROOT)) if path.exists() else str(path)
        if not path.exists():
            inventory.append(
                {
                    "backbone": backbone,
                    "source_file": rel,
                    "num_rows": 0,
                    "num_users": 0,
                    "candidate_pool_size_mean": None,
                    "has_backbone_score": False,
                    "has_label": False,
                    "has_valid_split": False,
                    "has_test_split": False,
                    "notes": "source file missing",
                }
            )
            continue
        df = pd.read_csv(path)
        frames[backbone] = df
        pool = df.groupby("user_id").size()
        splits = set(df.get("split", pd.Series(dtype=str)).astype(str).str.lower())
        inventory.append(
            {
                "backbone": backbone,
                "source_file": rel,
                "num_rows": len(df),
                "num_users": df["user_id"].nunique() if "user_id" in df else 0,
                "candidate_pool_size_mean": float(pool.mean()) if len(pool) else None,
                "has_backbone_score": "backbone_score" in df.columns,
                "has_label": "label" in df.columns,
                "has_valid_split": "valid" in splits,
                "has_test_split": "test" in splits,
                "notes": "fixed/full candidate_scores; used for local diagnostic only",
            }
        )
    return pd.DataFrame(inventory), frames


def add_proxy_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    out["sigmoid_global_score"] = sigmoid(out["backbone_score"])
    out["minmax_global_score"], global_fallback = minmax(out["backbone_score"])
    out["minmax_user_score"], user_minmax_fallback = per_user_minmax(out, "backbone_score")
    out["softmax_user_score"], softmax_fallback = per_user_softmax(out, "backbone_score")
    notes = {
        "sigmoid_global_score": "sigmoid(raw backbone_score); diagnostic proxy only",
        "minmax_global_score": f"global min-max diagnostic proxy; fallback_rows={global_fallback}",
        "minmax_user_score": f"per-user min-max diagnostic proxy; fallback_rows={user_minmax_fallback}",
        "softmax_user_score": f"per-user softmax diagnostic proxy; fallback_rows={softmax_fallback}",
    }
    return out, notes


def diagnostic_for_score(
    df: pd.DataFrame, backbone: str, score_col: str, source_file: str, notes: str
) -> dict[str, Any]:
    rank = ranking_metrics(df, score_col)
    y = df["label"].astype(int)
    p = df[score_col].astype(float)
    return {
        "backbone": backbone,
        "score_proxy": score_col,
        "ECE": ece_score(y, p),
        "Brier": brier_score(y, p),
        "AUROC": auroc_score(y.to_numpy(), p.to_numpy()),
        "NDCG@10": rank["NDCG@10"],
        "MRR": rank["MRR"],
        "HR@1": rank["HR@1"],
        "HR@3": rank["HR@3"],
        "NDCG@3": rank["NDCG@3"],
        "NDCG@5": rank["NDCG@5"],
        "num_users": rank["num_users"],
        "num_rows": len(df),
        "candidate_pool_size_mean": rank["candidate_pool_size_mean"],
        "hr10_trivial_flag": rank["hr10_trivial_flag"],
        "source_file": source_file,
        "notes": notes,
    }


def collect_backbone_diagnostics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for backbone, df in frames.items():
        path = BACKBONE_SOURCES[backbone]
        rel = str(path.relative_to(ROOT))
        df = df.dropna(subset=["backbone_score", "label", "user_id"]).copy()
        df["label"] = df["label"].astype(int)
        proxied, proxy_notes = add_proxy_scores(df)
        for score_col, notes in proxy_notes.items():
            rows.append(diagnostic_for_score(proxied, backbone, score_col, rel, notes))
    return pd.DataFrame(rows)


def collect_valid_test_calibration(inventory_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in inventory_df.itertuples(index=False):
        if not bool(row.has_valid_split):
            rows.append(
                {
                    "backbone": row.backbone,
                    "calibration_method": "not_run",
                    "fit_split": "valid",
                    "eval_split": "test",
                    "raw_or_proxy_ECE": None,
                    "calibrated_ECE": None,
                    "raw_or_proxy_Brier": None,
                    "calibrated_Brier": None,
                    "raw_or_proxy_AUROC": None,
                    "calibrated_AUROC": None,
                    "notes": "valid split candidate_scores are not available; no test-fit/test-eval calibration was performed",
                }
            )
        else:
            rows.append(
                {
                    "backbone": row.backbone,
                    "calibration_method": "pending",
                    "fit_split": "valid",
                    "eval_split": "test",
                    "notes": "valid split detected but explicit calibrator fitting is not implemented in this local diagnostic",
                }
            )
    return pd.DataFrame(rows)


def collect_comparison_table(backbone_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in backbone_diag.iterrows():
        rows.append(
            {
                "signal_type": "backbone_naive_probability_proxy",
                "source": row["backbone"],
                "score_name": row["score_proxy"],
                "ECE": row["ECE"],
                "Brier": row["Brier"],
                "AUROC": row["AUROC"],
                "NDCG@10": row["NDCG@10"],
                "MRR": row["MRR"],
                "interpretation": "backbone_score is a ranking signal, not a calibrated probability",
            }
        )

    # CEP and raw relevance are evaluated on the same Beauty pointwise candidate pool.
    joined = SUMMARY / "day19_sasrec_beauty_full_joined_candidates.csv"
    if joined.exists():
        df = pd.read_csv(joined)
        for score_col, signal_type, interpretation in [
            (
                "relevance_probability",
                "llm_raw_relevance",
                "raw LLM relevance is informative but miscalibrated",
            ),
            (
                "calibrated_relevance_probability",
                "cep_calibrated_relevance_posterior",
                "CEP provides calibrated relevance posterior",
            ),
        ]:
            if score_col not in df.columns:
                continue
            score_df = df.dropna(subset=[score_col, "label", "user_id"]).copy()
            rank = ranking_metrics(score_df, score_col)
            rows.append(
                {
                    "signal_type": signal_type,
                    "source": str(joined.relative_to(ROOT)),
                    "score_name": score_col,
                    "ECE": ece_score(score_df["label"], score_df[score_col]),
                    "Brier": brier_score(score_df["label"], score_df[score_col]),
                    "AUROC": auroc_score(
                        score_df["label"].astype(int).to_numpy(),
                        score_df[score_col].astype(float).to_numpy(),
                    ),
                    "NDCG@10": rank["NDCG@10"],
                    "MRR": rank["MRR"],
                    "interpretation": interpretation,
                }
            )

    day29b_raw = SUMMARY / "day29b_beauty_multimodel_raw_confidence_diagnostics.csv"
    if day29b_raw.exists():
        raw = pd.read_csv(day29b_raw)
        beauty = raw[raw["domain"] == "Beauty"]
        for row in beauty.itertuples(index=False):
            rows.append(
                {
                    "signal_type": "llm_raw_confidence",
                    "source": row.source_file,
                    "score_name": f"{row.model}_raw_confidence",
                    "ECE": row.raw_diagnostic_ece,
                    "Brier": row.raw_brier,
                    "AUROC": row.raw_auroc,
                    "NDCG@10": None,
                    "MRR": None,
                    "interpretation": "raw LLM confidence is informative but miscalibrated",
                }
            )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_report(
    inventory: pd.DataFrame,
    diag: pd.DataFrame,
    valid_test: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_lines = []
    for backbone in BACKBONE_SOURCES:
        sub = diag[diag["backbone"] == backbone].sort_values("Brier")
        if sub.empty:
            continue
        best = sub.iloc[0]
        rank_best = sub.sort_values("NDCG@10", ascending=False).iloc[0]
        best_lines.append(
            f"- {backbone}: best naive Brier proxy is `{best['score_proxy']}` "
            f"(ECE={fmt(best['ECE'])}, Brier={fmt(best['Brier'])}, AUROC={fmt(best['AUROC'])}); "
            f"best ranking proxy is `{rank_best['score_proxy']}` "
            f"(NDCG@10={fmt(rank_best['NDCG@10'])}, MRR={fmt(rank_best['MRR'])})."
        )

    cep = comparison[comparison["signal_type"] == "cep_calibrated_relevance_posterior"]
    cep_line = "CEP comparison source not available."
    if not cep.empty:
        row = cep.iloc[0]
        cep_line = (
            f"CEP calibrated relevance posterior on the Beauty candidate pool reports "
            f"ECE={fmt(row['ECE'])}, Brier={fmt(row['Brier'])}, "
            f"NDCG@10={fmt(row['NDCG@10'])}, and MRR={fmt(row['MRR'])}."
        )

    no_valid = valid_test["notes"].astype(str).str.contains("not available").all()
    valid_note = (
        "Valid split backbone scores are not available for the fixed full Beauty candidate-score files, "
        "so this report does not fit a proper backbone calibrator. It only reports naive probability diagnostics."
        if no_valid
        else "Some valid split backbone scores were detected; see the valid/test calibration table."
    )

    report = f"""# Day29c Backbone Score Calibration Diagnostic

## 1. Motivation

Day29b consolidated the observation that raw LLM confidence and raw relevance probability are informative but miscalibrated. Day29c checks the analogous question for recommender backbone scores: can SASRec-style, GRU4Rec, or Bert4Rec `backbone_score` be directly used as confidence or probability?

## 2. Clarification

The answer should be no by design. These backbone scores are ranking logits, dot products, or sequence scores. They are effective for ordering candidates, but they are not calibrated estimates of `P(relevant)`. This diagnostic is therefore not a critique that the backbones are broken; it is a reminder not to interpret raw ranking scores as probabilities.

## 3. Diagnostic Result

If we map backbone scores into probability-like quantities with naive transformations such as global sigmoid, global min-max, per-user min-max, or per-user softmax, calibration remains unreliable:

{chr(10).join(best_lines)}

The candidate pool size is approximately 6 for the Beauty full candidate-pool evaluation, so HR@10 is trivial and not used as evidence. NDCG and MRR remain meaningful because they depend on the positive item's exact rank.

## 4. Ranking vs Calibration

The backbone score's value is ranking ability, measured by NDCG/MRR/HR@1/HR@3. CEP's value is calibrated relevance posterior quality, measured by ECE/Brier, and then its usefulness as a plug-in decision signal. {cep_line}

## 5. Connection to Main Method

The final method does not replace the external backbone with CEP and does not treat backbone scores as confidence. The intended decomposition is:

- `backbone_score` provides ranking ability.
- `calibrated_relevance_probability` provides calibrated posterior relevance.
- `evidence_risk` provides secondary risk regularization.

This is the same interpretation used in the Day20/Day23/Day25 external backbone results.

## 6. Claim Boundary

{valid_note} We therefore do not claim that all backbone scores have been fully calibrated. The scoped conclusion is diagnostic: raw recommender scores should not be used as calibrated confidence without calibration, and CEP is better positioned as a calibrated posterior plug-in.

## Local-Only Execution Note

This analysis read existing `candidate_scores.csv`, joined-candidate summaries, and Day29b tables only. It did not call DeepSeek, did not train any backbone, did not change prompt/parser/formula code, and did not touch the running Day29 Movies inference process.
"""
    (SUMMARY / "day29c_backbone_score_miscalibration_report.md").write_text(report, encoding="utf-8")


def write_paper_snippet() -> None:
    snippet = """# Backbone Calibration Paper Snippet

Although sequential recommenders provide useful ranking scores, these scores should not be interpreted as calibrated probabilities. In our diagnostic, SASRec-style, GRU4Rec, and Bert4Rec candidate scores remain miscalibrated under naive probability mappings such as sigmoid, min-max normalization, and per-user softmax. This does not indicate that the backbones fail as recommenders; rather, it reflects that ranking logits and sequence scores are optimized for ordering candidates, not for estimating calibrated relevance probabilities.

This distinction motivates separating ranking ability from uncertainty estimation. External backbones provide candidate ranking signals, while CEP provides an evidence-grounded calibrated relevance posterior and an auxiliary evidence-risk signal. In downstream plug-in experiments, CEP is therefore used to complement backbone ranking scores rather than replacing the backbone or treating raw model scores as confidence.
"""
    (SUMMARY / "day29c_backbone_calibration_paper_snippet.md").write_text(snippet, encoding="utf-8")


def main() -> None:
    inventory, frames = collect_inventory()
    diagnostics = collect_backbone_diagnostics(frames)
    valid_test = collect_valid_test_calibration(inventory)
    comparison = collect_comparison_table(diagnostics)

    inventory.to_csv(SUMMARY / "day29c_backbone_score_source_inventory.csv", index=False)
    diagnostics.to_csv(SUMMARY / "day29c_backbone_score_calibration_diagnostics.csv", index=False)
    valid_test.to_csv(SUMMARY / "day29c_backbone_score_valid_test_calibration.csv", index=False)
    comparison.to_csv(SUMMARY / "day29c_backbone_vs_cep_calibration_comparison.csv", index=False)
    write_report(inventory, diagnostics, valid_test, comparison)
    write_paper_snippet()

    print(f"Wrote inventory rows: {len(inventory)}")
    print(f"Wrote backbone diagnostic rows: {len(diagnostics)}")
    print(f"Wrote valid/test calibration rows: {len(valid_test)}")
    print(f"Wrote comparison rows: {len(comparison)}")


if __name__ == "__main__":
    main()
