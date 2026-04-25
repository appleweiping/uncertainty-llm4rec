"""Day14 simple external backbone + Scheme-4 plug-in smoke test.

The first simple backbone is train-only item popularity. It is intentionally
lightweight and reproducible: no test labels, Day9 scores, or Day10 list ranks
are used as backbone scores.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EVIDENCE_COLUMNS = [
    "relevance_probability",
    "calibrated_relevance_probability",
    "evidence_risk",
    "ambiguity",
    "missing_information",
    "abs_evidence_margin",
    "positive_evidence",
    "negative_evidence",
]
SUMMARY_DIR = Path("output-repaired/summary")
BACKBONE_DIR = Path("output-repaired/backbone/simple_beauty_100")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_preview(path: Path, rows: list[dict], limit: int = 3) -> str:
    preview = rows[:limit]
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in preview)


def _build_train_popularity(train_path: Path) -> Counter:
    counts: Counter = Counter()
    for row in _read_jsonl(train_path):
        if int(row.get("label", 0)) == 1:
            item_id = row.get("candidate_item_id")
            if item_id:
                counts[str(item_id)] += 1
    return counts


def _select_candidate_pool(evidence_df: pd.DataFrame, max_users: int) -> pd.DataFrame:
    user_order = evidence_df["user_id"].drop_duplicates().head(max_users).tolist()
    return evidence_df[evidence_df["user_id"].isin(user_order)].copy()


def _export_candidate_scores(train_path: Path, evidence_path: Path, output_path: Path, max_users: int) -> pd.DataFrame:
    counts = _build_train_popularity(train_path)
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    pool = _select_candidate_pool(evidence, max_users)
    rows = []
    for _, row in pool.iterrows():
        candidate_id = str(row["candidate_item_id"])
        count = counts.get(candidate_id, 0)
        rows.append(
            {
                "user_id": row["user_id"],
                "candidate_item_id": candidate_id,
                "backbone_score": math.log1p(count),
                "label": int(row["label"]),
                "split": "test",
                "backbone_name": "train_popularity",
                "train_positive_count": count,
                "fallback_score": 1 if count == 0 else 0,
            }
        )
    scores = pd.DataFrame(rows)
    scores["backbone_rank"] = scores.groupby("user_id")["backbone_score"].rank(ascending=False, method="first").astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return scores


def _join_evidence(candidate_scores: pd.DataFrame, evidence_path: Path, output_path: Path) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    keep_cols = ["user_id", "candidate_item_id"] + [c for c in EVIDENCE_COLUMNS if c in evidence.columns]
    evidence = evidence[keep_cols].drop_duplicates(["user_id", "candidate_item_id"])
    joined = candidate_scores.merge(evidence, on=["user_id", "candidate_item_id"], how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return joined


def _normalize_per_user(values: pd.Series, users: pd.Series, method: str) -> pd.Series:
    out = pd.Series(index=values.index, dtype=float)
    for _, idx in users.groupby(users).groups.items():
        group = values.loc[idx].astype(float)
        if method == "minmax":
            lo = group.min()
            hi = group.max()
            denom = hi - lo
            out.loc[idx] = 0.0 if denom == 0 else (group - lo) / denom
        elif method == "zscore":
            std = group.std(ddof=0)
            out.loc[idx] = 0.0 if std == 0 else (group - group.mean()) / std
        else:
            raise ValueError(f"unknown normalization: {method}")
    return out.fillna(0.0)


def _rank_metrics(df: pd.DataFrame, score_col: str, k: int = 10) -> dict[str, float]:
    hits = []
    ndcgs = []
    mrrs = []
    recalls = []
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values(score_col, ascending=False).head(k)
        labels = ranked["label"].astype(int).tolist()
        total_pos = int(group["label"].astype(int).sum())
        hit = 1.0 if any(labels) else 0.0
        dcg = sum(label / math.log2(rank + 2) for rank, label in enumerate(labels))
        ideal_hits = min(total_pos, k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        rr = 0.0
        for rank, label in enumerate(labels, start=1):
            if label:
                rr = 1.0 / rank
                break
        hits.append(hit)
        ndcgs.append(0.0 if idcg == 0 else dcg / idcg)
        mrrs.append(rr)
        recalls.append(0.0 if total_pos == 0 else sum(labels) / total_pos)
    return {
        "HR@10": float(np.mean(hits)) if hits else math.nan,
        "NDCG@10": float(np.mean(ndcgs)) if ndcgs else math.nan,
        "MRR@10": float(np.mean(mrrs)) if mrrs else math.nan,
        "Recall@10": float(np.mean(recalls)) if recalls else math.nan,
    }


def _rank_change_stats(df: pd.DataFrame, base_col: str, final_col: str) -> dict[str, float]:
    rank_changed = []
    top10_changed = []
    taus = []
    spearmans = []
    for _, group in df.groupby("user_id", sort=False):
        g = group.copy()
        g["_base_rank"] = g[base_col].rank(ascending=False, method="first")
        g["_final_rank"] = g[final_col].rank(ascending=False, method="first")
        rank_changed.append(float((g["_base_rank"] != g["_final_rank"]).mean()))
        base_top = tuple(g.sort_values("_base_rank")["candidate_item_id"].head(10))
        final_top = tuple(g.sort_values("_final_rank")["candidate_item_id"].head(10))
        top10_changed.append(1.0 if base_top != final_top else 0.0)
        if len(g) > 1:
            taus.append(float(g["_base_rank"].corr(g["_final_rank"], method="kendall")))
            spearmans.append(float(g[base_col].corr(g["evidence_risk"], method="spearman")))
    return {
        "rank_change_rate": float(np.nanmean(rank_changed)) if rank_changed else math.nan,
        "top10_order_change_rate": float(np.nanmean(top10_changed)) if top10_changed else math.nan,
        "mean_kendall_tau": float(np.nanmean(taus)) if taus else math.nan,
        "base_risk_spearman": float(np.nanmean(spearmans)) if spearmans else math.nan,
    }


def _auc_binary(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _write_join_diagnostics(joined: pd.DataFrame, out_path: Path) -> None:
    rows = [
        {
            "num_backbone_rows": len(joined),
            "num_joined_rows": int(joined["evidence_risk"].notna().sum()),
            "join_coverage": float(joined["evidence_risk"].notna().mean()) if len(joined) else 0.0,
            "num_users": int(joined["user_id"].nunique()),
            "num_candidates": len(joined),
            "num_positive_labels": int(joined["label"].astype(int).sum()),
            "missing_evidence_rows": int(joined["evidence_risk"].isna().sum()),
            "missing_backbone_score_rows": int(joined["backbone_score"].isna().sum()),
            "fallback_score_rows": int(joined["fallback_score"].fillna(0).astype(int).sum()),
        }
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _rerank_grid(joined: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "evidence_risk", "calibrated_relevance_probability", "label"]).copy()
    rows = []
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5]
    normalizations = ["minmax", "zscore"]
    alpha = 0.75
    beta = 0.25
    for normalization in normalizations:
        norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        norm_rel = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)
        norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
        settings = [("A_Backbone_only", 0.0, norm_backbone)]
        for lam in lambdas:
            settings.append(("B_Backbone_plus_calibrated_relevance", lam, alpha * norm_backbone + (1 - alpha) * norm_rel))
            settings.append(("C_Backbone_plus_evidence_risk", lam, norm_backbone - lam * norm_risk))
            settings.append(("D_Backbone_plus_calibrated_relevance_plus_evidence_risk", lam, alpha * norm_backbone + beta * norm_rel - lam * norm_risk))
        for method, lam, final_score in settings:
            local = df.copy()
            local["final_score"] = final_score
            rows.append(
                {
                    "method": method,
                    "backbone_name": "train_popularity",
                    "lambda": lam,
                    "alpha": alpha,
                    "beta": beta,
                    "normalization": normalization,
                    **_rank_metrics(local, "final_score"),
                    **_rank_change_stats(local, "backbone_score", "final_score"),
                }
            )
    grid = pd.DataFrame(rows).drop_duplicates()
    grid.to_csv(out_path, index=False)
    return grid


def _write_plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "evidence_risk", "calibrated_relevance_probability", "label"]).copy()
    base_metrics = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    df["_base_rank"] = df.groupby("user_id")["backbone_score"].rank(ascending=False, method="first")
    wrong_high = df[(df["_base_rank"] <= 10) & (df["label"].astype(int) == 0)]
    correct_high = df[(df["_base_rank"] <= 10) & (df["label"].astype(int) == 1)]
    error_label = ((df["_base_rank"] <= 10) & (df["label"].astype(int) == 0)).astype(int)
    diag = pd.DataFrame(
        [
            {
                "backbone_score_AUROC": _auc_binary(df["label"].astype(int), df["backbone_score"]),
                "calibrated_relevance_AUROC": _auc_binary(df["label"].astype(int), df["calibrated_relevance_probability"]),
                "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(error_label, df["evidence_risk"]),
                "backbone_risk_spearman": float(df["backbone_score"].corr(df["evidence_risk"], method="spearman")),
                "risk_mean_for_wrong_high_rank": float(wrong_high["evidence_risk"].mean()) if len(wrong_high) else math.nan,
                "risk_mean_for_correct_high_rank": float(correct_high["evidence_risk"].mean()) if len(correct_high) else math.nan,
                "best_method": best["method"],
                "best_normalization": best["normalization"],
                "best_lambda": best["lambda"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR@10": best["MRR@10"],
                "best_relative_NDCG_vs_backbone": (best["NDCG@10"] - base_metrics["NDCG@10"]) / base_metrics["NDCG@10"] if base_metrics["NDCG@10"] else math.nan,
                "best_relative_MRR_vs_backbone": (best["MRR@10"] - base_metrics["MRR@10"]) / base_metrics["MRR@10"] if base_metrics["MRR@10"] else math.nan,
            }
        ]
    )
    diag.to_csv(out_path, index=False)
    return diag


def _write_report(candidate_scores: pd.DataFrame, joined: pd.DataFrame, grid: pd.DataFrame, diagnostics: pd.DataFrame, out_path: Path) -> None:
    join_coverage = float(joined["evidence_risk"].notna().mean()) if len(joined) else 0.0
    base = grid[grid["method"] == "A_Backbone_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    diag = diagnostics.iloc[0].to_dict()
    report = f"""# Day14 Simple External Backbone Plug-in Smoke Report

## 1. Why LLMEmb Is Temporarily Blocked

Day13 cloned and audited LLMEmb, and located the true score path in `external/LLMEmb/trainers/sequence_trainer.py`. However, the official clone does not include Beauty handled data, LLM item embeddings, SRS item embeddings, or a trained checkpoint. Therefore, LLMEmb remains a stronger external baseline target, but it should not block the first plug-in smoke test.

## 2. Why Simple Backbone First

Day14 uses a train-only item popularity backbone named `train_popularity`. It is not claimed as a strong SOTA baseline. Its role is to produce a real, reproducible `backbone_score` that is independent of Day9/Day10 scores and can be joined with Scheme-4 evidence fields.

## 3. Backbone Construction And Leakage Control

The backbone uses only positive rows from:

`data/processed/amazon_beauty/train.jsonl`

For each item:

`backbone_score = log(1 + train_positive_count)`

No valid/test labels are used to construct the score. Candidates unseen in train receive `backbone_score = 0` and `fallback_score = 1`.

## 4. Score Export Schema

Score export:

`output-repaired/backbone/simple_beauty_100/candidate_scores.csv`

Rows: `{len(candidate_scores)}`

Users: `{candidate_scores["user_id"].nunique()}`

Fields include:

`user_id, candidate_item_id, backbone_score, label, backbone_rank, split, backbone_name, train_positive_count, fallback_score`

## 5. Day9 Evidence Join

Joined table:

`output-repaired/summary/day14_simple_backbone_beauty_100_joined_candidates.csv`

Join coverage: `{join_coverage:.4f}`

Missing evidence rows: `{int(joined["evidence_risk"].isna().sum())}`

Fallback score rows: `{int(joined["fallback_score"].sum())}`

Because this backbone reuses Day9 candidate rows, candidate alignment is clean enough for smoke-test interpretation.

## 6. Plug-in Rerank Smoke Result

Backbone-only best row:

`{base}`

Best plug-in/grid row:

`{best}`

Diagnostics:

`{diag}`

The result should be read as a technical plug-in smoke test, not an external SOTA claim. If Scheme-4 improves over this weak backbone, it shows the adapter can exploit evidence fields. If not, the pipeline is still useful because score export, evidence join, and grid evaluation are now unblocked.

## 7. Day15 Recommendation

If the goal is quick scale-up, run the same simple backbone on full Beauty candidate pools to verify stability. If the goal is stronger evidence, implement BPR-MF or SASRec next. LLMEmb should resume only after its handled Beauty data, embeddings, checkpoint, and reversible id mapping are prepared.
"""
    out_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/amazon_beauty/train.jsonl"))
    parser.add_argument(
        "--evidence_path",
        type=Path,
        default=Path("output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl"),
    )
    parser.add_argument("--max_users", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)

    candidate_path = BACKBONE_DIR / "candidate_scores.csv"
    joined_path = SUMMARY_DIR / "day14_simple_backbone_beauty_100_joined_candidates.csv"
    join_diag_path = SUMMARY_DIR / "day14_simple_backbone_beauty_100_join_diagnostics.csv"
    grid_path = SUMMARY_DIR / "day14_simple_backbone_beauty_100_plugin_rerank_grid.csv"
    plugin_diag_path = SUMMARY_DIR / "day14_simple_backbone_beauty_100_plugin_diagnostics.csv"
    report_path = SUMMARY_DIR / "day14_simple_backbone_plugin_smoke_report.md"

    candidate_scores = _export_candidate_scores(args.train_path, args.evidence_path, candidate_path, args.max_users)
    joined = _join_evidence(candidate_scores, args.evidence_path, joined_path)
    _write_join_diagnostics(joined, join_diag_path)
    if float(joined["evidence_risk"].notna().mean()) < 0.8:
        report_path.write_text(
            "# Day14 Simple External Backbone Plug-in Smoke Report\n\n"
            "Blocked: join coverage is below 0.8, so performance is not interpreted.\n",
            encoding="utf-8",
        )
        print("Blocked: join coverage below 0.8")
        return
    grid = _rerank_grid(joined, grid_path)
    diagnostics = _write_plugin_diagnostics(joined, grid, plugin_diag_path)
    _write_report(candidate_scores, joined, grid, diagnostics, report_path)
    print("Day14 simple backbone plug-in smoke complete.")


if __name__ == "__main__":
    main()
