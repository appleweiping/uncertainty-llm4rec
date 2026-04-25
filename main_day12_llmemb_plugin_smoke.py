"""Day12 LLMEmb external backbone plug-in smoke test.

This script intentionally does not train or run LLMEmb. It consumes a
pre-exported external backbone score table and tests whether Scheme-4 evidence
signals from Day9 can be joined and used as a post-hoc plug-in reranker.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {"user_id", "candidate_item_id", "backbone_score", "label"}
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


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _write_blocked_outputs(reason: str, candidate_scores_path: Path, evidence_path: Path) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.DataFrame(
        [
            {
                "num_backbone_rows": 0,
                "num_joined_rows": 0,
                "join_coverage": 0.0,
                "num_users": 0,
                "num_candidates": 0,
                "num_positive_labels": 0,
                "missing_evidence_rows": "",
                "missing_backbone_score_rows": "",
                "id_mapping_issue_count": "",
                "status": "blocked",
                "blocked_reason": reason,
                "candidate_scores_path": str(candidate_scores_path),
                "evidence_path": str(evidence_path),
            }
        ]
    )
    diagnostics.to_csv(SUMMARY_DIR / "day12_llmemb_beauty_100_join_diagnostics.csv", index=False)

    report = f"""# Day12 LLMEmb Plug-in Smoke Report

## Status

Blocked before performance evaluation.

Reason: {reason}

The local project does not currently contain a real LLMEmb score export at:

`{candidate_scores_path}`

The plug-in reranker is implemented in `main_day12_llmemb_plugin_smoke.py`, but it requires a real external backbone table with this schema:

`user_id, candidate_item_id, backbone_score, label`

Optional columns are:

`backbone_rank, split, raw_user_id, raw_item_id, mapped_user_id, mapped_item_id, mapping_success`

## Evidence Source

Day9 evidence source expected by the script:

`{evidence_path}`

Required Day9 fields:

`{", ".join(EVIDENCE_COLUMNS)}`

## Next Step

Clone or place LLMEmb under an external-code directory, run a 100-user Beauty evaluation/export, and write:

`output-repaired/backbone/llmemb_beauty_100/candidate_scores.csv`

Then rerun:

```powershell
py -3.12 main_day12_llmemb_plugin_smoke.py
```

No synthetic backbone scores were generated, so no external-backbone performance claim is made.
"""
    (SUMMARY_DIR / "day12_llmemb_plugin_smoke_report.md").write_text(report, encoding="utf-8")


def _validate_score_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_SCORE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"candidate score table missing required columns: {sorted(missing)}")
    df["backbone_score"].map(_safe_float)


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
        base_top = set(g.nsmallest(10, "_base_rank")["candidate_item_id"])
        final_top = set(g.nsmallest(10, "_final_rank")["candidate_item_id"])
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
    order = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = order[y == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _build_joined(candidate_scores: Path, evidence_path: Path, output_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(candidate_scores)
    _validate_score_frame(scores)
    evidence = _read_jsonl(evidence_path)
    keep_cols = ["user_id", "candidate_item_id"] + [c for c in EVIDENCE_COLUMNS if c in evidence.columns]
    evidence = evidence[keep_cols].drop_duplicates(["user_id", "candidate_item_id"])
    joined = scores.merge(evidence, on=["user_id", "candidate_item_id"], how="left")
    for col in ["backbone_score", "label"] + EVIDENCE_COLUMNS:
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return joined


def _write_join_diagnostics(joined: pd.DataFrame, out_path: Path) -> None:
    missing_evidence = int(joined["evidence_risk"].isna().sum()) if "evidence_risk" in joined else len(joined)
    mapping_issue = (
        int((~joined["mapping_success"].astype(bool)).sum())
        if "mapping_success" in joined.columns
        else ""
    )
    diagnostics = pd.DataFrame(
        [
            {
                "num_backbone_rows": len(joined),
                "num_joined_rows": int(joined["evidence_risk"].notna().sum()) if "evidence_risk" in joined else 0,
                "join_coverage": float(joined["evidence_risk"].notna().mean()) if "evidence_risk" in joined and len(joined) else 0.0,
                "num_users": int(joined["user_id"].nunique()) if "user_id" in joined else 0,
                "num_candidates": len(joined),
                "num_positive_labels": int(joined["label"].fillna(0).astype(int).sum()) if "label" in joined else 0,
                "missing_evidence_rows": missing_evidence,
                "missing_backbone_score_rows": int(joined["backbone_score"].isna().sum()),
                "id_mapping_issue_count": mapping_issue,
                "status": "ready" if missing_evidence / max(len(joined), 1) <= 0.2 else "low_join_coverage",
                "blocked_reason": "",
            }
        ]
    )
    diagnostics.to_csv(out_path, index=False)


def _rerank_grid(joined: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "label", "evidence_risk"]).copy()
    if "calibrated_relevance_probability" not in df.columns:
        df["calibrated_relevance_probability"] = df.get("relevance_probability", 0.0)

    rows = []
    lambdas = [0.0, 0.05, 0.1, 0.2]
    normalizations = ["minmax", "zscore"]
    alpha = 0.75
    beta = 0.25

    for normalization in normalizations:
        norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
        norm_rel = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)

        setting_scores = []
        setting_scores.append(("Backbone only", 0.0, alpha, beta, norm_backbone, "backbone_score"))
        for lam in lambdas:
            setting_scores.append(("Backbone + evidence_risk", lam, alpha, beta, norm_backbone - lam * norm_risk, "backbone_score - lambda * evidence_risk"))
            setting_scores.append(("Backbone + calibrated relevance", lam, alpha, beta, alpha * norm_backbone + (1 - alpha) * norm_rel, "alpha * backbone_score + (1-alpha) * calibrated_relevance_probability"))
            setting_scores.append(("Backbone + calibrated relevance + evidence_risk", lam, alpha, beta, alpha * norm_backbone + beta * norm_rel - lam * norm_risk, "alpha * backbone_score + beta * calibrated_relevance_probability - lambda * evidence_risk"))

        for method, lam, a, b, scores, formula in setting_scores:
            local = df.copy()
            local["final_score"] = scores
            metrics = _rank_metrics(local, "final_score")
            changes = _rank_change_stats(local, "backbone_score", "final_score")
            rows.append(
                {
                    "method": method,
                    "lambda": lam,
                    "alpha": a,
                    "beta": b,
                    "normalization": normalization,
                    "formula": formula,
                    **metrics,
                    **changes,
                }
            )

    grid = pd.DataFrame(rows).drop_duplicates()
    grid.to_csv(out_path, index=False)
    return grid


def _write_plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, out_path: Path) -> None:
    df = joined.dropna(subset=["backbone_score", "label", "evidence_risk"]).copy()
    base_metrics = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict() if len(grid) else {}
    base_ndcg = base_metrics["NDCG@10"]
    base_mrr = base_metrics["MRR@10"]

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
                "best_method": best.get("method", ""),
                "best_normalization": best.get("normalization", ""),
                "best_lambda": best.get("lambda", ""),
                "best_NDCG@10": best.get("NDCG@10", math.nan),
                "best_MRR@10": best.get("MRR@10", math.nan),
                "best_relative_NDCG_vs_backbone": (best.get("NDCG@10", math.nan) - base_ndcg) / base_ndcg if base_ndcg else math.nan,
                "best_relative_MRR_vs_backbone": (best.get("MRR@10", math.nan) - base_mrr) / base_mrr if base_mrr else math.nan,
            }
        ]
    )
    diag.to_csv(out_path, index=False)


def _write_report(joined: pd.DataFrame, grid: pd.DataFrame, diagnostics_path: Path) -> None:
    join_coverage = joined["evidence_risk"].notna().mean() if len(joined) else 0.0
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict() if len(grid) else {}
    report = f"""# Day12 LLMEmb Plug-in Smoke Report

## 1. Motivation

Day11 selected LLMEmb as the first external backbone candidate because it is the most likely NH baseline to expose a candidate-level score table for Beauty-style sequential recommendation. Day12 tests only the plug-in path: external backbone score export, Day9 evidence join, and Scheme-4 post-hoc reranking.

## 2. Code Entrypoints

The detailed entrypoint audit is saved at:

`output-repaired/summary/day12_llmemb_code_entrypoints.md`

This smoke script does not train or modify LLMEmb. It consumes a score export with the unified schema.

## 3. Score Export Schema

Required score table:

`output-repaired/backbone/llmemb_beauty_100/candidate_scores.csv`

Required fields:

`user_id, candidate_item_id, backbone_score, label`

Optional fields:

`backbone_rank, split, raw_user_id, raw_item_id, mapped_user_id, mapped_item_id, mapping_success`

## 4. Join Diagnostics

Joined candidate rows: `{len(joined)}`

Join coverage with Day9 evidence: `{join_coverage:.4f}`

Diagnostics table:

`{diagnostics_path}`

## 5. Plug-in Rerank Smoke Result

Best smoke-test row:

`{best}`

This is a 100-user technical smoke test, not a full external SOTA comparison.

## 6. Failure Modes

If gains are absent or unstable, the next diagnosis should separate candidate alignment, external score quality, and evidence-risk relevance to the backbone's error mode. If join coverage is below 0.8, performance should not be interpreted until ID mapping is fixed.

## 7. Day13 Recommendation

If join coverage is high and the grid completes, Day13 can expand the same adapter to larger/full Beauty. If join coverage is low, Day13 should fix item/user mapping before any larger run.
"""
    (SUMMARY_DIR / "day12_llmemb_plugin_smoke_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_scores",
        type=Path,
        default=Path("output-repaired/backbone/llmemb_beauty_100/candidate_scores.csv"),
    )
    parser.add_argument(
        "--evidence_path",
        type=Path,
        default=Path("output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl"),
    )
    parser.add_argument(
        "--joined_output",
        type=Path,
        default=Path("output-repaired/summary/day12_llmemb_beauty_100_joined_candidates.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.candidate_scores.exists():
        _write_blocked_outputs(
            "LLMEmb candidate score export is not present locally; no synthetic backbone scores were generated.",
            args.candidate_scores,
            args.evidence_path,
        )
        print(f"Blocked: missing candidate score table: {args.candidate_scores}")
        return
    if not args.evidence_path.exists():
        _write_blocked_outputs(
            "Day9 evidence path is missing; cannot join Scheme-4 evidence fields.",
            args.candidate_scores,
            args.evidence_path,
        )
        print(f"Blocked: missing evidence file: {args.evidence_path}")
        return

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    joined = _build_joined(args.candidate_scores, args.evidence_path, args.joined_output)
    join_diag_path = SUMMARY_DIR / "day12_llmemb_beauty_100_join_diagnostics.csv"
    _write_join_diagnostics(joined, join_diag_path)
    grid = _rerank_grid(joined, SUMMARY_DIR / "day12_llmemb_beauty_100_plugin_rerank_grid.csv")
    _write_plugin_diagnostics(joined, grid, SUMMARY_DIR / "day12_llmemb_beauty_100_plugin_diagnostics.csv")
    _write_report(joined, grid, join_diag_path)
    print("Day12 LLMEmb plug-in smoke test complete.")


if __name__ == "__main__":
    main()
