"""Day15 BPR-MF personalized backbone + Scheme-4 plug-in smoke test."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


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
BACKBONE_DIR = Path("output-repaired/backbone/bprmf_beauty_100")
ARTIFACT_DIR = Path("artifacts/backbones/bprmf_beauty_100")


class BPRMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        return (user_vec * item_vec).sum(dim=-1)

    def forward(self, user_idx: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> torch.Tensor:
        pos_score = self.score(user_idx, pos_idx)
        neg_score = self.score(user_idx, neg_idx)
        return -F.logsigmoid(pos_score - neg_score).mean()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_training_data(train_path: Path) -> tuple[list[tuple[str, str]], dict[str, set[str]], list[str], list[str]]:
    positives: list[tuple[str, str]] = []
    user_pos: dict[str, set[str]] = defaultdict(set)
    all_users = set()
    all_items = set()
    for row in _read_jsonl(train_path):
        user_id = str(row["user_id"])
        item_id = str(row["candidate_item_id"])
        all_users.add(user_id)
        all_items.add(item_id)
        if int(row.get("label", 0)) == 1:
            positives.append((user_id, item_id))
            user_pos[user_id].add(item_id)
    return positives, user_pos, sorted(all_users), sorted(all_items)


def _sample_negatives(
    user_ids: np.ndarray,
    all_item_indices: np.ndarray,
    user_pos_idx: dict[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    negs = np.empty_like(user_ids)
    for i, user_idx in enumerate(user_ids):
        positives = user_pos_idx.get(int(user_idx), set())
        for _ in range(100):
            candidate = int(rng.choice(all_item_indices))
            if candidate not in positives:
                negs[i] = candidate
                break
        else:
            negs[i] = int(rng.choice(all_item_indices))
    return negs


def _train_bprmf(
    positives: list[tuple[str, str]],
    user_pos: dict[str, set[str]],
    users: list[str],
    items: list[str],
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[BPRMF, dict[str, int], dict[str, int], list[dict]]:
    _set_seed(seed)
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    train_user = np.asarray([user_to_idx[u] for u, _ in positives], dtype=np.int64)
    train_pos = np.asarray([item_to_idx[it] for _, it in positives], dtype=np.int64)
    user_pos_idx = {user_to_idx[u]: {item_to_idx[it] for it in its if it in item_to_idx} for u, its in user_pos.items()}
    all_item_indices = np.arange(len(items), dtype=np.int64)
    model = BPRMF(len(users), len(items), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    logs = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_user))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_user = train_user[idx]
            batch_pos = train_pos[idx]
            batch_neg = _sample_negatives(batch_user, all_item_indices, user_pos_idx, rng)
            user_t = torch.from_numpy(batch_user)
            pos_t = torch.from_numpy(batch_pos)
            neg_t = torch.from_numpy(batch_neg)
            optimizer.zero_grad()
            loss = model(user_t, pos_t, neg_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan})
    return model, user_to_idx, item_to_idx, logs


def _select_candidate_pool(evidence_df: pd.DataFrame, max_users: int) -> pd.DataFrame:
    user_order = evidence_df["user_id"].drop_duplicates().head(max_users).tolist()
    return evidence_df[evidence_df["user_id"].isin(user_order)].copy()


def _export_candidate_scores(
    model: BPRMF,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    evidence_path: Path,
    output_path: Path,
    max_users: int,
) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    pool = _select_candidate_pool(evidence, max_users)
    rows = []
    model.eval()
    with torch.no_grad():
        for _, row in pool.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["candidate_item_id"])
            fallback = 0
            fallback_reason = ""
            if user_id not in user_to_idx and item_id not in item_to_idx:
                score = 0.0
                fallback = 1
                fallback_reason = "unknown_user_and_item"
            elif user_id not in user_to_idx:
                score = 0.0
                fallback = 1
                fallback_reason = "unknown_user"
            elif item_id not in item_to_idx:
                score = 0.0
                fallback = 1
                fallback_reason = "unknown_item"
            else:
                score = float(
                    model.score(
                        torch.tensor([user_to_idx[user_id]], dtype=torch.long),
                        torch.tensor([item_to_idx[item_id]], dtype=torch.long),
                    ).item()
                )
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "backbone_score": score,
                    "label": int(row["label"]),
                    "split": "test",
                    "backbone_name": "bprmf",
                    "raw_user_id": user_id,
                    "raw_item_id": item_id,
                    "mapped_user_id": user_to_idx.get(user_id, ""),
                    "mapped_item_id": item_to_idx.get(item_id, ""),
                    "mapping_success": int(fallback == 0),
                    "fallback_score": fallback,
                    "fallback_reason": fallback_reason,
                }
            )
    scores = pd.DataFrame(rows)
    scores = scores.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
    scores["backbone_rank"] = scores.groupby("user_id").cumcount() + 1
    scores = scores.sort_index()
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
        ranked = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).head(k)
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
        base_order = group.sort_values([base_col, "candidate_item_id"], ascending=[False, True])["candidate_item_id"].tolist()
        final_order = group.sort_values([final_col, "candidate_item_id"], ascending=[False, True])["candidate_item_id"].tolist()
        base_rank_map = {item: rank + 1 for rank, item in enumerate(base_order)}
        final_rank_map = {item: rank + 1 for rank, item in enumerate(final_order)}
        g = group.copy()
        g["_base_rank"] = g["candidate_item_id"].map(base_rank_map)
        g["_final_rank"] = g["candidate_item_id"].map(final_rank_map)
        rank_changed.append(float((g["_base_rank"] != g["_final_rank"]).mean()))
        base_top = tuple(base_order[:10])
        final_top = tuple(final_order[:10])
        top10_changed.append(1.0 if base_top != final_top else 0.0)
        if len(g) > 1:
            taus.append(float(g["_base_rank"].corr(g["_final_rank"], method="kendall")))
            spearmans.append(_safe_spearman(g[base_col], g["evidence_risk"]))
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


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    mask = left.notna() & right.notna()
    if left[mask].nunique() < 2 or right[mask].nunique() < 2:
        return math.nan
    return float(left[mask].corr(right[mask], method="spearman"))


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
        settings = [("A_BPRMF_only", 0.0, norm_backbone)]
        for lam in lambdas:
            settings.append(("B_BPRMF_plus_calibrated_relevance", lam, alpha * norm_backbone + (1 - alpha) * norm_rel))
            settings.append(("C_BPRMF_plus_evidence_risk", lam, norm_backbone - lam * norm_risk))
            settings.append(("D_BPRMF_plus_calibrated_relevance_plus_evidence_risk", lam, alpha * norm_backbone + beta * norm_rel - lam * norm_risk))
        for method, lam, final_score in settings:
            local = df.copy()
            local["final_score"] = final_score
            rows.append(
                {
                    "method": method,
                    "backbone_name": "bprmf",
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


def _write_plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, day14_diag_path: Path, out_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "evidence_risk", "calibrated_relevance_probability", "label"]).copy()
    base_metrics = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    df["_base_rank"] = df.groupby("user_id")["backbone_score"].rank(ascending=False, method="first")
    df["_base_rank"] = (
        df.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
        .groupby("user_id")
        .cumcount()
        .add(1)
        .reindex(df.index)
    )
    wrong_high = df[(df["_base_rank"] <= 10) & (df["label"].astype(int) == 0)]
    correct_high = df[(df["_base_rank"] <= 10) & (df["label"].astype(int) == 1)]
    error_label = ((df["_base_rank"] <= 10) & (df["label"].astype(int) == 0)).astype(int)
    day14_ndcg = math.nan
    day14_mrr = math.nan
    if day14_diag_path.exists():
        day14 = pd.read_csv(day14_diag_path)
        if len(day14):
            day14_ndcg = float(day14.iloc[0].get("best_NDCG@10", math.nan))
            day14_mrr = float(day14.iloc[0].get("best_MRR@10", math.nan))
    diag = pd.DataFrame(
        [
            {
                "backbone_score_AUROC": _auc_binary(df["label"].astype(int), df["backbone_score"]),
                "calibrated_relevance_AUROC": _auc_binary(df["label"].astype(int), df["calibrated_relevance_probability"]),
                "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(error_label, df["evidence_risk"]),
                "backbone_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
                "risk_mean_for_wrong_high_rank": float(wrong_high["evidence_risk"].mean()) if len(wrong_high) else math.nan,
                "risk_mean_for_correct_high_rank": float(correct_high["evidence_risk"].mean()) if len(correct_high) else math.nan,
                "best_method": best["method"],
                "best_normalization": best["normalization"],
                "best_lambda": best["lambda"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR@10": best["MRR@10"],
                "best_relative_NDCG_vs_backbone": (best["NDCG@10"] - base_metrics["NDCG@10"]) / base_metrics["NDCG@10"] if base_metrics["NDCG@10"] else math.nan,
                "best_relative_MRR_vs_backbone": (best["MRR@10"] - base_metrics["MRR@10"]) / base_metrics["MRR@10"] if base_metrics["MRR@10"] else math.nan,
                "bprmf_backbone_NDCG@10": base_metrics["NDCG@10"],
                "bprmf_backbone_MRR@10": base_metrics["MRR@10"],
                "day14_popularity_best_NDCG@10": day14_ndcg,
                "day14_popularity_best_MRR@10": day14_mrr,
                "bprmf_vs_day14_popularity_NDCG_delta": base_metrics["NDCG@10"] - day14_ndcg if not math.isnan(day14_ndcg) else math.nan,
                "bprmf_vs_day14_popularity_MRR_delta": base_metrics["MRR@10"] - day14_mrr if not math.isnan(day14_mrr) else math.nan,
            }
        ]
    )
    diag.to_csv(out_path, index=False)
    return diag


def _write_report(
    train_logs: list[dict],
    candidate_scores: pd.DataFrame,
    joined: pd.DataFrame,
    grid: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    join_coverage = float(joined["evidence_risk"].notna().mean()) if len(joined) else 0.0
    base = grid[grid["method"] == "A_BPRMF_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    diag = diagnostics.iloc[0].to_dict()
    final_loss = train_logs[-1]["loss"] if train_logs else math.nan
    report = f"""# Day15 BPR-MF Personalized Backbone Plug-in Smoke Report

## 1. Day14 Recap

Day14 used a train-only item popularity backbone to unblock the external plug-in engineering path. That result should not be interpreted as Scheme 4 failing on external backbones, because popularity is not user-specific and its misranks are only weakly related to evidence risk.

## 2. Why BPR-MF

Day15 replaces popularity with BPR-MF. This is still lightweight, but it is a real personalized recommender with user and item embeddings:

`score(u, i) = dot(user_embedding[u], item_embedding[i])`

It is easier to train and export than SASRec, while being much closer to a recommendation backbone than global popularity.

## 3. Training Setup

Training data:

`{args.train_path}`

Only train split positive interactions are used as positives. Negative items are sampled from the train item universe excluding each user's train positives. No valid/test labels are used for training.

Hyperparameters:

- embedding_dim: `{args.embedding_dim}`
- epochs: `{args.epochs}`
- batch_size: `{args.batch_size}`
- learning_rate: `{args.learning_rate}`
- seed: `{args.seed}`
- final training loss: `{final_loss:.6f}`

Checkpoint path:

`artifacts/backbones/bprmf_beauty_100/bprmf.pt`

The checkpoint is an artifact and should not be committed.

## 4. Score Export

Score export:

`output-repaired/backbone/bprmf_beauty_100/candidate_scores.csv`

Rows: `{len(candidate_scores)}`

Users: `{candidate_scores["user_id"].nunique()}`

Fallback score rows: `{int(candidate_scores["fallback_score"].sum())}`

Fields include:

`user_id, candidate_item_id, backbone_score, label, backbone_rank, split, backbone_name, raw_user_id, raw_item_id, mapped_user_id, mapped_item_id, mapping_success, fallback_score, fallback_reason`

## 5. Join Diagnostics

Joined table:

`output-repaired/summary/day15_bprmf_beauty_100_joined_candidates.csv`

Join coverage: `{join_coverage:.4f}`

Missing evidence rows: `{int(joined["evidence_risk"].isna().sum())}`

## 6. Plug-in Result

BPR-MF only:

`{base}`

Best grid row:

`{best}`

Diagnostics:

`{diag}`

## 7. Interpretation

The main checks are whether BPR-MF is stronger than Day14 popularity, whether calibrated relevance is complementary to BPR-MF, and whether evidence risk identifies wrong high-rank candidates. This is still a 100-user smoke test, not a full external SOTA result.

## 8. Day16 Recommendation

If BPR-MF improves over popularity and any Scheme-4 variant improves over BPR-MF-only, Day16 should scale to larger/full Beauty. If BPR-MF is weak or Scheme-4 does not help, Day16 should either tune BPR-MF lightly or move to SASRec as the stronger sequential backbone.
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
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    positives, user_pos, users, items = _load_training_data(args.train_path)
    model, user_to_idx, item_to_idx, train_logs = _train_bprmf(
        positives,
        user_pos,
        users,
        items,
        args.embedding_dim,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "train_logs": train_logs,
            "args": vars(args),
        },
        ARTIFACT_DIR / "bprmf.pt",
    )
    pd.DataFrame(train_logs).to_csv(ARTIFACT_DIR / "train_log.csv", index=False)

    candidate_path = BACKBONE_DIR / "candidate_scores.csv"
    joined_path = SUMMARY_DIR / "day15_bprmf_beauty_100_joined_candidates.csv"
    join_diag_path = SUMMARY_DIR / "day15_bprmf_beauty_100_join_diagnostics.csv"
    grid_path = SUMMARY_DIR / "day15_bprmf_beauty_100_plugin_rerank_grid.csv"
    plugin_diag_path = SUMMARY_DIR / "day15_bprmf_beauty_100_plugin_diagnostics.csv"
    report_path = SUMMARY_DIR / "day15_bprmf_plugin_smoke_report.md"

    candidate_scores = _export_candidate_scores(model, user_to_idx, item_to_idx, args.evidence_path, candidate_path, args.max_users)
    joined = _join_evidence(candidate_scores, args.evidence_path, joined_path)
    _write_join_diagnostics(joined, join_diag_path)
    if float(joined["evidence_risk"].notna().mean()) < 0.8:
        report_path.write_text(
            "# Day15 BPR-MF Personalized Backbone Plug-in Smoke Report\n\n"
            "Blocked: join coverage is below 0.8, so performance is not interpreted.\n",
            encoding="utf-8",
        )
        print("Blocked: join coverage below 0.8")
        return
    grid = _rerank_grid(joined, grid_path)
    diagnostics = _write_plugin_diagnostics(
        joined,
        grid,
        SUMMARY_DIR / "day14_simple_backbone_beauty_100_plugin_diagnostics.csv",
        plugin_diag_path,
    )
    _write_report(train_logs, candidate_scores, joined, grid, diagnostics, args, report_path)
    print("Day15 BPR-MF plug-in smoke complete.")


if __name__ == "__main__":
    main()
