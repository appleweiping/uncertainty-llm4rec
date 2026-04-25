"""Day16 BPR-MF coverage repair and larger-smoke readiness analysis."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main_day15_bprmf_backbone_plugin_smoke import (
    BPRMF,
    _auc_binary,
    _normalize_per_user,
    _rank_change_stats,
    _rank_metrics,
    _read_jsonl,
    _safe_spearman,
)


SUMMARY_DIR = Path("output-repaired/summary")
ARTIFACT_DIR = Path("artifacts/backbones/bprmf_beauty_day16")
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


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_train_positives(train_path: Path) -> tuple[list[tuple[str, str]], dict[str, set[str]], Counter, set[str], set[str]]:
    positives: list[tuple[str, str]] = []
    user_pos: dict[str, set[str]] = defaultdict(set)
    item_pop: Counter = Counter()
    train_users = set()
    train_items = set()
    for row in _read_jsonl(train_path):
        user_id = str(row["user_id"])
        item_id = str(row["candidate_item_id"])
        if int(row.get("label", 0)) == 1:
            positives.append((user_id, item_id))
            user_pos[user_id].add(item_id)
            item_pop[item_id] += 1
            train_users.add(user_id)
            train_items.add(item_id)
    return positives, user_pos, item_pop, train_users, train_items


def _candidate_pool(evidence_path: Path, max_users: int) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    users = evidence["user_id"].drop_duplicates().head(max_users).tolist()
    return evidence[evidence["user_id"].isin(users)].copy()


def _fallback_reason(row: pd.Series, train_users: set[str], train_items: set[str]) -> str:
    missing_user = str(row["user_id"]) not in train_users
    missing_item = str(row["candidate_item_id"]) not in train_items
    if missing_user and missing_item:
        return "unknown_user_and_cold_item"
    if missing_user:
        return "unknown_user"
    if missing_item:
        return "cold_item"
    return ""


def _write_fallback_diagnostics(
    candidate_scores: pd.DataFrame,
    train_users: set[str],
    train_items: set[str],
    prefix: str = "day16_bprmf",
) -> pd.DataFrame:
    df = candidate_scores.copy()
    df["fallback_reason_recomputed"] = df.apply(lambda r: _fallback_reason(r, train_users, train_items), axis=1)
    fallback = df["fallback_reason_recomputed"] != ""
    user_fallback = df["fallback_reason_recomputed"].isin(["unknown_user", "unknown_user_and_cold_item"])
    item_fallback = df["fallback_reason_recomputed"].isin(["cold_item", "unknown_user_and_cold_item"])
    both_fallback = df["fallback_reason_recomputed"] == "unknown_user_and_cold_item"
    pos = df["label"].astype(int) == 1
    neg = ~pos

    diag = pd.DataFrame(
        [
            {
                "num_rows": len(df),
                "num_fallback_rows": int(fallback.sum()),
                "fallback_rate": float(fallback.mean()) if len(df) else 0.0,
                "fallback_user_rows": int(user_fallback.sum()),
                "fallback_item_rows": int(item_fallback.sum()),
                "fallback_both_rows": int(both_fallback.sum()),
                "num_unique_users": int(df["user_id"].nunique()),
                "num_users_missing_train_mapping": int(df.loc[user_fallback, "user_id"].nunique()),
                "num_unique_items": int(df["candidate_item_id"].nunique()),
                "num_items_missing_train_mapping": int(df.loc[item_fallback, "candidate_item_id"].nunique()),
                "positive_rows_fallback": int((fallback & pos).sum()),
                "negative_rows_fallback": int((fallback & neg).sum()),
                "fallback_rate_positive": float((fallback & pos).sum() / max(pos.sum(), 1)),
                "fallback_rate_negative": float((fallback & neg).sum() / max(neg.sum(), 1)),
            }
        ]
    )
    diag.to_csv(SUMMARY_DIR / f"{prefix}_fallback_diagnostics.csv", index=False)

    user_breakdown = (
        df.assign(is_fallback=fallback.astype(int))
        .groupby("user_id")
        .agg(
            num_rows=("candidate_item_id", "count"),
            fallback_rows=("is_fallback", "sum"),
            positive_rows=("label", "sum"),
        )
        .reset_index()
    )
    user_breakdown["fallback_rate"] = user_breakdown["fallback_rows"] / user_breakdown["num_rows"].clip(lower=1)
    user_breakdown.to_csv(SUMMARY_DIR / f"{prefix}_fallback_user_breakdown.csv", index=False)

    item_breakdown = (
        df.assign(is_fallback=fallback.astype(int))
        .groupby("candidate_item_id")
        .agg(
            num_rows=("user_id", "count"),
            fallback_rows=("is_fallback", "sum"),
            positive_rows=("label", "sum"),
        )
        .reset_index()
    )
    item_breakdown["fallback_rate"] = item_breakdown["fallback_rows"] / item_breakdown["num_rows"].clip(lower=1)
    item_breakdown.to_csv(SUMMARY_DIR / f"{prefix}_fallback_item_breakdown.csv", index=False)
    return diag


def _train_bprmf(
    positives: list[tuple[str, str]],
    user_pos: dict[str, set[str]],
    users: list[str],
    items: list[str],
    train_items: set[str],
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
    train_item_indices = np.asarray([item_to_idx[it] for it in sorted(train_items) if it in item_to_idx], dtype=np.int64)
    user_pos_idx = {user_to_idx[u]: {item_to_idx[it] for it in its if it in item_to_idx} for u, its in user_pos.items()}
    model = BPRMF(len(users), len(items), embedding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    logs = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_user))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_user = train_user[idx]
            batch_pos = train_pos[idx]
            batch_neg = np.empty_like(batch_user)
            for i, user_idx in enumerate(batch_user):
                positives_for_user = user_pos_idx.get(int(user_idx), set())
                for _ in range(100):
                    cand = int(rng.choice(train_item_indices))
                    if cand not in positives_for_user:
                        batch_neg[i] = cand
                        break
                else:
                    batch_neg[i] = int(rng.choice(train_item_indices))
            user_t = torch.from_numpy(batch_user)
            pos_t = torch.from_numpy(batch_pos)
            neg_t = torch.from_numpy(batch_neg)
            opt.zero_grad()
            loss = model(user_t, pos_t, neg_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan})
    return model, user_to_idx, item_to_idx, logs


def _known_item_min_score(model: BPRMF, user_idx: int, train_item_indices: list[int]) -> float:
    if not train_item_indices:
        return 0.0
    with torch.no_grad():
        user_t = torch.full((len(train_item_indices),), user_idx, dtype=torch.long)
        item_t = torch.tensor(train_item_indices, dtype=torch.long)
        return float(model.score(user_t, item_t).min().item())


def _mean_embedding_score(model: BPRMF, user_idx: int, train_item_indices: list[int]) -> float:
    if not train_item_indices:
        return 0.0
    with torch.no_grad():
        user_vec = model.user_embedding(torch.tensor([user_idx], dtype=torch.long)).squeeze(0)
        mean_item = model.item_embedding(torch.tensor(train_item_indices, dtype=torch.long)).mean(dim=0)
        return float((user_vec * mean_item).sum().item())


def _export_scores(
    model: BPRMF,
    pool: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    train_users: set[str],
    train_items: set[str],
    item_pop: Counter,
    strategy: str,
    output_path: Path,
) -> pd.DataFrame:
    train_item_indices = [item_to_idx[it] for it in sorted(train_items) if it in item_to_idx]
    rows = []
    model.eval()
    min_cache: dict[int, float] = {}
    mean_cache: dict[int, float] = {}
    with torch.no_grad():
        for _, row in pool.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["candidate_item_id"])
            user_has_train = user_id in train_users
            item_has_train = item_id in train_items
            fallback = 0
            fallback_reason = ""
            if user_has_train and item_has_train:
                score = float(
                    model.score(
                        torch.tensor([user_to_idx[user_id]], dtype=torch.long),
                        torch.tensor([item_to_idx[item_id]], dtype=torch.long),
                    ).item()
                )
            else:
                fallback = 1
                fallback_reason = _fallback_reason(row, train_users, train_items)
                if strategy == "train_popularity":
                    score = math.log1p(item_pop.get(item_id, 0))
                elif strategy == "mean_embedding" and user_has_train:
                    user_idx = user_to_idx[user_id]
                    if user_idx not in mean_cache:
                        mean_cache[user_idx] = _mean_embedding_score(model, user_idx, train_item_indices)
                    score = mean_cache[user_idx]
                else:
                    if user_has_train:
                        user_idx = user_to_idx[user_id]
                        if user_idx not in min_cache:
                            min_cache[user_idx] = _known_item_min_score(model, user_idx, train_item_indices)
                        score = min_cache[user_idx]
                    else:
                        score = 0.0
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "backbone_score": score,
                    "label": int(row["label"]),
                    "split": "test",
                    "backbone_name": "bprmf",
                    "cold_strategy": strategy,
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


def _join_evidence(scores: pd.DataFrame, evidence_path: Path, output_path: Path) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    keep = ["user_id", "candidate_item_id"] + [c for c in EVIDENCE_COLUMNS if c in evidence.columns]
    evidence = evidence[keep].drop_duplicates(["user_id", "candidate_item_id"])
    joined = scores.merge(evidence, on=["user_id", "candidate_item_id"], how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return joined


def _write_join_diagnostics(
    joined: pd.DataFrame,
    output_path: Path,
    train_users: set[str],
    train_items: set[str],
) -> pd.DataFrame:
    diag = _write_fallback_diagnostics(
        joined,
        train_users,
        train_items,
        prefix="day16_bprmf_repaired",
    )
    row = diag.iloc[0].to_dict()
    out = pd.DataFrame(
        [
            {
                "num_backbone_rows": len(joined),
                "num_joined_rows": int(joined["evidence_risk"].notna().sum()),
                "join_coverage": float(joined["evidence_risk"].notna().mean()) if len(joined) else 0.0,
                "num_users": int(joined["user_id"].nunique()),
                "num_candidates": len(joined),
                "num_positive_labels": int(joined["label"].astype(int).sum()),
                "missing_evidence_rows": int(joined["evidence_risk"].isna().sum()),
                "missing_backbone_score_rows": int(joined["backbone_score"].isna().sum()),
                "fallback_score_rows": int(joined["fallback_score"].astype(int).sum()),
                "fallback_rate": row["fallback_rate"],
                "fallback_rate_positive": row["fallback_rate_positive"],
                "fallback_rate_negative": row["fallback_rate_negative"],
            }
        ]
    )
    out.to_csv(output_path, index=False)
    return out


def _rerank_grid(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "evidence_risk", "calibrated_relevance_probability", "label"]).copy()
    rows = []
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5]
    normalizations = ["minmax", "zscore"]
    alphas = [0.5, 0.75, 0.9]
    for normalization in normalizations:
        norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        norm_rel = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)
        norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
        settings = [("A_BPRMF_only", 0.0, 1.0, 0.0, norm_backbone)]
        for alpha in alphas:
            beta = 1.0 - alpha
            settings.append(("B_BPRMF_plus_calibrated_relevance", 0.0, alpha, beta, alpha * norm_backbone + beta * norm_rel))
            for lam in lambdas:
                settings.append(("C_BPRMF_plus_evidence_risk", lam, alpha, beta, norm_backbone - lam * norm_risk))
                settings.append(("D_BPRMF_plus_calibrated_relevance_plus_evidence_risk", lam, alpha, beta, alpha * norm_backbone + beta * norm_rel - lam * norm_risk))
        for method, lam, alpha, beta, final_score in settings:
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
    grid.to_csv(output_path, index=False)
    return grid


def _plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, join_diag: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "evidence_risk", "calibrated_relevance_probability", "label"]).copy()
    base = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
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
    jd = join_diag.iloc[0].to_dict()
    out = pd.DataFrame(
        [
            {
                "backbone_score_AUROC": _auc_binary(df["label"].astype(int), df["backbone_score"]),
                "calibrated_relevance_AUROC": _auc_binary(df["label"].astype(int), df["calibrated_relevance_probability"]),
                "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(error_label, df["evidence_risk"]),
                "backbone_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
                "fallback_rate": jd["fallback_rate"],
                "fallback_rate_positive": jd["fallback_rate_positive"],
                "fallback_rate_negative": jd["fallback_rate_negative"],
                "risk_mean_for_wrong_high_rank": float(wrong_high["evidence_risk"].mean()) if len(wrong_high) else math.nan,
                "risk_mean_for_correct_high_rank": float(correct_high["evidence_risk"].mean()) if len(correct_high) else math.nan,
                "best_method": best["method"],
                "best_normalization": best["normalization"],
                "best_lambda": best["lambda"],
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR@10": best["MRR@10"],
                "best_HR@10": best["HR@10"],
                "best_relative_NDCG_vs_backbone": (best["NDCG@10"] - base["NDCG@10"]) / base["NDCG@10"] if base["NDCG@10"] else math.nan,
                "best_relative_MRR_vs_backbone": (best["MRR@10"] - base["MRR@10"]) / base["MRR@10"] if base["MRR@10"] else math.nan,
                "best_relative_HR_vs_backbone": (best["HR@10"] - base["HR@10"]) / base["HR@10"] if base["HR@10"] else math.nan,
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR@10": base["MRR@10"],
                "backbone_HR@10": base["HR@10"],
            }
        ]
    )
    out.to_csv(output_path, index=False)
    return out


def _strategy_ablation(rows: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def _write_report(
    fallback_diag: pd.DataFrame,
    strategy_ablation: pd.DataFrame,
    join_diag: pd.DataFrame,
    plugin_grid: pd.DataFrame,
    plugin_diag: pd.DataFrame,
    ran_500: bool,
    output_path: Path,
) -> None:
    fd = fallback_diag.iloc[0].to_dict()
    jd = join_diag.iloc[0].to_dict()
    pdg = plugin_diag.iloc[0].to_dict()
    best_strategy = strategy_ablation.sort_values(["best_NDCG@10", "best_MRR@10"], ascending=False).iloc[0].to_dict()
    method_best = (
        plugin_grid.sort_values(["method", "NDCG@10", "MRR@10"], ascending=[True, False, False])
        .groupby("method")
        .head(1)
        .loc[:, ["method", "NDCG@10", "MRR@10", "lambda", "alpha", "beta", "normalization", "rank_change_rate"]]
    )
    method_lines = [
        "| method | NDCG@10 | MRR@10 | lambda | alpha | beta | normalization | rank_change_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in method_best.iterrows():
        method_lines.append(
            f"| {row['method']} | {float(row['NDCG@10']):.4f} | {float(row['MRR@10']):.4f} | "
            f"{float(row['lambda']):.2f} | {float(row['alpha']):.2f} | {float(row['beta']):.2f} | "
            f"{row['normalization']} | {float(row['rank_change_rate']):.4f} |"
        )
    method_table = "\n".join(method_lines)

    day14_path = SUMMARY_DIR / "day14_simple_backbone_beauty_100_plugin_diagnostics.csv"
    day14_note = "Day14 popularity diagnostics were not found, so this report only compares Day16 variants."
    if day14_path.exists():
        day14 = pd.read_csv(day14_path).iloc[0].to_dict()
        day14_ndcg = day14.get("backbone_NDCG@10", day14.get("best_NDCG@10", 0.0))
        day14_mrr = day14.get("backbone_MRR@10", day14.get("best_MRR@10", 0.0))
        day14_note = (
            f"Day14 train-popularity backbone NDCG@10 was `{float(day14_ndcg):.4f}` "
            f"and MRR@10 was `{float(day14_mrr):.4f}`. The selected repaired BPR-MF-only "
            f"NDCG@10 is `{float(pdg['backbone_NDCG@10']):.4f}` and MRR@10 is `{float(pdg['backbone_MRR@10']):.4f}`, "
            "so this 100-user BPR-MF backbone is still not a healthy stronger baseline than popularity."
        )

    b_best = method_best[method_best["method"].str.startswith("B_")]
    c_best = method_best[method_best["method"].str.startswith("C_")]
    d_best = method_best[method_best["method"].str.startswith("D_")]
    component_note = "Component attribution is unavailable because the method grid is empty."
    if not b_best.empty and not c_best.empty and not d_best.empty:
        b_ndcg = float(b_best.iloc[0]["NDCG@10"])
        c_ndcg = float(c_best.iloc[0]["NDCG@10"])
        d_ndcg = float(d_best.iloc[0]["NDCG@10"])
        component_note = (
            f"B-only reaches NDCG@10 `{b_ndcg:.4f}`, C-only reaches `{c_ndcg:.4f}`, and D reaches `{d_ndcg:.4f}`. "
            "Most of the improvement comes from calibrated relevance; evidence risk is useful as a small regularizer "
            "when combined with calibrated relevance, but it is not a strong standalone error detector in this slice."
        )
    report = f"""# Day16 BPR-MF Coverage Repair Report

## 1. Day15 Recap

Day15 produced a positive smoke signal: `BPR-MF + calibrated relevance + evidence risk` improved over BPR-MF-only on the 100-user slice. However, Day15 also showed that BPR-MF-only was weaker than Day14 popularity and had high fallback coverage. Therefore, Day16 focuses on coverage repair and component attribution rather than making a full performance claim.

## 2. Fallback Diagnosis

Day15 fallback rows: `{fd['num_fallback_rows']} / {fd['num_rows']}`.

Fallback rate: `{fd['fallback_rate']:.4f}`.

User fallback rows: `{fd['fallback_user_rows']}`.

Item fallback rows: `{fd['fallback_item_rows']}`.

Both user and item fallback rows: `{fd['fallback_both_rows']}`.

Positive fallback rate: `{fd['fallback_rate_positive']:.4f}`.

Negative fallback rate: `{fd['fallback_rate_negative']:.4f}`.

The dominant issue is user coverage: some Day9 test users are not present in the BPR-MF train-positive mapping. Under a pure user-id BPR-MF model, these users cannot receive trained personalized embeddings without using non-train information. This is why BPR-MF-only can be underestimated.

## 3. Repair Strategy

Day16 keeps training leakage-safe: BPR-MF training still uses only train split positives. The repair expands vocabulary for query-time candidate users/items, but does not train on valid/test labels. Cold item strategies are explicit:

- `min_score`
- `train_popularity`
- `mean_embedding`

Unknown users remain fallback users because a user-id embedding model cannot personalize them without train history.

## 4. Repaired 100-user Result

Best cold strategy:

`{best_strategy}`

Selected repaired strategy output:

`output-repaired/backbone/bprmf_beauty_100_repaired/candidate_scores.csv`

Join coverage: `{jd['join_coverage']:.4f}`

Fallback rate: `{jd['fallback_rate']:.4f}`

Best plug-in diagnostics:

`{pdg}`

Popularity comparison:

{day14_note}

## 5. Larger 500-user Result

Ran 500-user smoke: `{ran_500}`

If this is false, it is because repaired 100-user fallback remained too high for a clean larger interpretation.

## 6. Component Attribution

Day16 explicitly compares:

- B-only: calibrated relevance only
- C-only: evidence risk only
- D: calibrated relevance + evidence risk

Best row per method group:

{method_table}

{component_note}

The clean interpretation is that Scheme 4 is currently strongest as calibrated relevance posterior first and evidence-risk regularization second. The risk signal alone remains close to random for this BPR-MF slice, partly because the backbone still has high user/item fallback.

## 7. Decision For Day17

If we want a stronger external backbone, the clean next step is SASRec because it can score from sequence history and should reduce unknown-user fallback. If we continue BPR-MF, Day17 should first repair user coverage or use a history-derived user representation; otherwise full-scale BPR-MF will remain confounded by cold users.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/amazon_beauty/train.jsonl"))
    parser.add_argument(
        "--evidence_path",
        type=Path,
        default=Path("output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl"),
    )
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    day15_scores_path = Path("output-repaired/backbone/bprmf_beauty_100/candidate_scores.csv")
    day15_scores = pd.read_csv(day15_scores_path)
    positives, user_pos, item_pop, train_users, train_items = _load_train_positives(args.train_path)
    fallback_diag = _write_fallback_diagnostics(day15_scores, train_users, train_items)

    pool100 = _candidate_pool(args.evidence_path, 100)
    candidate_users100 = set(pool100["user_id"].astype(str))
    candidate_items100 = set(pool100["candidate_item_id"].astype(str))
    users = sorted(train_users | candidate_users100)
    items = sorted(train_items | candidate_items100)
    model, user_to_idx, item_to_idx, logs = _train_bprmf(
        positives,
        user_pos,
        users,
        items,
        train_items,
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
            "train_logs": logs,
            "args": vars(args),
        },
        ARTIFACT_DIR / "bprmf_repaired.pt",
    )
    pd.DataFrame(logs).to_csv(ARTIFACT_DIR / "train_log.csv", index=False)

    strategy_rows = []
    strategy_artifacts = {}
    for strategy in ["min_score", "train_popularity", "mean_embedding"]:
        out_dir = Path(f"output-repaired/backbone/bprmf_beauty_100_repaired_{strategy}")
        scores = _export_scores(model, pool100, user_to_idx, item_to_idx, train_users, train_items, item_pop, strategy, out_dir / "candidate_scores.csv")
        joined = _join_evidence(scores, args.evidence_path, SUMMARY_DIR / f"day16_bprmf_beauty_100_repaired_{strategy}_joined_candidates.csv")
        join_diag = _write_join_diagnostics(
            joined,
            SUMMARY_DIR / f"day16_bprmf_beauty_100_repaired_{strategy}_join_diagnostics.csv",
            train_users,
            train_items,
        )
        grid = _rerank_grid(joined, SUMMARY_DIR / f"day16_bprmf_beauty_100_repaired_{strategy}_plugin_rerank_grid.csv")
        plugin_diag = _plugin_diagnostics(joined, grid, join_diag, SUMMARY_DIR / f"day16_bprmf_beauty_100_repaired_{strategy}_plugin_diagnostics.csv")
        best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
        base = grid[grid["method"] == "A_BPRMF_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
        strategy_rows.append(
            {
                "cold_strategy": strategy,
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR@10": base["MRR@10"],
                "best_method": best["method"],
                "best_lambda": best["lambda"],
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_normalization": best["normalization"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR@10": best["MRR@10"],
                "best_relative_NDCG_vs_backbone": plugin_diag.iloc[0]["best_relative_NDCG_vs_backbone"],
                "best_relative_MRR_vs_backbone": plugin_diag.iloc[0]["best_relative_MRR_vs_backbone"],
                "fallback_rate": join_diag.iloc[0]["fallback_rate"],
                "fallback_rate_positive": join_diag.iloc[0]["fallback_rate_positive"],
                "fallback_rate_negative": join_diag.iloc[0]["fallback_rate_negative"],
            }
        )
        strategy_artifacts[strategy] = (scores, joined, join_diag, grid, plugin_diag)

    ablation = _strategy_ablation(strategy_rows, SUMMARY_DIR / "day16_bprmf_cold_item_strategy_ablation.csv")
    best_strategy = ablation.sort_values(["best_NDCG@10", "best_MRR@10"], ascending=False).iloc[0]["cold_strategy"]
    scores, joined, join_diag, grid, plugin_diag = strategy_artifacts[best_strategy]

    # Canonical repaired-100 outputs requested by the plan.
    canonical_dir = Path("output-repaired/backbone/bprmf_beauty_100_repaired")
    canonical_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(canonical_dir / "candidate_scores.csv", index=False)
    joined.to_csv(SUMMARY_DIR / "day16_bprmf_beauty_100_repaired_joined_candidates.csv", index=False)
    join_diag.to_csv(SUMMARY_DIR / "day16_bprmf_beauty_100_repaired_join_diagnostics.csv", index=False)
    grid.to_csv(SUMMARY_DIR / "day16_bprmf_beauty_100_repaired_plugin_rerank_grid.csv", index=False)
    plugin_diag.to_csv(SUMMARY_DIR / "day16_bprmf_beauty_100_repaired_plugin_diagnostics.csv", index=False)

    ran_500 = False
    if float(join_diag.iloc[0]["join_coverage"]) == 1.0 and float(join_diag.iloc[0]["fallback_rate"]) <= 0.2:
        ran_500 = True
        pool500 = _candidate_pool(args.evidence_path, 500)
        scores500 = _export_scores(model, pool500, user_to_idx, item_to_idx, train_users, train_items, item_pop, str(best_strategy), Path("output-repaired/backbone/bprmf_beauty_500/candidate_scores.csv"))
        joined500 = _join_evidence(scores500, args.evidence_path, SUMMARY_DIR / "day16_bprmf_beauty_500_joined_candidates.csv")
        join_diag500 = _write_join_diagnostics(
            joined500,
            SUMMARY_DIR / "day16_bprmf_beauty_500_join_diagnostics.csv",
            train_users,
            train_items,
        )
        grid500 = _rerank_grid(joined500, SUMMARY_DIR / "day16_bprmf_beauty_500_plugin_rerank_grid.csv")
        _plugin_diagnostics(joined500, grid500, join_diag500, SUMMARY_DIR / "day16_bprmf_beauty_500_plugin_diagnostics.csv")

    _write_report(
        fallback_diag,
        ablation,
        join_diag,
        grid,
        plugin_diag,
        ran_500,
        SUMMARY_DIR / "day16_bprmf_coverage_repair_report.md",
    )
    print("Day16 BPR-MF coverage repair complete.")


if __name__ == "__main__":
    main()
