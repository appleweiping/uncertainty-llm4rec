from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main_day15_bprmf_backbone_plugin_smoke import _auc_binary, _normalize_per_user, _rank_change_stats, _safe_spearman


SEED = 42
NUM_NEGATIVES = 5
SUMMARY_DIR = Path("output-repaired/summary")
COLD_DIR = Path("data/processed/amazon_movies_medium_5neg")
WARM_DIR = Path("data/processed/amazon_movies_medium_5neg_warm")
CONTENT_BACKBONE_DIR = Path("output-repaired/backbone/content_movies_medium5_cold")
EVIDENCE_PATH = Path("output-repaired/movies_deepseek_relevance_evidence_medium_5neg_2000/calibrated/relevance_evidence_posterior_test.jsonl")

ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")
TOKEN_RE = re.compile(r"[a-z0-9]+")
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def extract_item_id(value: Any) -> str:
    text = str(value).strip()
    match = ASIN_RE.search(text)
    return match.group(0) if match else text


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(str(text).lower()) if len(tok) >= 2]


def history_text(row: dict[str, Any]) -> str:
    hist = row.get("history", [])
    if not isinstance(hist, list):
        return ""
    return " ".join(str(x) for x in hist)


def candidate_text(row: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            str(row.get("candidate_title", "") or ""),
            str(row.get("candidate_text", "") or ""),
        ]
        if part.strip()
    )


def vectorise_counts(tokens: list[str]) -> Counter:
    return Counter(tokens)


def tfidf_cosine(query_counts: Counter, doc_counts: Counter, idf: dict[str, float]) -> float:
    if not query_counts or not doc_counts:
        return 0.0
    common = set(query_counts) & set(doc_counts)
    numerator = sum(query_counts[t] * idf.get(t, 0.0) * doc_counts[t] * idf.get(t, 0.0) for t in common)
    q_norm = math.sqrt(sum((c * idf.get(t, 0.0)) ** 2 for t, c in query_counts.items()))
    d_norm = math.sqrt(sum((c * idf.get(t, 0.0)) ** 2 for t, c in doc_counts.items()))
    if q_norm <= 0 or d_norm <= 0:
        return 0.0
    return numerator / (q_norm * d_norm)


def bm25_score(query_tokens: list[str], doc_counts: Counter, idf: dict[str, float], doc_len: int, avgdl: float) -> float:
    if not query_tokens or not doc_counts:
        return 0.0
    k1 = 1.5
    b = 0.75
    score = 0.0
    for tok in set(query_tokens):
        tf = doc_counts.get(tok, 0)
        if tf <= 0:
            continue
        denom = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1e-12))
        score += idf.get(tok, 0.0) * (tf * (k1 + 1)) / denom
    return float(score)


def build_content_scores() -> pd.DataFrame:
    test_rows = read_jsonl(COLD_DIR / "test.jsonl")
    docs = []
    for row in test_rows:
        docs.append(tokenize(history_text(row)))
        docs.append(tokenize(candidate_text(row)))
    doc_freq = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))
    n_docs = len(docs)
    idf = {tok: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0) for tok, df in doc_freq.items()}
    doc_lens = [len(tokenize(candidate_text(row))) for row in test_rows]
    avgdl = float(np.mean(doc_lens)) if doc_lens else 1.0

    rows = []
    for row in test_rows:
        q_tokens = tokenize(history_text(row))
        d_tokens = tokenize(candidate_text(row))
        q_counts = vectorise_counts(q_tokens)
        d_counts = vectorise_counts(d_tokens)
        scores = {
            "tfidf_cosine": tfidf_cosine(q_counts, d_counts, idf),
            "bm25": bm25_score(q_tokens, d_counts, idf, len(d_tokens), avgdl),
        }
        for name, score in scores.items():
            rows.append(
                {
                    "user_id": str(row["user_id"]),
                    "candidate_item_id": extract_item_id(row["candidate_item_id"]),
                    "backbone_score": score,
                    "label": int(row.get("label", 0)),
                    "split": "test",
                    "backbone_name": name,
                    "fallback_score": 0,
                    "fallback_reason": "",
                }
            )
    scores = pd.DataFrame(rows)
    scores["backbone_rank"] = (
        scores.sort_values(["backbone_name", "user_id", "backbone_score", "candidate_item_id"], ascending=[True, True, False, True])
        .groupby(["backbone_name", "user_id"])
        .cumcount()
        + 1
    )
    write_csv(scores, CONTENT_BACKBONE_DIR / "candidate_scores.csv")
    return scores


def join_evidence(scores: pd.DataFrame) -> pd.DataFrame:
    evidence = pd.DataFrame(read_jsonl(EVIDENCE_PATH))
    evidence["user_id"] = evidence["user_id"].astype(str)
    evidence["candidate_item_id"] = evidence["candidate_item_id"].map(extract_item_id)
    cols = ["user_id", "candidate_item_id", *EVIDENCE_COLUMNS]
    joined = scores.merge(evidence[cols], on=["user_id", "candidate_item_id"], how="left")
    write_csv(joined, SUMMARY_DIR / "day34_movies_cold_content_carrier_joined_candidates.csv")
    diag_rows = []
    for backbone_name, group in joined.groupby("backbone_name"):
        fallback = group["fallback_score"].fillna(0).astype(int) == 1
        pos = group["label"].astype(int) == 1
        diag_rows.append(
            {
                "backbone_name": backbone_name,
                "num_backbone_rows": len(group),
                "num_joined_rows": int(group["calibrated_relevance_probability"].notna().sum()),
                "join_coverage": float(group["calibrated_relevance_probability"].notna().mean()) if len(group) else 0.0,
                "num_users": int(group["user_id"].nunique()),
                "num_candidates": int(group["candidate_item_id"].nunique()),
                "num_positive_labels": int(pos.sum()),
                "fallback_rate": float(fallback.mean()) if len(group) else 0.0,
                "fallback_rate_positive": float((fallback & pos).sum() / max(pos.sum(), 1)),
                "fallback_rate_negative": float((fallback & ~pos).sum() / max((~pos).sum(), 1)),
                "missing_evidence_rows": int(group["calibrated_relevance_probability"].isna().sum()),
                "missing_backbone_score_rows": int(group["backbone_score"].isna().sum()),
            }
        )
    diag = pd.DataFrame(diag_rows)
    write_csv(diag, SUMMARY_DIR / "day34_movies_cold_content_carrier_join_diagnostics.csv")
    return joined


def ranking_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    ranks = []
    pool_sizes = []
    for _, group in df.groupby("user_id"):
        ordered = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).reset_index(drop=True)
        labels = ordered["label"].astype(int).to_numpy()
        pool_sizes.append(len(ordered))
        pos_idx = np.where(labels == 1)[0]
        rank = int(pos_idx[0] + 1) if len(pos_idx) else len(ordered) + 1
        ranks.append(rank)
    ranks_arr = np.asarray(ranks, dtype=float)
    pool_arr = np.asarray(pool_sizes, dtype=float)

    def hr_at(k: int) -> float:
        return float(np.mean(ranks_arr <= k)) if len(ranks_arr) else 0.0

    def ndcg_at(k: int) -> float:
        vals = [1.0 / math.log2(r + 1) if r <= k else 0.0 for r in ranks_arr]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "NDCG@10": ndcg_at(10),
        "MRR": float(np.mean(1.0 / ranks_arr)) if len(ranks_arr) else 0.0,
        "HR@1": hr_at(1),
        "HR@3": hr_at(3),
        "NDCG@3": ndcg_at(3),
        "NDCG@5": ndcg_at(5),
        "HR@10": hr_at(10),
        "positive_rank_mean": float(np.mean(ranks_arr)) if len(ranks_arr) else 0.0,
        "positive_rank_median": float(np.median(ranks_arr)) if len(ranks_arr) else 0.0,
        "candidate_pool_size_mean": float(np.mean(pool_arr)) if len(pool_arr) else 0.0,
        "candidate_pool_size_min": float(np.min(pool_arr)) if len(pool_arr) else 0.0,
        "candidate_pool_size_max": float(np.max(pool_arr)) if len(pool_arr) else 0.0,
        "hr10_trivial_flag": bool(float(np.max(pool_arr)) <= 10 or float(np.mean(pool_arr)) <= 10) if len(pool_arr) else True,
    }


def rerank_grid(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone_name, base_df in joined.groupby("backbone_name"):
        base_df = base_df.copy()
        base_metrics = ranking_metrics(base_df, "backbone_score")
        base_sorted = base_df.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
        base_rank = base_sorted.groupby("user_id").cumcount() + 1
        base_rank_map = dict(zip(zip(base_sorted["user_id"], base_sorted["candidate_item_id"]), base_rank))
        rows.append(
            {
                "method": "A_Backbone_only",
                "backbone_name": backbone_name,
                "lambda": 0.0,
                "alpha": 1.0,
                "beta": 0.0,
                "normalization": "none",
                **base_metrics,
                "rank_change_rate": 0.0,
                "top10_order_change_rate": 0.0,
                "mean_kendall_tau": 1.0,
                "base_risk_spearman": _safe_spearman(base_df["backbone_score"], base_df["evidence_risk"]),
                "relative_NDCG_vs_backbone": 0.0,
                "relative_MRR_vs_backbone": 0.0,
            }
        )
        for normalization in ["minmax", "zscore"]:
            work = base_df.copy()
            work["norm_backbone"] = _normalize_per_user(work["backbone_score"], work["user_id"], normalization)
            work["norm_cal"] = _normalize_per_user(work["calibrated_relevance_probability"], work["user_id"], normalization)
            work["norm_risk"] = _normalize_per_user(work["evidence_risk"], work["user_id"], normalization)
            for alpha in [0.5, 0.75, 0.9]:
                beta = 1 - alpha
                b = work.copy()
                b["final_score"] = alpha * b["norm_backbone"] + (1 - alpha) * b["norm_cal"]
                metrics = ranking_metrics(b, "final_score")
                change = _rank_change_stats(b, "backbone_score", "final_score")
                rows.append(
                    {
                        "method": "B_Backbone_plus_calibrated_relevance",
                        "backbone_name": backbone_name,
                        "lambda": 0.0,
                        "alpha": alpha,
                        "beta": beta,
                        "normalization": normalization,
                        **metrics,
                        "rank_change_rate": change["rank_change_rate"],
                        "top10_order_change_rate": change["top10_order_change_rate"],
                        "mean_kendall_tau": change["mean_kendall_tau"],
                        "base_risk_spearman": change["base_risk_spearman"],
                        "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12),
                        "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
                    }
                )
            for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
                c = work.copy()
                c["final_score"] = c["norm_backbone"] - lam * c["norm_risk"]
                metrics = ranking_metrics(c, "final_score")
                change = _rank_change_stats(c, "backbone_score", "final_score")
                rows.append(
                    {
                        "method": "C_Backbone_plus_evidence_risk",
                        "backbone_name": backbone_name,
                        "lambda": lam,
                        "alpha": 1.0,
                        "beta": 0.0,
                        "normalization": normalization,
                        **metrics,
                        "rank_change_rate": change["rank_change_rate"],
                        "top10_order_change_rate": change["top10_order_change_rate"],
                        "mean_kendall_tau": change["mean_kendall_tau"],
                        "base_risk_spearman": change["base_risk_spearman"],
                        "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12),
                        "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
                    }
                )
            for alpha in [0.5, 0.75, 0.9]:
                beta = 1 - alpha
                for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
                    d = work.copy()
                    d["final_score"] = alpha * d["norm_backbone"] + beta * d["norm_cal"] - lam * d["norm_risk"]
                    metrics = ranking_metrics(d, "final_score")
                    change = _rank_change_stats(d, "backbone_score", "final_score")
                    rows.append(
                        {
                            "method": "D_Backbone_plus_calibrated_relevance_plus_evidence_risk",
                            "backbone_name": backbone_name,
                            "lambda": lam,
                            "alpha": alpha,
                            "beta": beta,
                            "normalization": normalization,
                            **metrics,
                            "rank_change_rate": change["rank_change_rate"],
                            "top10_order_change_rate": change["top10_order_change_rate"],
                            "mean_kendall_tau": change["mean_kendall_tau"],
                            "base_risk_spearman": change["base_risk_spearman"],
                            "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12),
                            "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
                        }
                    )
    grid = pd.DataFrame(rows)
    write_csv(grid, SUMMARY_DIR / "day34_movies_cold_content_carrier_plugin_rerank_grid.csv")
    return grid


def plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone_name, group in joined.groupby("backbone_name"):
        base = grid[(grid["backbone_name"] == backbone_name) & (grid["method"] == "A_Backbone_only")].iloc[0]
        best = grid[grid["backbone_name"] == backbone_name].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
        top_base = group.sort_values(["user_id", "backbone_score"], ascending=[True, False]).groupby("user_id").head(3)
        misrank = ((top_base["label"].astype(int) == 0)).astype(int)
        rows.append(
            {
                "backbone_name": backbone_name,
                "backbone_score_AUROC": _auc_binary(group["label"], group["backbone_score"]),
                "calibrated_relevance_AUROC": _auc_binary(group["label"], group["calibrated_relevance_probability"]),
                "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(misrank, top_base["evidence_risk"]),
                "backbone_risk_spearman": _safe_spearman(group["backbone_score"], group["evidence_risk"]),
                "fallback_rate": float(group["fallback_score"].fillna(0).astype(int).mean()),
                "best_method": best["method"],
                "best_relative_NDCG_vs_backbone": best["relative_NDCG_vs_backbone"],
                "best_relative_MRR_vs_backbone": best["relative_MRR_vs_backbone"],
                "best_lambda": best["lambda"],
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_normalization": best["normalization"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR": best["MRR"],
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR": base["MRR"],
                "candidate_pool_size_mean": base["candidate_pool_size_mean"],
                "hr10_trivial_flag": base["hr10_trivial_flag"],
            }
        )
    diag = pd.DataFrame(rows)
    write_csv(diag, SUMMARY_DIR / "day34_movies_cold_content_carrier_plugin_diagnostics.csv")
    return diag


def build_item_lookup(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for rows in rows_by_split.values():
        for row in rows:
            item_id = extract_item_id(row["candidate_item_id"])
            if item_id not in lookup or int(row.get("label", 0)) == 1:
                lookup[item_id] = {
                    "candidate_item_id": item_id,
                    "candidate_title": row.get("candidate_title", ""),
                    "candidate_text": row.get("candidate_text", f"Item ID: {item_id}"),
                    "target_popularity_group": row.get("target_popularity_group", "mid"),
                }
    return lookup


def user_seen_positive_items(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    seen = defaultdict(set)
    for rows in rows_by_split.values():
        for row in rows:
            if int(row.get("label", 0)) == 1:
                seen[str(row["user_id"])].add(extract_item_id(row["candidate_item_id"]))
                for hist_item in row.get("history", []) if isinstance(row.get("history", []), list) else []:
                    seen[str(row["user_id"])].add(extract_item_id(hist_item))
    return seen


def build_warm_split() -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame]:
    rows_by_split = {split: read_jsonl(COLD_DIR / f"{split}.jsonl") for split in ["train", "valid", "test"]}
    item_lookup = build_item_lookup(rows_by_split)
    train_candidate_vocab = {extract_item_id(row["candidate_item_id"]) for row in rows_by_split["train"]}
    warm_pool = sorted(train_candidate_vocab)
    seen_by_user = user_seen_positive_items(rows_by_split)
    rng = random.Random(SEED)
    # Keep the original train split as the warm train vocabulary source. The warm intervention
    # targets valid/test negative sampling, not the chronological training history.
    out = {"train": [dict(row) for row in rows_by_split["train"]], "valid": [], "test": []}
    insufficient = []
    for split in ["valid", "test"]:
        rows = rows_by_split[split]
        positives = [row for row in rows if int(row.get("label", 0)) == 1]
        for pos_row in positives:
            user_id = str(pos_row["user_id"])
            pos_item = extract_item_id(pos_row["candidate_item_id"])
            base = dict(pos_row)
            base["candidate_item_id"] = pos_item
            out[split].append(base)
            blocked = set(seen_by_user[user_id]) | {pos_item}
            candidates = [item for item in warm_pool if item not in blocked]
            if len(candidates) < NUM_NEGATIVES:
                insufficient.append({"split": split, "user_id": user_id, "available_negatives": len(candidates)})
            sampled = rng.sample(candidates, k=min(NUM_NEGATIVES, len(candidates)))
            for item_id in sampled:
                info = item_lookup.get(item_id, {})
                neg = {
                    "user_id": user_id,
                    "history": pos_row.get("history", []),
                    "candidate_item_id": item_id,
                    "candidate_title": info.get("candidate_title", ""),
                    "candidate_text": info.get("candidate_text", f"Item ID: {item_id}"),
                    "label": 0,
                    "target_popularity_group": info.get("target_popularity_group", "mid"),
                    "timestamp": pos_row.get("timestamp"),
                }
                out[split].append(neg)

    WARM_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        write_jsonl(rows, WARM_DIR / f"{split}.jsonl")
    users = sorted({str(row["user_id"]) for row in out["valid"]})
    (WARM_DIR / "sampled_users.json").write_text(json.dumps([{"user_id": u} for u in users], indent=2), encoding="utf-8")
    stats = {
        "source_dir": str(COLD_DIR),
        "output_dir": str(WARM_DIR),
        "seed": SEED,
        "num_negatives": NUM_NEGATIVES,
        "negative_sampling_pool": "train_candidate_vocab_minus_user_seen_items",
        "train_rows": len(out["train"]),
        "valid_rows": len(out["valid"]),
        "test_rows": len(out["test"]),
        "insufficient_negative_users": len(insufficient),
    }
    (WARM_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    schema_df = validate_schema(out)
    (WARM_DIR / "schema_validation.json").write_text(schema_df.to_json(orient="records", indent=2), encoding="utf-8")
    write_csv(schema_df, SUMMARY_DIR / "movies_medium_5neg_warm_schema_validation.csv")
    cold_df = cold_rate_for_rows(out)
    write_csv(cold_df, SUMMARY_DIR / "movies_medium_5neg_warm_cold_rate_diagnostics.csv")
    write_warm_report(schema_df, cold_df, stats)
    return out, schema_df, cold_df


def validate_schema(rows_by_split: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    required = ["user_id", "history", "candidate_item_id", "candidate_text", "label"]
    for split, rows_list in rows_by_split.items():
        df = pd.DataFrame(rows_list)
        labels = df["label"].astype(int)
        counts = df.groupby("user_id").size()
        missing = [c for c in required if c not in df.columns]
        rows.append(
            {
                "domain": "movies",
                "processed_path": str(WARM_DIR),
                "split": split,
                "num_rows": len(df),
                "num_users": df["user_id"].nunique(),
                "num_items": df["candidate_item_id"].nunique(),
                "positive_rows": int((labels == 1).sum()),
                "negative_rows": int((labels == 0).sum()),
                "avg_candidates_per_user": float(counts.mean()),
                "min_candidates_per_user": float(counts.min()),
                "max_candidates_per_user": float(counts.max()),
                "has_user_id": "user_id" in df.columns,
                "has_history": "history" in df.columns,
                "has_candidate_item_id": "candidate_item_id" in df.columns,
                "has_candidate_text": "candidate_text" in df.columns,
                "has_candidate_title": "candidate_title" in df.columns,
                "has_label": "label" in df.columns,
                "schema_compatible_with_beauty": not missing and "candidate_title" in df.columns,
                "missing_fields": ",".join(missing),
                "notes": "warm negative sampling from train_candidate_vocab - user_seen_items",
            }
        )
    return pd.DataFrame(rows)


def build_warm_vocabs(train_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    candidate_vocab = {extract_item_id(row["candidate_item_id"]) for row in train_rows}
    history_vocab = set()
    for row in train_rows:
        for item in row.get("history", []) if isinstance(row.get("history", []), list) else []:
            history_vocab.add(extract_item_id(item))
    return {
        "train_candidate_vocab": candidate_vocab,
        "train_history_vocab": history_vocab,
        "train_backbone_vocab": candidate_vocab | history_vocab,
    }


def cold_rate_for_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    vocabs = build_warm_vocabs(rows_by_split["train"])
    rows = []
    for split in ["valid", "test"]:
        df = pd.DataFrame(rows_by_split[split])
        df["candidate_item_id"] = df["candidate_item_id"].map(extract_item_id)
        df["label"] = df["label"].astype(int)
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        for name, vocab in vocabs.items():
            pos_cold = ~pos["candidate_item_id"].isin(vocab)
            neg_cold = ~neg["candidate_item_id"].isin(vocab)
            all_cold = ~df["candidate_item_id"].isin(vocab)
            rows.append(
                {
                    "split": split,
                    "vocab_definition": name,
                    "train_vocab_size": len(vocab),
                    "num_rows": len(df),
                    "num_positive_rows": len(pos),
                    "num_negative_rows": len(neg),
                    "positive_cold_rows": int(pos_cold.sum()),
                    "positive_cold_rate": float(pos_cold.mean()) if len(pos) else 0.0,
                    "negative_cold_rows": int(neg_cold.sum()),
                    "negative_cold_rate": float(neg_cold.mean()) if len(neg) else 0.0,
                    "all_candidate_cold_rows": int(all_cold.sum()),
                    "all_candidate_cold_rate": float(all_cold.mean()) if len(df) else 0.0,
                    "unique_candidate_items": df["candidate_item_id"].nunique(),
                    "unique_cold_candidate_items": len(set(df["candidate_item_id"]) - vocab),
                    "unique_positive_items": pos["candidate_item_id"].nunique(),
                    "unique_positive_cold_items": len(set(pos["candidate_item_id"]) - vocab),
                    "unique_negative_items": neg["candidate_item_id"].nunique(),
                    "unique_negative_cold_items": len(set(neg["candidate_item_id"]) - vocab),
                }
            )
    return pd.DataFrame(rows)


def write_warm_report(schema_df: pd.DataFrame, cold_df: pd.DataFrame, stats: dict[str, Any]) -> None:
    main = cold_df[cold_df["vocab_definition"] == "train_backbone_vocab"]
    valid = main[main["split"] == "valid"].iloc[0]
    test = main[main["split"] == "test"].iloc[0]
    report = f"""# Movies medium_5neg_warm Build Report

## 1. Purpose

This warm split is created separately from the existing cold-style `data/processed/amazon_movies_medium_5neg/`. The original directory is not overwritten. Warm negatives are sampled from `train_candidate_vocab - user_seen_items`, so ID-based backbones can evaluate mostly warm negative candidates.

## 2. Output

Output directory: `{WARM_DIR}`.

Rows: train `{stats['train_rows']}`, valid `{stats['valid_rows']}`, test `{stats['test_rows']}`.

## 3. Schema

Schema validation is saved to `output-repaired/summary/movies_medium_5neg_warm_schema_validation.csv`. The split remains Beauty-compatible and uses 1 positive + 5 negatives per user for valid/test.

## 4. Cold Rate

Using `train_backbone_vocab`:

- Valid positive cold rate: `{valid['positive_cold_rate']:.4f}`; valid negative cold rate: `{valid['negative_cold_rate']:.4f}`; valid all-candidate cold rate: `{valid['all_candidate_cold_rate']:.4f}`.
- Test positive cold rate: `{test['positive_cold_rate']:.4f}`; test negative cold rate: `{test['negative_cold_rate']:.4f}`; test all-candidate cold rate: `{test['all_candidate_cold_rate']:.4f}`.

Warm negative sampling sharply reduces negative cold rate. Any remaining positive cold rate is a chronological/domain cold-start limitation and should be reported separately before running ID-based backbones.

## 5. Day35 Recommendation

If positive cold rate is acceptable for the intended claim, Day35 can run Movies warm relevance evidence and then evaluate ID-based SASRec/GRU4Rec/Bert4Rec. If positive cold rate remains too high, Movies ID-backbone evaluation should be marked as cold-start limited rather than forced.
"""
    (SUMMARY_DIR / "movies_medium_5neg_warm_build_report.md").write_text(report, encoding="utf-8")


def write_content_report(grid: pd.DataFrame, diag: pd.DataFrame) -> None:
    best_rows = grid.sort_values(["NDCG@10", "MRR"], ascending=False).groupby("backbone_name").head(1)
    lines = []
    for _, row in best_rows.iterrows():
        lines.append(
            f"- `{row['backbone_name']}` best `{row['method']}`: NDCG@10 `{row['NDCG@10']:.4f}`, MRR `{row['MRR']:.4f}`, HR@1 `{row['HR@1']:.4f}`, HR@3 `{row['HR@3']:.4f}`."
        )
    report = f"""# Day34 Movies Cold-Style Content Carrier Report

## 1. Setting

This run uses the existing `data/processed/amazon_movies_medium_5neg/` as a cold-style sampling setting. It does not call DeepSeek API and reuses the existing Movies CEP evidence. HR@10 is retained in tables but marked trivial because each user has 6 candidates.

## 2. Backbone

The content carriers are TF-IDF cosine similarity and BM25-style similarity between user history text and candidate title/text. They do not depend on train item-id embeddings, so they can score cold candidates. They are diagnostic cold-aware content carriers, not SOTA recommender backbones.

## 3. Join

Join diagnostics are saved to `day34_movies_cold_content_carrier_join_diagnostics.csv`. Fallback is expected to be zero because the carrier scores from text rather than item-id vocab.

## 4. Plug-in Result

{chr(10).join(lines)}

## 5. Interpretation

This test asks whether CEP can plug into a cold-aware content carrier when ID-based sequential backbones are invalid. It should not be compared as a strong external recommender claim. The proper ID-backbone evaluation should use the separately constructed warm split.
"""
    (SUMMARY_DIR / "day34_movies_cold_content_carrier_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    scores = build_content_scores()
    joined = join_evidence(scores)
    grid = rerank_grid(joined)
    diag = plugin_diagnostics(joined, grid)
    write_content_report(grid, diag)
    build_warm_split()
    print("Wrote Day34 content carrier results and warm split diagnostics.")


if __name__ == "__main__":
    main()
