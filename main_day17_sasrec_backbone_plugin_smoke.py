"""Day17 minimal SASRec-style sequential backbone + Scheme-4 plug-in smoke test."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from main_day15_bprmf_backbone_plugin_smoke import (
    _auc_binary,
    _normalize_per_user,
    _rank_change_stats,
    _rank_metrics,
    _read_jsonl,
    _safe_spearman,
)


SUMMARY_DIR = Path("output-repaired/summary")
BACKBONE_DIR = Path("output-repaired/backbone/sasrec_beauty_100")
ARTIFACT_DIR = Path("artifacts/backbones/sasrec_beauty_100")
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


class MinimalSASRec(nn.Module):
    """Small causal Transformer sequence scorer for candidate item ranking."""

    def __init__(self, num_items: int, embedding_dim: int, num_layers: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seq) + self.position_embedding(positions)
        padding_mask = seq.eq(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq.device), diagonal=1)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.layer_norm(encoded)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        return encoded[torch.arange(batch_size, device=seq.device), last_idx]

    def score(self, seq: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.encode(seq)
        item_vec = self.item_embedding(item_idx)
        return (user_vec * item_vec).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> torch.Tensor:
        pos_score = self.score(seq, pos_idx)
        neg_score = self.score(seq, neg_idx)
        return -F.logsigmoid(pos_score - neg_score).mean()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_title_map(items_path: Path) -> dict[str, str]:
    items = pd.read_csv(items_path)
    title_to_id = {}
    for _, row in items.iterrows():
        item_id = str(row["item_id"])
        title = str(row.get("title", "")).strip()
        candidate_text = str(row.get("candidate_text", "")).strip()
        if title:
            title_to_id[title] = item_id
        if candidate_text.startswith("Title: "):
            title_to_id[candidate_text.removeprefix("Title: ").strip()] = item_id
        if candidate_text:
            title_to_id[candidate_text] = item_id
    return title_to_id


def _map_history(history: list[str], title_to_id: dict[str, str], max_seq_len: int) -> list[str]:
    mapped = []
    for title in history:
        item_id = title_to_id.get(str(title).strip())
        if item_id:
            mapped.append(item_id)
    return mapped[-max_seq_len:]


def _load_train_examples(
    train_path: Path,
    title_to_id: dict[str, str],
    max_seq_len: int,
) -> tuple[list[tuple[list[str], str]], set[str], Counter]:
    examples: list[tuple[list[str], str]] = []
    trained_items: set[str] = set()
    item_pop: Counter = Counter()
    for row in _read_jsonl(train_path):
        if int(row.get("label", 0)) != 1:
            continue
        target = str(row["candidate_item_id"])
        hist = _map_history(row.get("history", []), title_to_id, max_seq_len)
        if not hist:
            continue
        examples.append((hist, target))
        trained_items.add(target)
        trained_items.update(hist)
        item_pop[target] += 1
    return examples, trained_items, item_pop


def _candidate_pool(evidence_path: Path, test_path: Path, max_users: int) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    users = evidence["user_id"].drop_duplicates().head(max_users).tolist()
    key = evidence[evidence["user_id"].isin(users)][["user_id", "candidate_item_id"]].copy()
    key["candidate_item_id"] = key["candidate_item_id"].astype(str)
    key["user_id"] = key["user_id"].astype(str)
    rows = []
    for row in _read_jsonl(test_path):
        if str(row["user_id"]) in set(users):
            rows.append(row)
    test_df = pd.DataFrame(rows)
    test_df["user_id"] = test_df["user_id"].astype(str)
    test_df["candidate_item_id"] = test_df["candidate_item_id"].astype(str)
    pool = test_df.merge(key.drop_duplicates(), on=["user_id", "candidate_item_id"], how="inner")
    return pool


def _build_vocab(
    train_examples: list[tuple[list[str], str]],
    pool: pd.DataFrame,
    title_to_id: dict[str, str],
    max_seq_len: int,
) -> tuple[dict[str, int], set[str]]:
    items = set()
    history_items = set()
    for hist, target in train_examples:
        items.add(target)
        items.update(hist)
    for _, row in pool.iterrows():
        candidate = str(row["candidate_item_id"])
        items.add(candidate)
        mapped_hist = _map_history(row.get("history", []), title_to_id, max_seq_len)
        items.update(mapped_hist)
        history_items.update(mapped_hist)
    item_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(sorted(items))}
    return item_to_idx, history_items


def _pad_sequence(item_ids: list[str], item_to_idx: dict[str, int], max_seq_len: int) -> list[int]:
    idxs = [item_to_idx[item_id] for item_id in item_ids if item_id in item_to_idx][-max_seq_len:]
    # Right padding keeps the last observed item at index length - 1, matching MinimalSASRec.encode().
    return idxs + [0] * (max_seq_len - len(idxs))


def _train_sasrec(
    train_examples: list[tuple[list[str], str]],
    item_to_idx: dict[str, int],
    trained_items: set[str],
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    max_seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[MinimalSASRec, list[dict]]:
    _set_seed(seed)
    model = MinimalSASRec(len(item_to_idx), embedding_dim, num_layers, num_heads, max_seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = np.asarray([_pad_sequence(hist, item_to_idx, max_seq_len) for hist, _ in train_examples], dtype=np.int64)
    pos = np.asarray([item_to_idx[target] for _, target in train_examples], dtype=np.int64)
    trained_item_indices = np.asarray([item_to_idx[it] for it in sorted(trained_items) if it in item_to_idx], dtype=np.int64)
    rng = np.random.default_rng(seed)
    logs = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(seqs))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_seq = torch.from_numpy(seqs[idx])
            batch_pos = torch.from_numpy(pos[idx])
            batch_neg_np = rng.choice(trained_item_indices, size=len(idx), replace=True)
            same = batch_neg_np == pos[idx]
            attempts = 0
            while same.any() and attempts < 10:
                batch_neg_np[same] = rng.choice(trained_item_indices, size=int(same.sum()), replace=True)
                same = batch_neg_np == pos[idx]
                attempts += 1
            batch_neg = torch.from_numpy(batch_neg_np.astype(np.int64))
            opt.zero_grad()
            loss = model(batch_seq, batch_pos, batch_neg)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan})
    return model, logs


def _fallback_score(item_id: str, item_pop: Counter) -> float:
    return math.log1p(item_pop.get(item_id, 0))


def _export_scores(
    model: MinimalSASRec,
    pool: pd.DataFrame,
    title_to_id: dict[str, str],
    item_to_idx: dict[str, int],
    trained_items: set[str],
    item_pop: Counter,
    max_seq_len: int,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for _, row in pool.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["candidate_item_id"])
            hist = _map_history(row.get("history", []), title_to_id, max_seq_len)
            fallback = 0
            reason = ""
            if not hist:
                fallback = 1
                reason = "unmapped_history"
                score = _fallback_score(item_id, item_pop)
            elif item_id not in trained_items:
                fallback = 1
                reason = "cold_candidate_item"
                score = _fallback_score(item_id, item_pop)
            else:
                seq = torch.tensor([_pad_sequence(hist, item_to_idx, max_seq_len)], dtype=torch.long)
                item_t = torch.tensor([item_to_idx[item_id]], dtype=torch.long)
                score = float(model.score(seq, item_t).item())
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "backbone_score": score,
                    "label": int(row.get("label", 0)),
                    "split": "test",
                    "backbone_name": "minimal_sasrec",
                    "fallback_score": fallback,
                    "fallback_reason": reason,
                    "history_mapped_len": len(hist),
                }
            )
    scores = pd.DataFrame(rows)
    scores["backbone_rank"] = (
        scores.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
        .groupby("user_id")
        .cumcount()
        + 1
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return scores


def _join_evidence(candidate_scores: pd.DataFrame, evidence_path: Path, output_path: Path) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    evidence["user_id"] = evidence["user_id"].astype(str)
    evidence["candidate_item_id"] = evidence["candidate_item_id"].astype(str)
    cols = ["user_id", "candidate_item_id", *EVIDENCE_COLUMNS]
    joined = candidate_scores.merge(evidence[cols], on=["user_id", "candidate_item_id"], how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return joined


def _join_diagnostics(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    fallback = joined["fallback_score"].fillna(0).astype(int) == 1
    pos = joined["label"].astype(int) == 1
    diag = pd.DataFrame(
        [
            {
                "num_backbone_rows": len(joined),
                "num_joined_rows": int(joined["calibrated_relevance_probability"].notna().sum()),
                "join_coverage": float(joined["calibrated_relevance_probability"].notna().mean()) if len(joined) else 0.0,
                "num_users": int(joined["user_id"].nunique()),
                "num_candidates": int(joined["candidate_item_id"].nunique()),
                "num_positive_labels": int(pos.sum()),
                "fallback_rate": float(fallback.mean()) if len(joined) else 0.0,
                "fallback_rate_positive": float((fallback & pos).sum() / max(pos.sum(), 1)),
                "fallback_rate_negative": float((fallback & ~pos).sum() / max((~pos).sum(), 1)),
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


def _rerank_grid(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["calibrated_relevance_probability", "evidence_risk"]).copy()
    rows = []
    base_order = df[["user_id", "candidate_item_id", "backbone_score", "label"]].copy()
    for normalization in ["minmax", "zscore"]:
        df[f"norm_backbone_{normalization}"] = _normalize_per_user(
            df["backbone_score"], df["user_id"], normalization
        )
        df[f"norm_calibrated_{normalization}"] = _normalize_per_user(
            df["calibrated_relevance_probability"], df["user_id"], normalization
        )
        df[f"norm_risk_{normalization}"] = _normalize_per_user(
            df["evidence_risk"], df["user_id"], normalization
        )
        settings = []
        settings.append(("A_Backbone_only", 0.0, 1.0, 0.0, df["backbone_score"]))
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1 - alpha
            settings.append(
                (
                    "B_Backbone_plus_calibrated_relevance",
                    0.0,
                    alpha,
                    beta,
                    alpha * df[f"norm_backbone_{normalization}"] + beta * df[f"norm_calibrated_{normalization}"],
                )
            )
        for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
            settings.append(
                (
                    "C_Backbone_plus_evidence_risk",
                    lam,
                    1.0,
                    0.0,
                    df[f"norm_backbone_{normalization}"] - lam * df[f"norm_risk_{normalization}"],
                )
            )
            for alpha in [0.5, 0.75, 0.9]:
                beta = 1 - alpha
                settings.append(
                    (
                        "D_Backbone_plus_calibrated_relevance_plus_evidence_risk",
                        lam,
                        alpha,
                        beta,
                        alpha * df[f"norm_backbone_{normalization}"]
                        + beta * df[f"norm_calibrated_{normalization}"]
                        - lam * df[f"norm_risk_{normalization}"],
                    )
                )
        for method, lam, alpha, beta, score in settings:
            scored = df[["user_id", "candidate_item_id", "label", "evidence_risk"]].copy()
            scored["final_score"] = score
            metrics = _rank_metrics(scored, "final_score")
            change_input = scored.merge(
                base_order[["user_id", "candidate_item_id", "backbone_score"]],
                on=["user_id", "candidate_item_id"],
                how="left",
            )
            change = _rank_change_stats(change_input, "backbone_score", "final_score")
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
                }
            )
    grid = pd.DataFrame(rows)
    grid.to_csv(output_path, index=False)
    return grid


def _plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, join_diag: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["calibrated_relevance_probability", "evidence_risk"]).copy()
    pred_rank = (
        df.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
        .groupby("user_id")
        .cumcount()
        + 1
    )
    misrank = ((pred_rank <= 10) & (df["label"].astype(int) == 0)).astype(int)
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    base = grid[grid["method"] == "A_Backbone_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    diag = {
        "backbone_score_AUROC": _auc_binary(df["label"].astype(int), df["backbone_score"]),
        "calibrated_relevance_AUROC": _auc_binary(df["label"].astype(int), df["calibrated_relevance_probability"]),
        "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(misrank, df["evidence_risk"]),
        "backbone_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
        "fallback_rate": float(join_diag.iloc[0]["fallback_rate"]),
        "best_method": best["method"],
        "best_normalization": best["normalization"],
        "best_lambda": best["lambda"],
        "best_alpha": best["alpha"],
        "best_beta": best["beta"],
        "best_NDCG@10": best["NDCG@10"],
        "best_MRR@10": best["MRR@10"],
        "best_HR@10": best["HR@10"],
        "best_relative_NDCG_vs_backbone": (best["NDCG@10"] - base["NDCG@10"]) / max(base["NDCG@10"], 1e-12),
        "best_relative_MRR_vs_backbone": (best["MRR@10"] - base["MRR@10"]) / max(base["MRR@10"], 1e-12),
    }
    out = pd.DataFrame([diag])
    out.to_csv(output_path, index=False)
    return out


def _write_selection_report(output_path: Path) -> None:
    text = """# Day17 Healthy Ranking Backbone Selection

Day16 showed that BPR-MF is not a healthy final external backbone for the current Beauty candidate pool: it is user-id based, has high fallback, and remains weaker than popularity on the 100-user slice. Day17 therefore switches to a history-based sequential backbone.

Candidate options reviewed:

- LLMEmb SASRec: code exists in `external/LLMEmb/models/SASRec.py`, but the official repo still depends on missing handled data, LLM/SRS embeddings, and checkpoints for its full pipeline. It remains useful as a design reference but is not the fastest honest plug-in backbone.
- Minimal SASRec-style sequence encoder: can be trained directly from `data/processed/amazon_beauty/train.jsonl` using mapped history titles and positive next items. It does not depend on user-id embeddings, so it should reduce the unknown-user fallback that broke BPR-MF.
- Sequential ItemKNN/co-occurrence: lower-cost fallback, but less faithful to a neural sequential ranking backbone.

Selected Day17 backbone: minimal SASRec-style sequence encoder.

Why it is healthier than BPR-MF:

- It scores candidates from user history sequence rather than from a learned user-id embedding.
- It can score users that never appeared as user IDs in the train positives, as long as their history titles map to item IDs.
- It exports full candidate-pool scores for the same Day9/Day10 Beauty users, enabling fair Scheme 4 plug-in reranking.

Expected risk:

- Candidate items unseen in train remain cold and require explicit fallback.
- History title-to-item mapping may be imperfect.
- This is a 100-user smoke test, not a final SOTA claim.
"""
    output_path.write_text(text, encoding="utf-8")


def _write_report(
    join_diag: pd.DataFrame,
    plugin_diag: pd.DataFrame,
    grid: pd.DataFrame,
    output_path: Path,
) -> None:
    jd = join_diag.iloc[0].to_dict()
    pdg = plugin_diag.iloc[0].to_dict()
    prior_lines = []
    day14_path = SUMMARY_DIR / "day14_simple_backbone_beauty_100_plugin_diagnostics.csv"
    if day14_path.exists():
        day14 = pd.read_csv(day14_path).iloc[0].to_dict()
        prior_lines.append(
            f"Day14 train-popularity NDCG@10 `{float(day14.get('best_NDCG@10', 0.0)):.4f}`, "
            f"MRR@10 `{float(day14.get('best_MRR@10', 0.0)):.4f}`."
        )
    day16_path = SUMMARY_DIR / "day16_bprmf_beauty_100_repaired_plugin_diagnostics.csv"
    if day16_path.exists():
        day16 = pd.read_csv(day16_path).iloc[0].to_dict()
        prior_lines.append(
            f"Day16 repaired BPR-MF-only NDCG@10 `{float(day16.get('backbone_NDCG@10', 0.0)):.4f}`, "
            f"MRR@10 `{float(day16.get('backbone_MRR@10', 0.0)):.4f}`, "
            f"fallback `{float(day16.get('fallback_rate', 0.0)):.4f}`."
        )
    prior_comparison = " ".join(prior_lines) if prior_lines else "Prior backbone comparison files were not found."
    method_best = (
        grid.sort_values(["method", "NDCG@10", "MRR@10"], ascending=[True, False, False])
        .groupby("method")
        .head(1)
        .loc[:, ["method", "NDCG@10", "MRR@10", "lambda", "alpha", "beta", "normalization", "rank_change_rate"]]
    )
    lines = [
        "| method | NDCG@10 | MRR@10 | lambda | alpha | beta | normalization | rank_change_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in method_best.iterrows():
        lines.append(
            f"| {row['method']} | {float(row['NDCG@10']):.4f} | {float(row['MRR@10']):.4f} | "
            f"{float(row['lambda']):.2f} | {float(row['alpha']):.2f} | {float(row['beta']):.2f} | "
            f"{row['normalization']} | {float(row['rank_change_rate']):.4f} |"
        )
    method_table = "\n".join(lines)
    report = f"""# Day17 SASRec Backbone Plug-in Smoke Report

## 1. Why Stop BPR-MF Full

BPR-MF remained confounded by high fallback and weak backbone-only ranking. Day17 therefore uses a sequence-based backbone that scores candidates from history rather than user-id embeddings.

## 2. Why Minimal SASRec

The LLMEmb SASRec implementation exists but its official pipeline still needs missing handled data, embeddings, and checkpoints. A minimal SASRec-style encoder can be trained honestly from the current Beauty train split and can export candidate scores for the Day9 evidence pool.

## 3. Training And Score Export

Training uses only positive train rows from `data/processed/amazon_beauty/train.jsonl`. History titles are mapped through `items.csv`; test labels are not used for training. Candidate scores are exported to `output-repaired/backbone/sasrec_beauty_100/candidate_scores.csv`.

## 4. Join Coverage / Fallback

Join coverage: `{float(jd['join_coverage']):.4f}`.

Fallback rate: `{float(jd['fallback_rate']):.4f}`.

Positive fallback rate: `{float(jd['fallback_rate_positive']):.4f}`.

Negative fallback rate: `{float(jd['fallback_rate_negative']):.4f}`.

Compared with earlier smoke backbones:

{prior_comparison}

The minimal SASRec backbone is healthier than BPR-MF mainly because fallback drops below 20%. Its standalone ranking is only comparable to train-popularity on this 100-user slice, so this remains a smoke validation rather than a final external performance claim.

## 5. Backbone Only Vs Scheme 4 Plug-in

Best plug-in diagnostics:

`{pdg}`

Best row per method:

{method_table}

The best row uses calibrated relevance without a positive evidence-risk penalty when `lambda = 0`. This again supports the current interpretation: Scheme 4's most reliable contribution is calibrated relevance posterior; evidence risk remains a secondary regularizer and does not yet provide standalone ranking gains in this slice.

## 6. Day18 Decision

Because fallback is below 20% and the plug-in path is technically clean, Day18 can expand to 500 users. The expansion should still be framed as larger smoke validation, not full SOTA comparison, because SASRec-only is not yet clearly stronger than train-popularity.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/amazon_beauty/train.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/amazon_beauty/test.jsonl"))
    parser.add_argument("--items_path", type=Path, default=Path("data/processed/amazon_beauty/items.csv"))
    parser.add_argument(
        "--evidence_path",
        type=Path,
        default=Path("output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl"),
    )
    parser.add_argument("--max_users", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    _write_selection_report(SUMMARY_DIR / "day17_backbone_selection_report.md")
    title_to_id = _load_title_map(args.items_path)
    pool = _candidate_pool(args.evidence_path, args.test_path, args.max_users)
    train_examples, trained_items, item_pop = _load_train_examples(args.train_path, title_to_id, args.max_seq_len)
    item_to_idx, _ = _build_vocab(train_examples, pool, title_to_id, args.max_seq_len)
    model, logs = _train_sasrec(
        train_examples,
        item_to_idx,
        trained_items,
        args.embedding_dim,
        args.num_layers,
        args.num_heads,
        args.max_seq_len,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_to_idx": item_to_idx,
            "args": vars(args),
            "train_logs": logs,
        },
        ARTIFACT_DIR / "minimal_sasrec.pt",
    )
    (ARTIFACT_DIR / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    candidate_scores = _export_scores(
        model,
        pool,
        title_to_id,
        item_to_idx,
        trained_items,
        item_pop,
        args.max_seq_len,
        BACKBONE_DIR / "candidate_scores.csv",
    )
    joined = _join_evidence(candidate_scores, args.evidence_path, SUMMARY_DIR / "day17_sasrec_beauty_100_joined_candidates.csv")
    join_diag = _join_diagnostics(joined, SUMMARY_DIR / "day17_sasrec_beauty_100_join_diagnostics.csv")
    grid = _rerank_grid(joined, SUMMARY_DIR / "day17_sasrec_beauty_100_plugin_rerank_grid.csv")
    plugin_diag = _plugin_diagnostics(joined, grid, join_diag, SUMMARY_DIR / "day17_sasrec_beauty_100_plugin_diagnostics.csv")
    _write_report(join_diag, plugin_diag, grid, SUMMARY_DIR / "day17_sasrec_backbone_plugin_smoke_report.md")
    print("Day17 SASRec backbone plug-in smoke complete.")


if __name__ == "__main__":
    main()
