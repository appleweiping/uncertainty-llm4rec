"""Day24 third-backbone smoke using LLM-ESR Bert4Rec.

This script intentionally keeps Day24 as a 100-user smoke test. It trains a
real external Bert4Rec-style backbone from the LLM-ESR repo on Beauty train
interactions only, exports candidate-level scores, joins Day9 evidence fields,
and runs the same Scheme 4 plug-in grid used for SASRec/GRU4Rec.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from main_day15_bprmf_backbone_plugin_smoke import (
    _auc_binary,
    _normalize_per_user,
    _rank_change_stats,
    _rank_metrics,
    _safe_spearman,
)
from main_day17_sasrec_backbone_plugin_smoke import (
    EVIDENCE_COLUMNS,
    SUMMARY_DIR,
    _candidate_pool,
    _load_title_map,
    _load_train_examples,
)
from main_day21_second_backbone_plugin_smoke import _join_diagnostics, _join_evidence, _map_history, _pad_left


LLMESR_PATH = Path("external/LLM-ESR")
OPENP5_PATH = Path("external/OpenP5")
LLMEMB_PATH = Path("external/LLMEmb")
BACKBONE_DIR = Path("output-repaired/backbone/third_backbone_beauty_100")
ARTIFACT_DIR = Path("artifacts/backbones/llmesr_bert4rec_beauty_100")


def _load_bert4rec_class():
    sys.path.insert(0, str(LLMESR_PATH.resolve()))
    from models.Bert4Rec import Bert4Rec  # type: ignore

    return Bert4Rec


def _git_hash(path: Path) -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _build_vocab(train_examples: list[tuple[list[str], str]], pool: pd.DataFrame, title_to_id: dict[str, str]) -> tuple[dict[str, int], set[str]]:
    items: set[str] = set()
    trained_items: set[str] = set()
    for hist, target in train_examples:
        items.add(target)
        items.update(hist)
        trained_items.add(target)
        trained_items.update(hist)
    for _, row in pool.iterrows():
        items.add(str(row["candidate_item_id"]))
        for title in row.get("history", []):
            item_id = title_to_id.get(str(title).strip())
            if item_id:
                items.add(item_id)
    return {item_id: idx + 1 for idx, item_id in enumerate(sorted(items))}, trained_items


def _position_tensor(batch_size: int, max_seq_len: int) -> torch.Tensor:
    return torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)


def _train_external_bert4rec(
    train_examples: list[tuple[list[str], str]],
    item_to_idx: dict[str, int],
    trained_items: set[str],
    hidden_size: int,
    trm_num: int,
    num_heads: int,
    dropout_rate: float,
    max_seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    Bert4Rec = _load_bert4rec_class()
    args = SimpleNamespace(
        hidden_size=hidden_size,
        trm_num=trm_num,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        max_len=max_seq_len,
    )
    model = Bert4Rec(user_num=0, item_num=len(item_to_idx), device=torch.device("cpu"), args=args)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = np.asarray([_pad_left(hist, item_to_idx, max_seq_len) for hist, _ in train_examples], dtype=np.int64)
    targets = np.asarray([item_to_idx[target] for _, target in train_examples], dtype=np.int64)
    trained_item_indices = np.asarray([item_to_idx[item] for item in sorted(trained_items) if item in item_to_idx], dtype=np.int64)
    rng = np.random.default_rng(seed)
    logs = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(seqs))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_seq = torch.from_numpy(seqs[idx])
            batch_pos = torch.zeros((len(idx), max_seq_len), dtype=torch.long)
            batch_neg = torch.zeros((len(idx), max_seq_len), dtype=torch.long)
            batch_pos[:, -1] = torch.from_numpy(targets[idx])
            neg_np = rng.choice(trained_item_indices, size=len(idx), replace=True)
            same = neg_np == targets[idx]
            attempts = 0
            while same.any() and attempts < 10:
                neg_np[same] = rng.choice(trained_item_indices, size=int(same.sum()), replace=True)
                same = neg_np == targets[idx]
                attempts += 1
            batch_neg[:, -1] = torch.from_numpy(neg_np.astype(np.int64))
            positions = _position_tensor(len(idx), max_seq_len)
            opt.zero_grad()
            loss = model(seq=batch_seq, pos=batch_pos, neg=batch_neg, positions=positions)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan})
    return model, logs


def _export_scores(
    model,
    pool: pd.DataFrame,
    title_to_id: dict[str, str],
    item_to_idx: dict[str, int],
    trained_items: set[str],
    item_pop: dict[str, int],
    max_seq_len: int,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for _, row in pool.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["candidate_item_id"])
            hist = _map_history(row, title_to_id, trained_items, max_seq_len)
            fallback = 0
            reason = ""
            if not hist:
                fallback = 1
                reason = "unmapped_or_untrained_history"
                score = math.log1p(item_pop.get(item_id, 0))
            elif item_id not in trained_items:
                fallback = 1
                reason = "cold_candidate_item"
                score = math.log1p(item_pop.get(item_id, 0))
            else:
                seq = torch.tensor([_pad_left(hist, item_to_idx, max_seq_len)], dtype=torch.long)
                item_indices = torch.tensor([[item_to_idx[item_id]]], dtype=torch.long)
                positions = _position_tensor(1, max_seq_len)
                score = float(model.predict(seq=seq, item_indices=item_indices, positions=positions).squeeze().item())
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "backbone_score": score,
                    "label": int(row.get("label", 0)),
                    "split": "test",
                    "backbone_name": "llmesr_bert4rec",
                    "fallback_score": fallback,
                    "fallback_reason": reason,
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


def _rerank_grid(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    rows = []
    base_metrics = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    for normalization in ["minmax", "zscore"]:
        df[f"norm_backbone_{normalization}"] = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        df[f"norm_calibrated_{normalization}"] = _normalize_per_user(
            df["calibrated_relevance_probability"], df["user_id"], normalization
        )
        df[f"norm_risk_{normalization}"] = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
        settings = [("A_Backbone_only", 0.0, 1.0, 0.0, df["backbone_score"])]
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
            scored = df[["user_id", "candidate_item_id", "label", "backbone_score", "evidence_risk"]].copy()
            scored["final_score"] = score
            metrics = _rank_metrics(scored, "final_score")
            change = _rank_change_stats(scored, "backbone_score", "final_score")
            rows.append(
                {
                    "method": method,
                    "backbone_name": "llmesr_bert4rec",
                    "lambda": lam,
                    "alpha": alpha,
                    "beta": beta,
                    "normalization": normalization,
                    **metrics,
                    **change,
                    "base_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
                    "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                    / max(base_metrics["NDCG@10"], 1e-12),
                    "relative_MRR_vs_backbone": (metrics["MRR@10"] - base_metrics["MRR@10"])
                    / max(base_metrics["MRR@10"], 1e-12),
                }
            )
    grid = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_path, index=False)
    return grid


def _plugin_diagnostics(joined: pd.DataFrame, grid: pd.DataFrame, join_diag: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    pred_rank = (
        df.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
        .groupby("user_id")
        .cumcount()
        + 1
    )
    misrank = ((pred_rank <= 10) & (df["label"].astype(int) == 0)).astype(int)
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    base = grid[grid["method"] == "A_Backbone_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    diag = pd.DataFrame(
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
                "best_relative_MRR_vs_backbone": (best["MRR@10"] - base["MRR@10"]) / max(base["MRR@10"], 1e-12),
                "best_lambda": best["lambda"],
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_normalization": best["normalization"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR@10": best["MRR@10"],
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR@10": base["MRR@10"],
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


def _write_selection_report(output_path: Path) -> None:
    rows = [
        {
            "candidate_backbone": "LLM-ESR Bert4Rec",
            "repo_url_or_local_path": f"external/LLM-ESR ({_git_hash(LLMESR_PATH)})",
            "model_type": "masked-transformer sequential recommender",
            "requires_checkpoint": "no",
            "requires_special_embedding": "no",
            "supports_beauty_or_amazon": "yes",
            "can_export_candidate_score": "yes, Bert4Rec.predict returns candidate logits",
            "score_export_difficulty": "medium",
            "expected_fallback_risk": "same candidate cold-item risk as GRU4Rec; expected <20%",
            "recommended_choice": "yes",
            "reason": "Non-GRU external sequential backbone with real candidate logits and no missing checkpoint dependency.",
        },
        {
            "candidate_backbone": "OpenP5",
            "repo_url_or_local_path": f"external/OpenP5 ({_git_hash(OPENP5_PATH)})",
            "model_type": "generative recommendation",
            "requires_checkpoint": "yes",
            "requires_special_embedding": "no direct embedding, but generated data/checkpoint required",
            "supports_beauty_or_amazon": "yes",
            "can_export_candidate_score": "possible only with generative likelihood adapter",
            "score_export_difficulty": "high",
            "expected_fallback_risk": "blocked by missing scoring adapter/checkpoint",
            "recommended_choice": "no",
            "reason": "Not suitable for Day24 smoke because it would require new generative score export work.",
        },
        {
            "candidate_backbone": "LLMEmb",
            "repo_url_or_local_path": f"external/LLMEmb ({_git_hash(LLMEMB_PATH)})",
            "model_type": "LLM-enhanced sequential recommendation",
            "requires_checkpoint": "yes",
            "requires_special_embedding": "yes",
            "supports_beauty_or_amazon": "yes",
            "can_export_candidate_score": "yes in principle",
            "score_export_difficulty": "blocked",
            "expected_fallback_risk": "blocked by missing handled data/embedding/checkpoint",
            "recommended_choice": "no",
            "reason": "Day13 already found required artifacts missing.",
        },
        {
            "candidate_backbone": "ItemKNN/co-occurrence",
            "repo_url_or_local_path": "local implementation possible",
            "model_type": "history-based sanity baseline",
            "requires_checkpoint": "no",
            "requires_special_embedding": "no",
            "supports_beauty_or_amazon": "yes",
            "can_export_candidate_score": "yes",
            "score_export_difficulty": "low",
            "expected_fallback_risk": "low",
            "recommended_choice": "fallback only",
            "reason": "Useful as sanity, but too weak/coarse to serve as strong third backbone.",
        },
    ]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = "# Day24 Third Backbone Selection Report\n\n"
    md += _markdown_table(df)
    md += "\n\nRecommended Day24 choice: **LLM-ESR Bert4Rec**.\n"
    output_path.write_text(md, encoding="utf-8")


def _best_by_method(grid: pd.DataFrame) -> pd.DataFrame:
    return (
        grid.sort_values(["method", "NDCG@10", "MRR@10"], ascending=[True, False, False])
        .groupby("method")
        .head(1)
        .loc[
            :,
            [
                "method",
                "NDCG@10",
                "MRR@10",
                "HR@10",
                "lambda",
                "alpha",
                "beta",
                "normalization",
                "rank_change_rate",
                "relative_NDCG_vs_backbone",
                "relative_MRR_vs_backbone",
            ],
        ]
    )


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(join_diag: pd.DataFrame, plugin_diag: pd.DataFrame, grid: pd.DataFrame, output_path: Path) -> None:
    jd = join_diag.iloc[0].to_dict()
    pdg = plugin_diag.iloc[0].to_dict()
    best_table = _best_by_method(grid)
    report = f"""# Day24 Third Backbone Plug-in Smoke Report

## 1. Why A Third Backbone After Day20/Day23

Day20 and Day23 established full multi-seed support on SASRec-style and LLM-ESR GRU4Rec. Day24 adds a third 100-user smoke to check that the plug-in path is not only a SASRec/GRU pattern.

## 2. Third Backbone Choice

Selected backbone: **LLM-ESR Bert4Rec**.

It is an external masked-transformer sequential recommender. It is not the current minimal SASRec-style backbone and not the LLM-ESR GRU4Rec model. It exposes real candidate logits via `Bert4Rec.predict()` and can train from the Beauty train split without missing checkpoints, LLM embeddings, or generated data.

## 3. Score Export And Join Diagnostics

Candidate scores: `output-repaired/backbone/third_backbone_beauty_100/candidate_scores.csv`.

Join coverage: `{float(jd['join_coverage']):.4f}`.

Fallback rate: `{float(jd['fallback_rate']):.4f}`.

Positive fallback rate: `{float(jd['fallback_rate_positive']):.4f}`.

Negative fallback rate: `{float(jd['fallback_rate_negative']):.4f}`.

## 4. Backbone-only Vs B/C/D

Best row per method:

{_markdown_table(best_table)}

Diagnostics:

`{pdg}`

## 5. Component Pattern

This is a 100-user smoke, not a full result. The main question is whether calibrated relevance remains the dominant useful Scheme 4 signal and whether evidence risk provides secondary regularization when combined with it.

## 6. Day25 Decision

If join coverage is at least 0.95, fallback is below 20%, and B/D improve over the backbone-only row, Day25 can expand Bert4Rec to 500/full. If Bert4Rec is unhealthy, the SASRec and GRU4Rec full multi-seed conclusions remain the main evidence, and Day25 should switch to a graph/MF-style public backbone.
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
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--trm_num", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
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
    _write_selection_report(SUMMARY_DIR / "day24_third_backbone_selection_report.md")

    title_to_id = _load_title_map(args.items_path)
    pool = _candidate_pool(args.evidence_path, args.test_path, args.max_users)
    train_examples, trained_items, item_pop = _load_train_examples(args.train_path, title_to_id, args.max_seq_len)
    item_to_idx, trained_items = _build_vocab(train_examples, pool, title_to_id)
    model, logs = _train_external_bert4rec(
        train_examples,
        item_to_idx,
        trained_items,
        args.hidden_size,
        args.trm_num,
        args.num_heads,
        args.dropout_rate,
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
            "note": "Day24 LLM-ESR Bert4Rec smoke checkpoint; do not commit.",
        },
        ARTIFACT_DIR / "llmesr_bert4rec.pt",
    )
    (ARTIFACT_DIR / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    scores = _export_scores(
        model,
        pool,
        title_to_id,
        item_to_idx,
        trained_items,
        item_pop,
        args.max_seq_len,
        BACKBONE_DIR / "candidate_scores.csv",
    )
    joined = _join_evidence(scores, args.evidence_path, SUMMARY_DIR / "day24_third_backbone_beauty_100_joined_candidates.csv")
    join_diag = _join_diagnostics(joined, SUMMARY_DIR / "day24_third_backbone_beauty_100_join_diagnostics.csv")
    grid = _rerank_grid(joined, SUMMARY_DIR / "day24_third_backbone_beauty_100_plugin_rerank_grid.csv")
    plugin_diag = _plugin_diagnostics(joined, grid, join_diag, SUMMARY_DIR / "day24_third_backbone_beauty_100_plugin_diagnostics.csv")
    _write_report(join_diag, plugin_diag, grid, SUMMARY_DIR / "day24_third_backbone_plugin_smoke_report.md")
    print("Day24 third backbone Bert4Rec smoke complete.")


if __name__ == "__main__":
    main()
