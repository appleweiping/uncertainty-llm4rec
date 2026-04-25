"""Day21 second external backbone smoke using LLM-ESR GRU4Rec."""

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
    _read_jsonl,
    _safe_spearman,
)
from main_day17_sasrec_backbone_plugin_smoke import (
    EVIDENCE_COLUMNS,
    SUMMARY_DIR,
    _candidate_pool,
    _load_title_map,
    _load_train_examples,
)


BACKBONE_DIR = Path("output-repaired/backbone/second_backbone_beauty_100")
ARTIFACT_DIR = Path("artifacts/backbones/llmesr_gru4rec_beauty_100")
LLMESR_PATH = Path("external/LLM-ESR")
OPENP5_PATH = Path("external/OpenP5")


def _load_gru4rec_class():
    sys.path.insert(0, str(LLMESR_PATH.resolve()))
    from models.GRU4Rec import GRU4Rec  # type: ignore

    return GRU4Rec


def _pad_left(item_ids: list[str], item_to_idx: dict[str, int], max_seq_len: int) -> list[int]:
    idxs = [item_to_idx[item_id] for item_id in item_ids if item_id in item_to_idx][-max_seq_len:]
    return [0] * (max_seq_len - len(idxs)) + idxs


def _build_vocab(train_examples: list[tuple[list[str], str]], pool: pd.DataFrame, title_to_id: dict[str, str], max_seq_len: int) -> tuple[dict[str, int], set[str]]:
    items = set()
    trained_items = set()
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


def _train_external_gru4rec(
    train_examples: list[tuple[list[str], str]],
    item_to_idx: dict[str, int],
    trained_items: set[str],
    hidden_size: int,
    num_layers: int,
    max_seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    GRU4Rec = _load_gru4rec_class()
    args = SimpleNamespace(hidden_size=hidden_size, num_layers=num_layers)
    model = GRU4Rec(user_num=0, item_num=len(item_to_idx), device=torch.device("cpu"), args=args)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = np.asarray([_pad_left(hist, item_to_idx, max_seq_len) for hist, _ in train_examples], dtype=np.int64)
    pos = np.asarray([item_to_idx[target] for _, target in train_examples], dtype=np.int64)
    trained_item_indices = np.asarray([item_to_idx[item] for item in sorted(trained_items) if item in item_to_idx], dtype=np.int64)
    rng = np.random.default_rng(seed)
    logs = []
    model.train()
    positions = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(seqs))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_seq = torch.from_numpy(seqs[idx])
            batch_pos = torch.from_numpy(pos[idx])
            batch_neg_np = rng.choice(trained_item_indices, size=(len(idx), 1), replace=True)
            same = batch_neg_np[:, 0] == pos[idx]
            attempts = 0
            while same.any() and attempts < 10:
                batch_neg_np[same, 0] = rng.choice(trained_item_indices, size=int(same.sum()), replace=True)
                same = batch_neg_np[:, 0] == pos[idx]
                attempts += 1
            batch_neg = torch.from_numpy(batch_neg_np.astype(np.int64))
            opt.zero_grad()
            loss = model(seq=batch_seq, pos=batch_pos, neg=batch_neg, positions=positions[: len(idx)])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan})
    return model, logs


def _map_history(row: pd.Series, title_to_id: dict[str, str], trained_items: set[str], max_seq_len: int) -> list[str]:
    mapped = []
    for title in row.get("history", []):
        item_id = title_to_id.get(str(title).strip())
        if item_id and item_id in trained_items:
            mapped.append(item_id)
    return mapped[-max_seq_len:]


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
                positions = torch.zeros((1, max_seq_len), dtype=torch.long)
                score = float(model.predict(seq=seq, item_indices=item_indices, positions=positions).squeeze().item())
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "backbone_score": score,
                    "label": int(row.get("label", 0)),
                    "split": "test",
                    "backbone_name": "llmesr_gru4rec",
                    "fallback_score": fallback,
                    "fallback_reason": reason,
                    "mapping_success": int(bool(hist)),
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
                "missing_evidence_rows": int(joined["calibrated_relevance_probability"].isna().sum()),
                "missing_backbone_score_rows": int(joined["backbone_score"].isna().sum()),
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


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
                    "backbone_name": "llmesr_gru4rec",
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
    openp5_hash = ""
    llmesr_hash = ""
    try:
        import subprocess

        openp5_hash = subprocess.check_output(["git", "-C", str(OPENP5_PATH), "rev-parse", "HEAD"], text=True).strip()
        llmesr_hash = subprocess.check_output(["git", "-C", str(LLMESR_PATH), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    report = f"""# Day21 Second Backbone Selection Report

| candidate_backbone | repo_url | local_path | has_code | supports_beauty_or_amazon | requires_pretrained_checkpoint | requires_llm_embedding | requires_special_handled_data | can_export_candidate_score | score_export_difficulty | evaluation_metrics | recommended_choice | risk_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenP5 | https://github.com/agiresearch/OpenP5 | external/OpenP5 | yes | yes, README lists Beauty | yes for official eval | no direct embedding requirement, but generative checkpoint needed | yes, OpenP5 generated data format | possible via generation likelihood/rank adapter | high | HR/NDCG style platform metrics | no for Day21 smoke | Good platform, but candidate score export would require checkpoint and generative scoring adaptation. Commit `{openp5_hash}`. |
| LLM-ESR GRU4Rec | https://github.com/Applied-Machine-Learning-Lab/LLM-ESR | external/LLM-ESR | yes | yes, README lists beauty | no for base GRU4Rec | no for base GRU4Rec | official full LLM-ESR requires handled data, but GRU4Rec class can train from our split | yes, `models/GRU4Rec.py::predict` returns logits | low-medium | trainer reports HR/NDCG; our evaluator adds MRR/Recall | yes | Uses external repo's conventional sequential backbone without LLM enhancement; good for second plug-in smoke. Commit `{llmesr_hash}`. |
| LLMEmb | https://github.com/Applied-Machine-Learning-Lab/LLMEmb | external/LLMEmb | yes | yes | yes | yes | yes | yes in principle | blocked | HR/NDCG | no | Day13 already found missing handled data, embeddings, and checkpoint. |

Selected Day21 backbone: **LLM-ESR GRU4Rec**. It is selected because it is a real external sequential backbone implementation with a direct candidate-logit `predict()` method and does not require LLM embeddings or checkpoints for the base GRU4Rec smoke.
"""
    output_path.write_text(report, encoding="utf-8")


def _write_entrypoints(output_path: Path) -> None:
    report = """# Day21 Second Backbone Code Entrypoints

## Selected Repo

LLM-ESR: `external/LLM-ESR`

## Data Reading

Official data preprocessing is described in `external/LLM-ESR/README.md` and `external/LLM-ESR/data/data_process.py`. The full official pipeline expects handled files such as `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np.pkl`, `pca64_itm_emb_np.pkl`, and `sim_user_100.pkl`.

For Day21 smoke, we do not use the full LLM-ESR enhancement pipeline. We use the external repo's `GRU4Rec` class with our existing Beauty train split.

## Training / Eval Entry

- Official main: `external/LLM-ESR/main.py`
- Official Beauty command: `external/LLM-ESR/experiments/beauty.bash`
- Official trainer: `external/LLM-ESR/trainers/sequence_trainer.py`

## Candidate Score / Logit Entry

`external/LLM-ESR/models/GRU4Rec.py`

The key method is:

`GRU4Rec.predict(seq, item_indices, positions, **kwargs)`

It returns candidate logits with shape `(batch, num_candidates)`, which can be flattened into:

`user_id, candidate_item_id, backbone_score, label`

## Metrics Entry

Official ranking evaluation is in `external/LLM-ESR/trainers/sequence_trainer.py`, which ranks the positive item against negatives and calls `metric_report`. Day21 uses the project evaluator to keep HR@10, NDCG@10, MRR@10, and Recall@10 consistent with Day17-Day20.

## Export Strategy

The adapter trains LLM-ESR GRU4Rec on Beauty train positives only, maps user history titles to item IDs through `items.csv`, scores every candidate in the Day9 evidence-aligned 100-user pool, and writes `output-repaired/backbone/second_backbone_beauty_100/candidate_scores.csv`.
"""
    output_path.write_text(report, encoding="utf-8")


def _write_report(join_diag: pd.DataFrame, plugin_diag: pd.DataFrame, grid: pd.DataFrame, output_path: Path) -> None:
    jd = join_diag.iloc[0].to_dict()
    pdg = plugin_diag.iloc[0].to_dict()
    method_best = (
        grid.sort_values(["method", "NDCG@10", "MRR@10"], ascending=[True, False, False])
        .groupby("method")
        .head(1)
        .loc[:, ["method", "NDCG@10", "MRR@10", "lambda", "alpha", "beta", "normalization", "rank_change_rate", "relative_NDCG_vs_backbone", "relative_MRR_vs_backbone"]]
    )
    lines = [
        "| method | NDCG@10 | MRR@10 | rel NDCG | rel MRR | lambda | alpha | beta | normalization | rank_change_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in method_best.iterrows():
        lines.append(
            f"| {row['method']} | {float(row['NDCG@10']):.4f} | {float(row['MRR@10']):.4f} | "
            f"{float(row['relative_NDCG_vs_backbone']):.4f} | {float(row['relative_MRR_vs_backbone']):.4f} | "
            f"{float(row['lambda']):.2f} | {float(row['alpha']):.2f} | {float(row['beta']):.2f} | "
            f"{row['normalization']} | {float(row['rank_change_rate']):.4f} |"
        )
    table = "\n".join(lines)
    report = f"""# Day21 Second Backbone Plug-in Smoke Report

## 1. Why Second Backbone After Day20

Day20 showed that Scheme 4 works as a plug-in for the minimal SASRec-style backbone. Day21 checks whether the same plug-in path can attach to a second external/NH backbone implementation.

## 2. Selected Backbone And Why

Selected backbone: LLM-ESR GRU4Rec.

OpenP5 was audited first, but its useful candidate scoring path would require downloaded generated data/checkpoints and generative score adaptation. LLM-ESR's base GRU4Rec class exposes direct candidate logits through `predict()` and can be trained from the current Beauty train split without LLM embeddings.

## 3. Code Entrypoints

See `output-repaired/summary/day21_second_backbone_code_entrypoints.md`.

## 4. Score Export Schema

Candidate scores are exported to `output-repaired/backbone/second_backbone_beauty_100/candidate_scores.csv` with:

`user_id, candidate_item_id, backbone_score, label, backbone_rank, split, backbone_name, fallback_score, fallback_reason`.

## 5. Join / Fallback Diagnostics

Join coverage: `{float(jd['join_coverage']):.4f}`.

Fallback rate: `{float(jd['fallback_rate']):.4f}`.

Positive fallback rate: `{float(jd['fallback_rate_positive']):.4f}`.

Negative fallback rate: `{float(jd['fallback_rate_negative']):.4f}`.

## 6. Backbone Only Vs Scheme 4 Plug-in

Best diagnostics:

`{pdg}`

Best row per method:

{table}

## 7. Comparison With SASRec Result

This is a 100-user smoke on a second external implementation, not a full comparison. If join/fallback are healthy and gains are positive, Day22 can expand. If fallback is high or gains collapse, Day22 should diagnose history mapping and candidate cold-item behavior before scaling.

## 8. Day22 Recommendation

If the Day21 smoke is healthy, expand LLM-ESR GRU4Rec to 500 users with fixed settings. If it is unhealthy, switch to a different public sequential backbone with simpler data requirements.
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
    parser.add_argument("--num_layers", type=int, default=1)
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
    _write_selection_report(SUMMARY_DIR / "day21_second_backbone_selection_report.md")
    _write_entrypoints(SUMMARY_DIR / "day21_second_backbone_code_entrypoints.md")

    title_to_id = _load_title_map(args.items_path)
    pool = _candidate_pool(args.evidence_path, args.test_path, args.max_users)
    train_examples, trained_items, item_pop = _load_train_examples(args.train_path, title_to_id, args.max_seq_len)
    item_to_idx, trained_items = _build_vocab(train_examples, pool, title_to_id, args.max_seq_len)
    model, logs = _train_external_gru4rec(
        train_examples,
        item_to_idx,
        trained_items,
        args.hidden_size,
        args.num_layers,
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
            "note": "Day21 LLM-ESR GRU4Rec smoke checkpoint; do not commit.",
        },
        ARTIFACT_DIR / "llmesr_gru4rec.pt",
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
    joined = _join_evidence(scores, args.evidence_path, SUMMARY_DIR / "day21_second_backbone_beauty_100_joined_candidates.csv")
    join_diag = _join_diagnostics(joined, SUMMARY_DIR / "day21_second_backbone_beauty_100_join_diagnostics.csv")
    grid = _rerank_grid(joined, SUMMARY_DIR / "day21_second_backbone_beauty_100_plugin_rerank_grid.csv")
    plugin_diag = _plugin_diagnostics(joined, grid, join_diag, SUMMARY_DIR / "day21_second_backbone_beauty_100_plugin_diagnostics.csv")
    _write_report(join_diag, plugin_diag, grid, SUMMARY_DIR / "day21_second_backbone_plugin_smoke_report.md")
    print("Day21 second backbone plug-in smoke complete.")


if __name__ == "__main__":
    main()
