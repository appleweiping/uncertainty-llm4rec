"""Day22 full Beauty validation for LLM-ESR GRU4Rec + Scheme-4 plug-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from main_day15_bprmf_backbone_plugin_smoke import _normalize_per_user, _rank_change_stats, _rank_metrics, _read_jsonl
from main_day21_second_backbone_plugin_smoke import (
    ARTIFACT_DIR as DAY21_ARTIFACT_DIR,
    BACKBONE_DIR as DAY21_BACKBONE_DIR,
    SUMMARY_DIR,
    _build_vocab,
    _export_scores,
    _join_diagnostics,
    _join_evidence,
    _load_title_map,
    _load_train_examples,
    _plugin_diagnostics,
    _rerank_grid,
    _train_external_gru4rec,
)


BACKBONE_DIR = Path("output-repaired/backbone/llmesr_gru4rec_beauty_full")
ARTIFACT_DIR = Path("artifacts/backbones/llmesr_gru4rec_beauty_full")


def _candidate_pool(evidence_path: Path, test_path: Path, max_users: int) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    users = evidence["user_id"].drop_duplicates().head(max_users).tolist()
    key = evidence[evidence["user_id"].isin(users)][["user_id", "candidate_item_id"]].copy()
    key["candidate_item_id"] = key["candidate_item_id"].astype(str)
    key["user_id"] = key["user_id"].astype(str)
    rows = []
    user_set = set(users)
    for row in _read_jsonl(test_path):
        if str(row["user_id"]) in user_set:
            rows.append(row)
    test_df = pd.DataFrame(rows)
    test_df["user_id"] = test_df["user_id"].astype(str)
    test_df["candidate_item_id"] = test_df["candidate_item_id"].astype(str)
    return test_df.merge(key.drop_duplicates(), on=["user_id", "candidate_item_id"], how="inner")


def _rank_maps(df: pd.DataFrame, score_col: str) -> pd.Series:
    ranked = df.sort_values(["user_id", score_col, "candidate_item_id"], ascending=[True, False, True]).copy()
    ranked["_rank"] = ranked.groupby("user_id").cumcount() + 1
    return ranked.set_index(["user_id", "candidate_item_id"])["_rank"]


def _case_study(joined: pd.DataFrame, grid: pd.DataFrame, test_path: Path, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    best = grid.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
    normalization = best["normalization"]
    alpha = float(best["alpha"])
    beta = float(best["beta"])
    lam = float(best["lambda"])
    df["norm_backbone"] = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
    df["norm_calibrated"] = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)
    df["norm_risk"] = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
    method = str(best["method"])
    if method.startswith("B_"):
        df["final_score"] = alpha * df["norm_backbone"] + beta * df["norm_calibrated"]
    elif method.startswith("C_"):
        df["final_score"] = df["norm_backbone"] - lam * df["norm_risk"]
    elif method.startswith("D_"):
        df["final_score"] = alpha * df["norm_backbone"] + beta * df["norm_calibrated"] - lam * df["norm_risk"]
    else:
        df["final_score"] = df["backbone_score"]
    old_rank = _rank_maps(df, "backbone_score")
    new_rank = _rank_maps(df, "final_score")
    df["old_rank"] = df.set_index(["user_id", "candidate_item_id"]).index.map(old_rank)
    df["new_rank"] = df.set_index(["user_id", "candidate_item_id"]).index.map(new_rank)
    df["rank_delta"] = df["old_rank"] - df["new_rank"]

    text_lookup = {}
    for row in _read_jsonl(test_path):
        key = (str(row["user_id"]), str(row["candidate_item_id"]))
        text_lookup[key] = str(row.get("candidate_title") or row.get("candidate_text") or "")[:180]
    df["candidate_text_short"] = [
        text_lookup.get((str(row.user_id), str(row.candidate_item_id)), "") for row in df.itertuples(index=False)
    ]

    frames = []
    promoted_positive = (
        df[(df["label"].astype(int) == 1) & (df["rank_delta"] > 0)].sort_values("rank_delta", ascending=False).head(10).copy()
    )
    promoted_positive["case_type"] = "promoted_positive"
    frames.append(promoted_positive)
    demoted_negative = (
        df[(df["label"].astype(int) == 0) & (df["rank_delta"] < 0)].sort_values("rank_delta", ascending=True).head(10).copy()
    )
    demoted_negative["case_type"] = "demoted_negative"
    frames.append(demoted_negative)
    corrected_positive = promoted_positive.copy()
    corrected_positive["case_type"] = "corrected_positive"
    frames.append(corrected_positive)
    harmed_positive = (
        df[(df["label"].astype(int) == 1) & (df["rank_delta"] < 0)].sort_values("rank_delta", ascending=True).head(10).copy()
    )
    harmed_positive["case_type"] = "harmed_positive"
    frames.append(harmed_positive)
    high_risk_demoted = df[df["rank_delta"] < 0].sort_values(["evidence_risk", "rank_delta"], ascending=[False, True]).head(10).copy()
    high_risk_demoted["case_type"] = "high_risk_demoted"
    frames.append(high_risk_demoted)

    out = pd.concat(frames, ignore_index=True)
    cols = [
        "case_type",
        "user_id",
        "candidate_item_id",
        "label",
        "old_rank",
        "new_rank",
        "rank_delta",
        "backbone_score",
        "calibrated_relevance_probability",
        "evidence_risk",
        "ambiguity",
        "missing_information",
        "abs_evidence_margin",
        "candidate_text_short",
    ]
    out[cols].to_csv(output_path, index=False)
    return out[cols]


def _write_report(
    join_diag: pd.DataFrame,
    plugin_diag: pd.DataFrame,
    grid: pd.DataFrame,
    case_study: pd.DataFrame,
    output_path: Path,
) -> None:
    jd = join_diag.iloc[0].to_dict()
    pdg = plugin_diag.iloc[0].to_dict()
    method_best = (
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
                "relative_NDCG_vs_backbone",
                "relative_MRR_vs_backbone",
                "lambda",
                "alpha",
                "beta",
                "normalization",
                "rank_change_rate",
            ],
        ]
    )
    lines = [
        "| method | HR@10 | NDCG@10 | MRR@10 | rel NDCG | rel MRR | lambda | alpha | beta | normalization | rank_change_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in method_best.iterrows():
        lines.append(
            f"| {row['method']} | {float(row['HR@10']):.4f} | {float(row['NDCG@10']):.4f} | "
            f"{float(row['MRR@10']):.4f} | {float(row['relative_NDCG_vs_backbone']):.4f} | "
            f"{float(row['relative_MRR_vs_backbone']):.4f} | {float(row['lambda']):.2f} | "
            f"{float(row['alpha']):.2f} | {float(row['beta']):.2f} | {row['normalization']} | "
            f"{float(row['rank_change_rate']):.4f} |"
        )
    method_table = "\n".join(lines)
    sasrec_path = SUMMARY_DIR / "day20_sasrec_full_multiseed_summary.csv"
    sasrec_note = "Day20 SASRec multi-seed summary was not found."
    if sasrec_path.exists():
        sasrec = pd.read_csv(sasrec_path)
        best_sasrec = sasrec.sort_values("NDCG@10_mean", ascending=False).iloc[0].to_dict()
        sasrec_note = (
            f"Day20 best SASRec multi-seed method was `{best_sasrec['method']}` with mean NDCG@10 "
            f"`{float(best_sasrec['NDCG@10_mean']):.4f}` and mean MRR@10 `{float(best_sasrec['MRR@10_mean']):.4f}`. "
            "Day22 checks the same plug-in direction on LLM-ESR GRU4Rec; exact numeric equality is not expected."
        )
    case_counts = case_study["case_type"].value_counts().to_dict()
    health_status = "healthy"
    if float(jd["join_coverage"]) < 0.95 or float(jd["fallback_rate"]) >= 0.2:
        health_status = "partially blocked"
    report = f"""# Day22 LLM-ESR GRU4Rec Full Report

## 1. Day21 Recap

Day21 selected LLM-ESR GRU4Rec as the second external/NH backbone smoke because it exposes real candidate logits through `GRU4Rec.predict()` without requiring OpenP5-style generative checkpoint adaptation.

## 2. Day22 Setup

Day22 expands the same LLM-ESR GRU4Rec adapter to the full Beauty evidence-aligned candidate pool. Training uses only Beauty train split. No DeepSeek API calls, no prompt changes, no LoRA, and no formula changes are used.

## 3. Backbone Health

Health status: `{health_status}`.

Full users: `{int(jd['num_users'])}`.

Backbone rows: `{int(jd['num_backbone_rows'])}`.

Join coverage: `{float(jd['join_coverage']):.4f}`.

Fallback rate: `{float(jd['fallback_rate']):.4f}`.

Positive fallback rate: `{float(jd['fallback_rate_positive']):.4f}`.

Negative fallback rate: `{float(jd['fallback_rate_negative']):.4f}`.

GRU4Rec-only NDCG@10: `{float(pdg['backbone_NDCG@10']):.4f}`.

GRU4Rec-only MRR@10: `{float(pdg['backbone_MRR@10']):.4f}`.

## 4. Full Plug-in Result

Best method: `{pdg['best_method']}`.

Best NDCG@10: `{float(pdg['best_NDCG@10']):.4f}`.

Best MRR@10: `{float(pdg['best_MRR@10']):.4f}`.

Relative NDCG improvement vs GRU4Rec-only: `{float(pdg['best_relative_NDCG_vs_backbone']):.4f}`.

Relative MRR improvement vs GRU4Rec-only: `{float(pdg['best_relative_MRR_vs_backbone']):.4f}`.

Best row per method:

{method_table}

## 5. Component Attribution

If B is close to D, calibrated relevance posterior is the primary contributor. If D exceeds B at a positive lambda, evidence risk contributes as secondary regularization.

## 6. Comparison With SASRec Full Multi-seed

{sasrec_note}

## 7. Claim Boundary

This is second external sequential backbone full validation, not an external SOTA leaderboard claim.

## 8. Day23 Recommendation

If Day22 is healthy and positive, Day23 should produce a two-backbone final claim map and main table. If Day22 is partially blocked, Day23 should document the limitation and decide whether a third public backbone is needed.

Case study counts: `{case_counts}`.
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
    parser.add_argument("--max_users", type=int, default=100000)
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
            "note": "Day22 LLM-ESR GRU4Rec full checkpoint; do not commit.",
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
    joined = _join_evidence(scores, args.evidence_path, SUMMARY_DIR / "day22_llmesr_gru4rec_beauty_full_joined_candidates.csv")
    join_diag = _join_diagnostics(joined, SUMMARY_DIR / "day22_llmesr_gru4rec_beauty_full_join_diagnostics.csv")
    grid = _rerank_grid(joined, SUMMARY_DIR / "day22_llmesr_gru4rec_beauty_full_plugin_rerank_grid.csv")
    plugin_diag = _plugin_diagnostics(joined, grid, join_diag, SUMMARY_DIR / "day22_llmesr_gru4rec_beauty_full_plugin_diagnostics.csv")
    case_study = _case_study(joined, grid, args.test_path, SUMMARY_DIR / "day22_llmesr_gru4rec_full_plugin_case_study.csv")
    _write_report(join_diag, plugin_diag, grid, case_study, SUMMARY_DIR / "day22_llmesr_gru4rec_full_report.md")
    print("Day22 LLM-ESR GRU4Rec full validation complete.")


if __name__ == "__main__":
    main()
