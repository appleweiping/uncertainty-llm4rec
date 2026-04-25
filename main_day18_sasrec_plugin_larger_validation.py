"""Day18 SASRec-style backbone larger plug-in validation on 500 Beauty users."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    BACKBONE_DIR as DAY17_BACKBONE_DIR,
    SUMMARY_DIR,
    EVIDENCE_COLUMNS,
    _build_vocab,
    _candidate_pool,
    _export_scores,
    _load_title_map,
    _load_train_examples,
    _train_sasrec,
)


BACKBONE_DIR = Path("output-repaired/backbone/sasrec_beauty_500")
ARTIFACT_DIR = Path("artifacts/backbones/sasrec_beauty_500")


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
        settings = [("A_SASRec_only", 0.0, 1.0, 0.0, df["backbone_score"])]
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1 - alpha
            settings.append(
                (
                    "B_SASRec_plus_calibrated_relevance",
                    0.0,
                    alpha,
                    beta,
                    alpha * df[f"norm_backbone_{normalization}"] + beta * df[f"norm_calibrated_{normalization}"],
                )
            )
        for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
            settings.append(
                (
                    "C_SASRec_plus_evidence_risk",
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
                        "D_SASRec_plus_calibrated_relevance_plus_evidence_risk",
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
                    "backbone_name": "minimal_sasrec",
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
    base = grid[grid["method"] == "A_SASRec_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
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
    if str(best["method"]).startswith("B_"):
        df["final_score"] = alpha * df["norm_backbone"] + beta * df["norm_calibrated"]
    elif str(best["method"]).startswith("C_"):
        df["final_score"] = df["norm_backbone"] - lam * df["norm_risk"]
    elif str(best["method"]).startswith("D_"):
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
        text = str(row.get("candidate_title") or row.get("candidate_text") or "")[:180]
        text_lookup[key] = text
    df["candidate_text_short"] = [
        text_lookup.get((str(row.user_id), str(row.candidate_item_id)), "") for row in df.itertuples(index=False)
    ]

    frames = []
    promoted = df.sort_values("rank_delta", ascending=False).head(10).copy()
    promoted["case_type"] = "promoted"
    frames.append(promoted)
    demoted = df.sort_values("rank_delta", ascending=True).head(10).copy()
    demoted["case_type"] = "demoted"
    frames.append(demoted)
    corrected = df[(df["label"].astype(int) == 1) & (df["rank_delta"] > 0)].sort_values("rank_delta", ascending=False).head(10).copy()
    corrected["case_type"] = "corrected_positive"
    frames.append(corrected)
    harmed = df[(df["label"].astype(int) == 1) & (df["rank_delta"] < 0)].sort_values("rank_delta", ascending=True).head(10).copy()
    harmed["case_type"] = "harmed_positive"
    frames.append(harmed)
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
    method_table = "\n".join(lines)
    case_counts = case_study["case_type"].value_counts().to_dict()
    report = f"""# Day18 SASRec Plug-in Larger Validation Report

## 1. Day17 Recap

Day17 established a healthier 100-user sequential backbone smoke: join coverage was 1.0, fallback dropped below 20%, and adding calibrated relevance improved SASRec-style ranking.

## 2. Day18 Setup

Day18 expands the same minimal SASRec-style backbone to 500 Beauty users. Training still uses only the train split, and this is still a larger smoke validation rather than a full SOTA comparison.

## 3. Backbone Health

Join coverage: `{float(jd['join_coverage']):.4f}`.

Fallback rate: `{float(jd['fallback_rate']):.4f}`.

Positive fallback rate: `{float(jd['fallback_rate_positive']):.4f}`.

Negative fallback rate: `{float(jd['fallback_rate_negative']):.4f}`.

SASRec-only NDCG@10: `{float(pdg['backbone_NDCG@10']):.4f}`.

SASRec-only MRR@10: `{float(pdg['backbone_MRR@10']):.4f}`.

## 4. Plug-in Result

Best method: `{pdg['best_method']}`.

Best NDCG@10: `{float(pdg['best_NDCG@10']):.4f}`.

Best MRR@10: `{float(pdg['best_MRR@10']):.4f}`.

Relative NDCG improvement vs SASRec-only: `{float(pdg['best_relative_NDCG_vs_backbone']):.4f}`.

Relative MRR improvement vs SASRec-only: `{float(pdg['best_relative_MRR_vs_backbone']):.4f}`.

Best row per method:

{method_table}

## 5. Component Attribution

If the best B row remains close to or better than D, the gain is mainly from calibrated relevance posterior. If D improves over B at positive lambda, evidence risk is adding regularization. The diagnostics file records evidence-risk AUROC and base-risk Spearman for this check.

## 6. Case Study

Case study rows were written to `output-repaired/summary/day18_sasrec_plugin_case_study.csv`.

Case type counts: `{case_counts}`.

## 7. Decision For Day19

If join coverage stays above 0.95, fallback remains below 20%, and relative gains remain positive, Day19 can either expand to full or run multi-seed/slice stability. If evidence risk still contributes little beyond calibrated relevance, the paper should frame Scheme 4 as calibrated relevance posterior first and evidence-risk regularization second.
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
    parser.add_argument("--max_users", type=int, default=500)
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
            "note": "Day18 smoke checkpoint; do not commit.",
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
    joined = _join_evidence(candidate_scores, args.evidence_path, SUMMARY_DIR / "day18_sasrec_beauty_500_joined_candidates.csv")
    join_diag = _join_diagnostics(joined, SUMMARY_DIR / "day18_sasrec_beauty_500_join_diagnostics.csv")
    grid = _rerank_grid(joined, SUMMARY_DIR / "day18_sasrec_beauty_500_plugin_rerank_grid.csv")
    plugin_diag = _plugin_diagnostics(joined, grid, join_diag, SUMMARY_DIR / "day18_sasrec_beauty_500_plugin_diagnostics.csv")
    case_study = _case_study(joined, grid, args.test_path, SUMMARY_DIR / "day18_sasrec_plugin_case_study.csv")
    _write_report(join_diag, plugin_diag, grid, case_study, SUMMARY_DIR / "day18_sasrec_plugin_larger_report.md")
    print("Day18 SASRec plug-in larger validation complete.")


if __name__ == "__main__":
    main()
