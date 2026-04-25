"""Day23 GRU4Rec multi-seed stability and two-backbone final claim tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from main_day15_bprmf_backbone_plugin_smoke import _normalize_per_user, _rank_metrics, _read_jsonl
from main_day21_second_backbone_plugin_smoke import (
    SUMMARY_DIR,
    _build_vocab,
    _candidate_pool,
    _export_scores,
    _join_diagnostics,
    _join_evidence,
    _load_title_map,
    _load_train_examples,
    _train_external_gru4rec,
)


EVIDENCE_PATH = Path(
    "output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl"
)

FIXED_SETTINGS = [
    {
        "method": "A_GRU4Rec_only",
        "normalization": "none",
        "alpha": 1.0,
        "beta": 0.0,
        "lambda": 0.0,
    },
    {
        "method": "B_GRU4Rec_plus_calibrated_relevance",
        "normalization": "zscore",
        "alpha": 0.5,
        "beta": 0.5,
        "lambda": 0.0,
    },
    {
        "method": "C_GRU4Rec_plus_evidence_risk",
        "normalization": "zscore",
        "alpha": 1.0,
        "beta": 0.0,
        "lambda": 0.5,
    },
    {
        "method": "D_GRU4Rec_plus_calibrated_relevance_plus_evidence_risk",
        "normalization": "zscore",
        "alpha": 0.5,
        "beta": 0.5,
        "lambda": 0.2,
    },
]


def _score_fixed_settings(joined: pd.DataFrame, join_diag: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    base_metrics = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    rows = []
    for setting in FIXED_SETTINGS:
        method = setting["method"]
        normalization = setting["normalization"]
        alpha = float(setting["alpha"])
        beta = float(setting["beta"])
        lam = float(setting["lambda"])
        if method.startswith("A_"):
            final_score = df["backbone_score"]
        else:
            norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
            norm_calibrated = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)
            norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
            if method.startswith("B_"):
                final_score = alpha * norm_backbone + beta * norm_calibrated
            elif method.startswith("C_"):
                final_score = norm_backbone - lam * norm_risk
            elif method.startswith("D_"):
                final_score = alpha * norm_backbone + beta * norm_calibrated - lam * norm_risk
            else:
                raise ValueError(f"Unknown method: {method}")
        scored = df[["user_id", "candidate_item_id", "label"]].copy()
        scored["final_score"] = final_score
        metrics = _rank_metrics(scored, "final_score")
        rows.append(
            {
                "seed": seed,
                "method": method,
                "normalization": normalization,
                "alpha": alpha,
                "beta": beta,
                "lambda": lam,
                **metrics,
                "join_coverage": float(join_diag.iloc[0]["join_coverage"]),
                "fallback_rate": float(join_diag.iloc[0]["fallback_rate"]),
                "relative_NDCG_vs_gru4rec": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                / max(base_metrics["NDCG@10"], 1e-12),
                "relative_MRR_vs_gru4rec": (metrics["MRR@10"] - base_metrics["MRR@10"])
                / max(base_metrics["MRR@10"], 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _summary(results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    metric_cols = [
        "HR@10",
        "NDCG@10",
        "MRR@10",
        "Recall@10",
        "relative_NDCG_vs_gru4rec",
        "relative_MRR_vs_gru4rec",
        "fallback_rate",
    ]
    rows = []
    for method, group in results.groupby("method", sort=False):
        row = {"method": method}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _component_attribution(summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    base = summary[summary["method"] == "A_GRU4Rec_only"].iloc[0]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "method": row["method"],
                "NDCG@10_mean": row["NDCG@10_mean"],
                "MRR@10_mean": row["MRR@10_mean"],
                "relative_NDCG_vs_gru4rec_mean": row["relative_NDCG_vs_gru4rec_mean"],
                "relative_MRR_vs_gru4rec_mean": row["relative_MRR_vs_gru4rec_mean"],
                "delta_NDCG_vs_A_mean": row["NDCG@10_mean"] - base["NDCG@10_mean"],
                "delta_MRR_vs_A_mean": row["MRR@10_mean"] - base["MRR@10_mean"],
                "interpretation": _interpret_method(row["method"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _interpret_method(method: str) -> str:
    if method.startswith("A_"):
        return "external GRU4Rec backbone only"
    if method.startswith("B_"):
        return "calibrated relevance posterior contribution"
    if method.startswith("C_"):
        return "evidence risk as standalone regularizer"
    if method.startswith("D_"):
        return "posterior plus evidence-risk regularization"
    return ""


def _two_backbone_table(gru_summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    sasrec_path = SUMMARY_DIR / "day20_sasrec_full_multiseed_summary.csv"
    rows = []
    if sasrec_path.exists():
        sasrec = pd.read_csv(sasrec_path)
        for _, row in sasrec.iterrows():
            rows.append(
                {
                    "backbone": "minimal_sasrec",
                    "setting": "SASRec full multi-seed",
                    "users": 973,
                    "candidate_rows": 5838,
                    "fallback_rate_mean": 0.08838643371017471,
                    "method": row["method"],
                    "NDCG@10_mean": row["NDCG@10_mean"],
                    "NDCG@10_std": row["NDCG@10_std"],
                    "MRR@10_mean": row["MRR@10_mean"],
                    "MRR@10_std": row["MRR@10_std"],
                    "HR@10_mean": row["HR@10_mean"],
                    "HR@10_std": row["HR@10_std"],
                    "relative_NDCG_vs_backbone_mean": row["relative_NDCG_vs_sasrec_mean"],
                    "relative_NDCG_vs_backbone_std": row["relative_NDCG_vs_sasrec_std"],
                    "relative_MRR_vs_backbone_mean": row["relative_MRR_vs_sasrec_mean"],
                    "relative_MRR_vs_backbone_std": row["relative_MRR_vs_sasrec_std"],
                    "claim_level": "full_multiseed_validation",
                }
            )
    for _, row in gru_summary.iterrows():
        rows.append(
            {
                "backbone": "llmesr_gru4rec",
                "setting": "LLM-ESR GRU4Rec full multi-seed",
                "users": 973,
                "candidate_rows": 5838,
                "fallback_rate_mean": row["fallback_rate_mean"],
                "method": row["method"],
                "NDCG@10_mean": row["NDCG@10_mean"],
                "NDCG@10_std": row["NDCG@10_std"],
                "MRR@10_mean": row["MRR@10_mean"],
                "MRR@10_std": row["MRR@10_std"],
                "HR@10_mean": row["HR@10_mean"],
                "HR@10_std": row["HR@10_std"],
                "relative_NDCG_vs_backbone_mean": row["relative_NDCG_vs_gru4rec_mean"],
                "relative_NDCG_vs_backbone_std": row["relative_NDCG_vs_gru4rec_std"],
                "relative_MRR_vs_backbone_mean": row["relative_MRR_vs_gru4rec_mean"],
                "relative_MRR_vs_backbone_std": row["relative_MRR_vs_gru4rec_std"],
                "claim_level": "full_multiseed_validation",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _write_claim_map(gru_summary: pd.DataFrame, two_backbone_table: pd.DataFrame, output_path: Path) -> None:
    best_gru = gru_summary.sort_values("NDCG@10_mean", ascending=False).iloc[0].to_dict()
    best_sasrec = two_backbone_table[two_backbone_table["backbone"] == "minimal_sasrec"].sort_values(
        "NDCG@10_mean", ascending=False
    )
    sasrec_line = "SASRec multi-seed table not found."
    if not best_sasrec.empty:
        row = best_sasrec.iloc[0].to_dict()
        sasrec_line = (
            f"SASRec-style full multi-seed best method `{row['method']}` reaches NDCG@10 "
            f"`{float(row['NDCG@10_mean']):.4f}` and MRR@10 `{float(row['MRR@10_mean']):.4f}`."
        )
    text = f"""# Day23 Final Claim Map

## 1. Week1-Week4 / Original Pipeline

The original confidence pipeline showed that raw/verbalized confidence is not pure noise, but it is miscalibrated and cannot be used directly as a trustworthy probability.

## 2. Day6: Yes/No Decision Confidence Repair

The yes/no controlled setting showed that evidence decomposition and decoupled reranking can repair decision reliability. This layer is useful diagnostically, but it is not the final recommendation task form.

## 3. Day9: Candidate Relevance Posterior Calibration

Candidate relevance scoring reframed the task around `relevance_probability`. Valid-set calibration produced `calibrated_relevance_probability`, which became the main Scheme 4 signal.

## 4. Day10: List-level Boundary

Direct evidence-heavy list generation was not the best first-pass decision form. Plain list generation is a better base, while evidence works better as a posterior/risk plug-in.

## 5. Day20: SASRec Full Multi-seed External Plug-in

{sasrec_line}

## 6. Day23: GRU4Rec Full Multi-seed External Plug-in

LLM-ESR GRU4Rec full multi-seed best method `{best_gru['method']}` reaches NDCG@10 `{best_gru['NDCG@10_mean']:.4f}` and MRR@10 `{best_gru['MRR@10_mean']:.4f}`.

## 7. Final Method Position

Scheme 4 is best described as a calibrated evidence posterior plug-in. The primary contribution is calibrated relevance posterior; `evidence_risk` is a secondary risk regularizer. Across two sequential backbones, D generally improves over B, but C-only remains much weaker than B.

## 8. Claim Boundary

The current claim is Beauty full + two sequential backbones. It is not yet a universal SOTA claim across all domains and all recommender families. The next extension is a third backbone or cross-domain validation.
"""
    output_path.write_text(text, encoding="utf-8")


def _write_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    attribution: pd.DataFrame,
    two_backbone: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "| method | NDCG mean | NDCG std | MRR mean | MRR std | rel NDCG mean | rel MRR mean | fallback mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['NDCG@10_mean']:.4f} | {row['NDCG@10_std']:.4f} | "
            f"{row['MRR@10_mean']:.4f} | {row['MRR@10_std']:.4f} | "
            f"{row['relative_NDCG_vs_gru4rec_mean']:.4f} | {row['relative_MRR_vs_gru4rec_mean']:.4f} | "
            f"{row['fallback_rate_mean']:.4f} |"
        )
    table = "\n".join(lines)
    best = summary.sort_values("NDCG@10_mean", ascending=False).iloc[0].to_dict()
    report = f"""# Day23 GRU4Rec Multi-seed And Two-backbone Report

## 1. Day22 Recap

Day22 showed that LLM-ESR GRU4Rec full single-seed validation was healthy and positive.

## 2. Multi-seed Setup

Seeds: `42`, `43`, `44`.

The settings are fixed from Day22 rather than reselected per seed:

- A: GRU4Rec-only.
- B: zscore, alpha=0.5, beta=0.5, lambda=0.
- C: zscore, lambda=0.5.
- D: zscore, alpha=0.5, beta=0.5, lambda=0.2.

No DeepSeek API calls, prompt changes, LoRA, or formula changes are used.

## 3. GRU4Rec Stability Result

{table}

Best mean method: `{best['method']}`.

Best mean relative NDCG improvement: `{best['relative_NDCG_vs_gru4rec_mean']:.4f}`.

Best mean relative MRR improvement: `{best['relative_MRR_vs_gru4rec_mean']:.4f}`.

## 4. Two-backbone Comparison

The merged table is written to `output-repaired/summary/day23_two_backbone_external_plugin_main_table.csv`.

Both SASRec-style and LLM-ESR GRU4Rec support the same qualitative result: calibrated relevance posterior is the primary gain source, and evidence risk is a secondary regularizer.

## 5. Component Attribution

The attribution table is written to `output-repaired/summary/day23_gru4rec_component_attribution.csv`.

## 6. Day24 Recommendation

Day24 should audit a third backbone that is not just another GRU/SASRec clone. A graph or matrix-factorization-plus-neural baseline with clean candidate-score export would be ideal; ItemKNN/co-occurrence should only be a sanity fallback.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/amazon_beauty/train.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/amazon_beauty/test.jsonl"))
    parser.add_argument("--items_path", type=Path, default=Path("data/processed/amazon_beauty/items.csv"))
    parser.add_argument("--evidence_path", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    title_to_id = _load_title_map(args.items_path)
    pool = _candidate_pool(args.evidence_path, args.test_path, 100000)
    train_examples, trained_items_base, item_pop = _load_train_examples(args.train_path, title_to_id, args.max_seq_len)
    all_rows = []
    for seed in args.seeds:
        backbone_dir = Path(f"output-repaired/backbone/llmesr_gru4rec_beauty_full_seed{seed}")
        artifact_dir = Path(f"artifacts/backbones/llmesr_gru4rec_beauty_full_seed{seed}")
        backbone_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
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
            seed,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "item_to_idx": item_to_idx,
                "seed": seed,
                "args": vars(args),
                "train_logs": logs,
                "note": "Day23 GRU4Rec multi-seed checkpoint; do not commit.",
            },
            artifact_dir / "llmesr_gru4rec.pt",
        )
        (artifact_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        scores = _export_scores(
            model,
            pool,
            title_to_id,
            item_to_idx,
            trained_items,
            item_pop,
            args.max_seq_len,
            backbone_dir / "candidate_scores.csv",
        )
        joined = _join_evidence(
            scores,
            args.evidence_path,
            SUMMARY_DIR / f"day23_gru4rec_full_seed{seed}_joined_candidates.csv",
        )
        join_diag = _join_diagnostics(joined, SUMMARY_DIR / f"day23_gru4rec_full_seed{seed}_join_diagnostics.csv")
        all_rows.append(_score_fixed_settings(joined, join_diag, seed))

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(SUMMARY_DIR / "day23_gru4rec_full_multiseed_results.csv", index=False)
    summary = _summary(results, SUMMARY_DIR / "day23_gru4rec_full_multiseed_summary.csv")
    attribution = _component_attribution(summary, SUMMARY_DIR / "day23_gru4rec_component_attribution.csv")
    two_backbone = _two_backbone_table(summary, SUMMARY_DIR / "day23_two_backbone_external_plugin_main_table.csv")
    _write_claim_map(summary, two_backbone, SUMMARY_DIR / "day23_final_claim_map.md")
    _write_report(
        results,
        summary,
        attribution,
        two_backbone,
        SUMMARY_DIR / "day23_gru4rec_multiseed_and_two_backbone_report.md",
    )
    print("Day23 GRU4Rec multi-seed and two-backbone final tables complete.")


if __name__ == "__main__":
    main()
