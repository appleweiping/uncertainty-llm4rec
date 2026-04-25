"""Day20 multi-seed stability and final external backbone tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from main_day15_bprmf_backbone_plugin_smoke import _normalize_per_user, _rank_metrics, _read_jsonl
from main_day17_sasrec_backbone_plugin_smoke import (
    SUMMARY_DIR,
    _build_vocab,
    _candidate_pool,
    _export_scores,
    _load_title_map,
    _load_train_examples,
    _train_sasrec,
)
from main_day18_sasrec_plugin_larger_validation import _join_diagnostics, _join_evidence


EVIDENCE_PATH = Path(
    "output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl"
)


FIXED_SETTINGS = [
    {
        "method": "A_SASRec_only",
        "normalization": "none",
        "alpha": 1.0,
        "beta": 0.0,
        "lambda": 0.0,
    },
    {
        "method": "B_SASRec_plus_calibrated_relevance",
        "normalization": "minmax",
        "alpha": 0.5,
        "beta": 0.5,
        "lambda": 0.0,
    },
    {
        "method": "C_SASRec_plus_evidence_risk",
        "normalization": "zscore",
        "alpha": 1.0,
        "beta": 0.0,
        "lambda": 0.5,
    },
    {
        "method": "D_SASRec_plus_calibrated_relevance_plus_evidence_risk",
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
                "relative_NDCG_vs_sasrec": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                / max(base_metrics["NDCG@10"], 1e-12),
                "relative_MRR_vs_sasrec": (metrics["MRR@10"] - base_metrics["MRR@10"])
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
        "relative_NDCG_vs_sasrec",
        "relative_MRR_vs_sasrec",
    ]
    rows = []
    for method, group in results.groupby("method", sort=False):
        row = {"method": method}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    rename = {
        "HR@10_mean": "HR@10_mean",
        "HR@10_std": "HR@10_std",
        "NDCG@10_mean": "NDCG@10_mean",
        "NDCG@10_std": "NDCG@10_std",
        "MRR@10_mean": "MRR@10_mean",
        "MRR@10_std": "MRR@10_std",
        "Recall@10_mean": "Recall@10_mean",
        "Recall@10_std": "Recall@10_std",
        "relative_NDCG_vs_sasrec_mean": "relative_NDCG_vs_sasrec_mean",
        "relative_NDCG_vs_sasrec_std": "relative_NDCG_vs_sasrec_std",
        "relative_MRR_vs_sasrec_mean": "relative_MRR_vs_sasrec_mean",
        "relative_MRR_vs_sasrec_std": "relative_MRR_vs_sasrec_std",
    }
    out = out.rename(columns=rename)
    out.to_csv(output_path, index=False)
    return out


def _component_attribution(summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    base = summary[summary["method"] == "A_SASRec_only"].iloc[0]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "method": row["method"],
                "NDCG@10_mean": row["NDCG@10_mean"],
                "MRR@10_mean": row["MRR@10_mean"],
                "relative_NDCG_vs_sasrec_mean": row["relative_NDCG_vs_sasrec_mean"],
                "relative_MRR_vs_sasrec_mean": row["relative_MRR_vs_sasrec_mean"],
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
        return "sequential backbone only"
    if method.startswith("B_"):
        return "calibrated relevance posterior contribution"
    if method.startswith("C_"):
        return "evidence risk as standalone regularizer"
    if method.startswith("D_"):
        return "posterior plus evidence-risk regularization"
    return ""


def _read_first_row(path: Path) -> dict:
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def _main_table(multiseed_summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []

    def add_row(setting, users, candidate_rows, backbone, fallback_rate, method, ndcg, mrr, hr, rel_ndcg, rel_mrr, claim):
        rows.append(
            {
                "setting": setting,
                "users": users,
                "candidate_rows": candidate_rows,
                "backbone": backbone,
                "fallback_rate": fallback_rate,
                "method": method,
                "NDCG@10": ndcg,
                "MRR@10": mrr,
                "HR@10": hr,
                "relative_NDCG_vs_backbone": rel_ndcg,
                "relative_MRR_vs_backbone": rel_mrr,
                "claim_level": claim,
            }
        )

    day14 = _read_first_row(SUMMARY_DIR / "day14_simple_backbone_beauty_100_plugin_diagnostics.csv")
    if day14:
        add_row(
            "Popularity-only smoke",
            100,
            600,
            "train_popularity",
            "",
            day14.get("best_method", "A_Backbone_only"),
            day14.get("best_NDCG@10", ""),
            day14.get("best_MRR@10", ""),
            "",
            day14.get("best_relative_NDCG_vs_backbone", ""),
            day14.get("best_relative_MRR_vs_backbone", ""),
            "engineering_smoke",
        )

    day15 = _read_first_row(SUMMARY_DIR / "day15_bprmf_beauty_100_plugin_diagnostics.csv")
    if day15:
        add_row(
            "BPR-MF smoke",
            100,
            600,
            "bprmf",
            day15.get("fallback_rate", ""),
            day15.get("best_method", ""),
            day15.get("best_NDCG@10", ""),
            day15.get("best_MRR@10", ""),
            day15.get("best_HR@10", ""),
            day15.get("best_relative_NDCG_vs_backbone", ""),
            day15.get("best_relative_MRR_vs_backbone", ""),
            "positive_smoke",
        )

    day16 = _read_first_row(SUMMARY_DIR / "day16_bprmf_beauty_100_repaired_plugin_diagnostics.csv")
    if day16:
        add_row(
            "Repaired BPR-MF smoke",
            100,
            600,
            "bprmf_repaired",
            day16.get("fallback_rate", ""),
            day16.get("best_method", ""),
            day16.get("best_NDCG@10", ""),
            day16.get("best_MRR@10", ""),
            day16.get("best_HR@10", ""),
            day16.get("best_relative_NDCG_vs_backbone", ""),
            day16.get("best_relative_MRR_vs_backbone", ""),
            "positive_smoke",
        )

    for day, users, rows_count, claim in [
        ("day17", 100, 600, "positive_smoke"),
        ("day18", 500, 3000, "larger_validation"),
        ("day19", 973, 5838, "full_validation"),
    ]:
        diag = _read_first_row(SUMMARY_DIR / f"{day}_sasrec_beauty_{'100' if day == 'day17' else '500' if day == 'day18' else 'full'}_plugin_diagnostics.csv")
        if diag:
            add_row(
                f"SASRec {users} users" if day != "day19" else "SASRec full",
                users,
                rows_count,
                "minimal_sasrec",
                diag.get("fallback_rate", ""),
                diag.get("best_method", ""),
                diag.get("best_NDCG@10", ""),
                diag.get("best_MRR@10", ""),
                diag.get("best_HR@10", ""),
                diag.get("best_relative_NDCG_vs_backbone", ""),
                diag.get("best_relative_MRR_vs_backbone", ""),
                claim,
            )

    best_ms = multiseed_summary.sort_values("NDCG@10_mean", ascending=False).iloc[0]
    add_row(
        "SASRec full multi-seed",
        973,
        5838,
        "minimal_sasrec",
        "",
        best_ms["method"],
        best_ms["NDCG@10_mean"],
        best_ms["MRR@10_mean"],
        best_ms["HR@10_mean"],
        best_ms["relative_NDCG_vs_sasrec_mean"],
        best_ms["relative_MRR_vs_sasrec_mean"],
        "multiseed_validation",
    )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _write_report(results: pd.DataFrame, summary: pd.DataFrame, attribution: pd.DataFrame, main_table: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "| method | NDCG mean | NDCG std | MRR mean | MRR std | rel NDCG mean | rel MRR mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['NDCG@10_mean']:.4f} | {row['NDCG@10_std']:.4f} | "
            f"{row['MRR@10_mean']:.4f} | {row['MRR@10_std']:.4f} | "
            f"{row['relative_NDCG_vs_sasrec_mean']:.4f} | {row['relative_MRR_vs_sasrec_mean']:.4f} |"
        )
    table = "\n".join(lines)
    best = summary.sort_values("NDCG@10_mean", ascending=False).iloc[0].to_dict()
    fallback_mean = results.groupby("seed")["fallback_rate"].first().mean()
    report = f"""# Day20 SASRec Multi-seed Final Report

## 1. Day19 Recap

Day19 completed full Beauty SASRec plug-in validation with healthy join coverage and a stable positive gain from Scheme 4 plug-in scoring.

## 2. Multi-seed Setup

Seeds: `42`, `43`, `44`.

The comparison uses fixed Day19 settings rather than reselecting a best setting per seed:

- A: SASRec-only.
- B: minmax, alpha=0.5, beta=0.5, lambda=0.0.
- C: zscore, evidence-risk lambda=0.5.
- D: zscore, alpha=0.5, beta=0.5, lambda=0.2.

No DeepSeek API calls, prompt changes, LoRA, or formula tuning are used.

## 3. Stability Result

Mean fallback rate across seeds: `{fallback_mean:.4f}`.

{table}

Best mean method: `{best['method']}`.

Best mean relative NDCG improvement: `{best['relative_NDCG_vs_sasrec_mean']:.4f}`.

Best mean relative MRR improvement: `{best['relative_MRR_vs_sasrec_mean']:.4f}`.

## 4. Component Attribution

The component attribution table is written to `output-repaired/summary/day20_sasrec_component_attribution.csv`.

The expected interpretation is: calibrated relevance posterior is the primary contributor; evidence risk is a secondary regularizer when D improves over B, and a weak standalone scorer when C remains much smaller than B.

## 5. Final Claim

Scheme 4 can act as an external sequential backbone plug-in on full Beauty, primarily through calibrated relevance posterior, with evidence risk as secondary regularizer. This is not an external SOTA claim; it is a controlled full-domain plug-in validation.

## 6. Limitation

The backbone is a minimal SASRec-style implementation, not every NH/SOTA recommender. The next step is a second external ranking backbone or repository-level integration.

## 7. Day21 Recommendation

Day21 should select a second external backbone repository, preferably a healthier public sequential recommender that can export candidate scores without missing checkpoint or embedding dependencies.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/amazon_beauty/train.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/amazon_beauty/test.jsonl"))
    parser.add_argument("--items_path", type=Path, default=Path("data/processed/amazon_beauty/items.csv"))
    parser.add_argument("--evidence_path", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
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
    train_examples, trained_items, item_pop = _load_train_examples(args.train_path, title_to_id, args.max_seq_len)
    all_rows = []
    for seed in args.seeds:
        backbone_dir = Path(f"output-repaired/backbone/sasrec_beauty_full_seed{seed}")
        artifact_dir = Path(f"artifacts/backbones/sasrec_beauty_full_seed{seed}")
        backbone_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
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
            seed,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "item_to_idx": item_to_idx,
                "seed": seed,
                "args": vars(args),
                "train_logs": logs,
                "note": "Day20 multi-seed smoke checkpoint; do not commit.",
            },
            artifact_dir / "minimal_sasrec.pt",
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
            SUMMARY_DIR / f"day20_sasrec_full_seed{seed}_joined_candidates.csv",
        )
        join_diag = _join_diagnostics(joined, SUMMARY_DIR / f"day20_sasrec_full_seed{seed}_join_diagnostics.csv")
        all_rows.append(_score_fixed_settings(joined, join_diag, seed))

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(SUMMARY_DIR / "day20_sasrec_full_multiseed_results.csv", index=False)
    summary = _summary(results, SUMMARY_DIR / "day20_sasrec_full_multiseed_summary.csv")
    attribution = _component_attribution(summary, SUMMARY_DIR / "day20_sasrec_component_attribution.csv")
    main_table = _main_table(summary, SUMMARY_DIR / "day20_external_backbone_main_table.csv")
    _write_report(
        results,
        summary,
        attribution,
        main_table,
        SUMMARY_DIR / "day20_sasrec_multiseed_final_report.md",
    )
    print("Day20 SASRec multi-seed final tables complete.")


if __name__ == "__main__":
    main()
