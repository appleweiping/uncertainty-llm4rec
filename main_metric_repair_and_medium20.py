"""Metric repair and 20-negative medium split construction.

No API calls and no model reruns. This script:
1. recomputes Beauty external-backbone metrics with HR@1/3/5, NDCG@1/3/5/10,
   MRR, positive-rank statistics, and HR@10 trivial flags;
2. labels existing cross-domain medium as medium_5neg;
3. builds medium_20neg splits from regular domains.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd

from main_day15_bprmf_backbone_plugin_smoke import _normalize_per_user
from main_day28_build_cross_domain_medium_splits import (
    BEAUTY_DAY9_ROWS,
    DOMAINS,
    SUMMARY_DIR,
    build_domain,
    _markdown_table,
    _write_configs,
)


BACKBONE_SPECS = [
    {
        "backbone": "SASRec-style",
        "prefix": "sasrec",
        "results_path": SUMMARY_DIR / "day20_sasrec_full_multiseed_results.csv",
        "joined_template": SUMMARY_DIR / "day20_sasrec_full_seed{seed}_joined_candidates.csv",
        "rel_ndcg_name": "relative_NDCG_vs_sasrec",
        "rel_mrr_name": "relative_MRR_vs_sasrec",
    },
    {
        "backbone": "LLM-ESR GRU4Rec",
        "prefix": "gru4rec",
        "results_path": SUMMARY_DIR / "day23_gru4rec_full_multiseed_results.csv",
        "joined_template": SUMMARY_DIR / "day23_gru4rec_full_seed{seed}_joined_candidates.csv",
        "rel_ndcg_name": "relative_NDCG_vs_gru4rec",
        "rel_mrr_name": "relative_MRR_vs_gru4rec",
    },
    {
        "backbone": "LLM-ESR Bert4Rec",
        "prefix": "bert4rec",
        "results_path": SUMMARY_DIR / "day25_bert4rec_full_multiseed_results.csv",
        "joined_template": {
            42: SUMMARY_DIR / "day25_bert4rec_beauty_full_joined_candidates.csv",
            43: SUMMARY_DIR / "day25_bert4rec_full_seed43_joined_candidates.csv",
            44: SUMMARY_DIR / "day25_bert4rec_full_seed44_joined_candidates.csv",
        },
        "rel_ndcg_name": "relative_NDCG_vs_bert4rec",
        "rel_mrr_name": "relative_MRR_vs_bert4rec",
    },
]


def _joined_path(spec: dict, seed: int) -> Path:
    template = spec["joined_template"]
    if isinstance(template, dict):
        return template[seed]
    return Path(str(template).format(seed=seed))


def _final_score(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    method = str(row["method"])
    norm = str(row["normalization"])
    alpha = float(row["alpha"])
    beta = float(row["beta"])
    lam = float(row["lambda"])
    if method.startswith("A_"):
        return df["backbone_score"]
    norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], norm)
    norm_calibrated = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], norm)
    norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], norm)
    if method.startswith("B_"):
        return alpha * norm_backbone + beta * norm_calibrated
    if method.startswith("C_"):
        return norm_backbone - lam * norm_risk
    if method.startswith("D_"):
        return alpha * norm_backbone + beta * norm_calibrated - lam * norm_risk
    raise ValueError(f"Unknown method: {method}")


def _metric_from_ranks(ranks: list[int], pool_sizes: list[int]) -> dict[str, float | bool]:
    arr = pd.Series(ranks, dtype=float)
    pools = pd.Series(pool_sizes, dtype=float)

    def hr(k: int) -> float:
        return float((arr <= k).mean()) if len(arr) else math.nan

    def ndcg(k: int) -> float:
        vals = [1.0 / math.log2(rank + 1) if rank <= k else 0.0 for rank in ranks]
        return float(pd.Series(vals).mean()) if vals else math.nan

    return {
        "HR@1": hr(1),
        "HR@3": hr(3),
        "HR@5": hr(5),
        "HR@10": hr(10),
        "NDCG@1": ndcg(1),
        "NDCG@3": ndcg(3),
        "NDCG@5": ndcg(5),
        "NDCG@10": ndcg(10),
        "MRR": float((1.0 / arr).mean()) if len(arr) else math.nan,
        "positive_rank_mean": float(arr.mean()) if len(arr) else math.nan,
        "positive_rank_median": float(arr.median()) if len(arr) else math.nan,
        "candidate_pool_size_mean": float(pools.mean()) if len(pools) else math.nan,
        "candidate_pool_size_min": float(pools.min()) if len(pools) else math.nan,
        "candidate_pool_size_max": float(pools.max()) if len(pools) else math.nan,
        "hr10_trivial_flag": bool((pools.max() <= 10) or (pools.mean() <= 10)) if len(pools) else False,
    }


def _compute_metrics(df: pd.DataFrame, score_col: str = "final_score") -> dict[str, float | bool]:
    ranks: list[int] = []
    pool_sizes: list[int] = []
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).reset_index(drop=True)
        pool_sizes.append(len(ranked))
        positives = ranked.index[ranked["label"].astype(int) == 1].tolist()
        if positives:
            ranks.append(int(positives[0] + 1))
    return _metric_from_ranks(ranks, pool_sizes)


def repair_beauty_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in BACKBONE_SPECS:
        fixed = pd.read_csv(spec["results_path"])
        for _, setting in fixed.iterrows():
            seed = int(setting["seed"])
            joined = pd.read_csv(_joined_path(spec, seed))
            df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
            df["final_score"] = _final_score(df, setting)
            metrics = _compute_metrics(df, "final_score")
            rows.append(
                {
                    "backbone": spec["backbone"],
                    "seed": seed,
                    "method_original": setting["method"],
                    "method": _method_label(setting["method"]),
                    "normalization": setting["normalization"],
                    "alpha": setting["alpha"],
                    "beta": setting["beta"],
                    "lambda": setting["lambda"],
                    **metrics,
                    "fallback_rate": setting["fallback_rate"],
                }
            )
    results = pd.DataFrame(rows)
    summary_rows = []
    metric_cols = [
        "HR@1",
        "HR@3",
        "HR@5",
        "HR@10",
        "NDCG@1",
        "NDCG@3",
        "NDCG@5",
        "NDCG@10",
        "MRR",
        "positive_rank_mean",
        "positive_rank_median",
        "candidate_pool_size_mean",
        "candidate_pool_size_min",
        "candidate_pool_size_max",
        "fallback_rate",
    ]
    for (backbone, method), group in results.groupby(["backbone", "method"], sort=False):
        base = results[(results["backbone"] == backbone) & (results["method"] == "Backbone only")]
        row = {
            "backbone": backbone,
            "method": method,
            "claim_level": "full_multiseed_metric_repair",
            "hr10_trivial_flag": bool(group["hr10_trivial_flag"].any()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        row["relative_NDCG@10_vs_backbone_mean"] = (
            row["NDCG@10_mean"] - float(base["NDCG@10"].mean())
        ) / max(float(base["NDCG@10"].mean()), 1e-12)
        row["relative_MRR_vs_backbone_mean"] = (
            row["MRR_mean"] - float(base["MRR"].mean())
        ) / max(float(base["MRR"].mean()), 1e-12)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_DIR / "day26_three_backbone_external_plugin_main_table_metric_repaired.csv", index=False)
    attribution = _component_attribution(summary)
    attribution.to_csv(SUMMARY_DIR / "day26_component_attribution_summary_metric_repaired.csv", index=False)
    _write_metric_repair_report(summary, attribution)
    return summary, attribution


def _method_label(method: str) -> str:
    if method.startswith("A_"):
        return "Backbone only"
    if method.startswith("B_"):
        return "Backbone + calibrated relevance"
    if method.startswith("C_"):
        return "Backbone + evidence risk"
    if method.startswith("D_"):
        return "Backbone + calibrated relevance + evidence risk"
    return method


def _component_attribution(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone, group in summary.groupby("backbone", sort=False):
        by = group.set_index("method")
        a = by.loc["Backbone only"]
        b = by.loc["Backbone + calibrated relevance"]
        c = by.loc["Backbone + evidence risk"]
        d = by.loc["Backbone + calibrated relevance + evidence risk"]
        for method, row in by.iterrows():
            rows.append(
                {
                    "backbone": backbone,
                    "method": method,
                    "NDCG@10_mean": row["NDCG@10_mean"],
                    "MRR_mean": row["MRR_mean"],
                    "HR@1_mean": row["HR@1_mean"],
                    "HR@3_mean": row["HR@3_mean"],
                    "NDCG@3_mean": row["NDCG@3_mean"],
                    "NDCG@5_mean": row["NDCG@5_mean"],
                    "B_exceeds_A_NDCG": bool(b["NDCG@10_mean"] > a["NDCG@10_mean"]),
                    "C_weaker_than_B_NDCG": bool(c["NDCG@10_mean"] < b["NDCG@10_mean"]),
                    "D_exceeds_B_NDCG": bool(d["NDCG@10_mean"] > b["NDCG@10_mean"]),
                    "main_contributor": _contributor(method),
                    "hr10_trivial_flag": bool(row["hr10_trivial_flag"]),
                }
            )
    return pd.DataFrame(rows)


def _contributor(method: str) -> str:
    if method == "Backbone + calibrated relevance":
        return "calibrated_relevance_posterior"
    if method == "Backbone + evidence risk":
        return "evidence_risk"
    if method == "Backbone + calibrated relevance + evidence risk":
        return "combined"
    return "baseline"


def _write_metric_repair_report(summary: pd.DataFrame, attribution: pd.DataFrame) -> None:
    d = summary[summary["method"] == "Backbone + calibrated relevance + evidence risk"]
    lines = []
    for _, row in d.iterrows():
        lines.append(
            f"- {row['backbone']}: NDCG@10 `{row['NDCG@10_mean']:.4f}`, MRR `{row['MRR_mean']:.4f}`, "
            f"HR@1 `{row['HR@1_mean']:.4f}`, HR@3 `{row['HR@3_mean']:.4f}`, "
            f"NDCG@3 `{row['NDCG@3_mean']:.4f}`, NDCG@5 `{row['NDCG@5_mean']:.4f}`."
        )
    text = f"""# Day26 Metric Repair Report

## 1. HR@10 Triviality

Beauty and the first cross-domain medium builder use a 1 positive + 5 negatives candidate pool, so each user has only 6 candidates. Under this setup, HR@10 is trivial because top-10 covers the entire candidate pool. `HR@10 = 1.0` should not be interpreted as recommendation performance.

## 2. Valid Metrics Under 1+5 Candidates

NDCG@10 and MRR remain valid because they distinguish the exact rank of the positive item within the candidate pool. We additionally report HR@1, HR@3, HR@5, NDCG@1, NDCG@3, NDCG@5, positive-rank mean/median, and candidate-pool-size diagnostics.

## 3. Repaired Main Evidence

The main table should use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5. HR@10 is retained only with `hr10_trivial_flag=true`.

{chr(10).join(lines)}

## 4. Claim Text Repair

Use this wording: Scheme 4 / CEP improves NDCG and MRR over three sequential backbones under full Beauty candidate-pool evaluation. HR@10 is not used as primary evidence because the candidate pool contains fewer than 10 negatives.

Do not claim HR@10 improvement as evidence.
"""
    (SUMMARY_DIR / "day26_metric_repair_report.md").write_text(text, encoding="utf-8")


def _copy_medium_5neg() -> None:
    for domain in ["movies", "books", "electronics"]:
        src = Path(f"data/processed/amazon_{domain}_medium")
        dst = Path(f"data/processed/amazon_{domain}_medium_5neg")
        if not src.exists():
            continue
        if dst.exists():
            continue
        shutil.copytree(src, dst)
        stats_path = dst / "split_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["benchmark_variant"] = "medium_5neg"
            stats["hr10_trivial_flag"] = True
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _build_medium_20neg() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows = []
    validation_rows = []
    for domain in ["movies", "books", "electronics"]:
        old_dir = Path(f"data/processed/amazon_{domain}_medium")
        new_dir = Path(f"data/processed/amazon_{domain}_medium_20neg")
        five_dir = Path(f"data/processed/amazon_{domain}_medium_5neg")

        current_stats = _read_stats(old_dir)
        if current_stats.get("num_negatives") != 20:
            stats, validation = build_domain(domain, DOMAINS[domain], requested_users=500, seed=42, num_negatives=20)
        else:
            stats = current_stats
            validation = _validate_existing_medium(old_dir)

        # Copy instead of rename: Windows can deny directory renames if another
        # process briefly holds a handle. Copying keeps this recovery-friendly.
        if new_dir.exists():
            _copy_medium_files(old_dir, new_dir)
        else:
            shutil.copytree(old_dir, new_dir)

        stats["output_dir"] = str(new_dir)
        stats["benchmark_variant"] = "medium_20neg"
        stats["hr10_trivial_flag"] = False
        stats_path = new_dir / "split_stats.json"
        if stats_path.exists():
            stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
            stats_payload.update(
                {
                    "output_dir": str(new_dir),
                    "benchmark_variant": "medium_20neg",
                    "hr10_trivial_flag": False,
                }
            )
            stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
        for row in validation:
            row["processed_path"] = str(new_dir)
            row["benchmark_variant"] = "medium_20neg"

        # Restore the compatibility alias to the original 5-negative medium.
        if five_dir.exists():
            _copy_medium_files(five_dir, old_dir)
        stats_rows.append(stats)
        validation_rows.extend(validation)

    # Restore the 5neg alias as the default medium path for compatibility.
    for domain in ["movies", "books", "electronics"]:
        alias = Path(f"data/processed/amazon_{domain}_medium")
        src = Path(f"data/processed/amazon_{domain}_medium_5neg")
        if not alias.exists() and src.exists():
            shutil.copytree(src, alias)

    stats_df = pd.DataFrame(stats_rows)
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(SUMMARY_DIR / "day28_cross_domain_medium_20neg_schema_validation.csv", index=False)
    cost_rows = _load_5neg_cost_rows()
    for _, row in stats_df.iterrows():
        total = int(row.get("valid_rows", 0)) + int(row.get("test_rows", 0))
        cost_rows.append(
            {
                "domain": row["domain"],
                "variant": "medium_20neg",
                "medium_users": int(row.get("medium_users", 0)),
                "valid_rows": int(row.get("valid_rows", 0)),
                "test_rows": int(row.get("test_rows", 0)),
                "total_api_rows": total,
                "relative_to_beauty_day9": total / BEAUTY_DAY9_ROWS,
                "recommended_day29_mode": "movies_medium_20neg_first"
                if row["domain"] == "movies"
                else f"{row['domain']}_medium_20neg_later",
                "reason": "20-negative medium gives non-trivial HR@10 with API rows close to the 5neg medium.",
            }
        )
    pd.DataFrame(cost_rows).to_csv(SUMMARY_DIR / "day28_cross_domain_medium_5neg_vs_20neg_cost.csv", index=False)
    _write_medium_design_report(validation_df, pd.DataFrame(cost_rows))
    _write_20neg_configs()
    return stats_df, validation_df


def _build_medium_20neg_2000() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows = []
    validation_rows = []
    for domain in ["movies", "books", "electronics"]:
        stats, validation = build_domain(domain, DOMAINS[domain], requested_users=2000, seed=42, num_negatives=20)
        alias_dir = Path(f"data/processed/amazon_{domain}_medium")
        target_dir = Path(f"data/processed/amazon_{domain}_medium_20neg_2000")
        five_dir = Path(f"data/processed/amazon_{domain}_medium_5neg")

        if target_dir.exists():
            _copy_medium_files(alias_dir, target_dir)
        else:
            shutil.copytree(alias_dir, target_dir)

        stats["output_dir"] = str(target_dir)
        stats["benchmark_variant"] = "medium_20neg_2000"
        stats["hr10_trivial_flag"] = False
        stats_path = target_dir / "split_stats.json"
        if stats_path.exists():
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            payload.update(
                {
                    "output_dir": str(target_dir),
                    "benchmark_variant": "medium_20neg_2000",
                    "hr10_trivial_flag": False,
                }
            )
            stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for row in validation:
            row["processed_path"] = str(target_dir)
            row["benchmark_variant"] = "medium_20neg_2000"
        stats_rows.append(stats)
        validation_rows.extend(validation)

        # Keep the default medium alias on the 5-negative continuity split.
        if five_dir.exists():
            _copy_medium_files(five_dir, alias_dir)

    stats_df = pd.DataFrame(stats_rows)
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(SUMMARY_DIR / "day28_cross_domain_medium_20neg_2000_schema_validation.csv", index=False)

    cost_rows = []
    for _, row in stats_df.iterrows():
        total = int(row.get("valid_rows", 0)) + int(row.get("test_rows", 0))
        cost_rows.append(
            {
                "domain": row["domain"],
                "medium_users": int(row.get("medium_users", 0)),
                "negatives_per_positive": 20,
                "valid_rows": int(row.get("valid_rows", 0)),
                "test_rows": int(row.get("test_rows", 0)),
                "total_api_rows": total,
                "relative_to_beauty_day9": total / BEAUTY_DAY9_ROWS,
                "recommended_run_mode": "movies_medium_20neg_2000_first"
                if row["domain"] == "movies"
                else f"{row['domain']}_medium_20neg_2000_hold",
                "reason": "medium_20neg_2000 is the formal cross-domain medium benchmark: enough users and 21 candidates per user; run one domain at a time.",
            }
        )
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(SUMMARY_DIR / "day28_cross_domain_medium_20neg_2000_cost_estimate.csv", index=False)
    _write_20neg_2000_configs()
    _update_medium_design_report_2000()
    return stats_df, validation_df


def _read_stats(path: Path) -> dict:
    stats_path = path / "split_stats.json"
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _copy_medium_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["train.jsonl", "valid.jsonl", "test.jsonl", "sampled_users.json", "split_stats.json", "schema_validation.json"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def _validate_existing_medium(path: Path) -> list[dict]:
    schema_path = path / "schema_validation.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    rows = []
    for split in ["train", "valid", "test"]:
        split_path = path / f"{split}.jsonl"
        count = sum(1 for _ in split_path.open("r", encoding="utf-8")) if split_path.exists() else 0
        rows.append(
            {
                "domain": path.name.replace("amazon_", "").replace("_medium", ""),
                "processed_path": str(path),
                "split": split,
                "num_rows": count,
                "schema_compatible_with_beauty": split_path.exists(),
                "missing_fields": "",
                "notes": "Recovered from existing medium_20neg directory.",
            }
        )
    return rows


def _load_5neg_cost_rows() -> list[dict]:
    rows: list[dict] = []
    path = SUMMARY_DIR / "day28_cross_domain_medium_cost_estimate.csv"
    if not path.exists():
        return rows
    existing = pd.read_csv(path)
    for _, row in existing.iterrows():
        total = int(row.get("total_api_rows", int(row.get("valid_rows", 0)) + int(row.get("test_rows", 0))))
        rows.append(
            {
                "domain": row["domain"],
                "variant": "medium_5neg",
                "medium_users": int(row.get("medium_users", 0)),
                "valid_rows": int(row.get("valid_rows", 0)),
                "test_rows": int(row.get("test_rows", 0)),
                "total_api_rows": total,
                "relative_to_beauty_day9": float(row.get("relative_to_beauty_day9", total / BEAUTY_DAY9_ROWS)),
                "recommended_day29_mode": "do_not_use_hr10_as_primary",
                "reason": "5-negative medium keeps more users at controlled cost, but HR@10 is trivial because each user has 6 candidates.",
            }
        )
    return rows


def _write_20neg_configs() -> None:
    import yaml

    Path("configs/exp").mkdir(parents=True, exist_ok=True)
    Path("configs/external_backbone").mkdir(parents=True, exist_ok=True)
    for domain in ["movies", "books", "electronics"]:
        processed = f"data/processed/amazon_{domain}_medium_20neg"
        exp_config = {
            "exp_name": f"{domain}_deepseek_relevance_evidence_medium_20neg",
            "domain": domain,
            "train_input_path": f"{processed}/train.jsonl",
            "split_input_paths": {"valid": f"{processed}/valid.jsonl", "test": f"{processed}/test.jsonl"},
            "prompt_path": "prompts/candidate_relevance_evidence.txt",
            "output_root": "output-repaired",
            "output_dir": f"output-repaired/{domain}_deepseek_relevance_evidence_medium_20neg",
            "model_config": "configs/model/deepseek.yaml",
            "output_schema": "relevance_evidence",
            "method_variant": "candidate_relevance_evidence_posterior_medium_20neg",
            "resume": True,
            "concurrent": True,
            "max_workers": 4,
            "requests_per_minute": 120,
            "max_retries": 3,
            "retry_backoff_seconds": 2.0,
            "checkpoint_every": 1,
            "max_samples": None,
            "notes": "Day28 20neg medium config only. Do not launch automatically.",
        }
        Path(f"configs/exp/{domain}_deepseek_relevance_evidence_medium_20neg.yaml").write_text(
            yaml.safe_dump(exp_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        backbone_config = {
            "backbone_name": "sasrec",
            "domain": domain,
            "stage": "cross_domain_medium_20neg",
            "train_input_path": f"{processed}/train.jsonl",
            "valid_input_path": f"{processed}/valid.jsonl",
            "test_input_path": f"{processed}/test.jsonl",
            "score_output_path": f"output-repaired/backbone/sasrec_{domain}_medium_20neg/candidate_scores.csv",
            "evidence_table": {
                "path": f"output-repaired/{domain}_deepseek_relevance_evidence_medium_20neg/calibrated/relevance_evidence_posterior_test.jsonl",
                "join_keys": ["user_id", "candidate_item_id"],
                "fields": [
                    "relevance_probability",
                    "calibrated_relevance_probability",
                    "evidence_risk",
                    "ambiguity",
                    "missing_information",
                    "abs_evidence_margin",
                    "positive_evidence",
                    "negative_evidence",
                ],
            },
            "rerank": {
                "top_k": 10,
                "normalizations": ["minmax", "zscore"],
                "lambdas": [0.0, 0.05, 0.1, 0.2, 0.5],
                "alphas": [0.5, 0.75, 0.9],
                "settings": [
                    "Backbone only",
                    "Backbone + calibrated relevance",
                    "Backbone + evidence risk",
                    "Backbone + calibrated relevance + evidence risk",
                ],
                "metric_note": "medium_20neg has 21 candidates per user, so HR@10 is non-trivial.",
            },
        }
        Path(f"configs/external_backbone/{domain}_sasrec_plugin_medium_20neg.yaml").write_text(
            yaml.safe_dump(backbone_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def _write_20neg_2000_configs() -> None:
    import yaml

    Path("configs/exp").mkdir(parents=True, exist_ok=True)
    Path("configs/external_backbone").mkdir(parents=True, exist_ok=True)
    for domain in ["movies", "books", "electronics"]:
        processed = f"data/processed/amazon_{domain}_medium_20neg_2000"
        exp_config = {
            "exp_name": f"{domain}_deepseek_relevance_evidence_medium_20neg_2000",
            "domain": domain,
            "train_input_path": f"{processed}/train.jsonl",
            "split_input_paths": {"valid": f"{processed}/valid.jsonl", "test": f"{processed}/test.jsonl"},
            "prompt_path": "prompts/candidate_relevance_evidence.txt",
            "output_root": "output-repaired",
            "output_dir": f"output-repaired/{domain}_deepseek_relevance_evidence_medium_20neg_2000",
            "model_config": "configs/model/deepseek.yaml",
            "output_schema": "relevance_evidence",
            "method_variant": "candidate_relevance_evidence_posterior_medium_20neg_2000",
            "resume": True,
            "concurrent": True,
            "max_workers": 4,
            "requests_per_minute": 120,
            "max_retries": 3,
            "retry_backoff_seconds": 2.0,
            "checkpoint_every": 1,
            "max_samples": None,
            "notes": "Day28 2000-user 20neg medium-large config only. Do not launch automatically.",
        }
        Path(f"configs/exp/{domain}_deepseek_relevance_evidence_medium_20neg_2000.yaml").write_text(
            yaml.safe_dump(exp_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        backbone_config = {
            "backbone_name": "sasrec",
            "domain": domain,
            "stage": "cross_domain_medium_20neg_2000",
            "train_input_path": f"{processed}/train.jsonl",
            "valid_input_path": f"{processed}/valid.jsonl",
            "test_input_path": f"{processed}/test.jsonl",
            "score_output_path": f"output-repaired/backbone/sasrec_{domain}_medium_20neg_2000/candidate_scores.csv",
            "evidence_table": {
                "path": f"output-repaired/{domain}_deepseek_relevance_evidence_medium_20neg_2000/calibrated/relevance_evidence_posterior_test.jsonl",
                "join_keys": ["user_id", "candidate_item_id"],
                "fields": [
                    "relevance_probability",
                    "calibrated_relevance_probability",
                    "evidence_risk",
                    "ambiguity",
                    "missing_information",
                    "abs_evidence_margin",
                    "positive_evidence",
                    "negative_evidence",
                ],
            },
            "rerank": {
                "top_k": 10,
                "normalizations": ["minmax", "zscore"],
                "lambdas": [0.0, 0.05, 0.1, 0.2, 0.5],
                "alphas": [0.5, 0.75, 0.9],
                "settings": [
                    "Backbone only",
                    "Backbone + calibrated relevance",
                    "Backbone + evidence risk",
                    "Backbone + calibrated relevance + evidence risk",
                ],
                "metric_note": "medium_20neg_2000 has 21 candidates per user, so HR@10 is non-trivial.",
            },
        }
        Path(f"configs/external_backbone/{domain}_sasrec_plugin_medium_20neg_2000.yaml").write_text(
            yaml.safe_dump(backbone_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def _write_medium_design_report(validation_df: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    text = f"""# Day28 Cross-domain Medium Metric Design Report

## 1. Metric Issue

The existing medium_5neg split has one positive and five negatives per user for valid/test, so each user has 6 candidates. HR@10 is therefore trivial and should not be interpreted. The split remains useful for calibration, NDCG, and MRR.

## 2. medium_5neg

Advantages: more users at controlled API cost, stable calibration sample size, direct continuity with Beauty 1+5 candidate-pool evaluation.

Limitations: HR@10 is trivial because candidate_pool_size <= 10.

## 3. medium_20neg

Definition: 500 users per domain, one positive plus 20 negatives per valid/test user, seed=42, regular-domain source data.

Advantages: HR@10 becomes meaningful because each user has 21 candidates. It is closer to conventional ranking evaluation while keeping API rows close to medium_5neg.

Tradeoff: fewer users than medium_5neg, so there are fewer positives for calibration.

## 4. medium_20neg Schema Validation

{_markdown_table(validation_df)}

## 5. Cost Comparison

{_markdown_table(cost_df)}

## 6. Recommendation

Day29 should run Movies medium_20neg relevance evidence first. Books/Electronics medium_20neg are built and validated but should not be launched until Movies confirms the cross-domain pipeline.
"""
    (SUMMARY_DIR / "day28_cross_domain_medium_metric_design_report.md").write_text(text, encoding="utf-8")


def _update_medium_design_report_2000() -> None:
    validation_500 = pd.read_csv(SUMMARY_DIR / "day28_cross_domain_medium_20neg_schema_validation.csv")
    cost_500 = pd.read_csv(SUMMARY_DIR / "day28_cross_domain_medium_5neg_vs_20neg_cost.csv")
    validation_2000 = pd.read_csv(SUMMARY_DIR / "day28_cross_domain_medium_20neg_2000_schema_validation.csv")
    cost_2000 = pd.read_csv(SUMMARY_DIR / "day28_cross_domain_medium_20neg_2000_cost_estimate.csv")
    text = f"""# Day28 Cross-domain Medium Metric Design Report

## 1. Metric Repair

We no longer use HR@10 as a primary metric when the candidate pool has one positive plus five negatives. In that setting each user has 6 candidates, so top-10 covers the entire pool and HR@10 is trivial. HR@10 can be retained only with an explicit triviality flag.

## 2. medium_5neg_2000

The `medium_5neg_2000` continuity split is stored as `data/processed/amazon_*_medium_5neg/` and mirrored by the default `data/processed/amazon_*_medium/` alias. It has 2000 users per domain and 1 positive + 5 negatives per valid/test user. It remains useful for calibration, NDCG, and MRR continuity, but HR@10 must not be interpreted.

## 3. medium_20neg_500

The `medium_20neg_500` split is stored as `data/processed/amazon_*_medium_20neg/`. It has 500 users per domain and 1 positive + 20 negatives per valid/test user. It is a low-cost ranking smoke setting where HR@10 is non-trivial.

## 4. medium_20neg_2000

The `medium_20neg_2000` split is stored as `data/processed/amazon_*_medium_20neg_2000/`. It has 2000 users per domain and 21 candidates per valid/test user. This is the preferred formal cross-domain medium benchmark because both user count and candidate-pool size are large enough for HR@1/3/10, NDCG@3/5/10, and MRR.

## 5. Cost Comparison

`medium_20neg_500` costs about `1.80x` Beauty Day9 rows per domain. `medium_20neg_2000` costs about `7.19x` Beauty Day9 rows per domain. Day29 should not run all domains at once.

### 5neg and 20neg-500

{_markdown_table(cost_500)}

### 20neg-2000

{_markdown_table(cost_2000)}

## 6. Schema Validation: medium_20neg_500

{_markdown_table(validation_500)}

## 7. Schema Validation: medium_20neg_2000

{_markdown_table(validation_2000)}

## 8. Recommendation

Day29 should run Movies `medium_20neg_2000` relevance evidence first. If cost or time pressure is high, run Movies `medium_20neg_500` first as a smoke test, but the final cross-domain main result should prioritize the 2000-user 20-negative benchmark. Books and Electronics are constructed and validated but should wait until Movies confirms the pipeline.
"""
    (SUMMARY_DIR / "day28_cross_domain_medium_metric_design_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    repair_beauty_metrics()
    _copy_medium_5neg()
    _build_medium_20neg()
    _build_medium_20neg_2000()
    print("Metric repair and medium_20neg construction complete.")


if __name__ == "__main__":
    main()
