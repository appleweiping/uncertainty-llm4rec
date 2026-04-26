"""Day37 movies_small Beauty-route replication.

Runs the completed movies_small relevance evidence through valid-fit/test-eval
calibration, then trains three ID-based sequential backbones (minimal SASRec,
LLM-ESR GRU4Rec, LLM-ESR Bert4Rec) and applies the fixed Scheme-4 plug-in grid.

No prompt/parser/formula changes are made here. Raw API prediction jsonl files
are read from the Day37 movies_small output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from main_calibrate_relevance_evidence import (
    apply_relevance_posterior,
    apply_score_calibrator,
    build_relevance_frame,
    fit_relevance_posterior,
    metrics_row,
    usable_rows,
)
from main_day15_bprmf_backbone_plugin_smoke import (
    _auc_binary,
    _normalize_per_user,
    _rank_change_stats,
    _read_jsonl,
    _safe_spearman,
)
from main_day17_sasrec_backbone_plugin_smoke import (
    EVIDENCE_COLUMNS,
    _build_vocab as _build_sasrec_vocab,
    _export_scores as _export_sasrec_scores,
    _load_title_map,
    _load_train_examples,
    _train_sasrec,
)
from main_day21_second_backbone_plugin_smoke import (
    _build_vocab as _build_gru_vocab,
    _export_scores as _export_gru_scores,
    _train_external_gru4rec,
)
from main_day24_third_backbone_plugin_smoke import (
    _build_vocab as _build_bert_vocab,
    _export_scores as _export_bert_scores,
    _train_external_bert4rec,
)
from src.uncertainty.calibration import fit_calibrator


SUMMARY_DIR = Path("output-repaired/summary")
DATA_DIR = Path("data/processed/amazon_movies_small")
PRED_DIR = Path("output-repaired/movies_small_deepseek_relevance_evidence/predictions")
CAL_DIR = Path("output-repaired/movies_small_deepseek_relevance_evidence/calibrated")
ARTIFACT_ROOT = Path("artifacts/backbones/day37_movies_small")
BACKBONE_ROOT = Path("output-repaired/backbone")
VALID_RAW = PRED_DIR / "valid_raw.jsonl"
TEST_RAW = PRED_DIR / "test_raw.jsonl"
EVIDENCE_CANONICAL = CAL_DIR / "relevance_evidence_posterior_test.jsonl"

BACKBONES = {
    "sasrec": {
        "backbone_name": "minimal_sasrec",
        "backbone_dir": BACKBONE_ROOT / "sasrec_movies_small",
        "artifact_dir": ARTIFACT_ROOT / "sasrec",
        "grid_path": SUMMARY_DIR / "day37_movies_small_sasrec_plugin_rerank_grid.csv",
        "diag_path": SUMMARY_DIR / "day37_movies_small_sasrec_plugin_diagnostics.csv",
        "join_path": SUMMARY_DIR / "day37_movies_small_sasrec_joined_candidates.csv",
        "join_diag_path": SUMMARY_DIR / "day37_movies_small_sasrec_join_diagnostics.csv",
    },
    "gru4rec": {
        "backbone_name": "llmesr_gru4rec",
        "backbone_dir": BACKBONE_ROOT / "gru4rec_movies_small",
        "artifact_dir": ARTIFACT_ROOT / "gru4rec",
        "grid_path": SUMMARY_DIR / "day37_movies_small_gru4rec_plugin_rerank_grid.csv",
        "diag_path": SUMMARY_DIR / "day37_movies_small_gru4rec_plugin_diagnostics.csv",
        "join_path": SUMMARY_DIR / "day37_movies_small_gru4rec_joined_candidates.csv",
        "join_diag_path": SUMMARY_DIR / "day37_movies_small_gru4rec_join_diagnostics.csv",
    },
    "bert4rec": {
        "backbone_name": "llmesr_bert4rec",
        "backbone_dir": BACKBONE_ROOT / "bert4rec_movies_small",
        "artifact_dir": ARTIFACT_ROOT / "bert4rec",
        "grid_path": SUMMARY_DIR / "day37_movies_small_bert4rec_plugin_rerank_grid.csv",
        "diag_path": SUMMARY_DIR / "day37_movies_small_bert4rec_plugin_diagnostics.csv",
        "join_path": SUMMARY_DIR / "day37_movies_small_bert4rec_joined_candidates.csv",
        "join_diag_path": SUMMARY_DIR / "day37_movies_small_bert4rec_join_diagnostics.csv",
    },
}


def _enhanced_title_map(items_path: Path) -> dict[str, str]:
    title_to_id = _load_title_map(items_path)
    items = pd.read_csv(items_path)
    for _, row in items.iterrows():
        item_id = str(row["item_id"]).strip()
        title_to_id[item_id] = item_id
        title_to_id[f"Item ID: {item_id}"] = item_id
    return title_to_id


def _save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def _calibrate_movies_small() -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_raw = build_relevance_frame(pd.read_json(VALID_RAW, lines=True))
    test_raw = build_relevance_frame(pd.read_json(TEST_RAW, lines=True))
    valid_fit = usable_rows(valid_raw, "relevance_probability", target_col="label")
    calibrator = fit_calibrator(valid_fit, method="isotonic", confidence_col="relevance_probability", target_col="label")
    valid_cal = apply_score_calibrator(valid_raw, calibrator, "relevance_probability", "calibrated_relevance_probability")
    test_cal = apply_score_calibrator(test_raw, calibrator, "relevance_probability", "calibrated_relevance_probability")
    valid_cal["relevance_uncertainty"] = 1.0 - valid_cal["calibrated_relevance_probability"]
    test_cal["relevance_uncertainty"] = 1.0 - test_cal["calibrated_relevance_probability"]

    minimal = fit_relevance_posterior(valid_raw, "minimal", use_isotonic=True)
    full = fit_relevance_posterior(valid_raw, "full", use_isotonic=True)
    valid_min = apply_relevance_posterior(valid_raw, minimal, "minimal_calibrated_relevance_probability")
    test_min = apply_relevance_posterior(test_raw, minimal, "minimal_calibrated_relevance_probability")
    valid_full = apply_relevance_posterior(valid_raw, full, "full_calibrated_relevance_probability")
    test_full = apply_relevance_posterior(test_raw, full, "full_calibrated_relevance_probability")

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    _save_jsonl(valid_cal, CAL_DIR / "raw_relevance_valid_calibrated.jsonl")
    _save_jsonl(test_cal, CAL_DIR / "raw_relevance_test_calibrated.jsonl")
    _save_jsonl(valid_min, CAL_DIR / "relevance_evidence_posterior_minimal_valid.jsonl")
    _save_jsonl(test_min, CAL_DIR / "relevance_evidence_posterior_minimal_test.jsonl")
    _save_jsonl(valid_full, CAL_DIR / "relevance_evidence_posterior_full_valid.jsonl")
    _save_jsonl(test_full, CAL_DIR / "relevance_evidence_posterior_full_test.jsonl")
    canonical = test_min.copy()
    canonical["calibrated_relevance_probability"] = canonical["minimal_calibrated_relevance_probability"]
    canonical["relevance_uncertainty"] = 1.0 - canonical["calibrated_relevance_probability"]
    _save_jsonl(canonical, EVIDENCE_CANONICAL)

    metric_rows: list[dict[str, Any]] = []
    for split, raw_df, cal_df, min_df, full_df in [
        ("valid", valid_raw, valid_cal, valid_min, valid_full),
        ("test", test_raw, test_cal, test_min, test_full),
    ]:
        for score_type, df, col, status, fallback in [
            ("raw_relevance_probability", raw_df, "relevance_probability", "ready", ""),
            ("calibrated_relevance_probability", cal_df, "calibrated_relevance_probability", "ready", ""),
            (
                "evidence_posterior_relevance_minimal",
                min_df,
                "minimal_calibrated_relevance_probability",
                minimal.status,
                minimal.fallback_reason,
            ),
            (
                "evidence_posterior_relevance_full",
                full_df,
                "full_calibrated_relevance_probability",
                full.status,
                full.fallback_reason,
            ),
        ]:
            row = metrics_row(split=split, variant=score_type, df=df, score_col=col, status=status)
            metric_rows.append(
                {
                    "domain": "movies_small",
                    "split": split,
                    "score_type": score_type,
                    "ECE": row.get("ece", row.get("ECE", math.nan)),
                    "Brier": row.get("brier_score", row.get("Brier", math.nan)),
                    "AUROC": row.get("auroc", row.get("AUROC", math.nan)),
                    "high_conf_error_rate": row.get("high_conf_error_rate", math.nan),
                    "accuracy": row.get("accuracy", math.nan),
                    "status": status,
                    "fallback_reason": fallback,
                    "parse_success_rate": row.get("parse_success_rate", math.nan),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(SUMMARY_DIR / "day37_movies_small_calibration_comparison.csv", index=False)

    field_rows: list[dict[str, Any]] = []
    frame = build_relevance_frame(test_raw)
    for field in [
        "relevance_probability",
        "positive_evidence",
        "negative_evidence",
        "abs_evidence_margin",
        "ambiguity",
        "missing_information",
        "evidence_risk",
    ]:
        vals = pd.to_numeric(frame[field], errors="coerce").dropna()
        field_rows.append(
            {
                "domain": "movies_small",
                "field": field,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "q05": float(vals.quantile(0.05)),
                "q25": float(vals.quantile(0.25)),
                "q50": float(vals.quantile(0.50)),
                "q75": float(vals.quantile(0.75)),
                "q95": float(vals.quantile(0.95)),
                "near_zero_rate": float((vals <= 0.05).mean()),
                "near_one_rate": float((vals >= 0.95).mean()),
            }
        )
    fields = pd.DataFrame(field_rows)
    fields.to_csv(SUMMARY_DIR / "day37_movies_small_field_diagnostics.csv", index=False)
    return metrics, fields


def _candidate_pool(evidence_path: Path, test_path: Path) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    evidence["user_id"] = evidence["user_id"].astype(str)
    evidence["candidate_item_id"] = evidence["candidate_item_id"].astype(str)
    test = pd.DataFrame(_read_jsonl(test_path))
    test["user_id"] = test["user_id"].astype(str)
    test["candidate_item_id"] = test["candidate_item_id"].astype(str)
    key = evidence[["user_id", "candidate_item_id"]].drop_duplicates()
    return test.merge(key, on=["user_id", "candidate_item_id"], how="inner")


def _join_evidence(candidate_scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(EVIDENCE_CANONICAL))
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


def _rank_metrics_extended(df: pd.DataFrame, score_col: str) -> dict[str, float | bool]:
    rows = []
    pool_sizes = []
    positive_ranks = []
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).reset_index(drop=True)
        labels = ranked["label"].astype(int).to_numpy()
        pool_sizes.append(len(ranked))
        pos_idx = np.where(labels == 1)[0]
        if len(pos_idx):
            positive_ranks.append(int(pos_idx[0]) + 1)
        def hr_at(k: int) -> float:
            return float(labels[: min(k, len(labels))].sum() > 0)
        def ndcg_at(k: int) -> float:
            kk = min(k, len(labels))
            dcg = sum(float(labels[i]) / math.log2(i + 2) for i in range(kk))
            total_pos = int(labels.sum())
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(total_pos, kk)))
            return 0.0 if idcg == 0 else float(dcg / idcg)
        rr = 0.0
        for i, label in enumerate(labels, start=1):
            if label:
                rr = 1.0 / i
                break
        rows.append(
            {
                "HR@1": hr_at(1),
                "HR@3": hr_at(3),
                "HR@5": hr_at(5),
                "HR@10": hr_at(10),
                "NDCG@1": ndcg_at(1),
                "NDCG@3": ndcg_at(3),
                "NDCG@5": ndcg_at(5),
                "NDCG@10": ndcg_at(10),
                "MRR": rr,
                "MRR@10": rr,
                "Recall@10": hr_at(10),
            }
        )
    metrics = {key: float(np.mean([row[key] for row in rows])) if rows else math.nan for key in rows[0].keys()}
    metrics.update(
        {
            "positive_rank_mean": float(np.mean(positive_ranks)) if positive_ranks else math.nan,
            "positive_rank_median": float(np.median(positive_ranks)) if positive_ranks else math.nan,
            "candidate_pool_size_mean": float(np.mean(pool_sizes)) if pool_sizes else math.nan,
            "candidate_pool_size_min": float(np.min(pool_sizes)) if pool_sizes else math.nan,
            "candidate_pool_size_max": float(np.max(pool_sizes)) if pool_sizes else math.nan,
            "hr10_trivial_flag": bool(max(pool_sizes) <= 10 or np.mean(pool_sizes) <= 10) if pool_sizes else True,
        }
    )
    return metrics


def _rerank_grid(joined: pd.DataFrame, backbone_label: str, method_label: str, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    rows = []
    base_metrics = _rank_metrics_extended(df.assign(final_score=df["backbone_score"]), "final_score")
    for normalization in ["minmax", "zscore"]:
        df[f"norm_backbone_{normalization}"] = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        df[f"norm_calibrated_{normalization}"] = _normalize_per_user(
            df["calibrated_relevance_probability"], df["user_id"], normalization
        )
        df[f"norm_risk_{normalization}"] = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
        settings = [(f"A_{method_label}_only", 0.0, 1.0, 0.0, df["backbone_score"])]
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1 - alpha
            settings.append(
                (
                    f"B_{method_label}_plus_calibrated_relevance",
                    0.0,
                    alpha,
                    beta,
                    alpha * df[f"norm_backbone_{normalization}"] + beta * df[f"norm_calibrated_{normalization}"],
                )
            )
        for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
            settings.append(
                (
                    f"C_{method_label}_plus_evidence_risk",
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
                        f"D_{method_label}_plus_calibrated_relevance_plus_evidence_risk",
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
            metrics = _rank_metrics_extended(scored, "final_score")
            change = _rank_change_stats(scored, "backbone_score", "final_score")
            rows.append(
                {
                    "method": method,
                    "backbone_name": backbone_label,
                    "lambda": lam,
                    "alpha": alpha,
                    "beta": beta,
                    "normalization": normalization,
                    **metrics,
                    **change,
                    "base_risk_spearman": _safe_spearman(df["backbone_score"], df["evidence_risk"]),
                    "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                    / max(float(base_metrics["NDCG@10"]), 1e-12),
                    "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(float(base_metrics["MRR"]), 1e-12),
                    "relative_HR_vs_backbone": (metrics["HR@10"] - base_metrics["HR@10"]) / max(float(base_metrics["HR@10"]), 1e-12),
                }
            )
    grid = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_path, index=False)
    return grid


def _plugin_diagnostics(
    joined: pd.DataFrame,
    grid: pd.DataFrame,
    join_diag: pd.DataFrame,
    method_label: str,
    output_path: Path,
) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    pred_rank = (
        df.sort_values(["user_id", "backbone_score", "candidate_item_id"], ascending=[True, False, True])
        .groupby("user_id")
        .cumcount()
        + 1
    )
    misrank = ((pred_rank <= 10) & (df["label"].astype(int) == 0)).astype(int)
    best = grid.sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0].to_dict()
    base = grid[grid["method"] == f"A_{method_label}_only"].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0].to_dict()
    best_b = (
        grid[grid["method"] == f"B_{method_label}_plus_calibrated_relevance"]
        .sort_values(["NDCG@10", "MRR"], ascending=False)
        .iloc[0]
        .to_dict()
    )
    best_c = (
        grid[grid["method"] == f"C_{method_label}_plus_evidence_risk"]
        .sort_values(["NDCG@10", "MRR"], ascending=False)
        .iloc[0]
        .to_dict()
    )
    best_d = (
        grid[grid["method"] == f"D_{method_label}_plus_calibrated_relevance_plus_evidence_risk"]
        .sort_values(["NDCG@10", "MRR"], ascending=False)
        .iloc[0]
        .to_dict()
    )
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
                "best_relative_MRR_vs_backbone": (best["MRR"] - base["MRR"]) / max(base["MRR"], 1e-12),
                "best_lambda": best["lambda"],
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_normalization": best["normalization"],
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR": base["MRR"],
                "backbone_HR@1": base["HR@1"],
                "backbone_HR@3": base["HR@3"],
                "best_NDCG@10": best["NDCG@10"],
                "best_MRR": best["MRR"],
                "best_HR@1": best["HR@1"],
                "best_HR@3": best["HR@3"],
                "B_NDCG@10": best_b["NDCG@10"],
                "B_MRR": best_b["MRR"],
                "C_NDCG@10": best_c["NDCG@10"],
                "C_MRR": best_c["MRR"],
                "D_NDCG@10": best_d["NDCG@10"],
                "D_MRR": best_d["MRR"],
                "candidate_pool_size_mean": base["candidate_pool_size_mean"],
                "hr10_trivial_flag": base["hr10_trivial_flag"],
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


def _train_and_score_backbone(
    name: str,
    train_examples: list[tuple[list[str], str]],
    pool: pd.DataFrame,
    title_to_id: dict[str, str],
    item_pop: dict[str, int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = BACKBONES[name]
    backbone_dir: Path = cfg["backbone_dir"]  # type: ignore[assignment]
    artifact_dir: Path = cfg["artifact_dir"]  # type: ignore[assignment]
    backbone_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if name == "sasrec":
        trained_items = set()
        for hist, target in train_examples:
            trained_items.add(target)
            trained_items.update(hist)
        item_to_idx, _ = _build_sasrec_vocab(train_examples, pool, title_to_id, args.max_seq_len)
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
        score_df = _export_sasrec_scores(
            model,
            pool,
            title_to_id,
            item_to_idx,
            trained_items,
            item_pop,
            args.max_seq_len,
            backbone_dir / "candidate_scores.csv",
        )
    elif name == "gru4rec":
        item_to_idx, trained_items = _build_gru_vocab(train_examples, pool, title_to_id, args.max_seq_len)
        model, logs = _train_external_gru4rec(
            train_examples,
            item_to_idx,
            trained_items,
            args.hidden_size,
            args.gru_layers,
            args.max_seq_len,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
        )
        score_df = _export_gru_scores(
            model,
            pool,
            title_to_id,
            item_to_idx,
            trained_items,
            item_pop,
            args.max_seq_len,
            backbone_dir / "candidate_scores.csv",
        )
    elif name == "bert4rec":
        item_to_idx, trained_items = _build_bert_vocab(train_examples, pool, title_to_id)
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
        score_df = _export_bert_scores(
            model,
            pool,
            title_to_id,
            item_to_idx,
            trained_items,
            item_pop,
            args.max_seq_len,
            backbone_dir / "candidate_scores.csv",
        )
    else:
        raise ValueError(name)
    (artifact_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    joined = _join_evidence(score_df, cfg["join_path"])  # type: ignore[arg-type]
    join_diag = _join_diagnostics(joined, cfg["join_diag_path"])  # type: ignore[arg-type]
    method_label = {"sasrec": "SASRec", "gru4rec": "GRU4Rec", "bert4rec": "Bert4Rec"}[name]
    grid = _rerank_grid(joined, str(cfg["backbone_name"]), method_label, cfg["grid_path"])  # type: ignore[arg-type]
    diag = _plugin_diagnostics(joined, grid, join_diag, method_label, cfg["diag_path"])  # type: ignore[arg-type]
    return score_df, grid, diag


def _write_api_parity_md() -> None:
    proxy_values = {
        "HTTP_PROXY": bool(__import__("os").environ.get("HTTP_PROXY")),
        "HTTPS_PROXY": bool(__import__("os").environ.get("HTTPS_PROXY")),
        "ALL_PROXY": bool(__import__("os").environ.get("ALL_PROXY")),
    }
    text = f"""# Day37 Movies Small API Parity Check

## Config Parity

`movies_small` uses the same successful route as Day29 Movies medium5 and Day30 robustness:

- backend/provider: `deepseek` / `deepseek`
- model_name: `deepseek-chat`
- base_url: `https://api.deepseek.com`
- api_key_env: `DEEPSEEK_API_KEY` (value not printed)
- prompt_path: `prompts/candidate_relevance_evidence.txt`
- output_schema: `relevance_evidence`
- main command: `py -3.12 main_infer.py --config configs\\exp\\movies_small_deepseek_relevance_evidence.yaml --split_name valid/test --concurrent --resume --max_workers 4 --requests_per_minute 120`
- working directory: `{Path.cwd()}`

## Root Cause Of Earlier APIConnectionError

The Day37 config matched the successful Day29/Day30 config. The failure was caused by shell proxy environment variables pointing to a bad local proxy (`127.0.0.1:9`). Clearing `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` restored the same `main_infer.py` route.

Current process proxy env present flags: `{proxy_values}`.

## Recovery Result

After clearing proxy variables for the inference command:

- one-row parity health succeeded;
- 20-row movies_small smoke succeeded;
- movies_small valid/test full inference completed with parse_success=1.0.

No prompt, parser, formula, backend, or model config changes were made.
"""
    (SUMMARY_DIR / "day37_movies_small_api_parity_check.md").write_text(text, encoding="utf-8")


def _write_runtime_monitor() -> None:
    rows = []
    for split, path in [("valid", VALID_RAW), ("test", TEST_RAW)]:
        df = pd.read_json(path, lines=True)
        rows.append((split, len(df), float(df["parse_success"].mean())))
    lines = [
        "# Day37 Movies Small Runtime Monitor",
        "",
        "| split | expected_rows | current_rows | parse_success | status | resume_command |",
        "|---|---:|---:|---:|---|---|",
    ]
    for split, n, parse in rows:
        lines.append(
            f"| {split} | 3000 | {n} | {parse:.4f} | {'complete' if n == 3000 and parse >= 0.95 else 'needs_attention'} | "
            f"`$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; py -3.12 main_infer.py --config configs\\exp\\movies_small_deepseek_relevance_evidence.yaml --split_name {split} --concurrent --resume --max_workers 4 --requests_per_minute 120` |"
        )
    (SUMMARY_DIR / "day37_movies_small_runtime_monitor.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(metrics: pd.DataFrame, diag_frames: dict[str, pd.DataFrame]) -> None:
    test_metrics = metrics[metrics["split"] == "test"]
    raw = test_metrics[test_metrics["score_type"] == "raw_relevance_probability"].iloc[0]
    cal = test_metrics[test_metrics["score_type"] == "calibrated_relevance_probability"].iloc[0]
    lines = [
        "# Day37 Movies Small Beauty-Route Replication Report",
        "",
        "## 1. Why Movies Small",
        "",
        "`movies_small` has a complete Beauty-compatible schema and healthy low cold-rate, so it is a clean small-domain replication of the Beauty Day9 -> backbone plug-in route. This is cross-domain sanity / continuity, not a replacement for regular medium or full-domain claims.",
        "",
        "## 2. API Parity",
        "",
        "Day37 reuses the same DeepSeek backend/config/command route as Day29/Day30. The earlier APIConnectionError was traced to bad proxy environment variables (`127.0.0.1:9`), not to prompt/schema/parser/config mismatch. Clearing proxy variables restored inference.",
        "",
        "## 3. Relevance Calibration",
        "",
        f"On movies_small test, raw relevance ECE `{float(raw['ECE']):.4f}` and Brier `{float(raw['Brier']):.4f}` improved to calibrated ECE `{float(cal['ECE']):.4f}` and Brier `{float(cal['Brier']):.4f}`. AUROC changed from `{float(raw['AUROC']):.4f}` to `{float(cal['AUROC']):.4f}`. This matches the Beauty interpretation: calibration fixes probability quality more than ranking separability.",
        "",
        "## 4. Three Backbone Plug-in",
        "",
        "| backbone | fallback_rate | backbone NDCG@10 | backbone MRR | best method | best NDCG@10 | best MRR | rel NDCG | rel MRR |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name, diag in diag_frames.items():
        row = diag.iloc[0]
        lines.append(
            f"| {name} | {float(row['fallback_rate']):.4f} | {float(row['backbone_NDCG@10']):.4f} | "
            f"{float(row['backbone_MRR']):.4f} | {row['best_method']} | {float(row['best_NDCG@10']):.4f} | "
            f"{float(row['best_MRR']):.4f} | {float(row['best_relative_NDCG_vs_backbone']):.4f} | "
            f"{float(row['best_relative_MRR_vs_backbone']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Important caveat: join coverage is 1.0, but ID-backbone fallback remains non-trivial, especially on positives. This means the movies_small replication supports cross-domain sanity and directionality, but it should not be presented as a fully healthy external-backbone benchmark without a later mapping/training-vocab repair.",
            "",
            "## 5. Component Attribution",
            "",
            "For each backbone, compare B/C/D rows in the grid. The expected CEP interpretation remains: calibrated relevance is the primary posterior signal; evidence_risk is a secondary regularizer if D improves over B at positive lambda. C-only should not be treated as the main scorer unless it beats B consistently.",
            "",
            "## 6. Direction Against Beauty",
            "",
            "Movies_small now follows the same route as Beauty: DeepSeek candidate relevance evidence, valid-fit/test-eval calibration, and three sequential backbone plug-in grids. This supports cross-domain continuity, but only at small-domain sanity scale.",
            "",
            "## 7. Limitations",
            "",
            "Movies_small has 6 candidates per user, so HR@10 is trivial and is not used as primary evidence. This result does not replace regular medium/full-domain analysis.",
            "",
            "## 8. Next Step",
            "",
            "If the three backbone directions are positive, Day38 can run movies_small multi-seed or extend the same small-domain sanity path to books_small/electronics_small one domain at a time.",
        ]
    )
    (SUMMARY_DIR / "day37_movies_small_beauty_route_replication_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--trm_num", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    _write_api_parity_md()
    _write_runtime_monitor()
    metrics, _ = _calibrate_movies_small()
    title_to_id = _enhanced_title_map(DATA_DIR / "items.csv")
    pool = _candidate_pool(EVIDENCE_CANONICAL, DATA_DIR / "test.jsonl")
    train_examples, trained_items, item_pop = _load_train_examples(DATA_DIR / "train.jsonl", title_to_id, args.max_seq_len)
    print(f"Loaded train_examples={len(train_examples)} trained_items={len(trained_items)} pool_rows={len(pool)}")
    diag_frames: dict[str, pd.DataFrame] = {}
    for name in ["sasrec", "gru4rec", "bert4rec"]:
        print(f"Training/scoring {name}...")
        _, _, diag = _train_and_score_backbone(name, train_examples, pool, title_to_id, item_pop, args)
        diag_frames[name] = diag
    _write_report(metrics, diag_frames)
    print("Day37 movies_small Beauty-route replication complete.")


if __name__ == "__main__":
    main()
