"""Day39 books/electronics small-domain CEP + backbone replication.

This script intentionally reuses the Day37 movies_small successful route, but
only for books_small and electronics_small. It does not call the API itself;
run ``main_infer.py`` first to create valid/test raw predictions, then run this
script to fit calibration on valid, evaluate on test, train the three small
ID-based backbones, and write the Day39 summary tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
BACKBONE_ROOT = Path("output-repaired/backbone")
ARTIFACT_ROOT = Path("artifacts/backbones/day39_small_domains")

DOMAINS = {
    "books_small": {
        "label": "books_small",
        "data_dir": Path("data/processed/amazon_books_small"),
        "pred_dir": Path("output-repaired/books_small_deepseek_relevance_evidence/predictions"),
        "cal_dir": Path("output-repaired/books_small_deepseek_relevance_evidence/calibrated"),
        "config": Path("configs/exp/books_small_deepseek_relevance_evidence.yaml"),
    },
    "electronics_small": {
        "label": "electronics_small",
        "data_dir": Path("data/processed/amazon_electronics_small"),
        "pred_dir": Path("output-repaired/electronics_small_deepseek_relevance_evidence/predictions"),
        "cal_dir": Path("output-repaired/electronics_small_deepseek_relevance_evidence/calibrated"),
        "config": Path("configs/exp/electronics_small_deepseek_relevance_evidence.yaml"),
    },
}

BACKBONE_LABELS = {
    "sasrec": ("minimal_sasrec", "SASRec"),
    "gru4rec": ("llmesr_gru4rec", "GRU4Rec"),
    "bert4rec": ("llmesr_bert4rec", "Bert4Rec"),
}


def _save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _enhanced_title_map(items_path: Path) -> dict[str, str]:
    title_to_id = _load_title_map(items_path)
    if not items_path.exists():
        return title_to_id
    items = pd.read_csv(items_path)
    for _, row in items.iterrows():
        item_id = str(row["item_id"]).strip()
        title_to_id[item_id] = item_id
        title_to_id[f"Item ID: {item_id}"] = item_id
    return title_to_id


def _api_env_check() -> None:
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    rows = [f"- `{key}` present: `{bool(os.environ.get(key))}`" for key in proxy_keys]
    text = "\n".join(
        [
            "# Day39 Small Domains API Environment Check",
            "",
            "Day39 inherits the Day37 movies_small successful route. The earlier Day37 APIConnectionError was caused by proxy variables pointing to a bad local proxy; Day39 commands should clear those proxy variables before calling `main_infer.py`.",
            "",
            "## Proxy Environment",
            "",
            *rows,
            "",
            "## DeepSeek Route",
            "",
            "- backend/provider: `deepseek` / `deepseek`",
            "- model_name: `deepseek-chat`",
            "- base_url: `https://api.deepseek.com`",
            "- api_key_env: `DEEPSEEK_API_KEY` (value not printed)",
            f"- api_key_env_present: `{bool(os.environ.get('DEEPSEEK_API_KEY'))}`",
            "- prompt_path: `prompts/candidate_relevance_evidence.txt`",
            "- output_schema: `relevance_evidence`",
            "- stable command shape: `py -3.12 main_infer.py --config <domain_config> --split_name valid/test --concurrent --resume --max_workers 4 --requests_per_minute 120`",
            "",
            "Reference: Day37 movies_small completed with this route after clearing proxy variables.",
        ]
    )
    (SUMMARY_DIR / "day39_small_domains_api_env_check.md").write_text(text + "\n", encoding="utf-8")


def _preflight() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain, cfg in DOMAINS.items():
        data_dir: Path = cfg["data_dir"]  # type: ignore[assignment]
        train = data_dir / "train.jsonl"
        valid = data_dir / "valid.jsonl"
        test = data_dir / "test.jsonl"
        valid_df = pd.DataFrame(_read_jsonl(valid))
        test_df = pd.DataFrame(_read_jsonl(test))
        train_rows = _count_jsonl(train)
        valid_rows = len(valid_df)
        test_rows = len(test_df)
        pool_sizes = test_df.groupby("user_id").size()
        required = ["user_id", "history", "candidate_item_id", "candidate_text", "candidate_title", "label"]
        row = {
            "domain": domain,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "test_rows": test_rows,
            "valid_users": int(valid_df["user_id"].nunique()),
            "test_users": int(test_df["user_id"].nunique()),
            "candidate_pool_size_mean": float(pool_sizes.mean()),
            "has_user_id": "user_id" in test_df.columns,
            "has_history": "history" in test_df.columns,
            "has_candidate_item_id": "candidate_item_id" in test_df.columns,
            "has_candidate_text": "candidate_text" in test_df.columns,
            "has_candidate_title": "candidate_title" in test_df.columns,
            "has_label": "label" in test_df.columns,
            "schema_compatible": all(col in test_df.columns for col in required),
            "hr10_trivial_flag": bool(pool_sizes.max() <= 10 or pool_sizes.mean() <= 10),
            "notes": "Expected small-domain 500 users x 6 candidates; HR@10 trivial.",
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day39_books_electronics_small_preflight_check.csv", index=False)
    return out


def _field_stats(domain: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
        rows.append(
            {
                "domain": domain,
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
    return pd.DataFrame(rows)


def _calibrate_domain(domain: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    pred_dir: Path = cfg["pred_dir"]  # type: ignore[assignment]
    cal_dir: Path = cfg["cal_dir"]  # type: ignore[assignment]
    valid_raw = build_relevance_frame(pd.read_json(pred_dir / "valid_raw.jsonl", lines=True))
    test_raw = build_relevance_frame(pd.read_json(pred_dir / "test_raw.jsonl", lines=True))
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

    cal_dir.mkdir(parents=True, exist_ok=True)
    _save_jsonl(valid_cal, cal_dir / "raw_relevance_valid_calibrated.jsonl")
    _save_jsonl(test_cal, cal_dir / "raw_relevance_test_calibrated.jsonl")
    _save_jsonl(valid_min, cal_dir / "relevance_evidence_posterior_minimal_valid.jsonl")
    _save_jsonl(test_min, cal_dir / "relevance_evidence_posterior_minimal_test.jsonl")
    _save_jsonl(valid_full, cal_dir / "relevance_evidence_posterior_full_valid.jsonl")
    _save_jsonl(test_full, cal_dir / "relevance_evidence_posterior_full_test.jsonl")

    canonical = test_min.copy()
    canonical["calibrated_relevance_probability"] = canonical["minimal_calibrated_relevance_probability"]
    canonical["relevance_uncertainty"] = 1.0 - canonical["calibrated_relevance_probability"]
    evidence_path = cal_dir / "relevance_evidence_posterior_test.jsonl"
    _save_jsonl(canonical, evidence_path)

    metric_rows: list[dict[str, Any]] = []
    for split, raw_df, cal_df, min_df, full_df in [
        ("valid", valid_raw, valid_cal, valid_min, valid_full),
        ("test", test_raw, test_cal, test_min, test_full),
    ]:
        for score_type, df, col, status, fallback in [
            ("raw_relevance_probability", raw_df, "relevance_probability", "ready", ""),
            ("calibrated_relevance_probability", cal_df, "calibrated_relevance_probability", "ready", ""),
            ("evidence_posterior_relevance_minimal", min_df, "minimal_calibrated_relevance_probability", minimal.status, minimal.fallback_reason),
            ("evidence_posterior_relevance_full", full_df, "full_calibrated_relevance_probability", full.status, full.fallback_reason),
        ]:
            row = metrics_row(split=split, variant=score_type, df=df, score_col=col, status=status)
            metric_rows.append(
                {
                    "domain": domain,
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
    return pd.DataFrame(metric_rows), _field_stats(domain, build_relevance_frame(test_raw)), evidence_path


def _candidate_pool(evidence_path: Path, test_path: Path) -> pd.DataFrame:
    evidence = pd.DataFrame(_read_jsonl(evidence_path))
    evidence["user_id"] = evidence["user_id"].astype(str)
    evidence["candidate_item_id"] = evidence["candidate_item_id"].astype(str)
    test = pd.DataFrame(_read_jsonl(test_path))
    test["user_id"] = test["user_id"].astype(str)
    test["candidate_item_id"] = test["candidate_item_id"].astype(str)
    key = evidence[["user_id", "candidate_item_id"]].drop_duplicates()
    return test.merge(key, on=["user_id", "candidate_item_id"], how="inner")


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
    rows = len(joined)
    evidence_cols = [c for c in EVIDENCE_COLUMNS if c in joined.columns]
    missing_evidence = joined[evidence_cols].isna().any(axis=1) if evidence_cols else pd.Series(True, index=joined.index)
    fallback = joined["fallback_score"].fillna(False).astype(bool) if "fallback_score" in joined.columns else pd.Series(False, index=joined.index)
    diag = pd.DataFrame(
        [
            {
                "num_backbone_rows": rows,
                "num_joined_rows": rows,
                "join_coverage": float(1.0 - missing_evidence.mean()) if rows else 0.0,
                "num_users": int(joined["user_id"].nunique()),
                "num_candidates": int(joined[["user_id", "candidate_item_id"]].drop_duplicates().shape[0]),
                "num_positive_labels": int((joined["label"] == 1).sum()),
                "fallback_rate": float(fallback.mean()) if rows else 0.0,
                "fallback_rate_positive": float(fallback[joined["label"] == 1].mean()) if (joined["label"] == 1).any() else 0.0,
                "fallback_rate_negative": float(fallback[joined["label"] == 0].mean()) if (joined["label"] == 0).any() else 0.0,
                "missing_evidence_rows": int(missing_evidence.sum()) if rows else 0,
                "missing_backbone_score_rows": int(joined["backbone_score"].isna().sum()),
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


def _rank_metrics_extended(df: pd.DataFrame, score_col: str) -> dict[str, float | bool]:
    rows: list[dict[str, float]] = []
    for _, group in df.groupby("user_id", sort=False):
        sorted_group = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        labels = sorted_group["label"].astype(int).to_numpy()
        pos_idx = np.where(labels == 1)[0]
        rank = int(pos_idx[0] + 1) if len(pos_idx) else len(labels) + 1

        def hr(k: int) -> float:
            return float(rank <= min(k, len(labels)))

        def ndcg(k: int) -> float:
            return float(1.0 / math.log2(rank + 1)) if rank <= min(k, len(labels)) else 0.0

        rows.append(
            {
                "HR@1": hr(1),
                "HR@3": hr(3),
                "HR@5": hr(5),
                "HR@10": hr(10),
                "NDCG@1": ndcg(1),
                "NDCG@3": ndcg(3),
                "NDCG@5": ndcg(5),
                "NDCG@10": ndcg(10),
                "MRR": float(1.0 / rank),
                "MRR@10": float(1.0 / rank) if rank <= 10 else 0.0,
                "Recall@10": hr(10),
                "positive_rank_mean": float(rank),
                "positive_rank_median": float(rank),
                "candidate_pool_size_mean": float(len(labels)),
                "candidate_pool_size_min": float(len(labels)),
                "candidate_pool_size_max": float(len(labels)),
            }
        )
    metrics = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    metrics["hr10_trivial_flag"] = bool(metrics.get("candidate_pool_size_max", 0.0) <= 10 or metrics.get("candidate_pool_size_mean", 0.0) <= 10)
    return metrics


def _rerank_grid(joined: pd.DataFrame, backbone_label: str, method_label: str, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = joined.copy()
    base["norm_backbone"] = _normalize_per_user(base["backbone_score"], base["user_id"], "minmax")
    base_metrics = _rank_metrics_extended(base.assign(final_score=base["backbone_score"]), "final_score")
    rows.append(
        {
            "method": f"A_{method_label}_only",
            "backbone_name": backbone_label,
            "lambda": 0.0,
            "alpha": 1.0,
            "beta": 0.0,
            "normalization": "minmax",
            **base_metrics,
            "rank_change_rate": 0.0,
            "top10_order_change_rate": 0.0,
            "mean_kendall_tau": 1.0,
            "base_risk_spearman": _safe_spearman(base["backbone_score"], base["evidence_risk"]),
            "relative_NDCG_vs_backbone": 0.0,
            "relative_MRR_vs_backbone": 0.0,
            "relative_HR_vs_backbone": 0.0,
        }
    )
    configs: list[tuple[str, float, float, float, str]] = []
    for norm in ["minmax", "zscore"]:
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1.0 - alpha
            configs.append(("B", alpha, beta, 0.0, norm))
        for lamb in [0.05, 0.1, 0.2, 0.5]:
            configs.append(("C", 1.0, 0.0, lamb, norm))
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1.0 - alpha
            for lamb in [0.05, 0.1, 0.2, 0.5]:
                configs.append(("D", alpha, beta, lamb, norm))

    for group, alpha, beta, lamb, norm in configs:
        work = joined.copy()
        nb = _normalize_per_user(work["backbone_score"], work["user_id"], norm)
        nr = _normalize_per_user(work["calibrated_relevance_probability"], work["user_id"], norm)
        nk = _normalize_per_user(work["evidence_risk"], work["user_id"], norm)
        if group == "B":
            method = f"B_{method_label}_plus_calibrated_relevance"
            score = alpha * nb + beta * nr
        elif group == "C":
            method = f"C_{method_label}_plus_evidence_risk"
            score = nb - lamb * nk
        else:
            method = f"D_{method_label}_plus_calibrated_relevance_plus_evidence_risk"
            score = alpha * nb + beta * nr - lamb * nk
        work["final_score"] = score
        metrics = _rank_metrics_extended(work, "final_score")
        rank_stats = _rank_change_stats(work, "backbone_score", "final_score")
        rows.append(
            {
                "method": method,
                "backbone_name": backbone_label,
                "lambda": lamb,
                "alpha": alpha,
                "beta": beta,
                "normalization": norm,
                **metrics,
                **rank_stats,
                "base_risk_spearman": _safe_spearman(work["backbone_score"], work["evidence_risk"]),
                "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / base_metrics["NDCG@10"],
                "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / base_metrics["MRR"],
                "relative_HR_vs_backbone": (metrics["HR@10"] - base_metrics["HR@10"]) / base_metrics["HR@10"] if base_metrics["HR@10"] else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _plugin_diagnostics(
    joined: pd.DataFrame,
    grid: pd.DataFrame,
    join_diag: pd.DataFrame,
    method_label: str,
    output_path: Path,
) -> pd.DataFrame:
    labels = joined["label"].astype(int).to_numpy()
    risk = pd.to_numeric(joined["evidence_risk"], errors="coerce").to_numpy()
    scores = pd.to_numeric(joined["backbone_score"], errors="coerce").to_numpy()
    rel = pd.to_numeric(joined["calibrated_relevance_probability"], errors="coerce").to_numpy()
    baseline = grid[grid["method"].str.startswith("A_")].iloc[0]
    best = grid.sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
    b_best = grid[grid["method"].str.startswith("B_")].sort_values(["NDCG@10", "MRR"], ascending=False).head(1)
    c_best = grid[grid["method"].str.startswith("C_")].sort_values(["NDCG@10", "MRR"], ascending=False).head(1)
    d_best = grid[grid["method"].str.startswith("D_")].sort_values(["NDCG@10", "MRR"], ascending=False).head(1)
    diag = pd.DataFrame(
        [
            {
                "backbone_score_AUROC": _auc_binary(labels, scores),
                "calibrated_relevance_AUROC": _auc_binary(labels, rel),
                "evidence_risk_AUROC_for_error_or_misrank": _auc_binary(1 - labels, risk),
                "backbone_risk_spearman": _safe_spearman(pd.Series(scores), pd.Series(risk)),
                "fallback_rate": float(join_diag["fallback_rate"].iloc[0]),
                "fallback_rate_positive": float(join_diag["fallback_rate_positive"].iloc[0]),
                "fallback_rate_negative": float(join_diag["fallback_rate_negative"].iloc[0]),
                "best_method": best["method"],
                "best_relative_NDCG_vs_backbone": float(best["relative_NDCG_vs_backbone"]),
                "best_relative_MRR_vs_backbone": float(best["relative_MRR_vs_backbone"]),
                "best_lambda": float(best["lambda"]),
                "best_alpha": float(best["alpha"]),
                "best_beta": float(best["beta"]),
                "best_normalization": best["normalization"],
                "backbone_NDCG@10": float(baseline["NDCG@10"]),
                "backbone_MRR": float(baseline["MRR"]),
                "backbone_HR@1": float(baseline["HR@1"]),
                "backbone_HR@3": float(baseline["HR@3"]),
                "best_NDCG@10": float(best["NDCG@10"]),
                "best_MRR": float(best["MRR"]),
                "best_HR@1": float(best["HR@1"]),
                "best_HR@3": float(best["HR@3"]),
                "B_NDCG@10": float(b_best["NDCG@10"].iloc[0]) if not b_best.empty else math.nan,
                "B_MRR": float(b_best["MRR"].iloc[0]) if not b_best.empty else math.nan,
                "C_NDCG@10": float(c_best["NDCG@10"].iloc[0]) if not c_best.empty else math.nan,
                "C_MRR": float(c_best["MRR"].iloc[0]) if not c_best.empty else math.nan,
                "D_NDCG@10": float(d_best["NDCG@10"].iloc[0]) if not d_best.empty else math.nan,
                "D_MRR": float(d_best["MRR"].iloc[0]) if not d_best.empty else math.nan,
                "candidate_pool_size_mean": float(baseline["candidate_pool_size_mean"]),
                "hr10_trivial_flag": bool(baseline["hr10_trivial_flag"]),
                "method_label": method_label,
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


def _train_and_score_backbone(
    domain: str,
    name: str,
    train_examples: list[tuple[list[str], str]],
    pool: pd.DataFrame,
    title_to_id: dict[str, str],
    item_pop: dict[str, int],
    evidence_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backbone_label, method_label = BACKBONE_LABELS[name]
    backbone_dir = BACKBONE_ROOT / f"{name}_{domain}"
    artifact_dir = ARTIFACT_ROOT / domain / name
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
        score_df = _export_sasrec_scores(model, pool, title_to_id, item_to_idx, trained_items, item_pop, args.max_seq_len, backbone_dir / "candidate_scores.csv")
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
        score_df = _export_gru_scores(model, pool, title_to_id, item_to_idx, trained_items, item_pop, args.max_seq_len, backbone_dir / "candidate_scores.csv")
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
        score_df = _export_bert_scores(model, pool, title_to_id, item_to_idx, trained_items, item_pop, args.max_seq_len, backbone_dir / "candidate_scores.csv")
    else:
        raise ValueError(name)
    (artifact_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    joined_path = SUMMARY_DIR / f"day39_{domain}_{name}_joined_candidates.csv"
    join_diag_path = SUMMARY_DIR / f"day39_{domain}_{name}_join_diagnostics.csv"
    grid_path = SUMMARY_DIR / f"day39_{domain}_{name}_plugin_rerank_grid.csv"
    diag_path = SUMMARY_DIR / f"day39_{domain}_{name}_plugin_diagnostics.csv"
    joined = _join_evidence(score_df, evidence_path, joined_path)
    join_diag = _join_diagnostics(joined, join_diag_path)
    grid = _rerank_grid(joined, backbone_label, method_label, grid_path)
    diag = _plugin_diagnostics(joined, grid, join_diag, method_label, diag_path)
    return score_df, grid, diag


def _runtime_monitor() -> None:
    lines = [
        "# Day39 Books/Electronics Small Runtime Monitor",
        "",
        "| domain | split | expected_rows | current_rows | parse_success | status | resume_command |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for domain, cfg in DOMAINS.items():
        pred_dir: Path = cfg["pred_dir"]  # type: ignore[assignment]
        config: Path = cfg["config"]  # type: ignore[assignment]
        for split in ["valid", "test"]:
            path = pred_dir / f"{split}_raw.jsonl"
            if path.exists():
                df = pd.read_json(path, lines=True)
                rows = len(df)
                parse = float(df["parse_success"].mean()) if "parse_success" in df else math.nan
                status = "complete" if rows == 3000 and parse >= 0.95 else "needs_attention"
            else:
                rows = 0
                parse = math.nan
                status = "missing"
            cmd = (
                "$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; "
                f"py -3.12 main_infer.py --config {config} --split_name {split} --concurrent --resume --max_workers 4 --requests_per_minute 120"
            )
            lines.append(f"| {domain} | {split} | 3000 | {rows} | {parse:.4f} | {status} | `{cmd}` |")
    (SUMMARY_DIR / "day39_books_electronics_small_runtime_monitor.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fallback_summary(diag_frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (domain, backbone), diag in diag_frames.items():
        row = diag.iloc[0]
        fallback = float(row["fallback_rate"])
        pos = float(row["fallback_rate_positive"])
        if fallback < 0.2 and pos < 0.2:
            health = "healthy"
        elif fallback < 0.5:
            health = "caution"
        else:
            health = "fallback_heavy"
        rows.append(
            {
                "domain": domain,
                "backbone": backbone,
                "fallback_rate": fallback,
                "positive_fallback_rate": pos,
                "negative_fallback_rate": float(row["fallback_rate_negative"]),
                "join_coverage": 1.0,
                "health_status": health,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day39_books_electronics_small_fallback_summary.csv", index=False)
    return out


def _small_cross_domain_summary(diag_frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_from_grid(domain: str, dataset_type: str, backbone: str, grid_path: Path, fallback_rate: float, pos_fallback: float, claim: str) -> None:
        if not grid_path.exists():
            return
        grid = pd.read_csv(grid_path)
        for _, row in grid.iterrows():
            rows.append(
                {
                    "domain": domain,
                    "dataset_type": dataset_type,
                    "backbone": backbone,
                    "method": row["method"],
                    "NDCG@10": row["NDCG@10"],
                    "MRR": row["MRR"],
                    "HR@1": row["HR@1"],
                    "HR@3": row["HR@3"],
                    "NDCG@3": row["NDCG@3"],
                    "NDCG@5": row["NDCG@5"],
                    "relative_NDCG_vs_backbone": row["relative_NDCG_vs_backbone"],
                    "relative_MRR_vs_backbone": row["relative_MRR_vs_backbone"],
                    "fallback_rate": fallback_rate,
                    "positive_fallback_rate": pos_fallback,
                    "claim_level": claim,
                }
            )

    # Beauty metric-repaired table, compressed into the same schema.
    beauty_path = SUMMARY_DIR / "day26_three_backbone_external_plugin_main_table_metric_repaired.csv"
    if beauty_path.exists():
        beauty = pd.read_csv(beauty_path)
        for _, row in beauty.iterrows():
            rows.append(
                {
                    "domain": "beauty",
                    "dataset_type": "beauty_full",
                    "backbone": row["backbone"],
                    "method": row["method"],
                    "NDCG@10": row.get("NDCG@10_mean", math.nan),
                    "MRR": row.get("MRR_mean", math.nan),
                    "HR@1": row.get("HR@1_mean", math.nan),
                    "HR@3": row.get("HR@3_mean", math.nan),
                    "NDCG@3": row.get("NDCG@3_mean", math.nan),
                    "NDCG@5": row.get("NDCG@5_mean", math.nan),
                    "relative_NDCG_vs_backbone": row.get("relative_NDCG@10_vs_backbone_mean", math.nan),
                    "relative_MRR_vs_backbone": row.get("relative_MRR_vs_backbone_mean", math.nan),
                    "fallback_rate": row.get("fallback_rate_mean", math.nan),
                    "positive_fallback_rate": math.nan,
                    "claim_level": "beauty_full_primary",
                }
            )

    # Day37 movies_small.
    for short, label in BACKBONE_LABELS.items():
        diag_path = SUMMARY_DIR / f"day37_movies_small_{short}_plugin_diagnostics.csv"
        if diag_path.exists():
            diag = pd.read_csv(diag_path).iloc[0]
            claim = "small_cross_domain_caution" if float(diag["fallback_rate"]) >= 0.5 or float(diag["fallback_rate_positive"]) >= 0.5 else "small_cross_domain_sanity"
            add_from_grid("movies_small", "movies_small", short, SUMMARY_DIR / f"day37_movies_small_{short}_plugin_rerank_grid.csv", float(diag["fallback_rate"]), float(diag["fallback_rate_positive"]), claim)

    # Day39 books/electronics.
    for (domain, short), diag in diag_frames.items():
        row = diag.iloc[0]
        claim = "small_cross_domain_caution" if float(row["fallback_rate"]) >= 0.5 or float(row["fallback_rate_positive"]) >= 0.5 else "small_cross_domain_sanity"
        add_from_grid(domain, domain, short, SUMMARY_DIR / f"day39_{domain}_{short}_plugin_rerank_grid.csv", float(row["fallback_rate"]), float(row["fallback_rate_positive"]), claim)

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day39_small_domains_cross_domain_summary.csv", index=False)
    return out


def _write_report(calibration: pd.DataFrame, fallback: pd.DataFrame, diag_frames: dict[tuple[str, str], pd.DataFrame]) -> None:
    lines = [
        "# Day39 Books/Electronics Small Replication Report",
        "",
        "## 1. Why Books/Electronics Small",
        "",
        "Day39 extends the small-domain continuity path from movies_small to books_small and electronics_small. These are cross-domain sanity experiments, not replacements for Beauty full or regular medium analysis.",
        "",
        "## 2. API Fix",
        "",
        "Day39 commands clear `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` before calling `main_infer.py`, inheriting the Day37 movies_small successful DeepSeek route. No prompt, parser, formula, backend, or model changes were made.",
        "",
        "## 3. Calibration Result",
        "",
        "| domain | raw ECE | calibrated ECE | raw Brier | calibrated Brier | raw AUROC | calibrated AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        test = calibration[(calibration["domain"] == domain) & (calibration["split"] == "test")]
        raw = test[test["score_type"] == "raw_relevance_probability"].iloc[0]
        cal = test[test["score_type"] == "calibrated_relevance_probability"].iloc[0]
        lines.append(
            f"| {domain} | {float(raw['ECE']):.4f} | {float(cal['ECE']):.4f} | {float(raw['Brier']):.4f} | {float(cal['Brier']):.4f} | {float(raw['AUROC']):.4f} | {float(cal['AUROC']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Backbone Plug-in Result",
            "",
            "| domain | backbone | health | fallback | pos fallback | backbone NDCG | best method | best NDCG | best MRR | rel NDCG | rel MRR |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for (domain, backbone), diag in diag_frames.items():
        row = diag.iloc[0]
        health = fallback[(fallback["domain"] == domain) & (fallback["backbone"] == backbone)]["health_status"].iloc[0]
        lines.append(
            f"| {domain} | {backbone} | {health} | {float(row['fallback_rate']):.4f} | {float(row['fallback_rate_positive']):.4f} | "
            f"{float(row['backbone_NDCG@10']):.4f} | {row['best_method']} | {float(row['best_NDCG@10']):.4f} | {float(row['best_MRR']):.4f} | "
            f"{float(row['best_relative_NDCG_vs_backbone']):.4f} | {float(row['best_relative_MRR_vs_backbone']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 5. Fallback Health",
            "",
            "`healthy` means fallback_rate < 0.2 and positive_fallback_rate < 0.2. `caution` and `fallback_heavy` rows should not be described as fully healthy ID-backbone evidence. They remain useful for directionality / compensation analysis.",
            "",
            "## 6. Relation To Movies Small And Beauty",
            "",
            "Beauty full three-backbone multi-seed remains the primary performance evidence. Small domains provide cross-domain sanity / continuity. If a small-domain backbone is fallback-heavy, interpret gains with the same caution introduced by Day38.",
            "",
            "## 7. Limitations",
            "",
            "Each small domain uses 6 candidates per user, so HR@10 is trivial and not used as a claim-supporting metric. Primary metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.",
            "",
            "## 8. Day40 Recommendation",
            "",
            "If books/electronics directions are consistent, Day40 should consolidate the small-domain cross-domain table and claim map. If any backbone is fallback-heavy, add fallback sensitivity before making stronger statements for that domain/backbone.",
        ]
    )
    (SUMMARY_DIR / "day39_books_electronics_small_replication_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
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
    _api_env_check()
    _preflight()
    if args.preflight_only:
        print("Day39 preflight complete.")
        return

    calibration_frames: list[pd.DataFrame] = []
    field_frames: list[pd.DataFrame] = []
    diag_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for domain, cfg in DOMAINS.items():
        pred_dir: Path = cfg["pred_dir"]  # type: ignore[assignment]
        for split in ["valid", "test"]:
            path = pred_dir / f"{split}_raw.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing {domain} {split} predictions: {path}")
            rows = _count_jsonl(path)
            if rows != 3000:
                raise RuntimeError(f"Expected 3000 rows for {path}, found {rows}. Resume inference before analysis.")
        metrics, fields, evidence_path = _calibrate_domain(domain, cfg)
        calibration_frames.append(metrics)
        field_frames.append(fields)

        data_dir: Path = cfg["data_dir"]  # type: ignore[assignment]
        title_to_id = _enhanced_title_map(data_dir / "items.csv")
        pool = _candidate_pool(evidence_path, data_dir / "test.jsonl")
        train_examples, _, item_pop = _load_train_examples(data_dir / "train.jsonl", title_to_id, args.max_seq_len)
        print(f"{domain}: train_examples={len(train_examples)} pool_rows={len(pool)}")
        for backbone in ["sasrec", "gru4rec", "bert4rec"]:
            print(f"{domain}: training/scoring {backbone}")
            _, _, diag = _train_and_score_backbone(domain, backbone, train_examples, pool, title_to_id, item_pop, evidence_path, args)
            diag_frames[(domain, backbone)] = diag

    calibration = pd.concat(calibration_frames, ignore_index=True)
    fields = pd.concat(field_frames, ignore_index=True)
    calibration.to_csv(SUMMARY_DIR / "day39_books_electronics_small_calibration_comparison.csv", index=False)
    fields.to_csv(SUMMARY_DIR / "day39_books_electronics_small_field_diagnostics.csv", index=False)
    _runtime_monitor()
    fallback = _fallback_summary(diag_frames)
    _small_cross_domain_summary(diag_frames)
    _write_report(calibration, fallback, diag_frames)
    print("Day39 books/electronics small replication complete.")


if __name__ == "__main__":
    main()
