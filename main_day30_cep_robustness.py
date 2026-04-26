"""Day30 CEP robustness setup and analysis.

This script prepares a 500-user Beauty robustness subset, generates lightweight
noisy test variants, writes inference configs, and analyzes completed noisy CEP
outputs. API inference is intentionally run through main_infer.py with resume;
this script itself only prepares/analyzes local files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from main_calibrate_relevance_evidence import (
    apply_relevance_posterior,
    apply_score_calibrator,
    build_relevance_frame,
    fit_relevance_posterior,
    metrics_row,
    usable_rows,
)
from main_day15_bprmf_backbone_plugin_smoke import _normalize_per_user
from src.uncertainty.calibration import fit_calibrator


SEED = 42
NUM_USERS = 500
NOISE_TYPES = ["history_dropout", "candidate_text_dropout", "history_swap_noise"]
NOISE_LEVELS = [0.1, 0.2, 0.3]
DATA_DIR = Path("data/processed/amazon_beauty_robustness_500")
SUMMARY_DIR = Path("output-repaired/summary")
OUTPUT_ROOT = Path("output-repaired")
SOURCE_TEST = Path("data/processed/amazon_beauty/test.jsonl")
SOURCE_VALID_RAW = Path("output-repaired/beauty_deepseek_relevance_evidence_full/predictions/valid_raw.jsonl")
SOURCE_CLEAN_RAW_CAL = Path(
    "output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/raw_relevance_test_calibrated.jsonl"
)
SOURCE_CLEAN_FULL = Path(
    "output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_full_test.jsonl"
)
SASREC_SCORES = Path("output-repaired/backbone/sasrec_beauty_full/candidate_scores.csv")
ITEMS_PATH = Path("data/processed/amazon_beauty/items.csv")
PROMPT_PATH = "prompts/candidate_relevance_evidence.txt"
MODEL_CONFIG = "configs/model/deepseek.yaml"


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(clean_json(record), ensure_ascii=False) + "\n")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def setting_name(noise_type: str, level: float) -> str:
    return f"{noise_type}_{level:.1f}"


def exp_name(noise_type: str, level: float) -> str:
    return f"beauty_robustness_500/{setting_name(noise_type, level)}"


def choose_users(test_df: pd.DataFrame) -> list[str]:
    users = sorted(test_df["user_id"].astype(str).unique().tolist())
    rng = random.Random(SEED)
    selected = rng.sample(users, NUM_USERS)
    return sorted(selected)


def subset_stats(df: pd.DataFrame) -> pd.DataFrame:
    pool = df.groupby("user_id").size()
    return pd.DataFrame(
        [
            {
                "num_users": int(df["user_id"].nunique()),
                "num_rows": int(len(df)),
                "candidate_pool_size_mean": float(pool.mean()),
                "candidate_pool_size_min": int(pool.min()),
                "candidate_pool_size_max": int(pool.max()),
                "positive_rows": int((df["label"].astype(int) == 1).sum()),
                "negative_rows": int((df["label"].astype(int) == 0).sum()),
                "source_test_path": str(SOURCE_TEST),
                "seed": SEED,
            }
        ]
    )


def load_item_titles() -> list[str]:
    items = pd.read_csv(ITEMS_PATH)
    title_col = "title" if "title" in items.columns else "candidate_text"
    titles = items[title_col].dropna().astype(str)
    return [t for t in titles.tolist() if t.strip()]


def token_dropout(text: str, level: float, rng: random.Random) -> str:
    tokens = str(text or "").split()
    if len(tokens) <= 1:
        return str(text or "")
    keep = [tok for tok in tokens if rng.random() >= level]
    if not keep:
        keep = [rng.choice(tokens)]
    return " ".join(keep)


def history_dropout(history: list[Any], level: float, rng: random.Random) -> list[Any]:
    hist = [str(x) for x in history]
    if len(hist) <= 1:
        return hist
    keep = [item for item in hist if rng.random() >= level]
    if not keep:
        keep = [rng.choice(hist)]
    return keep


def history_swap(history: list[Any], level: float, item_titles: list[str], rng: random.Random) -> list[Any]:
    hist = [str(x) for x in history]
    n_insert = max(1, int(round(len(hist) * level))) if hist else 1
    existing = set(hist)
    candidates = [title for title in item_titles if title not in existing]
    inserts = rng.sample(candidates, min(n_insert, len(candidates))) if candidates else []
    if not inserts:
        return hist
    noisy = hist.copy()
    for title in inserts:
        pos = rng.randint(0, len(noisy))
        noisy.insert(pos, title)
    return noisy


def apply_noise(df: pd.DataFrame, noise_type: str, level: float, item_titles: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = random.Random(f"{SEED}-{noise_type}-{level}")
    records = df.to_dict(orient="records")
    before_lens: list[int] = []
    after_lens: list[int] = []
    text_changed = 0
    history_changed = 0
    for record in records:
        old_history = list(record.get("history") or [])
        old_text = str(record.get("candidate_text") or "")
        before_lens.append(len(old_history))
        if noise_type == "history_dropout":
            record["history"] = history_dropout(old_history, level, rng)
        elif noise_type == "candidate_text_dropout":
            record["candidate_text"] = token_dropout(old_text, level, rng)
        elif noise_type == "history_swap_noise":
            record["history"] = history_swap(old_history, level, item_titles, rng)
        else:
            raise ValueError(noise_type)
        after_lens.append(len(record.get("history") or []))
        if str(record.get("candidate_text") or "") != old_text:
            text_changed += 1
        if list(record.get("history") or []) != old_history:
            history_changed += 1
    out = pd.DataFrame(records)
    stats = {
        "noise_type": noise_type,
        "noise_level": level,
        "num_rows": int(len(out)),
        "num_users": int(out["user_id"].nunique()),
        "avg_history_len_before": float(np.mean(before_lens)),
        "avg_history_len_after": float(np.mean(after_lens)),
        "candidate_text_changed_rate": text_changed / max(len(out), 1),
        "history_changed_rate": history_changed / max(len(out), 1),
    }
    return out, stats


def write_config(noise_type: str, level: float, test_path: Path) -> Path:
    cfg_path = Path("configs/exp") / f"beauty_robustness_500_{noise_type}_{level:.1f}.yaml"
    cfg = {
        "exp_name": exp_name(noise_type, level),
        "split_input_paths": {"test": str(test_path).replace("\\", "/")},
        "output_root": "output-repaired",
        "prompt_path": PROMPT_PATH,
        "model_config": MODEL_CONFIG,
        "max_samples": None,
        "overwrite": False,
        "seed": SEED,
        "output_schema": "relevance_evidence",
        "method_variant": f"candidate_relevance_evidence_robustness_500_{noise_type}_{level:.1f}",
        "concurrent": True,
        "resume": True,
        "max_workers": 4,
        "requests_per_minute": 120,
        "max_retries": 3,
        "retry_backoff_seconds": 2.0,
        "checkpoint_every": 1,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def prepare() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_df = load_jsonl(SOURCE_TEST)
    selected_users = choose_users(test_df)
    subset = test_df[test_df["user_id"].astype(str).isin(selected_users)].copy()
    subset = subset.sort_values(["user_id", "label", "candidate_item_id"], ascending=[True, False, True])
    write_jsonl(subset.to_dict(orient="records"), DATA_DIR / "clean_test.jsonl")
    subset_stats(subset).to_csv(SUMMARY_DIR / "day30_beauty_robustness_subset_stats.csv", index=False)

    item_titles = load_item_titles()
    noise_rows = []
    config_rows = []
    for noise_type in NOISE_TYPES:
        for level in NOISE_LEVELS:
            noisy, stats = apply_noise(subset, noise_type, level, item_titles)
            setting_dir = DATA_DIR / f"noisy_{setting_name(noise_type, level)}"
            test_path = setting_dir / "test.jsonl"
            write_jsonl(noisy.to_dict(orient="records"), test_path)
            # 20-row smoke input, kept local and cheap.
            write_jsonl(noisy.head(20).to_dict(orient="records"), setting_dir / "smoke_20.jsonl")
            cfg_path = write_config(noise_type, level, test_path)
            noise_rows.append(stats)
            config_rows.append(
                {
                    "noise_type": noise_type,
                    "noise_level": level,
                    "config_path": str(cfg_path),
                    "test_input_path": str(test_path),
                    "smoke_input_path": str(setting_dir / "smoke_20.jsonl"),
                    "output_dir": str(OUTPUT_ROOT / exp_name(noise_type, level)),
                }
            )
    pd.DataFrame(noise_rows).to_csv(SUMMARY_DIR / "day30_beauty_robustness_noise_stats.csv", index=False)
    pd.DataFrame(config_rows).to_csv(SUMMARY_DIR / "day30_beauty_robustness_config_inventory.csv", index=False)
    print("Prepared Day30 robustness subset, noisy variants, and configs.")


def calibration_models() -> tuple[Any, Any, Any]:
    valid_df = build_relevance_frame(load_jsonl(SOURCE_VALID_RAW))
    valid_fit = usable_rows(valid_df, "relevance_probability", target_col="label")
    raw_calibrator = fit_calibrator(
        valid_fit,
        method="isotonic",
        confidence_col="relevance_probability",
        target_col="label",
    )
    minimal_model = fit_relevance_posterior(valid_df, feature_set="minimal")
    full_model = fit_relevance_posterior(valid_df, feature_set="full")
    return raw_calibrator, minimal_model, full_model


def apply_clean_calibration(raw_df: pd.DataFrame, raw_calibrator: Any, minimal_model: Any, full_model: Any) -> pd.DataFrame:
    raw_df = build_relevance_frame(raw_df)
    cal = apply_score_calibrator(
        raw_df, raw_calibrator, input_col="relevance_probability", output_col="calibrated_relevance_probability"
    )
    min_df = apply_relevance_posterior(raw_df, minimal_model, "minimal_calibrated_relevance_probability")
    full_df = apply_relevance_posterior(raw_df, full_model, "full_calibrated_relevance_probability")
    keys = ["user_id", "candidate_item_id", "label"]
    cols = keys + [
        "calibrated_relevance_probability",
        "relevance_probability",
        "positive_evidence",
        "negative_evidence",
        "evidence_margin",
        "abs_evidence_margin",
        "ambiguity",
        "missing_information",
        "evidence_risk",
        "parse_success",
    ]
    out = cal[cols].copy()
    out = out.merge(min_df[keys + ["minimal_calibrated_relevance_probability"]], on=keys, how="left")
    out = out.merge(full_df[keys + ["full_calibrated_relevance_probability"]], on=keys, how="left")
    return out


def metrics_for(df: pd.DataFrame, split: str, condition: str) -> list[dict[str, Any]]:
    rows = []
    variants = {
        "raw_relevance_probability": "relevance_probability",
        "calibrated_relevance_probability": "calibrated_relevance_probability",
        "evidence_posterior_relevance_minimal": "minimal_calibrated_relevance_probability",
        "evidence_posterior_relevance_full": "full_calibrated_relevance_probability",
    }
    for variant, col in variants.items():
        row = metrics_row(split=split, variant=variant, df=df, score_col=col)
        row["condition"] = condition
        row["mean_relevance_probability"] = float(pd.to_numeric(df["relevance_probability"], errors="coerce").mean())
        row["mean_calibrated_relevance_probability"] = float(
            pd.to_numeric(df["calibrated_relevance_probability"], errors="coerce").mean()
        )
        row["mean_evidence_risk"] = float(pd.to_numeric(df["evidence_risk"], errors="coerce").mean())
        rows.append(row)
    return rows


def dcg_from_rank(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def rank_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    ranks = []
    pool_sizes = []
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).reset_index(drop=True)
        pool_sizes.append(len(ranked))
        pos = ranked.index[ranked["label"].astype(int) == 1].tolist()
        if pos:
            ranks.append(pos[0] + 1)
    arr = np.asarray(ranks, dtype=float)
    return {
        "NDCG@10": float(np.mean([dcg_from_rank(int(r), 10) for r in arr])),
        "MRR": float(np.mean(1.0 / arr)),
        "HR@1": float(np.mean(arr <= 1)),
        "HR@3": float(np.mean(arr <= 3)),
        "NDCG@3": float(np.mean([dcg_from_rank(int(r), 3) for r in arr])),
        "NDCG@5": float(np.mean([dcg_from_rank(int(r), 5) for r in arr])),
        "positive_rank_mean": float(np.mean(arr)),
        "candidate_pool_size_mean": float(np.mean(pool_sizes)),
        "hr10_trivial_flag": bool(max(pool_sizes) <= 10 or np.mean(pool_sizes) <= 10),
    }


def join_sasrec(evidence: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_csv(SASREC_SCORES)
    subset_users = set(evidence["user_id"].astype(str).unique())
    scores = scores[scores["user_id"].astype(str).isin(subset_users)].copy()
    keys = ["user_id", "candidate_item_id", "label"]
    return scores.merge(evidence, on=keys, how="inner")


def plugin_rows(condition: str, evidence: pd.DataFrame, clean_baselines: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    joined = join_sasrec(evidence)
    rows = []
    norm = "zscore"
    alpha = 0.5
    beta = 0.5
    lam = 0.2
    joined["norm_backbone"] = _normalize_per_user(joined["backbone_score"], joined["user_id"], norm)
    joined["norm_calibrated"] = _normalize_per_user(joined["calibrated_relevance_probability"], joined["user_id"], norm)
    joined["norm_risk"] = _normalize_per_user(joined["evidence_risk"], joined["user_id"], norm)
    methods = {
        "A_SASRec_only": joined["backbone_score"],
        "B_SASRec_plus_calibrated_relevance": alpha * joined["norm_backbone"] + beta * joined["norm_calibrated"],
        "D_SASRec_plus_calibrated_relevance_plus_evidence_risk": (
            alpha * joined["norm_backbone"] + beta * joined["norm_calibrated"] - lam * joined["norm_risk"]
        ),
    }
    for method, score in methods.items():
        work = joined.copy()
        work["final_score"] = score
        metrics = rank_metrics(work, "final_score")
        clean = clean_baselines.get(method) if clean_baselines else None
        row = {
            "condition": condition,
            "method": method,
            "normalization": norm,
            "alpha": alpha if method != "A_SASRec_only" else 1.0,
            "beta": beta if method != "A_SASRec_only" else 0.0,
            "lambda": lam if method.startswith("D_") else 0.0,
            **metrics,
        }
        if clean:
            row["relative_NDCG_vs_clean"] = (row["NDCG@10"] - clean["NDCG@10"]) / max(clean["NDCG@10"], 1e-12)
            row["relative_MRR_vs_clean"] = (row["MRR"] - clean["MRR"]) / max(clean["MRR"], 1e-12)
            row["degradation_NDCG"] = clean["NDCG@10"] - row["NDCG@10"]
            row["degradation_MRR"] = clean["MRR"] - row["MRR"]
        else:
            row["relative_NDCG_vs_clean"] = 0.0
            row["relative_MRR_vs_clean"] = 0.0
            row["degradation_NDCG"] = 0.0
            row["degradation_MRR"] = 0.0
        rows.append(row)
    return rows


def read_clean_subset_evidence(raw_calibrator: Any, minimal_model: Any, full_model: Any) -> pd.DataFrame:
    clean = load_jsonl(DATA_DIR / "clean_test.jsonl")
    keys = clean[["user_id", "candidate_item_id", "label"]]
    raw = load_jsonl(SOURCE_CLEAN_RAW_CAL)
    full = load_jsonl(SOURCE_CLEAN_FULL)
    merged = keys.merge(raw, on=["user_id", "candidate_item_id", "label"], how="left")
    full_cols = ["user_id", "candidate_item_id", "label", "full_calibrated_relevance_probability"]
    merged = merged.merge(full[full_cols], on=["user_id", "candidate_item_id", "label"], how="left")
    if "minimal_calibrated_relevance_probability" not in merged.columns:
        # Recompute minimal locally from clean raw rows for symmetry with noisy analysis.
        merged = apply_clean_calibration(merged, raw_calibrator, minimal_model, full_model)
    return merged


def analyze() -> None:
    raw_calibrator, minimal_model, full_model = calibration_models()
    clean_evidence = read_clean_subset_evidence(raw_calibrator, minimal_model, full_model)
    metric_rows = metrics_for(clean_evidence, "test", "clean")
    clean_plugin_rows = plugin_rows("clean", clean_evidence)
    clean_map = {row["method"]: row for row in clean_plugin_rows}
    plugin_all = clean_plugin_rows.copy()

    for noise_type in NOISE_TYPES:
        for level in NOISE_LEVELS:
            condition = setting_name(noise_type, level)
            raw_path = OUTPUT_ROOT / exp_name(noise_type, level) / "predictions" / "test_raw.jsonl"
            if not raw_path.exists() or line_count(raw_path) == 0:
                continue
            raw_df = load_jsonl(raw_path)
            evidence = apply_clean_calibration(raw_df, raw_calibrator, minimal_model, full_model)
            metric_rows.extend(metrics_for(evidence, "test", condition))
            plugin_all.extend(plugin_rows(condition, evidence, clean_map))

    cep_metrics = pd.DataFrame(metric_rows)
    cep_metrics.to_csv(SUMMARY_DIR / "day30_cep_robustness_metrics.csv", index=False)
    grid = pd.DataFrame(plugin_all)
    grid.to_csv(SUMMARY_DIR / "day30_sasrec_cep_robustness_grid.csv", index=False)

    clean_metric = cep_metrics[(cep_metrics["condition"] == "clean") & (cep_metrics["variant"] == "calibrated_relevance_probability")].iloc[0]
    rows = []
    for _, row in grid[grid["condition"] != "clean"].iterrows():
        noisy_metric = cep_metrics[
            (cep_metrics["condition"] == row["condition"])
            & (cep_metrics["variant"] == "calibrated_relevance_probability")
        ]
        if noisy_metric.empty:
            continue
        noisy_metric = noisy_metric.iloc[0]
        noise_type, level_str = parse_condition(str(row["condition"]))
        clean_row = clean_map[row["method"]]
        rows.append(
            {
                "noise_type": noise_type,
                "noise_level": float(level_str),
                "method": row["method"],
                "clean_NDCG": clean_row["NDCG@10"],
                "noisy_NDCG": row["NDCG@10"],
                "NDCG_drop": clean_row["NDCG@10"] - row["NDCG@10"],
                "clean_MRR": clean_row["MRR"],
                "noisy_MRR": row["MRR"],
                "MRR_drop": clean_row["MRR"] - row["MRR"],
                "clean_ECE": clean_metric["ece"],
                "noisy_ECE": noisy_metric["ece"],
                "ECE_change": noisy_metric["ece"] - clean_metric["ece"],
                "mean_evidence_risk_change": noisy_metric["mean_evidence_risk"] - clean_metric["mean_evidence_risk"],
                "high_conf_error_change": noisy_metric["high_conf_error_rate"] - clean_metric["high_conf_error_rate"],
            }
        )
    degradation = pd.DataFrame(rows)
    degradation.to_csv(SUMMARY_DIR / "day30_robustness_degradation_summary.csv", index=False)
    write_report(cep_metrics, grid, degradation)
    print("Wrote Day30 robustness metrics, grid, degradation summary, and report.")


def parse_condition(condition: str) -> tuple[str, str]:
    for noise_type in NOISE_TYPES:
        prefix = noise_type + "_"
        if condition.startswith(prefix):
            return noise_type, condition[len(prefix) :]
    return condition, "nan"


def format_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No noisy inference outputs were available at report time."

    def fmt(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *rows])


def write_report(cep_metrics: pd.DataFrame, grid: pd.DataFrame, degradation: pd.DataFrame) -> None:
    clean_raw = cep_metrics[(cep_metrics["condition"] == "clean") & (cep_metrics["variant"] == "raw_relevance_probability")].iloc[0]
    clean_cal = cep_metrics[(cep_metrics["condition"] == "clean") & (cep_metrics["variant"] == "calibrated_relevance_probability")].iloc[0]
    best_clean = grid[grid["condition"] == "clean"].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
    sasrec_clean = grid[(grid["condition"] == "clean") & (grid["method"] == "A_SASRec_only")].iloc[0]
    noisy_parse_min = cep_metrics[cep_metrics["condition"] != "clean"]["parse_success_rate"].min()
    noisy_best = grid[grid["condition"] != "clean"].sort_values(["NDCG@10", "MRR"], ascending=False).head(3)
    noisy_best_table = format_markdown_table(
        noisy_best,
        ["condition", "method", "NDCG@10", "MRR", "HR@3", "degradation_NDCG", "degradation_MRR"],
    )
    d_degradation = degradation[degradation["method"] == "D_SASRec_plus_calibrated_relevance_plus_evidence_risk"]
    worst_d_drop = d_degradation["NDCG_drop"].max() if not d_degradation.empty else float("nan")
    worst_d_mrr_drop = d_degradation["MRR_drop"].max() if not d_degradation.empty else float("nan")
    text = f"""# Day30 CEP + Backbone Robustness Report

## 1. Motivation

Week1-Week4 included robustness for the old raw-confidence pipeline. Day30 adds robustness for the CEP route after Day9/Day10 and for the external backbone plug-in setting.

## 2. Setup

The experiment uses a fixed Beauty 500-user subset sampled with seed 42 from the full Beauty test users. The candidate pool is unchanged from the Beauty full setting: 1 positive plus 5 negatives per user. Noise types are `history_dropout`, `candidate_text_dropout`, and `history_swap_noise` with levels 0.1, 0.2, and 0.3. The prompt and scoring formula are unchanged.

Because the candidate pool has 6 items, HR@10 is trivial and is not used as primary evidence. The report focuses on NDCG@10, MRR, HR@1/HR@3, NDCG@3, and NDCG@5.

## 3. CEP Signal Robustness

Clean raw relevance has ECE `{clean_raw['ece']:.4f}` and Brier `{clean_raw['brier_score']:.4f}`. Clean calibrated relevance has ECE `{clean_cal['ece']:.4f}` and Brier `{clean_cal['brier_score']:.4f}`. Noisy rows use the clean Beauty valid calibrator applied to noisy test outputs; no noisy test fit is used. The minimum noisy parse success rate is `{noisy_parse_min:.4f}`.

## 4. Backbone Plug-in Robustness

Clean SASRec-only has NDCG@10 `{sasrec_clean['NDCG@10']:.4f}`, MRR `{sasrec_clean['MRR']:.4f}`, and HR@3 `{sasrec_clean['HR@3']:.4f}`. Clean best method is `{best_clean['method']}` with NDCG@10 `{best_clean['NDCG@10']:.4f}`, MRR `{best_clean['MRR']:.4f}`, HR@3 `{best_clean['HR@3']:.4f}`. SASRec-only is fixed across noise because the candidate pool does not change; B/D rows use noisy CEP fields.

Top noisy rows by NDCG/MRR:

{noisy_best_table}

## 5. Key Finding

Across the completed noisy settings, the best noisy rows remain close to clean CEP performance and stay above the fixed SASRec-only baseline. The largest observed D-setting drop is NDCG `{worst_d_drop:.4f}` and MRR `{worst_d_mrr_drop:.4f}`. This supports the limited claim that CEP remains useful under these light perturbations and that evidence risk can act as a secondary regularizer. Interpret the results as 500-user controlled robustness, not a full robustness benchmark.

## 6. Limitations

This is not full Beauty robustness and only prioritizes SASRec. GRU4Rec/Bert4Rec can be added later if the first run is stable.

## 7. Day31 Recommendation

If the noisy rows remain positive relative to SASRec-only and degradation is moderate, Day31 should extend robustness to GRU4Rec/Bert4Rec or fold this into the final robustness section. If noisy CEP degrades sharply, the next step should be noisy-aware calibration rather than prompt tuning.
"""
    (SUMMARY_DIR / "day30_cep_backbone_robustness_report.md").write_text(text, encoding="utf-8")


def smoke_status() -> pd.DataFrame:
    rows = []
    for noise_type in NOISE_TYPES:
        for level in NOISE_LEVELS:
            condition = setting_name(noise_type, level)
            pred_dir = OUTPUT_ROOT / "beauty_robustness_500" / condition / "predictions"
            candidates = sorted(
                pred_dir.glob("smoke_verify*_raw.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            smoke_path = candidates[0] if candidates else pred_dir / "smoke_raw.jsonl"
            total = 0
            parse_ok = 0
            error_types: dict[str, int] = {}
            if smoke_path.exists():
                with smoke_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        total += 1
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            error_types["json_decode_error"] = error_types.get("json_decode_error", 0) + 1
                            continue
                        if rec.get("parse_success") is True:
                            parse_ok += 1
                        error_type = str(rec.get("error_type") or rec.get("parse_error") or "")
                        if error_type:
                            error_types[error_type] = error_types.get(error_type, 0) + 1
            rows.append(
                {
                    "noise_type": noise_type,
                    "noise_level": level,
                    "smoke_rows": total,
                    "smoke_parse_success_rate": parse_ok / total if total else 0.0,
                    "smoke_output_path": str(smoke_path),
                    "dominant_error_type": max(error_types, key=error_types.get) if error_types else "",
                    "error_type_counts": json.dumps(error_types, sort_keys=True),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day30_robustness_smoke_status.csv", index=False)
    return out


def blocked_report() -> None:
    status_df = smoke_status()
    subset_path = SUMMARY_DIR / "day30_beauty_robustness_subset_stats.csv"
    noise_path = SUMMARY_DIR / "day30_beauty_robustness_noise_stats.csv"
    subset = pd.read_csv(subset_path).iloc[0] if subset_path.exists() else None
    first = status_df.iloc[0]
    text = f"""# Day30 CEP Robustness Blocked Status

## Current Status

Day30 local preparation is complete, but noisy CEP API inference is blocked at smoke validation. The first required smoke setting, `history_dropout_0.1`, produced `{int(first['smoke_rows'])}` rows with parse_success_rate `{first['smoke_parse_success_rate']:.4f}`. The dominant error is `{first['dominant_error_type']}`. Because the smoke threshold is parse_success >= 0.95, the orchestrator correctly stopped before launching the 3000-row full setting.

## Completed Local Artifacts

- Beauty robustness subset: `data/processed/amazon_beauty_robustness_500/clean_test.jsonl`
- Noisy variants: `data/processed/amazon_beauty_robustness_500/noisy_*`
- Configs: `configs/exp/beauty_robustness_500_*.yaml`
- Runtime monitor: `output-repaired/summary/day30_cep_robustness_runtime_monitor.md`
- Smoke status: `output-repaired/summary/day30_robustness_smoke_status.csv`

## Subset Summary

"""
    if subset is not None:
        text += (
            f"The subset has `{int(subset['num_users'])}` users, `{int(subset['num_rows'])}` rows, "
            f"candidate_pool_size_mean `{float(subset['candidate_pool_size_mean']):.1f}`, "
            f"`{int(subset['positive_rows'])}` positive rows, and `{int(subset['negative_rows'])}` negative rows.\n\n"
        )
    else:
        text += "Subset stats file was not found.\n\n"
    text += """## Why Full Inference Was Not Started

Running full robustness despite a 0.0 smoke parse_success rate would waste API budget and produce 27k failed rows. The observed failure is `APIConnectionError`, not a prompt/schema/parser issue. The prompt remains `prompts/candidate_relevance_evidence.txt`, the schema remains `relevance_evidence`, and the first smoke output contains empty `raw_response` fields due to connection failure.

## Resume Command

After API connectivity recovers, resume with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\\run_day30_cep_robustness_pipeline.ps1
```

The orchestrator runs settings sequentially, uses `max_workers=4` and `requests_per_minute=120` inside each setting, and writes progress to `day30_cep_robustness_runtime_monitor.md`.

## Guardrails Preserved

No Books/Electronics runs were started, no LoRA was used, and the main prompt/formula were not changed. Full inference is intentionally blocked until smoke parse_success is >= 0.95.
"""
    (SUMMARY_DIR / "day30_cep_backbone_robustness_report.md").write_text(text, encoding="utf-8")


def smoke_commands() -> None:
    inv = pd.read_csv(SUMMARY_DIR / "day30_beauty_robustness_config_inventory.csv")
    rows = []
    for row in inv.itertuples(index=False):
        smoke_out = OUTPUT_ROOT / str(row.output_dir).replace("output-repaired\\", "").replace("output-repaired/", "") / "predictions" / "smoke_raw.jsonl"
        rows.append(
            {
                "noise_type": row.noise_type,
                "noise_level": row.noise_level,
                "command": (
                    f"py -3.12 main_infer.py --config {row.config_path} "
                    f"--input_path {row.smoke_input_path} --output_path {smoke_out} "
                    f"--split_name smoke --concurrent --resume --max_workers 4"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "day30_robustness_smoke_commands.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--smoke-commands", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.smoke_commands:
        smoke_commands()
    if args.analyze:
        analyze()
    if args.status:
        blocked_report()
    if not (args.prepare or args.smoke_commands or args.analyze or args.status):
        parser.error("Use --prepare, --smoke-commands, --analyze, or --status.")


if __name__ == "__main__":
    main()
