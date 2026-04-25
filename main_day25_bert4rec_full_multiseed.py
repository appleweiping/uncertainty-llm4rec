"""Day25 LLM-ESR Bert4Rec full + multi-seed Scheme 4 validation."""

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
    SUMMARY_DIR,
    _candidate_pool,
    _load_title_map,
    _load_train_examples,
)
from main_day21_second_backbone_plugin_smoke import _join_diagnostics, _join_evidence
from main_day24_third_backbone_plugin_smoke import (
    _build_vocab,
    _export_scores,
    _markdown_table,
    _train_external_bert4rec,
)


BACKBONE_DIR = Path("output-repaired/backbone/llmesr_bert4rec_beauty_full")
ARTIFACT_DIR = Path("artifacts/backbones/llmesr_bert4rec_beauty_full")


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
        settings = [("A_Bert4Rec_only", 0.0, 1.0, 0.0, df["backbone_score"])]
        for alpha in [0.5, 0.75, 0.9]:
            beta = 1 - alpha
            settings.append(
                (
                    "B_Bert4Rec_plus_calibrated_relevance",
                    0.0,
                    alpha,
                    beta,
                    alpha * df[f"norm_backbone_{normalization}"] + beta * df[f"norm_calibrated_{normalization}"],
                )
            )
        for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
            settings.append(
                (
                    "C_Bert4Rec_plus_evidence_risk",
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
                        "D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk",
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
                    "relative_HR_vs_backbone": (metrics["HR@10"] - base_metrics["HR@10"])
                    / max(base_metrics["HR@10"], 1e-12),
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
    base = grid[grid["method"] == "A_Bert4Rec_only"].sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0].to_dict()
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
                "best_HR@10": best["HR@10"],
                "backbone_NDCG@10": base["NDCG@10"],
                "backbone_MRR@10": base["MRR@10"],
                "backbone_HR@10": base["HR@10"],
            }
        ]
    )
    diag.to_csv(output_path, index=False)
    return diag


def _train_score_join(
    *,
    seed: int,
    pool: pd.DataFrame,
    title_to_id: dict[str, str],
    train_examples: list[tuple[list[str], str]],
    item_pop: dict[str, int],
    args: argparse.Namespace,
    backbone_dir: Path,
    artifact_dir: Path,
    joined_path: Path,
    join_diag_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backbone_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
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
        seed,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_to_idx": item_to_idx,
            "seed": seed,
            "args": vars(args),
            "train_logs": logs,
            "note": "Day25 Bert4Rec checkpoint; do not commit.",
        },
        artifact_dir / "llmesr_bert4rec.pt",
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
    joined = _join_evidence(scores, args.evidence_path, joined_path)
    join_diag = _join_diagnostics(joined, join_diag_path)
    return scores, joined, join_diag


def _rank_maps(df: pd.DataFrame, score_col: str) -> pd.Series:
    ranked = df.sort_values(["user_id", score_col, "candidate_item_id"], ascending=[True, False, True]).copy()
    ranked["_rank"] = ranked.groupby("user_id").cumcount() + 1
    return ranked.set_index(["user_id", "candidate_item_id"])["_rank"]


def _score_for_setting(df: pd.DataFrame, setting: dict) -> pd.Series:
    method = setting["method"]
    normalization = setting["normalization"]
    alpha = float(setting["alpha"])
    beta = float(setting["beta"])
    lam = float(setting["lambda"])
    if method.startswith("A_"):
        return df["backbone_score"]
    norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
    norm_calibrated = _normalize_per_user(df["calibrated_relevance_probability"], df["user_id"], normalization)
    norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
    if method.startswith("B_"):
        return alpha * norm_backbone + beta * norm_calibrated
    if method.startswith("C_"):
        return norm_backbone - lam * norm_risk
    if method.startswith("D_"):
        return alpha * norm_backbone + beta * norm_calibrated - lam * norm_risk
    raise ValueError(f"Unknown method: {method}")


def _fixed_settings_from_grid(grid: pd.DataFrame) -> list[dict]:
    best_d = (
        grid[grid["method"] == "D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk"]
        .sort_values(["NDCG@10", "MRR@10"], ascending=False)
        .iloc[0]
        .to_dict()
    )
    norm = best_d["normalization"]
    alpha = float(best_d["alpha"])
    beta = float(best_d["beta"])
    lam = float(best_d["lambda"])
    return [
        {"method": "A_Bert4Rec_only", "normalization": "none", "alpha": 1.0, "beta": 0.0, "lambda": 0.0},
        {
            "method": "B_Bert4Rec_plus_calibrated_relevance",
            "normalization": norm,
            "alpha": alpha,
            "beta": beta,
            "lambda": 0.0,
        },
        {
            "method": "C_Bert4Rec_plus_evidence_risk",
            "normalization": norm,
            "alpha": 1.0,
            "beta": 0.0,
            "lambda": lam,
        },
        {
            "method": "D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk",
            "normalization": norm,
            "alpha": alpha,
            "beta": beta,
            "lambda": lam,
        },
    ]


def _score_fixed_settings(joined: pd.DataFrame, join_diag: pd.DataFrame, seed: int, fixed_settings: list[dict]) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    base_metrics = _rank_metrics(df.assign(final_score=df["backbone_score"]), "final_score")
    rows = []
    for setting in fixed_settings:
        final_score = _score_for_setting(df, setting)
        scored = df[["user_id", "candidate_item_id", "label"]].copy()
        scored["final_score"] = final_score
        metrics = _rank_metrics(scored, "final_score")
        rows.append(
            {
                "seed": seed,
                "method": setting["method"],
                "normalization": setting["normalization"],
                "alpha": float(setting["alpha"]),
                "beta": float(setting["beta"]),
                "lambda": float(setting["lambda"]),
                **metrics,
                "join_coverage": float(join_diag.iloc[0]["join_coverage"]),
                "fallback_rate": float(join_diag.iloc[0]["fallback_rate"]),
                "relative_NDCG_vs_bert4rec": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                / max(base_metrics["NDCG@10"], 1e-12),
                "relative_MRR_vs_bert4rec": (metrics["MRR@10"] - base_metrics["MRR@10"])
                / max(base_metrics["MRR@10"], 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _summary(results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    cols = [
        "HR@10",
        "NDCG@10",
        "MRR@10",
        "Recall@10",
        "relative_NDCG_vs_bert4rec",
        "relative_MRR_vs_bert4rec",
        "fallback_rate",
    ]
    rows = []
    for method, group in results.groupby("method", sort=False):
        row = {"method": method}
        for col in cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _component_attribution(summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    base = summary[summary["method"] == "A_Bert4Rec_only"].iloc[0]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "method": row["method"],
                "NDCG@10_mean": row["NDCG@10_mean"],
                "MRR@10_mean": row["MRR@10_mean"],
                "relative_NDCG_vs_bert4rec_mean": row["relative_NDCG_vs_bert4rec_mean"],
                "relative_MRR_vs_bert4rec_mean": row["relative_MRR_vs_bert4rec_mean"],
                "delta_NDCG_vs_A_mean": row["NDCG@10_mean"] - base["NDCG@10_mean"],
                "delta_MRR_vs_A_mean": row["MRR@10_mean"] - base["MRR@10_mean"],
                "interpretation": _interpret(row["method"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _interpret(method: str) -> str:
    if method.startswith("A_"):
        return "external Bert4Rec backbone only"
    if method.startswith("B_"):
        return "calibrated relevance posterior contribution"
    if method.startswith("C_"):
        return "evidence risk as standalone regularizer"
    if method.startswith("D_"):
        return "posterior plus evidence-risk regularization"
    return ""


def _case_study(joined: pd.DataFrame, fixed_settings: list[dict], test_path: Path, output_path: Path) -> pd.DataFrame:
    df = joined.dropna(subset=["backbone_score", "calibrated_relevance_probability", "evidence_risk"]).copy()
    d_setting = [s for s in fixed_settings if s["method"].startswith("D_")][0]
    df["final_score"] = _score_for_setting(df, d_setting)
    old_rank = _rank_maps(df, "backbone_score")
    new_rank = _rank_maps(df, "final_score")
    idx = df.set_index(["user_id", "candidate_item_id"]).index
    df["old_rank"] = idx.map(old_rank)
    df["new_rank"] = idx.map(new_rank)
    df["rank_delta"] = df["old_rank"] - df["new_rank"]
    text_lookup = {}
    for row in _read_jsonl(test_path):
        key = (str(row["user_id"]), str(row["candidate_item_id"]))
        text_lookup[key] = str(row.get("candidate_title") or row.get("candidate_text") or "")[:180]
    df["candidate_text_short"] = [
        text_lookup.get((str(row.user_id), str(row.candidate_item_id)), "") for row in df.itertuples(index=False)
    ]
    frames = []
    specs = [
        ("promoted_positive", (df["label"].astype(int) == 1) & (df["rank_delta"] > 0), ["rank_delta"], [False]),
        ("demoted_negative", (df["label"].astype(int) == 0) & (df["rank_delta"] < 0), ["rank_delta"], [True]),
        ("corrected_positive", (df["label"].astype(int) == 1) & (df["rank_delta"] > 0), ["old_rank", "rank_delta"], [True, False]),
        ("harmed_positive", (df["label"].astype(int) == 1) & (df["rank_delta"] < 0), ["rank_delta"], [True]),
        ("high-risk_demoted", df["rank_delta"] < 0, ["evidence_risk", "rank_delta"], [False, True]),
    ]
    for case_type, mask, sort_cols, ascending in specs:
        part = df[mask].sort_values(sort_cols, ascending=ascending).head(10).copy()
        part["case_type"] = case_type
        frames.append(part)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
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


def _comparison_lines() -> str:
    lines = []
    day20 = SUMMARY_DIR / "day20_sasrec_full_multiseed_summary.csv"
    day23 = SUMMARY_DIR / "day23_gru4rec_full_multiseed_summary.csv"
    if day20.exists():
        sas = pd.read_csv(day20).sort_values("NDCG@10_mean", ascending=False).iloc[0]
        lines.append(
            f"SASRec full multi-seed best `{sas['method']}`: NDCG `{sas['NDCG@10_mean']:.4f}`, "
            f"MRR `{sas['MRR@10_mean']:.4f}`."
        )
    if day23.exists():
        gru = pd.read_csv(day23).sort_values("NDCG@10_mean", ascending=False).iloc[0]
        lines.append(
            f"GRU4Rec full multi-seed best `{gru['method']}`: NDCG `{gru['NDCG@10_mean']:.4f}`, "
            f"MRR `{gru['MRR@10_mean']:.4f}`."
        )
    return " ".join(lines) if lines else "Prior SASRec/GRU4Rec multi-seed summaries were not found."


def _write_report(
    join_diag: pd.DataFrame,
    plugin_diag: pd.DataFrame,
    grid: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    attribution: pd.DataFrame,
    fixed_settings: list[dict],
    output_path: Path,
) -> None:
    jd = join_diag.iloc[0].to_dict()
    pdg = plugin_diag.iloc[0].to_dict()
    best_by_method = (
        grid.sort_values(["method", "NDCG@10", "MRR@10"], ascending=[True, False, False])
        .groupby("method")
        .head(1)
        .loc[
            :,
            [
                "method",
                "HR@10",
                "NDCG@10",
                "MRR@10",
                "relative_NDCG_vs_backbone",
                "relative_MRR_vs_backbone",
                "lambda",
                "alpha",
                "beta",
                "normalization",
            ],
        ]
    )
    fixed_df = pd.DataFrame(fixed_settings)
    text = f"""# Day25 Bert4Rec Full Multi-seed Report

## 1. Day24 Recap

Day24 showed a positive 100-user smoke for LLM-ESR Bert4Rec. It is a third external backbone: not minimal SASRec-style and not GRU4Rec, while still exposing real candidate logits via `Bert4Rec.predict()`.

## 2. Day25 Setup

Full Beauty candidate pool is scored with LLM-ESR Bert4Rec trained only on the Beauty train split. No DeepSeek API calls, prompt changes, LoRA, or formula changes are used. Day9 full evidence is reused for calibrated relevance and evidence risk.

## 3. Backbone Health

Users: `{int(jd['num_users'])}`.

Candidate rows: `{int(jd['num_backbone_rows'])}`.

Join coverage: `{float(jd['join_coverage']):.4f}`.

Fallback rate: `{float(jd['fallback_rate']):.4f}`.

Positive fallback rate: `{float(jd['fallback_rate_positive']):.4f}`.

Negative fallback rate: `{float(jd['fallback_rate_negative']):.4f}`.

Bert4Rec-only NDCG@10: `{float(pdg['backbone_NDCG@10']):.4f}`.

Bert4Rec-only MRR@10: `{float(pdg['backbone_MRR@10']):.4f}`.

## 4. Full Plug-in Grid

Best method: `{pdg['best_method']}`.

Best NDCG@10: `{float(pdg['best_NDCG@10']):.4f}`.

Best MRR@10: `{float(pdg['best_MRR@10']):.4f}`.

Relative NDCG improvement vs Bert4Rec-only: `{float(pdg['best_relative_NDCG_vs_backbone']):.4f}`.

Relative MRR improvement vs Bert4Rec-only: `{float(pdg['best_relative_MRR_vs_backbone']):.4f}`.

Best row per method:

{_markdown_table(best_by_method)}

## 5. Multi-seed Stability

Fixed settings selected from the seed-42 full grid, then reused for seeds 42/43/44:

{_markdown_table(fixed_df)}

Multi-seed summary:

{_markdown_table(summary)}

## 6. Component Attribution

{_markdown_table(attribution)}

## 7. Comparison With SASRec And GRU4Rec

{_comparison_lines()}

The expected pattern is consistent if B carries most of the improvement, C remains weaker, and D is at least competitive with or above B. This supports the position that calibrated relevance posterior is the main contribution and evidence risk is a secondary regularizer.

## 8. Day26 Recommendation

Day26 should build the three-backbone final table and final paper-facing claim map. Keep the claim bounded to Beauty full and these external/sequential backbones unless cross-domain experiments are added.
"""
    output_path.write_text(text, encoding="utf-8")


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--trm_num", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    title_to_id = _load_title_map(args.items_path)
    pool = _candidate_pool(args.evidence_path, args.test_path, 100000)
    train_examples, _, item_pop = _load_train_examples(args.train_path, title_to_id, args.max_seq_len)

    seed0 = args.seeds[0]
    _, joined0, join_diag0 = _train_score_join(
        seed=seed0,
        pool=pool,
        title_to_id=title_to_id,
        train_examples=train_examples,
        item_pop=item_pop,
        args=args,
        backbone_dir=BACKBONE_DIR,
        artifact_dir=ARTIFACT_DIR,
        joined_path=SUMMARY_DIR / "day25_bert4rec_beauty_full_joined_candidates.csv",
        join_diag_path=SUMMARY_DIR / "day25_bert4rec_beauty_full_join_diagnostics.csv",
    )
    grid = _rerank_grid(joined0, SUMMARY_DIR / "day25_bert4rec_beauty_full_plugin_rerank_grid.csv")
    plugin_diag = _plugin_diagnostics(joined0, grid, join_diag0, SUMMARY_DIR / "day25_bert4rec_beauty_full_plugin_diagnostics.csv")
    fixed_settings = _fixed_settings_from_grid(grid)
    case_study = _case_study(joined0, fixed_settings, args.test_path, SUMMARY_DIR / "day25_bert4rec_full_plugin_case_study.csv")

    result_frames = [_score_fixed_settings(joined0, join_diag0, seed0, fixed_settings)]
    for seed in args.seeds[1:]:
        _, joined, join_diag = _train_score_join(
            seed=seed,
            pool=pool,
            title_to_id=title_to_id,
            train_examples=train_examples,
            item_pop=item_pop,
            args=args,
            backbone_dir=Path(f"output-repaired/backbone/llmesr_bert4rec_beauty_full_seed{seed}"),
            artifact_dir=Path(f"artifacts/backbones/llmesr_bert4rec_beauty_full_seed{seed}"),
            joined_path=SUMMARY_DIR / f"day25_bert4rec_full_seed{seed}_joined_candidates.csv",
            join_diag_path=SUMMARY_DIR / f"day25_bert4rec_full_seed{seed}_join_diagnostics.csv",
        )
        result_frames.append(_score_fixed_settings(joined, join_diag, seed, fixed_settings))

    results = pd.concat(result_frames, ignore_index=True)
    results.to_csv(SUMMARY_DIR / "day25_bert4rec_full_multiseed_results.csv", index=False)
    summary = _summary(results, SUMMARY_DIR / "day25_bert4rec_full_multiseed_summary.csv")
    attribution = _component_attribution(summary, SUMMARY_DIR / "day25_bert4rec_component_attribution.csv")
    _write_report(
        join_diag0,
        plugin_diag,
        grid,
        results,
        summary,
        attribution,
        fixed_settings,
        SUMMARY_DIR / "day25_bert4rec_full_multiseed_report.md",
    )
    print("Day25 Bert4Rec full multi-seed validation complete.")


if __name__ == "__main__":
    main()
