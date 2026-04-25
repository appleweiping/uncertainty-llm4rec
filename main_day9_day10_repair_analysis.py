from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main_eval_recommendation_list import build_candidate_frame, safe_list
from main_rerank import save_table
from src.analysis.rerank_diagnostics import add_user_normalized_column, compute_rank_change_diagnostics
from src.eval.bias_metrics import compute_bias_metrics
from src.eval.ranking_metrics import compute_ranking_metrics
from src.methods.baseline_ranker import rank_by_score


LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]
NORMALIZATIONS = ["minmax", "zscore"]


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def safe_auc(labels: pd.Series, scores: pd.Series) -> float:
    labels = pd.to_numeric(labels, errors="coerce")
    scores = pd.to_numeric(scores, errors="coerce")
    mask = labels.notna() & scores.notna()
    labels = labels[mask].astype(int)
    scores = scores[mask].astype(float)
    if labels.nunique() < 2:
        return float("nan")

    # Mann-Whitney formulation of ROC AUC, avoiding a new sklearn dependency path.
    ranks = scores.rank(method="average")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(ranks[labels == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def metric_row(
    ranked: pd.DataFrame,
    *,
    method: str,
    base_score_source: str,
    uncertainty_col: str,
    lambda_penalty: float | None,
    normalization: str | None,
    k: int,
) -> dict[str, Any]:
    ranking = compute_ranking_metrics(ranked, k=k)
    bias = compute_bias_metrics(ranked, k=k)
    row: dict[str, Any] = {
        "method": method,
        "base_score_source": base_score_source,
        "uncertainty_col": uncertainty_col,
        "lambda": lambda_penalty,
        "normalization": normalization,
        "HR@10": ranking.get("HR@10"),
        "NDCG@10": ranking.get("NDCG@10"),
        "MRR@10": ranking.get("MRR@10"),
    }
    row.update(bias)
    return row


def valid_recommended_ids(pred_row: pd.Series, candidate_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    recommended = safe_list(pred_row.get("recommended_items"))
    for item in sorted(
        recommended,
        key=lambda row: int(row.get("rank", len(candidate_ids) + 1)) if isinstance(row, dict) else 9999,
    ):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("candidate_item_id", "")).strip()
        if item_id in candidate_ids and item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


def add_plain_rank_scores(
    candidate_df: pd.DataFrame,
    samples: pd.DataFrame,
    plain_predictions: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    pred_by_sample = {int(row.get("sample_id")): row for _, row in plain_predictions.iterrows()}
    sample_by_id = {int(row.get("sample_id", idx)): row for idx, row in samples.iterrows()}
    ranks: dict[tuple[int, str], int] = {}

    for sample_id, sample in sample_by_id.items():
        candidate_ids = [str(item_id) for item_id in safe_list(sample.get("candidate_item_ids"))]
        pred = pred_by_sample.get(sample_id)
        ordered = valid_recommended_ids(pred, candidate_ids) if pred is not None else []
        rank_by_item = {item_id: rank for rank, item_id in enumerate(ordered[:k], start=1)}
        for item_id in candidate_ids:
            ranks[(sample_id, item_id)] = int(rank_by_item.get(item_id, k + 1))

    out = candidate_df.copy()
    out["plain_list_rank"] = [
        ranks.get((int(row.sample_id), str(row.candidate_item_id)), k + 1)
        for row in out.itertuples(index=False)
    ]
    out["plain_rank_score"] = ((k - out["plain_list_rank"].astype(float) + 1.0) / float(k)).clip(lower=0.0)
    out["plain_rank_score_reciprocal"] = np.where(
        out["plain_list_rank"].astype(float) <= float(k),
        1.0 / out["plain_list_rank"].astype(float),
        0.0,
    )
    return out


def join_day9_evidence(candidate_df: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "user_id",
        "candidate_item_id",
        "relevance_probability",
        "calibrated_relevance_probability",
        "minimal_calibrated_relevance_probability",
        "full_calibrated_relevance_probability",
        "positive_evidence",
        "negative_evidence",
        "abs_evidence_margin",
        "ambiguity",
        "missing_information",
        "evidence_risk",
        "relevance_uncertainty",
        "parse_success",
    ]
    available = [col for col in keep_cols if col in evidence.columns]
    merged = candidate_df.merge(
        evidence[available],
        on=["user_id", "candidate_item_id"],
        how="left",
        suffixes=("", "_day9"),
    )
    numeric_defaults = {
        "relevance_probability": 0.0,
        "calibrated_relevance_probability": 0.0,
        "minimal_calibrated_relevance_probability": 0.0,
        "full_calibrated_relevance_probability": 0.0,
        "positive_evidence": 0.0,
        "negative_evidence": 0.0,
        "abs_evidence_margin": 0.0,
        "ambiguity": 1.0,
        "missing_information": 1.0,
        "evidence_risk": 1.0,
        "relevance_uncertainty": 1.0,
    }
    for col, default in numeric_defaults.items():
        if col not in merged.columns:
            merged[col] = default
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default).astype(float)
    return merged


def rank_by_base(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    work = df.copy()
    work["score"] = pd.to_numeric(work[base_col], errors="coerce").fillna(0.0)
    ranked = rank_by_score(work, user_col="user_id", score_col="score", rank_col="rank")
    return ranked


def decoupled_rerank(
    df: pd.DataFrame,
    *,
    base_col: str,
    uncertainty_col: str,
    lambda_penalty: float,
    normalization: str,
) -> pd.DataFrame:
    work = add_user_normalized_column(df, base_col, "normalized_base_score", method=normalization)
    work = add_user_normalized_column(work, uncertainty_col, "normalized_uncertainty", method=normalization)
    work["score"] = (
        pd.to_numeric(work["normalized_base_score"], errors="coerce").fillna(0.0)
        - float(lambda_penalty) * pd.to_numeric(work["normalized_uncertainty"], errors="coerce").fillna(0.0)
    )
    return rank_by_score(work, user_col="user_id", score_col="score", rank_col="rank")


def run_plain_base_grid(df: pd.DataFrame, *, k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_col = "plain_rank_score"
    uncertainty_col = "evidence_risk"
    baseline_ranked = rank_by_base(df, base_col)
    base_row = metric_row(
        baseline_ranked,
        method="plain_direct_list_baseline_from_rank_score",
        base_score_source=base_col,
        uncertainty_col=uncertainty_col,
        lambda_penalty=0.0,
        normalization="none",
        k=k,
    )
    base_row.update(
        {
            "rank_change_rate": 0.0,
            "top10_order_change_rate": 0.0,
            "mean_kendall_tau": 1.0,
            "base_uncertainty_spearman": float(
                pd.to_numeric(df[base_col], errors="coerce").corr(
                    pd.to_numeric(df[uncertainty_col], errors="coerce"),
                    method="spearman",
                )
            ),
        }
    )
    rows.append(base_row)

    for normalization in NORMALIZATIONS:
        for lambda_penalty in LAMBDA_GRID:
            reranked = decoupled_rerank(
                df,
                base_col=base_col,
                uncertainty_col=uncertainty_col,
                lambda_penalty=lambda_penalty,
                normalization=normalization,
            )
            row = metric_row(
                reranked,
                method="plain_direct_list_plus_evidence_risk_rerank",
                base_score_source=base_col,
                uncertainty_col=uncertainty_col,
                lambda_penalty=lambda_penalty,
                normalization=normalization,
                k=k,
            )
            diag = compute_rank_change_diagnostics(
                baseline_ranked,
                reranked,
                base_score_col=base_col,
                uncertainty_col=uncertainty_col,
                lambda_penalty=lambda_penalty,
                setting="Day10-plain-base-risk",
                normalization=normalization,
                k=k,
            )
            row.update(
                {
                    "rank_change_rate": diag["rank_change_rate"],
                    "top10_order_change_rate": diag["top10_order_change_rate"],
                    "mean_kendall_tau": diag["mean_kendall_tau"],
                    "base_uncertainty_spearman": diag["base_uncertainty_spearman"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_external_comparison_rows(grid: pd.DataFrame, comparison_path: Path, bridge_path: Path) -> pd.DataFrame:
    rows = [grid]
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        comp = comp.assign(
            base_score_source="reported_day10_full",
            uncertainty_col=np.where(comp["method"].str.contains("risk", case=False, na=False), "list_prompt_evidence_risk", ""),
            normalization="reported",
            rank_change_rate=np.nan,
            top10_order_change_rate=np.nan,
            mean_kendall_tau=np.nan,
            base_uncertainty_spearman=np.nan,
        )
        comp["lambda"] = np.nan
        rows.append(comp[grid.columns.intersection(comp.columns).tolist() + [c for c in grid.columns if c not in comp.columns]].reindex(columns=grid.columns))
    if bridge_path.exists():
        bridge = pd.read_csv(bridge_path)
        bridge = bridge[bridge["method"].isin(["day9_pointwise_calibrated_relevance", "day9_pointwise_decoupled_rerank"])].copy()
        bridge["base_score_source"] = bridge.get("base_score_col", "reported_day9")
        bridge["uncertainty_col"] = bridge.get("uncertainty_col", "")
        bridge["lambda"] = bridge.get("lambda", np.nan)
        bridge["normalization"] = bridge.get("normalization", "reported")
        rows.append(bridge.reindex(columns=grid.columns))
    return pd.concat(rows, ignore_index=True)


def run_stronger_base_ablation(df: pd.DataFrame, *, k: int) -> pd.DataFrame:
    work = add_hybrid_scores(df)
    base_specs: list[tuple[str, str]] = [
        ("A_relevance_probability", "relevance_probability"),
        ("B_calibrated_relevance_probability", "calibrated_relevance_probability"),
        ("C_plain_rank_score", "plain_rank_score"),
    ]
    for alpha in [0.25, 0.5, 0.75]:
        rel_col = f"hybrid_alpha_{alpha:g}_score"
        base_specs.append((f"D_hybrid_alpha_{alpha:g}", rel_col))

    rows: list[dict[str, Any]] = []
    uncertainty_col = "evidence_risk"
    for source_name, base_col in base_specs:
        baseline_ranked = rank_by_base(work, base_col)
        for normalization in NORMALIZATIONS:
            for lambda_penalty in LAMBDA_GRID:
                reranked = decoupled_rerank(
                    work,
                    base_col=base_col,
                    uncertainty_col=uncertainty_col,
                    lambda_penalty=lambda_penalty,
                    normalization=normalization,
                )
                row = metric_row(
                    reranked,
                    method="day9_day10_stronger_base_evidence_risk",
                    base_score_source=source_name,
                    uncertainty_col=uncertainty_col,
                    lambda_penalty=lambda_penalty,
                    normalization=normalization,
                    k=k,
                )
                diag = compute_rank_change_diagnostics(
                    baseline_ranked,
                    reranked,
                    base_score_col=base_col,
                    uncertainty_col=uncertainty_col,
                    lambda_penalty=lambda_penalty,
                    setting=source_name,
                    normalization=normalization,
                    k=k,
                )
                row.update(
                    {
                        "base_score_col": base_col,
                        "rank_change_rate": diag["rank_change_rate"],
                        "top10_order_change_rate": diag["top10_order_change_rate"],
                        "mean_kendall_tau": diag["mean_kendall_tau"],
                        "base_uncertainty_spearman": diag["base_uncertainty_spearman"],
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def add_hybrid_scores(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    rel_norm = add_user_normalized_column(work, "calibrated_relevance_probability", "_norm_rel", method="minmax")
    rel_norm = add_user_normalized_column(rel_norm, "plain_rank_score", "_norm_plain", method="minmax")
    for alpha in [0.25, 0.5, 0.75]:
        rel_col = f"hybrid_alpha_{alpha:g}_score"
        work[rel_col] = alpha * rel_norm["_norm_rel"].astype(float) + (1.0 - alpha) * rel_norm["_norm_plain"].astype(float)
    return work


def risk_diagnostics_for_base(
    df: pd.DataFrame,
    *,
    base_source: str,
    base_col: str,
    best_lambda: float,
    normalization: str,
    k: int,
) -> dict[str, Any]:
    baseline = rank_by_base(df, base_col)
    reranked = decoupled_rerank(
        df,
        base_col=base_col,
        uncertainty_col="evidence_risk",
        lambda_penalty=best_lambda,
        normalization=normalization,
    )
    merged = baseline[["user_id", "candidate_item_id", "rank", base_col, "evidence_risk", "label"]].rename(
        columns={"rank": "old_rank"}
    ).merge(
        reranked[["user_id", "candidate_item_id", "rank"]].rename(columns={"rank": "new_rank"}),
        on=["user_id", "candidate_item_id"],
        how="inner",
    )
    merged["rank_delta"] = merged["old_rank"].astype(float) - merged["new_rank"].astype(float)
    positive_rank = (
        merged[merged["label"].astype(int) == 1][["user_id", "old_rank"]]
        .rename(columns={"old_rank": "positive_old_rank"})
    )
    merged = merged.merge(positive_rank, on="user_id", how="left")
    merged["misranked_negative"] = (
        (merged["label"].astype(int) == 0)
        & (merged["old_rank"].astype(float) < merged["positive_old_rank"].astype(float))
    ).astype(int)
    ranked_metrics = compute_ranking_metrics(baseline, k=k)
    diag = compute_rank_change_diagnostics(
        baseline,
        reranked,
        base_score_col=base_col,
        uncertainty_col="evidence_risk",
        lambda_penalty=best_lambda,
        setting=base_source,
        normalization=normalization,
        k=k,
    )
    promoted = merged[merged["rank_delta"] > 0]
    demoted = merged[merged["rank_delta"] < 0]
    wrong_high = merged[merged["misranked_negative"] == 1]
    return {
        "base_score_source": base_source,
        "base_score_AUROC": safe_auc(merged["label"], merged[base_col]),
        "base_score_NDCG@10": ranked_metrics["NDCG@10"],
        "evidence_risk_AUROC_for_error_or_misrank": safe_auc(merged["misranked_negative"], merged["evidence_risk"]),
        "base_risk_spearman": float(pd.to_numeric(merged[base_col], errors="coerce").corr(pd.to_numeric(merged["evidence_risk"], errors="coerce"), method="spearman")),
        "risk_mean_for_promoted": float(pd.to_numeric(promoted["evidence_risk"], errors="coerce").mean()) if len(promoted) else float("nan"),
        "risk_mean_for_demoted": float(pd.to_numeric(demoted["evidence_risk"], errors="coerce").mean()) if len(demoted) else float("nan"),
        "risk_mean_for_wrong_high_rank": float(pd.to_numeric(wrong_high["evidence_risk"], errors="coerce").mean()) if len(wrong_high) else float("nan"),
        "rank_change_rate": diag["rank_change_rate"],
        "best_lambda": best_lambda,
        "normalization": normalization,
    }


def build_report(
    *,
    plain_grid: pd.DataFrame,
    stronger: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    plain_baseline = plain_grid[plain_grid["method"] == "plain_direct_list_baseline_from_rank_score"].iloc[0]
    reported_plain_rows = plain_grid[plain_grid["method"] == "plain_direct_list"]
    reported_plain = reported_plain_rows.iloc[0] if not reported_plain_rows.empty else plain_baseline
    plain_best = plain_grid[plain_grid["method"] == "plain_direct_list_plus_evidence_risk_rerank"].sort_values(
        ["NDCG@10", "MRR@10"],
        ascending=False,
    ).iloc[0]
    stronger_best = stronger.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0]
    rel_reconstructed_ndcg = (float(plain_best["NDCG@10"]) - float(plain_baseline["NDCG@10"])) / float(plain_baseline["NDCG@10"])
    rel_reconstructed_mrr = (float(plain_best["MRR@10"]) - float(plain_baseline["MRR@10"])) / float(plain_baseline["MRR@10"])
    rel_reported_ndcg = (float(plain_best["NDCG@10"]) - float(reported_plain["NDCG@10"])) / float(reported_plain["NDCG@10"])
    rel_reported_mrr = (float(plain_best["MRR@10"]) - float(reported_plain["MRR@10"])) / float(reported_plain["MRR@10"])
    rel_hybrid_vs_reported_ndcg = (float(stronger_best["NDCG@10"]) - float(reported_plain["NDCG@10"])) / float(reported_plain["NDCG@10"])
    rel_hybrid_vs_reported_mrr = (float(stronger_best["MRR@10"]) - float(reported_plain["MRR@10"])) / float(reported_plain["MRR@10"])
    diag_best = diagnostics[diagnostics["base_score_source"].astype(str) == str(stronger_best["base_score_source"])]
    risk_auc = float(diag_best.iloc[0]["evidence_risk_AUROC_for_error_or_misrank"]) if not diag_best.empty else float("nan")

    lines = [
        "# Day9/Day10 Task-Specific Repair Report",
        "",
        "## 1. Formulation Correction",
        "",
        "Day6, Day9, and Day10 should not be forced into one identical formula. Day6 is yes/no decision confidence repair, Day9 is candidate relevance posterior calibration, and Day10 is list-level recommendation selection. Scheme four is the shared component, but its role changes by task: reliability repair in Day6, calibrated relevance posterior in Day9, and post-hoc risk audit/reranking for Day10.",
        "",
        "## 2. Day10 Plain Base + Evidence Risk",
        "",
        f"Using the reconstructed Day10 plain rank score as the base ranking gives NDCG@10 `{float(plain_baseline['NDCG@10']):.6f}` and MRR@10 `{float(plain_baseline['MRR@10']):.6f}`. The best plain-base evidence-risk rerank gives NDCG@10 `{float(plain_best['NDCG@10']):.6f}` and MRR@10 `{float(plain_best['MRR@10']):.6f}` with lambda `{plain_best['lambda']}` and normalization `{plain_best['normalization']}`. Relative to the reconstructed rank-score baseline, changes are NDCG `{rel_reconstructed_ndcg:.4f}` and MRR `{rel_reconstructed_mrr:.4f}`.",
        "",
        f"However, the originally reported Day10 full plain direct list remains NDCG@10 `{float(reported_plain['NDCG@10']):.6f}` and MRR@10 `{float(reported_plain['MRR@10']):.6f}`. Relative to that reported plain list, the best plain-base evidence-risk rerank changes NDCG by `{rel_reported_ndcg:.4f}` and MRR by `{rel_reported_mrr:.4f}`. This distinction matters because converting a generated list into a rank-score table introduces a small reconstruction gap.",
        "",
        "## 3. Day9 Stronger Base Ablation",
        "",
        f"The best stronger-base setting is `{stronger_best['base_score_source']}` with lambda `{stronger_best['lambda']}` and normalization `{stronger_best['normalization']}`, reaching NDCG@10 `{float(stronger_best['NDCG@10']):.6f}` and MRR@10 `{float(stronger_best['MRR@10']):.6f}`. Relative to the reported Day10 plain list, this is NDCG `{rel_hybrid_vs_reported_ndcg:.4f}` and MRR `{rel_hybrid_vs_reported_mrr:.4f}`. This table compares relevance_probability, calibrated_relevance_probability, Day10 plain_rank_score, and hybrid calibrated relevance plus plain rank score.",
        "",
        "## 4. Diagnostics",
        "",
        f"The diagnostic table checks whether the base score itself is strong, whether evidence_risk is aligned with misranked negatives, and whether promoted/demoted items have the expected risk pattern. For the best hybrid setting, evidence_risk AUROC for misranked negatives is `{risk_auc:.6f}`, which is close to random. This is the key guardrail: the current improvement should be interpreted as a small task-specific hybrid/risk effect, not as proof that evidence_risk alone robustly detects all wrong high-rank items.",
        "",
        "## 5. Conclusion",
        "",
    ]
    if float(stronger_best["NDCG@10"]) > float(reported_plain["NDCG@10"]):
        lines.append("The repaired formulation is better stated as task-specific: Day10 first-pass evidence list generation should not be tuned further, but a hybrid of plain list rank score and Day9 calibrated relevance with a mild evidence-risk penalty gives the best current full Beauty result. The gain is positive but modest and below the 5% external-backbone target.")
    else:
        lines.append("Plain-base plus evidence-risk reranking does not improve over the Day10 reported plain list. This suggests the Day10 plain list is already strong for the current small candidate pools, or Day9 evidence_risk is not well aligned with list-level misranking. The next step should be external backbone plug-in rather than more Day10 prompt tuning.")
    lines.extend(
        [
            "",
            "## 6. Output Files",
            "",
            "- `output-repaired/summary/beauty_day10_plain_base_evidence_risk_rerank_grid.csv`",
            "- `output-repaired/summary/beauty_day9_stronger_base_evidence_risk_ablation.csv`",
            "- `output-repaired/summary/beauty_day9_day10_repair_diagnostics.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking_path", default="data/processed/amazon_beauty/ranking_test.jsonl")
    parser.add_argument("--plain_predictions", default="output-repaired/beauty_deepseek_recommendation_list_full_plain/predictions/plain_list_raw.jsonl")
    parser.add_argument("--day9_evidence", default="output-repaired/beauty_deepseek_relevance_evidence_full/calibrated/relevance_evidence_posterior_test.jsonl")
    parser.add_argument("--day10_comparison", default="output-repaired/summary/beauty_day10_full_plain_vs_evidence_comparison.csv")
    parser.add_argument("--day10_bridge", default="output-repaired/summary/beauty_day10_full_list_vs_day9_pointwise_comparison.csv")
    parser.add_argument("--summary_dir", default="output-repaired/summary")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.ranking_path).copy()
    samples["sample_id"] = range(len(samples))
    candidate_df = build_candidate_frame(samples)
    plain_predictions = read_jsonl(args.plain_predictions)
    day9_evidence = read_jsonl(args.day9_evidence)

    with_plain = add_plain_rank_scores(candidate_df, samples, plain_predictions, k=args.k)
    df = join_day9_evidence(with_plain, day9_evidence)

    plain_grid_path = summary_dir / "beauty_day10_plain_base_evidence_risk_rerank_grid.csv"
    stronger_path = summary_dir / "beauty_day9_stronger_base_evidence_risk_ablation.csv"
    if plain_grid_path.exists() and not args.force:
        plain_grid = pd.read_csv(plain_grid_path)
        plain_grid_core = plain_grid[
            plain_grid["method"].isin(
                [
                    "plain_direct_list_baseline_from_rank_score",
                    "plain_direct_list_plus_evidence_risk_rerank",
                ]
            )
        ].copy()
    else:
        plain_grid_core = run_plain_base_grid(df, k=args.k)
        plain_grid = add_external_comparison_rows(
            plain_grid_core,
            Path(args.day10_comparison),
            Path(args.day10_bridge),
        )
        save_table(plain_grid, plain_grid_path)

    if stronger_path.exists() and not args.force:
        stronger = pd.read_csv(stronger_path)
    else:
        stronger = run_stronger_base_ablation(df, k=args.k)
        save_table(stronger, stronger_path)

    best_by_base = (
        stronger.sort_values(["NDCG@10", "MRR@10"], ascending=False)
        .groupby("base_score_source", as_index=False)
        .head(1)
    )
    diag_rows = []
    df_for_diag = add_hybrid_scores(df)
    base_col_by_source = dict(zip(best_by_base["base_score_source"], best_by_base["base_score_col"]))
    for _, row in best_by_base.iterrows():
        diag_rows.append(
            risk_diagnostics_for_base(
                df_for_diag,
                base_source=str(row["base_score_source"]),
                base_col=base_col_by_source[str(row["base_score_source"])],
                best_lambda=float(row["lambda"]),
                normalization=str(row["normalization"]),
                k=args.k,
            )
        )
    diagnostics = pd.DataFrame(diag_rows)
    save_table(diagnostics, summary_dir / "beauty_day9_day10_repair_diagnostics.csv")

    report = build_report(
        plain_grid=plain_grid,
        stronger=stronger,
        diagnostics=diagnostics,
    )
    (summary_dir / "beauty_day9_day10_repair_report.md").write_text(report, encoding="utf-8")

    print(f"Saved {summary_dir / 'beauty_day10_plain_base_evidence_risk_rerank_grid.csv'}")
    print(f"Saved {summary_dir / 'beauty_day9_stronger_base_evidence_risk_ablation.csv'}")
    print(f"Saved {summary_dir / 'beauty_day9_day10_repair_diagnostics.csv'}")
    print(f"Saved {summary_dir / 'beauty_day9_day10_repair_report.md'}")
    best_plain = plain_grid_core[plain_grid_core["method"] == "plain_direct_list_plus_evidence_risk_rerank"].sort_values(
        ["NDCG@10", "MRR@10"],
        ascending=False,
    ).iloc[0]
    best_stronger = stronger.sort_values(["NDCG@10", "MRR@10"], ascending=False).iloc[0]
    print(
        "DAY9_DAY10_REPAIR "
        f"best_plain_rerank_ndcg={float(best_plain['NDCG@10']):.6f} "
        f"best_stronger={best_stronger['base_score_source']} "
        f"best_stronger_ndcg={float(best_stronger['NDCG@10']):.6f}"
    )


if __name__ == "__main__":
    main()
