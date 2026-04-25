from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main_rerank import save_table
from src.analysis.rerank_diagnostics import add_user_normalized_column, compute_rank_change_diagnostics
from src.eval.bias_metrics import compute_bias_metrics
from src.eval.ranking_metrics import compute_ranking_metrics
from src.methods.baseline_ranker import rank_by_score
from src.utils.paths import ensure_exp_dirs


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def candidate_title(sample: pd.Series, item_id: str) -> str:
    ids = safe_list(sample.get("candidate_item_ids"))
    titles = safe_list(sample.get("candidate_titles"))
    try:
        idx = ids.index(item_id)
    except ValueError:
        return ""
    return str(titles[idx] if idx < len(titles) else "")[:180]


def build_candidate_frame(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sample in samples.iterrows():
        ids = safe_list(sample.get("candidate_item_ids"))
        labels = safe_list(sample.get("candidate_labels"))
        groups = safe_list(sample.get("candidate_popularity_groups"))
        titles = safe_list(sample.get("candidate_titles"))
        for idx, item_id in enumerate(ids):
            rows.append(
                {
                    "sample_id": int(sample.get("sample_id", sample.name)),
                    "source_event_id": sample.get("source_event_id", ""),
                    "user_id": sample.get("user_id", ""),
                    "candidate_item_id": str(item_id),
                    "label": int(labels[idx]) if idx < len(labels) else int(str(item_id) == str(sample.get("positive_item_id", ""))),
                    "target_popularity_group": str(groups[idx]) if idx < len(groups) else "unknown",
                    "candidate_title_or_text": str(titles[idx])[:180] if idx < len(titles) else "",
                    "candidate_pool_order": idx + 1,
                }
            )
    return pd.DataFrame(rows)


def flatten_list_predictions(
    predictions: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    setting: str,
    rerank_evidence: bool = False,
    lambda_penalty: float = 0.2,
) -> pd.DataFrame:
    candidate_df = build_candidate_frame(samples)
    rows: list[dict[str, Any]] = []
    sample_by_id = {int(row.get("sample_id", idx)): row for idx, row in samples.iterrows()}

    for _, pred in predictions.iterrows():
        sample_id = int(pred.get("sample_id"))
        sample = sample_by_id.get(sample_id)
        if sample is None:
            continue
        candidate_ids = [str(item_id) for item_id in safe_list(sample.get("candidate_item_ids"))]
        recommended = safe_list(pred.get("recommended_items"))
        valid_seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for item in sorted(recommended, key=lambda row: int(row.get("rank", len(candidate_ids) + 1)) if isinstance(row, dict) else 9999):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("candidate_item_id", "")).strip()
            if item_id not in candidate_ids or item_id in valid_seen:
                continue
            valid_seen.add(item_id)
            ordered.append(item)

        for item_id in candidate_ids:
            if item_id not in valid_seen:
                ordered.append({"candidate_item_id": item_id, "rank": len(ordered) + 1})
                valid_seen.add(item_id)

        if rerank_evidence:
            work = pd.DataFrame(ordered)
            for col, default in [
                ("relevance_probability", 0.0),
                ("evidence_risk", 1.0),
            ]:
                if col not in work.columns:
                    work[col] = default
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)
            work["base_score"] = work["relevance_probability"].astype(float)
            work["uncertainty"] = work["evidence_risk"].astype(float)
            if float(work["base_score"].max() - work["base_score"].min()) > 0:
                work["base_score"] = (work["base_score"] - work["base_score"].min()) / (
                    work["base_score"].max() - work["base_score"].min()
                )
            if float(work["uncertainty"].max() - work["uncertainty"].min()) > 0:
                work["uncertainty"] = (work["uncertainty"] - work["uncertainty"].min()) / (
                    work["uncertainty"].max() - work["uncertainty"].min()
                )
            work["final_score"] = work["base_score"] - float(lambda_penalty) * work["uncertainty"]
            work = work.sort_values(["final_score", "rank"], ascending=[False, True]).reset_index(drop=True)
            ordered = work.to_dict(orient="records")

        for rank, item in enumerate(ordered, start=1):
            item_id = str(item.get("candidate_item_id", "")).strip()
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_event_id": sample.get("source_event_id", ""),
                    "user_id": sample.get("user_id", ""),
                    "candidate_item_id": item_id,
                    "rank": rank,
                    "setting": setting,
                    "relevance_probability": item.get("relevance_probability"),
                    "evidence_risk": item.get("evidence_risk"),
                    "positive_evidence": item.get("positive_evidence"),
                    "negative_evidence": item.get("negative_evidence"),
                    "abs_evidence_margin": item.get("abs_evidence_margin"),
                    "ambiguity": item.get("ambiguity"),
                    "missing_information": item.get("missing_information"),
                    "reason": item.get("reason", ""),
                }
            )

    ranked = pd.DataFrame(rows)
    return ranked.merge(
        candidate_df.drop(columns=["candidate_pool_order"]),
        on=["sample_id", "source_event_id", "user_id", "candidate_item_id"],
        how="left",
    )


def list_quality_row(
    predictions: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    method: str,
    k: int,
    include_evidence: bool = False,
) -> dict[str, Any]:
    ranking = compute_ranking_metrics(ranked, k=k)
    bias = compute_bias_metrics(ranked, k=k)
    total_items = int(sum(len(safe_list(items)) for items in predictions["recommended_items"]))
    invalid = int(pd.to_numeric(predictions["invalid_item_count"], errors="coerce").fillna(0).sum())
    duplicate = int(pd.to_numeric(predictions["duplicate_item_count"], errors="coerce").fillna(0).sum())
    row: dict[str, Any] = {
        "method": method,
        "HR@10": ranking.get("HR@10"),
        "NDCG@10": ranking.get("NDCG@10"),
        "MRR@10": ranking.get("MRR@10"),
        "Recall@10": ranking.get("HR@10"),
        "parse_success_rate": float(predictions["parse_success"].astype(bool).mean()),
        "schema_valid_rate": float(predictions["schema_valid"].astype(bool).mean()),
        "invalid_item_rate": float(invalid / total_items) if total_items else 0.0,
        "duplicate_item_rate": float(duplicate / total_items) if total_items else 0.0,
    }
    row.update(bias)
    if include_evidence:
        risks = pd.to_numeric(ranked["evidence_risk"], errors="coerce").dropna()
        row["mean_global_uncertainty"] = float(pd.to_numeric(predictions["global_uncertainty"], errors="coerce").mean())
        row["mean_item_evidence_risk"] = float(risks.mean()) if len(risks) else float("nan")
        row["mean_evidence_risk"] = row["mean_item_evidence_risk"]
    return row


def build_pointwise_bridge(
    samples: pd.DataFrame,
    pointwise_path: str | Path,
    *,
    lambda_penalty: float,
    normalization: str,
    k: int,
) -> pd.DataFrame:
    candidate_df = build_candidate_frame(samples)
    pointwise = read_jsonl(pointwise_path)
    subset = candidate_df.merge(
        pointwise,
        on=["user_id", "candidate_item_id"],
        how="left",
        suffixes=("", "_pointwise"),
    )
    subset["calibrated_relevance_probability"] = pd.to_numeric(
        subset["calibrated_relevance_probability"],
        errors="coerce",
    ).fillna(0.0)
    subset["evidence_risk"] = pd.to_numeric(subset["evidence_risk"], errors="coerce").fillna(1.0)

    pointwise_base = subset.copy()
    pointwise_base["score"] = pointwise_base["calibrated_relevance_probability"]
    pointwise_ranked = rank_by_score(pointwise_base, user_col="user_id", score_col="score", rank_col="rank")
    pointwise_ranked["method"] = "day9_pointwise_calibrated_relevance"

    decoupled = add_user_normalized_column(
        subset,
        "calibrated_relevance_probability",
        "normalized_base_score",
        method=normalization,
    )
    decoupled = add_user_normalized_column(
        decoupled,
        "evidence_risk",
        "normalized_uncertainty",
        method=normalization,
    )
    decoupled["score"] = decoupled["normalized_base_score"] - float(lambda_penalty) * decoupled["normalized_uncertainty"]
    decoupled_ranked = rank_by_score(decoupled, user_col="user_id", score_col="score", rank_col="rank")
    decoupled_ranked["method"] = "day9_pointwise_decoupled_rerank"

    rows = []
    for method, ranked in [
        ("day9_pointwise_calibrated_relevance", pointwise_ranked),
        ("day9_pointwise_decoupled_rerank", decoupled_ranked),
    ]:
        metric = compute_ranking_metrics(ranked, k=k)
        bias = compute_bias_metrics(ranked, k=k)
        row = {
            "method": method,
            "HR@10": metric["HR@10"],
            "NDCG@10": metric["NDCG@10"],
            "MRR@10": metric["MRR@10"],
            "Recall@10": metric["HR@10"],
        }
        row.update(bias)
        if method == "day9_pointwise_decoupled_rerank":
            diag = compute_rank_change_diagnostics(
                pointwise_ranked,
                decoupled_ranked,
                base_score_col="calibrated_relevance_probability",
                uncertainty_col="evidence_risk",
                lambda_penalty=lambda_penalty,
                setting="Day9-bridge",
                normalization=normalization,
                k=k,
            )
            row.update(diag)
        rows.append(row)
    return pd.DataFrame(rows)


def add_relative_improvements(comparison: pd.DataFrame) -> pd.DataFrame:
    out = comparison.copy()
    plain = out[out["method"] == "plain_direct_list"]
    if plain.empty:
        out["relative_NDCG_improvement_vs_plain"] = np.nan
        out["relative_MRR_improvement_vs_plain"] = np.nan
        return out
    ndcg_base = float(plain.iloc[0]["NDCG@10"])
    mrr_base = float(plain.iloc[0]["MRR@10"])
    out["relative_NDCG_improvement_vs_plain"] = out["NDCG@10"].astype(float).apply(
        lambda value: (value - ndcg_base) / ndcg_base if ndcg_base else np.nan
    )
    out["relative_MRR_improvement_vs_plain"] = out["MRR@10"].astype(float).apply(
        lambda value: (value - mrr_base) / mrr_base if mrr_base else np.nan
    )
    return out


def build_report(
    *,
    plain_row: dict[str, Any],
    evidence_row: dict[str, Any],
    comparison: pd.DataFrame,
    bridge: pd.DataFrame,
    max_samples: int,
) -> str:
    best = comparison.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]
    ev_vs_plain = comparison[comparison["method"] == "evidence_guided_list"]
    ndcg_rel = float(ev_vs_plain.iloc[0]["relative_NDCG_improvement_vs_plain"]) if not ev_vs_plain.empty else float("nan")
    mrr_rel = float(ev_vs_plain.iloc[0]["relative_MRR_improvement_vs_plain"]) if not ev_vs_plain.empty else float("nan")
    lines = [
        "# Day10 Plain Vs Evidence List Recommendation Report",
        "",
        "## 1. Motivation",
        "",
        "Day9 is candidate-level relevance posterior. Day10 tests whether the same idea transfers to list-level closed-catalog recommendation decisions.",
        "",
        "## 2. Plain Baseline",
        "",
        "The plain baseline uses the same user histories and candidate pools but forbids evidence fields. It only returns ranked candidate_item_id values and short reasons, so it is the no-scheme-four list recommendation control.",
        "",
        "## 3. Evidence-Guided Method",
        "",
        "The evidence-guided setting returns the same ranked list plus relevance_probability, positive/negative evidence, ambiguity, missing_information, and global_uncertainty. evidence_risk is derived after parsing.",
        "",
        "## 4. Fair Comparison Setup",
        "",
        f"Both settings use the same `{max_samples}` Beauty ranking users, identical candidate pools, the same DeepSeek backend, and the same top-K/evaluation code. The only intended difference is whether scheme-four evidence decomposition is available.",
        "",
        "## 5. Evaluation",
        "",
        f"Plain direct list reaches HR@10 `{plain_row['HR@10']:.4f}`, NDCG@10 `{plain_row['NDCG@10']:.4f}`, and MRR@10 `{plain_row['MRR@10']:.4f}`. Evidence-guided list reaches HR@10 `{evidence_row['HR@10']:.4f}`, NDCG@10 `{evidence_row['NDCG@10']:.4f}`, and MRR@10 `{evidence_row['MRR@10']:.4f}`. Evidence vs plain relative NDCG change is `{ndcg_rel:.4f}` and relative MRR change is `{mrr_rel:.4f}`.",
        "",
        "## 6. Bridge To Day9",
        "",
        "The bridge table compares Day9 pointwise calibrated relevance, Day9 decoupled pointwise rerank, Day10 plain direct list generation, Day10 evidence-guided list generation, and Day10 evidence-guided list plus evidence-risk rerank on the same users and candidate pools.",
        "",
        "## 7. Findings",
        "",
        f"The best method in this smoke test is `{best['method']}` with NDCG@10 `{float(best['NDCG@10']):.4f}` and MRR@10 `{float(best['MRR@10']):.4f}`. If evidence-guided generation does not dominate plain directly, the result should be read as evidence decomposition helping mainly through calibrated/risk-aware decision modules rather than automatically improving the first-pass list prompt.",
        "",
        "## 8. Limitations",
        "",
        "This is a 100/200-user smoke test, not full Beauty, not external SOTA, and not Qwen-LoRA. HR@10 is less informative because each Beauty candidate pool is small; NDCG/MRR are the primary list-order signals.",
        "",
        "## 9. Next Step",
        "",
        "If this smoke test is stable, the next Day10-full run should keep the same plain-vs-evidence control and then decide whether Qwen-LoRA should distill the evidence list schema, the Day9 pointwise schema, or both.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary_dir", default="output-repaired/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    exp_name = str(cfg.get("exp_name", "beauty_day10_recommendation_list_200"))
    output_root = str(cfg.get("output_root", "output-repaired"))
    paths = ensure_exp_dirs(exp_name, output_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(cfg.get("max_samples", 200))
    k = int(cfg.get("top_k", 10))
    samples = read_jsonl(cfg["input_path"]).head(max_samples).copy()
    samples["sample_id"] = range(len(samples))

    plain_predictions = read_jsonl(paths.predictions_dir / "plain_list_raw.jsonl")
    evidence_predictions = read_jsonl(paths.predictions_dir / "evidence_list_raw.jsonl")

    plain_ranked = flatten_list_predictions(plain_predictions, samples, setting="plain_direct_list")
    evidence_ranked = flatten_list_predictions(evidence_predictions, samples, setting="evidence_guided_list")
    evidence_reranked = flatten_list_predictions(
        evidence_predictions,
        samples,
        setting="evidence_guided_list_risk_rerank",
        rerank_evidence=True,
        lambda_penalty=float(cfg.get("bridge_lambda", 0.1)),
    )

    plain_row = list_quality_row(plain_predictions, plain_ranked, method="plain_direct_list", k=k)
    evidence_row = list_quality_row(
        evidence_predictions,
        evidence_ranked,
        method="evidence_guided_list",
        k=k,
        include_evidence=True,
    )
    evidence_rerank_row = list_quality_row(
        evidence_predictions,
        evidence_reranked,
        method="evidence_guided_list_risk_rerank",
        k=k,
        include_evidence=True,
    )

    save_table(pd.DataFrame([plain_row]), summary_dir / "beauty_day10_plain_list_eval.csv")
    save_table(pd.DataFrame([evidence_row]), summary_dir / "beauty_day10_evidence_list_eval.csv")

    comparison = add_relative_improvements(pd.DataFrame([plain_row, evidence_row, evidence_rerank_row]))
    save_table(comparison, summary_dir / "beauty_day10_plain_vs_evidence_comparison.csv")

    pointwise_paths = ensure_exp_dirs(
        str(cfg.get("pointwise_exp_name", "beauty_deepseek_relevance_evidence_full")),
        str(cfg.get("pointwise_output_root", output_root)),
    )
    bridge = build_pointwise_bridge(
        samples,
        pointwise_paths.calibrated_dir / "relevance_evidence_posterior_test.jsonl",
        lambda_penalty=float(cfg.get("bridge_lambda", 0.1)),
        normalization=str(cfg.get("bridge_normalization", "zscore")),
        k=k,
    )
    list_bridge_rows = comparison.copy()
    for col in bridge.columns:
        if col not in list_bridge_rows.columns:
            list_bridge_rows[col] = np.nan
    for col in list_bridge_rows.columns:
        if col not in bridge.columns:
            bridge[col] = np.nan
    bridge_comparison = pd.concat([bridge[list_bridge_rows.columns], list_bridge_rows], ignore_index=True)
    save_table(bridge_comparison, summary_dir / "beauty_day10_list_vs_pointwise_comparison.csv")

    report = build_report(
        plain_row=plain_row,
        evidence_row=evidence_row,
        comparison=comparison,
        bridge=bridge_comparison,
        max_samples=max_samples,
    )
    (summary_dir / "beauty_day10_plain_vs_evidence_list_report.md").write_text(report, encoding="utf-8")

    print(f"Saved {summary_dir / 'beauty_day10_plain_list_eval.csv'}")
    print(f"Saved {summary_dir / 'beauty_day10_evidence_list_eval.csv'}")
    print(f"Saved {summary_dir / 'beauty_day10_plain_vs_evidence_comparison.csv'}")
    print(f"Saved {summary_dir / 'beauty_day10_list_vs_pointwise_comparison.csv'}")
    print(f"Saved {summary_dir / 'beauty_day10_plain_vs_evidence_list_report.md'}")
    best = comparison.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).iloc[0]
    print(
        "BEST_DAY10_LIST "
        f"method={best['method']} NDCG@10={float(best['NDCG@10']):.6f} "
        f"MRR@10={float(best['MRR@10']):.6f}"
    )


if __name__ == "__main__":
    main()
