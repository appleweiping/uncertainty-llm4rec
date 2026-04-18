from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
)
from src.methods.uncertainty_ranker import (
    build_ranker_rows,
    build_reranked_predictions,
    rank_candidates_by_score,
    summarize_rerank_effect,
)
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


UNCERTAINTY_SOURCE_DEFAULTS = {
    "pointwise_calibrated": {
        "subdir": "calibrated",
        "filename": "test_calibrated.jsonl",
        "uncertainty_col": "uncertainty",
        "confidence_col": "calibrated_confidence",
    },
    "self_consistency": {
        "subdir": "self_consistency",
        "filename": "test_self_consistency.jsonl",
        "uncertainty_col": "consistency_uncertainty",
        "confidence_col": "consistency_confidence",
    },
    "fused": {
        "subdir": "calibrated",
        "filename": "test_fused.jsonl",
        "uncertainty_col": "fused_uncertainty",
        "confidence_col": "fused_confidence",
    },
}


def infer_domain_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    ignore_tokens = {
        "deepseek",
        "qwen",
        "kimi",
        "doubao",
        "glm",
        "gpt",
        "openai",
        "local",
        "small",
        "mini",
        "large",
        "clean",
        "noisy",
        "robustness",
        "calibration",
        "rerank",
        "reranking",
        "pointwise",
        "rank",
        "pairwise",
    }
    domain_tokens = [token for token in tokens if token not in ignore_tokens and not token.startswith("lambda")]
    if not domain_tokens:
        return "unknown"

    domain = domain_tokens[0]
    if domain.startswith("movie"):
        return "movies"
    if domain.startswith("book"):
        return "books"
    if domain.startswith("electronic"):
        return "electronics"
    return domain


def infer_model_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    known_models = ["deepseek", "qwen", "kimi", "doubao", "glm", "gpt", "openai", "local"]
    for token in reversed(tokens):
        if token in known_models:
            return token
    return "unknown"


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_uncertainty_path(
    *,
    output_root: str | Path,
    uncertainty_exp_name: str,
    uncertainty_source: str,
    explicit_path: str | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    if uncertainty_source not in UNCERTAINTY_SOURCE_DEFAULTS:
        raise ValueError(f"Unsupported uncertainty source: {uncertainty_source}")

    source_spec = UNCERTAINTY_SOURCE_DEFAULTS[uncertainty_source]
    return Path(output_root) / uncertainty_exp_name / source_spec["subdir"] / source_spec["filename"]


def build_result_row(
    *,
    method_name: str,
    prediction_df: pd.DataFrame,
    topk: int,
    lambda_penalty: float | None = None,
    uncertainty_source: str | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    eval_df = build_ranking_eval_frame(prediction_df)
    metrics = compute_ranking_task_metrics(eval_df, k=topk)
    metrics["samples"] = metrics.get("sample_count")
    metrics["coverage"] = metrics.get(f"coverage@{topk}")
    metrics["head_exposure_ratio"] = metrics.get(f"head_exposure_ratio@{topk}")
    metrics["longtail_coverage"] = metrics.get(f"longtail_coverage@{topk}")

    row: dict[str, float | str] = {"method": method_name}
    if lambda_penalty is not None:
        row["lambda_penalty"] = float(lambda_penalty)
    if uncertainty_source is not None:
        row["uncertainty_source"] = uncertainty_source
    row.update(metrics)
    if extra_metrics:
        row.update(extra_metrics)
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="beauty_qwen_rank", help="Direct ranking experiment name.")
    parser.add_argument("--new_exp_name", type=str, default=None, help="Output experiment name for rank rerank results.")
    parser.add_argument("--uncertainty_exp_name", type=str, default=None, help="Pointwise uncertainty experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Optional explicit path to rank_predictions.jsonl")
    parser.add_argument("--uncertainty_input_path", type=str, default=None, help="Optional explicit path to uncertainty jsonl")
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument(
        "--uncertainty_source",
        type=str,
        default="pointwise_calibrated",
        choices=["pointwise_calibrated", "self_consistency", "fused"],
        help="Which pointwise-derived uncertainty source to use.",
    )
    parser.add_argument("--uncertainty_col", type=str, default=None, help="Optional explicit uncertainty column name.")
    parser.add_argument("--uncertainty_confidence_col", type=str, default=None, help="Optional explicit uncertainty confidence column name.")
    parser.add_argument("--lambda_penalty", type=float, default=0.5, help="Penalty coefficient in final_score = relevance - lambda * uncertainty.")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for rerank evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    new_exp_name = args.new_exp_name or f"{args.exp_name}_rerank"

    if args.uncertainty_exp_name is not None:
        uncertainty_exp_name = args.uncertainty_exp_name
    elif args.exp_name.endswith("_rank"):
        uncertainty_exp_name = args.exp_name[: -len("_rank")]
    else:
        uncertainty_exp_name = args.exp_name

    source_defaults = UNCERTAINTY_SOURCE_DEFAULTS[args.uncertainty_source]
    uncertainty_col = args.uncertainty_col or source_defaults["uncertainty_col"]
    uncertainty_confidence_col = args.uncertainty_confidence_col or source_defaults["confidence_col"]

    ranking_paths = ensure_exp_dirs(args.exp_name, args.output_root)
    rerank_paths = ensure_exp_dirs(new_exp_name, args.output_root)

    input_path = Path(args.input_path) if args.input_path else ranking_paths.predictions_dir / "rank_predictions.jsonl"
    uncertainty_path = resolve_uncertainty_path(
        output_root=args.output_root,
        uncertainty_exp_name=uncertainty_exp_name,
        uncertainty_source=args.uncertainty_source,
        explicit_path=args.uncertainty_input_path,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Ranking prediction file not found: {input_path}")
    if not uncertainty_path.exists():
        raise FileNotFoundError(f"Uncertainty file not found: {uncertainty_path}")

    print(f"[{args.exp_name}] Loading ranking predictions from: {input_path}")
    ranking_df = load_jsonl(input_path)
    print(f"[{args.exp_name}] Loaded {len(ranking_df)} ranking events.")

    print(f"[{new_exp_name}] Loading uncertainty source from: {uncertainty_path}")
    uncertainty_df = load_jsonl(uncertainty_path)
    print(f"[{new_exp_name}] Loaded {len(uncertainty_df)} pointwise uncertainty rows.")
    if args.seed is not None:
        print(f"[{new_exp_name}] Seed: {args.seed}")

    scored_rows = build_ranker_rows(
        ranking_predictions_df=ranking_df,
        uncertainty_df=uncertainty_df,
        lambda_penalty=args.lambda_penalty,
        topk=args.k,
        uncertainty_col=uncertainty_col,
        uncertainty_confidence_col=uncertainty_confidence_col,
        uncertainty_source=args.uncertainty_source,
    )
    ranked_rows = rank_candidates_by_score(scored_rows)
    reranked_predictions = build_reranked_predictions(ranked_rows, ranking_df, topk=args.k)

    save_jsonl(reranked_predictions, rerank_paths.reranked_dir / "rank_reranked.jsonl")
    save_table(ranked_rows, rerank_paths.tables_dir / "rank_reranked_rows.csv")

    baseline_row = build_result_row(
        method_name="direct_candidate_ranking",
        prediction_df=ranking_df,
        topk=args.k,
        extra_metrics={"avg_uncertainty_coverage_rate": float("nan"), "changed_ranking_fraction": 0.0, "avg_position_shift": 0.0},
    )
    rerank_effect_metrics = summarize_rerank_effect(ranking_df, reranked_predictions)
    rerank_row = build_result_row(
        method_name="uncertainty_aware_rank_rerank",
        prediction_df=reranked_predictions,
        topk=args.k,
        lambda_penalty=args.lambda_penalty,
        uncertainty_source=args.uncertainty_source,
        extra_metrics=rerank_effect_metrics,
    )

    results_df = pd.concat([baseline_row, rerank_row], ignore_index=True)
    save_table(results_df, rerank_paths.tables_dir / "rerank_results.csv")

    baseline_exposure_df = compute_ranking_exposure_distribution(build_ranking_eval_frame(ranking_df), k=args.k)
    baseline_exposure_df["method"] = "direct_candidate_ranking"
    rerank_exposure_df = compute_ranking_exposure_distribution(build_ranking_eval_frame(reranked_predictions), k=args.k)
    rerank_exposure_df["method"] = "uncertainty_aware_rank_rerank"
    exposure_df = pd.concat([baseline_exposure_df, rerank_exposure_df], ignore_index=True)
    save_table(exposure_df, rerank_paths.tables_dir / "topk_exposure_distribution.csv")

    compare_df = results_df.copy()
    compare_df.insert(0, "model", infer_model_name(args.exp_name))
    compare_df.insert(0, "domain", infer_domain_name(args.exp_name))
    compare_df.insert(2, "task", "candidate_ranking")
    compare_df.insert(3, "base_exp_name", args.exp_name)
    compare_df.insert(4, "rerank_exp_name", new_exp_name)
    compare_df.insert(5, "uncertainty_exp_name", uncertainty_exp_name)
    save_table(compare_df, Path(args.output_root) / "summary" / "week6_day1_rank_rerank_compare.csv")

    print(f"[{new_exp_name}] Ranking rerank done.")
    print(f"[{new_exp_name}] Reranked predictions saved to: {rerank_paths.reranked_dir / 'rank_reranked.jsonl'}")
    print(f"[{new_exp_name}] Tables saved to: {rerank_paths.tables_dir}")


if __name__ == "__main__":
    main()
