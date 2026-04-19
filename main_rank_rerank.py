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
    apply_local_margin_swaps,
    build_ranker_rows,
    build_reranked_predictions,
    rank_candidates_by_score,
    summarize_rerank_effect,
)
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed
from src.utils.exp_io import load_yaml


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
    rerank_variant: str | None = None,
    gate_topk: int | None = None,
    tau: float | None = None,
    gamma: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    delta: float | None = None,
    coverage_fallback_scale: float | None = None,
    eta: float | None = None,
    m_rel: float | None = None,
    m_unc: float | None = None,
    swap_a: float | None = None,
    swap_b: float | None = None,
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
        row["lambda"] = float(lambda_penalty)
    if uncertainty_source is not None:
        row["uncertainty_source"] = uncertainty_source
    if rerank_variant is not None:
        row["rerank_variant"] = rerank_variant
    if gate_topk is not None:
        row["gate_topk"] = int(gate_topk)
    if tau is not None:
        row["tau"] = float(tau)
    if gamma is not None:
        row["gamma"] = float(gamma)
    if alpha is not None:
        row["alpha"] = float(alpha)
    if beta is not None:
        row["beta"] = float(beta)
    if delta is not None:
        row["delta"] = float(delta)
    if coverage_fallback_scale is not None:
        row["coverage_fallback_scale"] = float(coverage_fallback_scale)
    if eta is not None:
        row["eta"] = float(eta)
    if m_rel is not None:
        row["m_rel"] = float(m_rel)
    if m_unc is not None:
        row["m_unc"] = float(m_unc)
    if swap_a is not None:
        row["swap_a"] = float(swap_a)
    if swap_b is not None:
        row["swap_b"] = float(swap_b)
    row.update(metrics)
    if extra_metrics:
        row.update(extra_metrics)
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional rerank config YAML.")
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
    parser.add_argument(
        "--rerank_variant",
        type=str,
        default="linear",
        choices=[
            "linear",
            "coverage_aware_linear",
            "topk_gated_linear",
            "nonlinear_structured_risk_rerank",
            "local_margin_swap_rerank",
            "structured_risk_plus_local_margin_swap_rerank",
        ],
        help="Ranking rerank variant.",
    )
    parser.add_argument(
        "--gate_topk",
        type=int,
        default=3,
        help="Top-k decision band for gated and structured-risk variants.",
    )
    parser.add_argument("--tau", type=float, default=0.35, help="Structured-risk threshold before non-linear uncertainty activation.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Structured-risk non-linear exponent.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Structured-risk uncertainty amplification coefficient.")
    parser.add_argument("--beta", type=float, default=0.7, help="Structured-risk positional gate decay.")
    parser.add_argument("--delta", type=float, default=0.5, help="Structured-risk front-position boost factor.")
    parser.add_argument(
        "--coverage_fallback_scale",
        type=float,
        default=0.5,
        help="Penalty/protection down-scaling factor when uncertainty is only available through fallback.",
    )
    parser.add_argument("--eta", type=float, default=0.02, help="Structured-risk protection bonus coefficient.")
    parser.add_argument("--m_rel", type=float, default=0.05, help="Local margin swap relevance gap threshold.")
    parser.add_argument("--m_unc", type=float, default=0.15, help="Local margin swap uncertainty gap threshold.")
    parser.add_argument("--swap_a", type=float, default=1.5, help="Local margin swap uncertainty gain coefficient.")
    parser.add_argument("--swap_b", type=float, default=1.0, help="Local margin swap relevance penalty coefficient.")
    parser.add_argument("--swap_iterations", type=int, default=2, help="Maximum local swap passes for swap-based variants.")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for rerank evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    cfg = load_yaml(args.config)
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> None:
    args = merge_config(parse_args())
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
        rerank_variant=args.rerank_variant,
        gate_topk=args.gate_topk,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        coverage_fallback_scale=args.coverage_fallback_scale,
        eta=args.eta,
        m_rel=args.m_rel,
        m_unc=args.m_unc,
        swap_a=args.swap_a,
        swap_b=args.swap_b,
        uncertainty_col=uncertainty_col,
        uncertainty_confidence_col=uncertainty_confidence_col,
        uncertainty_source=args.uncertainty_source,
    )
    ranked_rows = rank_candidates_by_score(scored_rows)
    ranked_rows = apply_local_margin_swaps(
        ranked_rows,
        rerank_variant=args.rerank_variant,
        m_rel=args.m_rel,
        m_unc=args.m_unc,
        swap_a=args.swap_a,
        swap_b=args.swap_b,
        max_iterations=args.swap_iterations,
    )
    reranked_predictions = build_reranked_predictions(ranked_rows, ranking_df, topk=args.k)

    save_jsonl(reranked_predictions, rerank_paths.reranked_dir / "rank_reranked.jsonl")
    save_table(ranked_rows, rerank_paths.tables_dir / "rank_reranked_rows.csv")

    baseline_row = build_result_row(
        method_name="direct_candidate_ranking",
        prediction_df=ranking_df,
        topk=args.k,
        extra_metrics={
            "avg_uncertainty_coverage_rate": float("nan"),
            "uncertainty_coverage": float("nan"),
            "changed_ranking_fraction": 0.0,
            "avg_position_shift": 0.0,
            "local_swap_event_fraction": 0.0,
        },
    )
    rerank_effect_metrics = summarize_rerank_effect(ranking_df, reranked_predictions)
    rerank_row = build_result_row(
        method_name=f"uncertainty_aware_rank_rerank_{args.rerank_variant}",
        prediction_df=reranked_predictions,
        topk=args.k,
        lambda_penalty=args.lambda_penalty,
        uncertainty_source=args.uncertainty_source,
        rerank_variant=args.rerank_variant,
        gate_topk=args.gate_topk,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        coverage_fallback_scale=args.coverage_fallback_scale,
        eta=args.eta,
        m_rel=args.m_rel,
        m_unc=args.m_unc,
        swap_a=args.swap_a,
        swap_b=args.swap_b,
        extra_metrics=rerank_effect_metrics,
    )

    results_df = pd.concat([baseline_row, rerank_row], ignore_index=True)
    save_table(results_df, rerank_paths.tables_dir / "rerank_results.csv")

    baseline_exposure_df = compute_ranking_exposure_distribution(build_ranking_eval_frame(ranking_df), k=args.k)
    baseline_exposure_df["method"] = "direct_candidate_ranking"
    rerank_exposure_df = compute_ranking_exposure_distribution(build_ranking_eval_frame(reranked_predictions), k=args.k)
    rerank_exposure_df["method"] = f"uncertainty_aware_rank_rerank_{args.rerank_variant}"
    exposure_df = pd.concat([baseline_exposure_df, rerank_exposure_df], ignore_index=True)
    save_table(exposure_df, rerank_paths.tables_dir / "topk_exposure_distribution.csv")

    compare_df = results_df.copy()
    compare_df.insert(0, "model", infer_model_name(args.exp_name))
    compare_df.insert(0, "domain", infer_domain_name(args.exp_name))
    compare_df.insert(2, "task", "candidate_ranking")
    compare_df.insert(3, "base_exp_name", args.exp_name)
    compare_df.insert(4, "rerank_exp_name", new_exp_name)
    compare_df.insert(5, "uncertainty_exp_name", uncertainty_exp_name)
    summary_dir = Path(args.output_root) / "summary"
    save_table(compare_df, summary_dir / f"{new_exp_name}_compare.csv")
    if args.rerank_variant == "linear" and new_exp_name == f"{args.exp_name}_rerank":
        save_table(compare_df, summary_dir / "week6_day1_rank_rerank_compare.csv")

    print(f"[{new_exp_name}] Ranking rerank done.")
    print(f"[{new_exp_name}] Reranked predictions saved to: {rerank_paths.reranked_dir / 'rank_reranked.jsonl'}")
    print(f"[{new_exp_name}] Tables saved to: {rerank_paths.tables_dir}")


if __name__ == "__main__":
    main()
