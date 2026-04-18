from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.confidence_correctness import (
    compute_confidence_bins_accuracy,
    compute_confidence_correctness_summary,
    prepare_prediction_dataframe,
)
from src.analysis.exposure_analysis import compute_high_confidence_exposure
from src.analysis.plotting import (
    plot_confidence_histogram,
    plot_high_confidence_exposure_shift,
    plot_popularity_avg_confidence,
    plot_popularity_confidence_boxplot,
    plot_reliability_diagram,
)
from src.analysis.popularity_bias import compute_popularity_group_stats
from src.eval.calibration_metrics import (
    compute_calibration_metrics,
    get_reliability_dataframe,
)
from src.eval.ranking_metrics import compute_ranking_metrics
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_dict(summary: dict[str, Any], path: str | Path) -> None:
    save_table(pd.DataFrame([summary]), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name, e.g. beauty_deepseek")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional prediction file. Defaults to outputs/{exp_name}/predictions/test_raw.jsonl or test_ranking_raw.jsonl.",
    )
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for pointwise reliability evaluation.")
    parser.add_argument("--high_conf_threshold", type=float, default=0.8, help="High confidence threshold.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument(
        "--task_type",
        type=str,
        default="auto",
        help="Evaluation mode: auto, pointwise_yesno, or candidate_ranking.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for ranking metrics.")
    return parser.parse_args()


def infer_task_type(raw_df: pd.DataFrame, explicit_task_type: str) -> str:
    task_type = str(explicit_task_type).strip().lower()
    if task_type in {"pointwise_yesno", "candidate_ranking"}:
        return task_type

    if "task_type" in raw_df.columns:
        values = {
            str(value).strip().lower()
            for value in raw_df["task_type"].dropna().tolist()
            if str(value).strip()
        }
        if "candidate_ranking" in values:
            return "candidate_ranking"

    ranking_columns = {"ranked_item_ids", "top_k_item_ids", "candidate_scores", "selected_item_id"}
    if ranking_columns.intersection(set(raw_df.columns)):
        return "candidate_ranking"

    return "pointwise_yesno"


def resolve_input_path(paths, explicit_input_path: str | None, task_type: str) -> Path:
    if explicit_input_path:
        return Path(explicit_input_path)

    if task_type == "candidate_ranking":
        return paths.predictions_dir / "test_ranking_raw.jsonl"

    pointwise_default = paths.predictions_dir / "test_raw.jsonl"
    ranking_default = paths.predictions_dir / "test_ranking_raw.jsonl"
    if pointwise_default.exists():
        return pointwise_default
    if ranking_default.exists():
        return ranking_default
    return pointwise_default


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_candidate_scores(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            out.append(
                {
                    "item_id": item_id,
                    "score": _safe_float(row.get("score", -1.0), default=-1.0),
                    "reason": str(row.get("reason", "")).strip(),
                }
            )
        return out
    return []


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _softmax_entropy(scores: list[float]) -> float:
    if not scores:
        return float("nan")
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0:
        return float("nan")
    probs = [value / total for value in exps]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return float(entropy)


def build_ranked_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in raw_df.to_dict(orient="records"):
        user_id = str(record.get("user_id", "")).strip()
        target_item_id = str(record.get("target_item_id", "")).strip()
        popularity_group = str(record.get("target_popularity_group", "unknown")).strip().lower() or "unknown"

        ranked_item_ids = _normalize_string_list(record.get("ranked_item_ids"))
        candidate_scores = _normalize_candidate_scores(record.get("candidate_scores"))

        if not ranked_item_ids and candidate_scores:
            ranked_item_ids = [row["item_id"] for row in sorted(candidate_scores, key=lambda r: r.get("score", -1.0), reverse=True)]

        score_map = {row["item_id"]: row.get("score", -1.0) for row in candidate_scores}
        candidates = record.get("candidates", [])
        label_map: dict[str, int] = {}
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                item_id = str(candidate.get("item_id", "")).strip()
                if not item_id:
                    continue
                label_map[item_id] = int(candidate.get("label", 0))

        for rank, item_id in enumerate(ranked_item_ids, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "target_item_id": target_item_id,
                    "candidate_item_id": item_id,
                    "label": int(label_map.get(item_id, int(item_id == target_item_id))),
                    "rank": rank,
                    "score": _safe_float(score_map.get(item_id, float("nan"))),
                    "target_popularity_group": popularity_group,
                }
            )

    return pd.DataFrame(rows)


def compute_ranking_proxy_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in raw_df.to_dict(orient="records"):
        candidate_scores = _normalize_candidate_scores(record.get("candidate_scores"))
        sorted_scores = sorted(candidate_scores, key=lambda row: row.get("score", -1.0), reverse=True)
        scores = [row.get("score", -1.0) for row in sorted_scores]
        top1_score = scores[0] if len(scores) >= 1 else float("nan")
        top2_score = scores[1] if len(scores) >= 2 else float("nan")
        margin = (top1_score - top2_score) if len(scores) >= 2 else float("nan")
        selected_item_id = str(
            record.get("selected_item_id")
            or (sorted_scores[0]["item_id"] if sorted_scores else "")
        ).strip()
        target_item_id = str(record.get("target_item_id", "")).strip()

        rows.append(
            {
                "user_id": str(record.get("user_id", "")).strip(),
                "target_item_id": target_item_id,
                "selected_item_id": selected_item_id,
                "candidate_count": int(record.get("candidate_count", len(candidate_scores) or 0)),
                "top1_score": _safe_float(top1_score),
                "top2_score": _safe_float(top2_score),
                "score_margin": _safe_float(margin),
                "score_entropy": _safe_float(_softmax_entropy(scores)),
                "is_top1_correct": int(selected_item_id == target_item_id and selected_item_id != ""),
                "target_popularity_group": str(record.get("target_popularity_group", "unknown")).strip().lower() or "unknown",
            }
        )

    return pd.DataFrame(rows)


def evaluate_pointwise(args: argparse.Namespace, raw_df: pd.DataFrame, paths) -> None:
    df = prepare_prediction_dataframe(raw_df)

    print(f"[{args.exp_name}] Loaded {len(df)} pointwise samples.")
    if args.seed is not None:
        print(f"[{args.exp_name}] Seed: {args.seed}")

    metrics = compute_calibration_metrics(df, confidence_col="confidence", n_bins=args.n_bins)
    save_summary_dict(metrics, paths.tables_dir / "diagnostic_metrics.csv")

    cc_summary = compute_confidence_correctness_summary(
        df,
        high_conf_threshold=args.high_conf_threshold,
    )
    save_summary_dict(cc_summary, paths.tables_dir / "confidence_correctness_summary.csv")

    bins_df = compute_confidence_bins_accuracy(df, n_bins=args.n_bins)
    save_table(bins_df, paths.tables_dir / "confidence_bins_accuracy.csv")

    reliability_df = get_reliability_dataframe(
        df["label"].to_numpy(),
        df["confidence"].to_numpy(),
        n_bins=args.n_bins,
    )
    save_table(reliability_df, paths.tables_dir / "reliability_bins.csv")

    pop_df = compute_popularity_group_stats(
        df,
        high_conf_threshold=args.high_conf_threshold,
    )
    save_table(pop_df, paths.tables_dir / "popularity_group_stats.csv")

    exposure_df = compute_high_confidence_exposure(
        df,
        high_conf_threshold=args.high_conf_threshold,
    )
    save_table(exposure_df, paths.tables_dir / "high_confidence_exposure.csv")

    plot_confidence_histogram(
        df,
        paths.figures_dir / "confidence_histogram_correct_vs_wrong.png",
    )
    plot_reliability_diagram(
        reliability_df,
        paths.figures_dir / "reliability_diagram.png",
    )
    plot_popularity_avg_confidence(
        pop_df,
        paths.figures_dir / "popularity_avg_confidence.png",
    )
    plot_popularity_confidence_boxplot(
        df,
        paths.figures_dir / "popularity_confidence_boxplot.png",
    )
    plot_high_confidence_exposure_shift(
        exposure_df,
        paths.figures_dir / "high_confidence_exposure_shift.png",
    )

    print(f"[{args.exp_name}] Pointwise evaluation done.")
    print(f"[{args.exp_name}] Tables saved to:  {paths.tables_dir}")
    print(f"[{args.exp_name}] Figures saved to: {paths.figures_dir}")


def evaluate_candidate_ranking(args: argparse.Namespace, raw_df: pd.DataFrame, paths) -> None:
    ranked_df = build_ranked_dataframe(raw_df)
    proxy_df = compute_ranking_proxy_rows(raw_df)

    if ranked_df.empty:
        raise ValueError("Ranking prediction file did not produce any ranked rows.")

    print(f"[{args.exp_name}] Loaded {len(raw_df)} ranking predictions for {ranked_df['user_id'].nunique()} users.")
    if args.seed is not None:
        print(f"[{args.exp_name}] Seed: {args.seed}")

    ranking_metrics = compute_ranking_metrics(ranked_df, k=args.k)
    save_summary_dict(ranking_metrics, paths.tables_dir / "ranking_metrics.csv")
    save_table(ranked_df, paths.tables_dir / "ranking_rows.csv")

    if not proxy_df.empty:
        proxy_summary = {
            "num_users": int(len(proxy_df)),
            "top1_accuracy": float(proxy_df["is_top1_correct"].mean()),
            "avg_top1_score": float(proxy_df["top1_score"].mean()),
            "avg_top2_score": float(proxy_df["top2_score"].mean()) if proxy_df["top2_score"].notna().any() else float("nan"),
            "avg_score_margin": float(proxy_df["score_margin"].mean()) if proxy_df["score_margin"].notna().any() else float("nan"),
            "avg_score_entropy": float(proxy_df["score_entropy"].mean()) if proxy_df["score_entropy"].notna().any() else float("nan"),
        }
        save_summary_dict(proxy_summary, paths.tables_dir / "ranking_proxy_summary.csv")
        save_table(proxy_df, paths.tables_dir / "ranking_proxy_rows.csv")

    print(f"[{args.exp_name}] Ranking evaluation done.")
    print(f"[{args.exp_name}] Tables saved to: {paths.tables_dir}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    input_path = resolve_input_path(paths, args.input_path, args.task_type)

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    print(f"[{args.exp_name}] Loading predictions from: {input_path}")
    raw_df = load_jsonl(input_path)
    task_type = infer_task_type(raw_df, args.task_type)
    print(f"[{args.exp_name}] Evaluation task type: {task_type}")

    if task_type == "candidate_ranking":
        evaluate_candidate_ranking(args, raw_df, paths)
        return

    evaluate_pointwise(args, raw_df, paths)


if __name__ == "__main__":
    main()
