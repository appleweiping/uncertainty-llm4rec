from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.exp_io import load_yaml
from src.utils.io import ensure_dir


DOMAIN_CONFIGS = {
    "movies": {
        "pointwise": "configs/exp/movies_deepseek_pointwise.yaml",
        "rank": "configs/exp/movies_deepseek_rank.yaml",
        "pointwise_fallback_exp": "movies_small_deepseek",
    },
    "beauty": {
        "pointwise": "configs/exp/beauty_deepseek_pointwise.yaml",
        "rank": "configs/exp/beauty_deepseek_rank.yaml",
        "pointwise_fallback_exp": "beauty_deepseek",
    },
    "books": {
        "pointwise": "configs/exp/books_deepseek_pointwise.yaml",
        "rank": "configs/exp/books_deepseek_rank.yaml",
        "pointwise_fallback_exp": "books_small_deepseek",
    },
    "electronics": {
        "pointwise": "configs/exp/electronics_deepseek_pointwise.yaml",
        "rank": "configs/exp/electronics_deepseek_rank.yaml",
        "pointwise_fallback_exp": "electronics_small_deepseek",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--status_filename", type=str, default="week6_magic7_4domain_deepseek_status.csv")
    parser.add_argument("--compare_filename", type=str, default="week6_magic7_4domain_deepseek_compare.csv")
    parser.add_argument("--notes_filename", type=str, default="week6_magic7_cross_domain_notes.md")
    return parser.parse_args()


def _read_csv_optional(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _count_jsonl(path: Path, max_lines: int | None = None) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for count, _ in enumerate(f, start=1):
            if max_lines is not None and count >= max_lines:
                break
    return count


def _metric(metrics: pd.DataFrame | None, column: str, *aliases: str) -> Any:
    if metrics is None or metrics.empty or column not in metrics.columns:
        for alias in aliases:
            if metrics is not None and not metrics.empty and alias in metrics.columns:
                return metrics.iloc[0].get(alias, pd.NA)
        return pd.NA
    return metrics.iloc[0].get(column, pd.NA)


def _parse_success_rate(path: Path) -> Any:
    if not path.exists():
        return pd.NA
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        return pd.NA
    if df.empty or "parse_success" not in df.columns:
        return pd.NA
    return float(pd.to_numeric(df["parse_success"], errors="coerce").fillna(0).mean())


def _resolve_exp_dir(output_root: Path, exp_name: str, cfg: dict[str, Any]) -> Path:
    output_dir = cfg.get("output_dir")
    if output_dir:
        return Path(output_dir)
    return output_root / exp_name


def _task_paths(output_root: Path, cfg_path: Path, task: str, domain: str, fallback_exp: str | None) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    exp_name = str(cfg.get("exp_name", cfg_path.stem))
    exp_dir = _resolve_exp_dir(output_root, exp_name, cfg)
    input_path = Path(str(cfg.get("input_path", "")))

    if task == "pointwise":
        prediction_path = exp_dir / "predictions" / "test_raw.jsonl"
        metric_path = exp_dir / "tables" / "diagnostic_metrics.csv"
        fallback_metric_path = output_root / str(fallback_exp or "") / "tables" / "diagnostic_metrics.csv"
        fallback_prediction_path = output_root / str(fallback_exp or "") / "predictions" / "test_raw.jsonl"
    else:
        prediction_path = exp_dir / "predictions" / "rank_predictions.jsonl"
        metric_path = exp_dir / "tables" / "ranking_metrics.csv"
        fallback_metric_path = Path("")
        fallback_prediction_path = Path("")

    metrics = _read_csv_optional(metric_path)
    metric_source = exp_name
    if metrics is None and task == "pointwise" and fallback_exp:
        metrics = _read_csv_optional(fallback_metric_path)
        if metrics is not None:
            metric_source = str(fallback_exp)
            prediction_path = fallback_prediction_path

    prediction_exists = prediction_path.exists()
    metric_exists = metrics is not None
    input_exists = input_path.exists()

    if prediction_exists and metric_exists:
        status = "eval_ready"
    elif prediction_exists:
        status = "prediction_ready_eval_missing"
    elif metric_exists:
        status = "fallback_eval_ready" if metric_source != exp_name else "metric_ready_prediction_missing"
    elif input_exists:
        status = "input_ready_needs_run"
    else:
        status = "input_missing"

    return {
        "domain": domain,
        "task": task,
        "model": "deepseek",
        "config_path": str(cfg_path),
        "exp_name": exp_name,
        "result_source_exp_name": metric_source if metric_exists else "",
        "input_path": str(input_path),
        "output_dir": str(exp_dir),
        "prediction_path": str(prediction_path),
        "input_exists": input_exists,
        "prediction_exists": prediction_exists,
        "eval_ready": metric_exists,
        "status": status,
        "metrics": metrics,
        "sample_cap": cfg.get("max_samples", pd.NA),
        "raw_prediction_count": _count_jsonl(prediction_path),
    }


def build_status_and_compare(output_root: str | Path = "outputs") -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = Path(output_root)
    status_records: list[dict[str, Any]] = []
    compare_records: list[dict[str, Any]] = []

    for domain, specs in DOMAIN_CONFIGS.items():
        for task in ["pointwise", "rank"]:
            info = _task_paths(
                output_root=output_root,
                cfg_path=Path(specs[task]),
                task=task,
                domain=domain,
                fallback_exp=specs.get("pointwise_fallback_exp") if task == "pointwise" else None,
            )
            metrics = info.pop("metrics")
            status_records.append(info.copy())

            compare_records.append(
                {
                    "domain": domain,
                    "task": "pointwise_yesno" if task == "pointwise" else "candidate_ranking",
                    "model": "deepseek",
                    "samples": _metric(metrics, "sample_count") if task == "rank" else _metric(metrics, "num_samples"),
                    "ECE": _metric(metrics, "ECE", "ece"),
                    "Brier": _metric(metrics, "Brier", "brier_score"),
                    "HR@10": _metric(metrics, "HR@10"),
                    "NDCG@10": _metric(metrics, "NDCG@10"),
                    "MRR": _metric(metrics, "MRR"),
                    "parse_success_rate": (
                        _metric(metrics, "parse_success_rate")
                        if task == "rank"
                        else _parse_success_rate(Path(info["prediction_path"]))
                    ),
                    "status": info["status"],
                    "exp_name": info["exp_name"],
                    "result_source_exp_name": info["result_source_exp_name"],
                    "output_dir": info["output_dir"],
                }
            )

    return pd.DataFrame(status_records), pd.DataFrame(compare_records)


def write_notes(compare_df: pd.DataFrame, status_df: pd.DataFrame, output_path: Path) -> None:
    ready_pointwise = int(((compare_df["task"] == "pointwise_yesno") & compare_df["status"].astype(str).str.contains("ready")).sum())
    ready_rank = int(((compare_df["task"] == "candidate_ranking") & (compare_df["status"] == "eval_ready")).sum())
    rank_boundary = (
        "The current matrix now contains completed 100-sample DeepSeek candidate-ranking rows for all four domains, "
        "so it can support a first compact cross-domain listwise sanity check. It still cannot support final large-scale "
        "claims because the sample size is intentionally small and literature-aligned baselines are only at the schema-bridge stage."
        if ready_rank == 4
        else "The current matrix can support whether the project structure is ready for four-domain replication and whether existing pointwise evidence can be compared under one schema. It cannot yet support a strong cross-domain listwise conclusion until the missing candidate-ranking inference/evaluation rows are actually produced."
    )
    next_step = (
        "The next execution step is to connect the structured-risk current-best family and stronger literature-aligned baselines to this four-domain matrix, then scale beyond the compact 100-sample setting. Pairwise remains outside the hard four-domain requirement at this stage so that coverage repair does not block the main cross-domain matrix."
        if ready_rank == 4
        else "The next execution step is to run the pending DeepSeek candidate-ranking configs, evaluate them with `main_eval_rank.py`, and then regenerate this table. Pairwise remains outside the hard four-domain requirement at this stage so that coverage repair does not block the main cross-domain matrix."
    )
    lines = [
        "# Week6 Magic7 Cross-Domain Minimal Notes",
        "",
        "This note records the four-domain DeepSeek minimal evidence matrix handoff. It is a compact cross-domain bridge, not the final large-scale experiment.",
        "",
        f"Pointwise diagnosis currently has {ready_pointwise}/4 domains with reusable or direct evaluation rows. Candidate ranking currently has {ready_rank}/4 domains with completed ranking evaluation rows under the new DeepSeek rank configs.",
        "",
        rank_boundary,
        "",
        next_step,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)

    status_df, compare_df = build_status_and_compare(output_root)
    status_path = summary_dir / args.status_filename
    compare_path = summary_dir / args.compare_filename
    notes_path = summary_dir / args.notes_filename

    status_df.to_csv(status_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    write_notes(compare_df, status_df, notes_path)

    print(f"Saved status table to: {status_path}")
    print(f"Saved compare table to: {compare_path}")
    print(f"Saved notes to: {notes_path}")


if __name__ == "__main__":
    main()
