from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.utils.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pointwise_path",
        type=str,
        default="outputs/beauty_qwen_pointwise/predictions/test_raw.jsonl",
        help="Pointwise prediction file path.",
    )
    parser.add_argument(
        "--ranking_path",
        type=str,
        default="outputs/beauty_qwen_rank/predictions/rank_predictions.jsonl",
        help="Candidate ranking prediction file path.",
    )
    parser.add_argument(
        "--pairwise_path",
        type=str,
        default="outputs/beauty_qwen_pairwise/predictions/pairwise_predictions.jsonl",
        help="Pairwise prediction file path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary/week5_day3_inference_summary.csv",
        help="Summary CSV output path.",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _summarize_records(task: str, rows: list[dict[str, Any]], source_path: str | Path) -> dict[str, Any]:
    latencies: list[float] = []
    parse_flags: list[int] = []
    response_lengths: list[int] = []
    empty_outputs: list[int] = []

    for row in rows:
        latencies.append(float(row.get("latency", row.get("response_latency", 0.0)) or 0.0))
        raw_response = str(row.get("raw_response", "") or "")
        response_lengths.append(len(raw_response))
        empty_outputs.append(int(raw_response.strip() == ""))

        parse_success = row.get("parse_success")
        if parse_success is None:
            parse_success = str(row.get("recommend", "unknown")).strip().lower() in {"yes", "no"}
        parse_flags.append(int(bool(parse_success)))

    return {
        "task": task,
        "sample_count": len(rows),
        "avg_latency": _safe_mean(latencies),
        "parse_success_rate": _safe_mean(parse_flags),
        "avg_response_length": _safe_mean(response_lengths),
        "failure_ratio": round(1.0 - _safe_mean(parse_flags), 4) if rows else 0.0,
        "empty_output_ratio": _safe_mean(empty_outputs),
        "source_path": str(source_path),
    }


def main() -> None:
    args = parse_args()
    summary_rows = []

    for task, path in [
        ("pointwise_yesno", args.pointwise_path),
        ("candidate_ranking", args.ranking_path),
        ("pairwise_preference", args.pairwise_path),
    ]:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Prediction file not found for {task}: {path_obj}")
        rows = load_jsonl(path_obj)
        summary_rows.append(_summarize_records(task, rows, path_obj))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
