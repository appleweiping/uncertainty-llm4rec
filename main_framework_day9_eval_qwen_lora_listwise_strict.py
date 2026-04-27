from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_day6_eval_qwen_lora_listwise import (
    _generate_lora_predictions,
    _oracle_rankings,
    _random_rankings,
    _read_jsonl,
    _write_jsonl,
    evaluate_rankings,
)


PRED_PATH = Path("output-repaired/framework/day9_qwen_lora_beauty_listwise_strict_eval512_predictions.jsonl")
SUMMARY_CSV = Path("data_done/framework_day9_listwise_strict_eval512_summary.csv")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day9 listwise-v2 strict adapter.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_strict_small.yaml")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_listwise_strict_day9_small")
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    test_file = Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_listwise_json_strict.jsonl"))
    samples = _read_jsonl(test_file, limit=args.eval_samples)
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    pred_rows, rankings = _generate_lora_predictions(cfg, samples, args.adapter_path, args.max_new_tokens)
    _write_jsonl(PRED_PATH, pred_rows)
    rows = [
        evaluate_rankings("day9_listwise_v2_strict_train_strict_infer", samples, rankings, pred_rows),
        evaluate_rankings("random_ranking_same_samples", samples, _random_rankings(samples), None),
        evaluate_rankings("oracle_positive_upper_bound", samples, _oracle_rankings(samples), None),
    ]
    _write_csv(SUMMARY_CSV, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
