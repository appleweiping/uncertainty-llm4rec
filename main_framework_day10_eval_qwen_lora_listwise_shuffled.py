from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_day6_eval_qwen_lora_listwise import _generate_lora_predictions
from src.framework.safe_ranking_eval import (
    evaluate_rankings,
    oracle_rankings,
    random_rankings,
    rankings_from_listwise_predictions,
    read_jsonl,
    write_csv,
    write_jsonl,
)


PRED_PATH = Path("output-repaired/framework/day10_qwen_lora_beauty_listwise_strict_shuffled_eval512_predictions.jsonl")
SUMMARY_CSV = Path("data_done/framework_day10_listwise_strict_shuffled_eval512_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day10 listwise strict shuffled Qwen-LoRA adapter.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_strict_shuffled_small.yaml")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_listwise_strict_shuffled_day10_small")
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--tie_break_policy", default="lexical")
    args = parser.parse_args()

    cfg = _read_config(args.config)
    samples = read_jsonl(Path(str(cfg.get("test_file"))), limit=args.eval_samples)
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    pred_rows, _ = _generate_lora_predictions(cfg, samples, args.adapter_path, args.max_new_tokens)
    write_jsonl(PRED_PATH, pred_rows)
    rankings, parse_rows = rankings_from_listwise_predictions(samples, pred_rows, tie_break_policy=args.tie_break_policy)
    rows = [
        evaluate_rankings(
            "day10_listwise_v2_strict_shuffled_safe_eval",
            samples,
            rankings,
            parse_rows,
            tie_break_policy=args.tie_break_policy,
        ),
        evaluate_rankings(
            "random_seeded_tiebreak_same_samples",
            samples,
            random_rankings(samples, seed=42),
            None,
            tie_break_policy="seeded_random:42",
        ),
        evaluate_rankings(
            "oracle_positive_upper_bound",
            samples,
            oracle_rankings(samples),
            None,
            tie_break_policy="label_oracle",
        ),
    ]
    write_csv(SUMMARY_CSV, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
