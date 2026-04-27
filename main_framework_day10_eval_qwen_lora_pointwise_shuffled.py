from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_day9_eval_qwen_lora_pointwise import _generate_pointwise_predictions, parse_relevance
from src.framework.safe_ranking_eval import (
    evaluate_rankings,
    oracle_rankings,
    random_rankings,
    read_jsonl,
    write_csv,
    write_jsonl,
)


PRED_PATH = Path("output-repaired/framework/day10_qwen_lora_beauty_pointwise_shuffled_predictions.jsonl")
SUMMARY_CSV = Path("data_done/framework_day10_pointwise_shuffled_safe_eval_summary.csv")
AUDIT_MD = Path("data_done/framework_day10_pointwise_shuffled_leakage_audit.md")


def _candidate_id(row: dict) -> str:
    return str(row.get("input", {}).get("candidate_item", {}).get("candidate_item_id", "")).strip()


def _group_rows(rows: list[dict], pool_size: int = 6) -> list[list[dict]]:
    return [rows[i : i + pool_size] for i in range(0, len(rows), pool_size) if len(rows[i : i + pool_size]) == pool_size]


def _samples_from_groups(groups: list[list[dict]]) -> list[dict]:
    samples = []
    for idx, group in enumerate(groups):
        pool = [{"candidate_item_id": _candidate_id(row)} for row in group]
        positives = [_candidate_id(row) for row in group if int(row.get("output", {}).get("relevance_label", 0)) == 1]
        target = positives[0] if positives else _candidate_id(group[0])
        samples.append(
            {
                "sample_id": f"day10_pointwise_group_{idx}",
                "input": {"candidate_pool": pool},
                "output": {"target_item_id": target},
            }
        )
    return samples


def _safe_rankings(groups: list[list[dict]], preds: list[dict], parse_failure_score: float) -> list[list[str]]:
    rankings = []
    offset = 0
    for group in groups:
        scored = []
        for row in group:
            pred = preds[offset]
            parsed = parse_relevance(str(pred.get("raw_response", "")))
            score = float(parsed["score"]) if parsed["schema_valid"] else parse_failure_score
            cid = _candidate_id(row)
            scored.append((cid, score, cid))
            offset += 1
        scored.sort(key=lambda x: (-x[1], x[2]))
        rankings.append([x[0] for x in scored])
    return rankings


def write_audit(summary_rows: list[dict], prediction_count: int, parse_failure_score: float) -> None:
    main = next((r for r in summary_rows if r["method"] == "day10_pointwise_shuffled_safe_eval"), {})
    report = f"""# Framework-Day10 Pointwise Shuffled Leakage Audit

## Scope

Pointwise is an audited comparison route only. It is not the main baseline unless safe evaluation beats random after candidate-order randomization.

## Evaluation Rules

- scores come only from parsed model output;
- parse failure score: `{parse_failure_score}`;
- tie-break policy: lexical `candidate_item_id`;
- labels are used only for metrics, never for scoring or tie-break.

## Result

- prediction rows: `{prediction_count}`
- NDCG@10: `{main.get('NDCG@10', 'NA')}`
- MRR: `{main.get('MRR', 'NA')}`
- HR@1: `{main.get('HR@1', 'NA')}`

If this result is not clearly above random, pointwise remains diagnostic and should not be used as the Day11 baseline.
"""
    AUDIT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day10 pointwise shuffled adapter with safe aggregation.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_pointwise_shuffled_small.yaml")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_pointwise_shuffled_day10_small")
    parser.add_argument("--eval_users", type=int, default=512)
    parser.add_argument("--candidate_pool_size", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--parse_failure_score", type=float, default=0.5)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    rows = read_jsonl(Path(str(cfg["test_file"])))
    groups = _group_rows(rows, pool_size=args.candidate_pool_size)[: args.eval_users]
    pointwise_rows = [row for group in groups for row in group]
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    pred_rows = _generate_pointwise_predictions(cfg, pointwise_rows, args.adapter_path, args.max_new_tokens)
    write_jsonl(PRED_PATH, pred_rows)
    samples = _samples_from_groups(groups)
    rankings = _safe_rankings(groups, pred_rows, parse_failure_score=args.parse_failure_score)
    parse_rows = [
        {
            "parse_success": bool(row.get("parse_success", False)),
            "schema_valid": bool(row.get("schema_valid", False)),
            "invalid_item_count": 0,
            "duplicate_item_count": 0,
            "raw_output_items": 1,
        }
        for row in pred_rows
    ]
    summary_rows = [
        evaluate_rankings("day10_pointwise_shuffled_safe_eval", samples, rankings, parse_rows, tie_break_policy="lexical"),
        evaluate_rankings("random_seeded_tiebreak_same_samples", samples, random_rankings(samples, seed=42), None, tie_break_policy="seeded_random:42"),
        evaluate_rankings("oracle_positive_upper_bound", samples, oracle_rankings(samples), None, tie_break_policy="label_oracle"),
    ]
    write_csv(SUMMARY_CSV, summary_rows)
    write_audit(summary_rows, prediction_count=len(pred_rows), parse_failure_score=args.parse_failure_score)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
