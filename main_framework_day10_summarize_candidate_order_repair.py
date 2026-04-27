from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


OUT_CSV = Path("data_done/framework_day10_candidate_order_repaired_baseline_comparison.csv")
REPORT_MD = Path("data_done/framework_day10_candidate_order_randomization_report.md")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    method: str,
    dataset_version: str,
    candidate_order: str,
    tie_break_policy: str,
    task_type: str,
    max_steps: str,
    parse_success_rate: str,
    schema_valid_rate: str,
    ndcg: str,
    mrr: str,
    hr1: str,
    hr3: str,
    ndcg3: str,
    ndcg5: str,
    safe_to_use: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "dataset_version": dataset_version,
        "candidate_order": candidate_order,
        "tie_break_policy": tie_break_policy,
        "task_type": task_type,
        "max_steps": max_steps,
        "parse_success_rate": parse_success_rate,
        "schema_valid_rate": schema_valid_rate,
        "NDCG@10": ndcg,
        "MRR": mrr,
        "HR@1": hr1,
        "HR@3": hr3,
        "NDCG@3": ndcg3,
        "NDCG@5": ndcg5,
        "HR@10": "1.0",
        "hr10_trivial_flag": "true",
        "safe_to_use": safe_to_use,
        "notes": notes,
    }


def _from_summary(path: Path, method: str) -> dict[str, str] | None:
    return next((r for r in _read_csv(path) if r.get("method") == method), None)


def build_rows() -> list[dict[str, Any]]:
    rows = [
        _row("random", "beauty_5neg", "seeded_random", "seeded_random:42", "baseline", "0", "1.0", "1.0", "0.5519015474641107", "0.4102864583333333", "0.171875", "0.494140625", "0.3560228082449828", "0.48441698271903616", "true", "Random same-users baseline from Day9.5 safe eval."),
        _row("day6_listwise_v1_original", "data_done_lora/beauty", "original", "original_parser", "candidate_ranking_listwise", "100", "0.8320", "0.8027", "0.5747", "0.4197", "0.2559", "0.4727", "0.3783", "0.4749", "caution", "Weak signal but parse/schema unstable."),
        _row("day9_listwise_v2_strict_original_order", "data_done_lora/beauty", "listwise_not_fixed_but_not_day10_v2", "original_eval", "candidate_ranking_listwise", "300", "0.912109375", "0.794921875", "0.5880492311033212", "0.4470703125", "0.263671875", "0.5078125", "0.4008297958217109", "0.5107478719032167", "caution", "Best current listwise result, but not trained on Day10 shuffled v2 files."),
        _row("day9_pointwise_original_unsafe_eval", "data_done_lora/beauty", "positive_fixed_pos1", "original_candidate_order", "candidate_relevance_pointwise", "300", "0.84765625", "0.84765625", "0.9985583193498885", "0.998046875", "0.99609375", "1.0", "0.9985583193498885", "0.9985583193498885", "false", "Unsafe: near-oracle caused by candidate-order bias/tie-break artifact."),
        _row("day9_pointwise_independent_safe_eval", "data_done_lora/beauty", "positive_fixed_pos1", "lexical", "candidate_relevance_pointwise", "300", "0.84765625", "0.84765625", "0.4672424593050481", "0.30345052083333335", "0.08984375", "0.265625", "0.1879632619977701", "0.33366476413953977", "false", "Safe eval falls below random; pointwise-v1 is not usable."),
        _row("oracle", "beauty_5neg", "label_based", "label_oracle", "oracle", "0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "false", "Upper bound only, not a method."),
    ]
    day10_list = _from_summary(Path("data_done/framework_day10_listwise_strict_shuffled_eval512_summary.csv"), "day10_listwise_v2_strict_shuffled_safe_eval")
    if day10_list:
        rows.insert(
            3,
            _row(
                "day10_listwise_v2_strict_shuffled",
                "data_done_lora_v2/beauty",
                "shuffled_seed42",
                day10_list.get("tie_break_policy", "lexical"),
                "candidate_ranking_listwise",
                "300",
                day10_list.get("parse_success_rate", "NA"),
                day10_list.get("schema_valid_rate", "NA"),
                day10_list.get("NDCG@10", "NA"),
                day10_list.get("MRR", "NA"),
                day10_list.get("HR@1", "NA"),
                day10_list.get("HR@3", "NA"),
                day10_list.get("NDCG@3", "NA"),
                day10_list.get("NDCG@5", "NA"),
                "review",
                "Server-computed Day10 listwise shuffled safe eval.",
            ),
        )
    else:
        rows.insert(
            3,
            _row("day10_listwise_v2_strict_shuffled", "data_done_lora_v2/beauty", "shuffled_seed42", "lexical", "candidate_ranking_listwise", "300", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending", "Run Day10 listwise shuffled train/eval on server."),
        )
    day10_point = _from_summary(Path("data_done/framework_day10_pointwise_shuffled_safe_eval_summary.csv"), "day10_pointwise_shuffled_safe_eval")
    if day10_point:
        rows.insert(
            -1,
            _row(
                "day10_pointwise_shuffled_safe_eval",
                "data_done_lora_v2/beauty",
                "shuffled_seed42",
                day10_point.get("tie_break_policy", "lexical"),
                "candidate_relevance_pointwise",
                "300",
                day10_point.get("parse_success_rate", "NA"),
                day10_point.get("schema_valid_rate", "NA"),
                day10_point.get("NDCG@10", "NA"),
                day10_point.get("MRR", "NA"),
                day10_point.get("HR@1", "NA"),
                day10_point.get("HR@3", "NA"),
                day10_point.get("NDCG@3", "NA"),
                day10_point.get("NDCG@5", "NA"),
                "review",
                "Server-computed Day10 pointwise shuffled safe eval.",
            ),
        )
    else:
        rows.insert(
            -1,
            _row("day10_pointwise_shuffled_safe_eval", "data_done_lora_v2/beauty", "shuffled_seed42", "lexical", "candidate_relevance_pointwise", "300", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending", "Audited comparison only; run after listwise shuffled."),
        )
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    listwise = next((r for r in rows if r["method"] == "day10_listwise_v2_strict_shuffled"), {})
    pointwise = next((r for r in rows if r["method"] == "day10_pointwise_shuffled_safe_eval"), {})
    report = f"""# Framework-Day10 Candidate Order Randomization Report

## 1. Day9.5 Recap

Day9.5 showed that the pointwise near-oracle result was an order-bias artifact: old pointwise positives were fixed at position 1, and the original evaluator used original candidate order as the tie-break.

## 2. Why Candidate-Order Randomization Is Necessary

Candidate order must be neutral before training or evaluating Qwen-LoRA baselines. Otherwise parse failures, tied scores, or label-like ordering can create inflated ranking metrics.

## 3. Shuffled Data Construction

Day10 writes Beauty-only shuffled instruction data under `data_done_lora_v2/beauty/` and does not overwrite old `data_done_lora/beauty/`.

## 4. Positive Position Diagnostics

See `data_done/framework_day10_candidate_order_diagnostics.csv`. Old pointwise has positive position-1 rate `1.0`; shuffled pointwise spreads positives across positions 1-6.

## 5. Listwise Strict Shuffled Train/Eval

- status / NDCG@10: `{listwise.get('NDCG@10', 'pending_server_run')}`
- tie-break: `{listwise.get('tie_break_policy', 'lexical')}`

This is the primary Day10 candidate baseline path.

## 6. Pointwise Shuffled Audited Eval

- status / NDCG@10: `{pointwise.get('NDCG@10', 'pending_server_run')}`
- role: audited comparison only, not the main baseline unless safe eval clearly beats random.

## 7. Comparison

See `data_done/framework_day10_candidate_order_repaired_baseline_comparison.csv`.

## 8. Decision

If listwise strict shuffled is stable and beats random under safe eval, use it as the Day11 baseline candidate. If both listwise and pointwise remain weak, Day11 should redesign the baseline target rather than enter CEP.

## 9. Boundary

Day10 does not call APIs, train four domains, or implement confidence/evidence/CEP framework.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    _write_csv(OUT_CSV, rows)
    write_report(rows)


if __name__ == "__main__":
    main()
