from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


COMPARISON_CSV = Path("data_done/framework_day9_qwen_lora_baseline_formulation_comparison.csv")
REPORT_PATH = Path("data_done/framework_day9_qwen_lora_baseline_formulation_report.md")


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    method: str,
    train_samples: str,
    max_steps: str,
    task_type: str,
    prompt_style: str,
    parse_success_rate: str,
    schema_valid_rate: str,
    ndcg: str,
    mrr: str,
    hr1: str,
    hr3: str,
    ndcg3: str,
    ndcg5: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "train_samples": train_samples,
        "max_steps": max_steps,
        "task_type": task_type,
        "prompt_style": prompt_style,
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
        "notes": notes,
    }


def build_rows() -> list[dict[str, Any]]:
    rows = [
        _row("random_ranking_same_512", "0", "0", "baseline", "random", "1.0", "1.0", "0.5532", "0.4114", "0.1680", "0.5234", "0.3677", "0.4955", "Day7/Day8 random baseline on same 512 Beauty 5neg samples."),
        _row("day6_listwise_v1_original_prompt", "512", "100", "candidate_ranking_listwise", "original", "0.8320", "0.8027", "0.5747", "0.4197", "0.2559", "0.4727", "0.3783", "0.4749", "Existing Day6 adapter evaluated on Day7 512 samples."),
        _row("day8_listwise_v1_strict_inference_prompt", "512", "100", "candidate_ranking_listwise", "strict_infer_only", "0.8086", "0.7910", "0.5618", "0.3995", "NA", "NA", "NA", "NA", "Strict inference prompt without strict training caused prompt mismatch."),
    ]
    strict_rows = _read_csv(Path("data_done/framework_day9_listwise_strict_eval512_summary.csv"))
    strict = next((r for r in strict_rows if r.get("method") == "day9_listwise_v2_strict_train_strict_infer"), None)
    if strict:
        rows.append(
            _row(
                "day9_listwise_v2_strict_train_strict_infer",
                "622",
                "300",
                "candidate_ranking_listwise",
                "json_strict_train_and_infer",
                strict.get("parse_success_rate", "NA"),
                strict.get("schema_valid_rate", "NA"),
                strict.get("NDCG@10", "NA"),
                strict.get("MRR", "NA"),
                strict.get("HR@1", "NA"),
                strict.get("HR@3", "NA"),
                strict.get("NDCG@3", "NA"),
                strict.get("NDCG@5", "NA"),
                "Server-computed strict listwise-v2 result.",
            )
        )
    else:
        rows.append(_row("day9_listwise_v2_strict_train_strict_infer", "622", "300", "candidate_ranking_listwise", "json_strict_train_and_infer", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "Run Day9 listwise strict train/eval on server."))

    pointwise_rows = _read_csv(Path("data_done/framework_day9_pointwise_eval_summary.csv"))
    pointwise = next((r for r in pointwise_rows if r.get("method") == "day9_pointwise_v1_aggregated_ranking"), None)
    if pointwise:
        rows.append(
            _row(
                "day9_pointwise_v1_aggregated_ranking",
                "2000",
                "300",
                "candidate_relevance_pointwise",
                "pointwise_relevance",
                pointwise.get("parse_success_rate", "NA"),
                pointwise.get("schema_valid_rate", "NA"),
                pointwise.get("NDCG@10", "NA"),
                pointwise.get("MRR", "NA"),
                pointwise.get("HR@1", "NA"),
                pointwise.get("HR@3", "NA"),
                pointwise.get("NDCG@3", "NA"),
                pointwise.get("NDCG@5", "NA"),
                "Server-computed pointwise-v1 aggregated ranking result.",
            )
        )
    else:
        rows.append(_row("day9_pointwise_v1_aggregated_ranking", "2000", "300", "candidate_relevance_pointwise", "pointwise_relevance", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "pending_server_run", "Run Day9 pointwise train/eval on server."))
    rows.append(_row("oracle_positive_upper_bound", "0", "0", "oracle", "label_based", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0", "Sanity upper bound only, not a method."))
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    strict = next((r for r in rows if r["method"] == "day9_listwise_v2_strict_train_strict_infer"), {})
    pointwise = next((r for r in rows if r["method"] == "day9_pointwise_v1_aggregated_ranking"), {})
    report = f"""# Framework-Day9 Qwen-LoRA Baseline Formulation Report

## 1. Day8 Recap

Day8 showed that parser-only repair was marginal and strict inference prompt hurt the existing adapter because of prompt mismatch. The baseline problem is therefore formulation stability, not just parsing.

## 2. Why Formulation Repair

The CEP/confidence/evidence framework should not be layered on top of an unstable local recommender. Day9 compares listwise-v1, listwise-v2 strict train/infer, and pointwise-v1 aggregation before any framework fusion.

## 3. Listwise-v2 Strict Train/Infer

- status: `{strict.get('NDCG@10', 'pending_server_run')}`
- prompt style: `json_strict_train_and_infer`
- target: closed-catalog full candidate ranking JSON

If this row remains `pending_server_run`, run the Day9 server commands after pulling the branch.

## 4. Pointwise-v1 Train/Aggregate

- status: `{pointwise.get('NDCG@10', 'pending_server_run')}`
- formulation: candidate relevance label per item, aggregated into per-user ranking
- note: this is a raw relevance baseline, not calibrated probability and not CEP.

## 5. Comparison With Random and Day6

See `data_done/framework_day9_qwen_lora_baseline_formulation_comparison.csv`. HR@10 remains trivial for Beauty 5neg; use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## 6. Baseline Choice

Choose the next baseline only after server rows are filled. If pointwise-v1 is more stable, use it for larger Beauty training. If listwise-v2 is stronger, keep listwise. If both are weak, revise training target/data volume before entering CEP.

## 7. Day10 Recommendation

Do not enter confidence/evidence/CEP framework yet unless a baseline formulation clearly beats random with good parse/schema stability. Day10 should either scale the better formulation or revise the baseline prompt/data target.

## 8. Boundary

Day9 is still baseline-only. It does not call APIs, train four domains, use calibrated confidence, use evidence risk, or implement CEP fusion.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    _write_csv(COMPARISON_CSV, rows)
    write_report(rows)


if __name__ == "__main__":
    main()
