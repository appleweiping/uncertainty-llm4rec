from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from main_framework_day9_eval_qwen_lora_pointwise import parse_relevance


DATA_ROOT = Path("data_done")
LORA_ROOT = Path("data_done_lora/beauty")
PRED_PATH = Path("output-repaired/framework/day9_qwen_lora_beauty_pointwise_predictions.jsonl")
POINTWISE_SUMMARY = Path("data_done/framework_day9_pointwise_eval_summary.csv")
COMPARISON_CSV = Path("data_done/framework_day9_qwen_lora_baseline_formulation_comparison.csv")

PROMPT_LEAKAGE_CSV = DATA_ROOT / "framework_day95_pointwise_prompt_leakage_check.csv"
SUSPICIOUS_EXAMPLES = DATA_ROOT / "framework_day95_pointwise_suspicious_examples.jsonl"
OVERLAP_CSV = DATA_ROOT / "framework_day95_pointwise_split_overlap_check.csv"
OVERLAP_EXAMPLES = DATA_ROOT / "framework_day95_pointwise_overlap_examples.jsonl"
EVAL_LOGIC_MD = DATA_ROOT / "framework_day95_pointwise_eval_logic_audit.md"
ORDER_BIAS_CSV = DATA_ROOT / "framework_day95_candidate_order_bias.csv"
PRED_DIST_CSV = DATA_ROOT / "framework_day95_pointwise_prediction_distribution.csv"
INDEPENDENT_EVAL_CSV = DATA_ROOT / "framework_day95_pointwise_independent_eval.csv"
REPORT_MD = DATA_ROOT / "framework_day95_pointwise_leakage_audit_report.md"
INGEST_REPORT_MD = DATA_ROOT / "framework_day9_server_result_ingest_report.md"


SUSPICIOUS_KEYS = {
    "label",
    "relevance_label",
    "target",
    "target_item_id",
    "positive",
    "is_positive",
    "answer",
    "gold",
    "ground_truth",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _iter_keys(obj: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            p = f"{prefix}.{key}" if prefix else str(key)
            keys.append(p)
            keys.extend(_iter_keys(value, p))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj[:20]):
            keys.extend(_iter_keys(value, f"{prefix}[{idx}]"))
    return keys


def _has_suspicious_key(obj: Any) -> tuple[int, list[str]]:
    keys = _iter_keys(obj)
    hits = []
    for key in keys:
        leaf = key.split(".")[-1].split("[")[0]
        if leaf in SUSPICIOUS_KEYS:
            hits.append(key)
    return len(hits), hits


def prompt_leakage_check() -> list[dict[str, Any]]:
    rows = []
    examples = []
    for split in ["train", "valid", "test"]:
        data = _read_jsonl(LORA_ROOT / f"{split}_pointwise.jsonl")
        checked = data[: min(len(data), 5000)]
        label_in_input = 0
        label_in_meta = 0
        target_in_input = 0
        suspicious_count = 0
        for row in checked:
            input_hits, input_keys = _has_suspicious_key(row.get("input", {}))
            meta_hits, meta_keys = _has_suspicious_key(row.get("metadata", {}))
            if "label" in {k.split(".")[-1] for k in input_keys} or "relevance_label" in {k.split(".")[-1] for k in input_keys}:
                label_in_input += 1
            if "label" in {k.split(".")[-1] for k in meta_keys} or "relevance_label" in {k.split(".")[-1] for k in meta_keys}:
                label_in_meta += 1
            if any(k.split(".")[-1] in {"target", "target_item_id"} for k in input_keys):
                target_in_input += 1
            if input_hits or meta_hits:
                suspicious_count += 1
                if len(examples) < 50:
                    examples.append(
                        {
                            "split": split,
                            "sample_id": row.get("sample_id"),
                            "input_suspicious_keys": input_keys,
                            "metadata_suspicious_keys": meta_keys,
                            "input": row.get("input", {}),
                            "metadata": row.get("metadata", {}),
                        }
                    )
        rows.append(
            {
                "split": split,
                "num_samples_checked": len(checked),
                "label_in_input_count": label_in_input,
                "label_in_metadata_count": label_in_meta,
                "target_in_input_count": target_in_input,
                "suspicious_field_count": suspicious_count,
                "suspicious_examples_path": str(SUSPICIOUS_EXAMPLES) if examples else "",
                "status": "pass" if suspicious_count == 0 else "needs_review",
            }
        )
    _write_jsonl(SUSPICIOUS_EXAMPLES, examples)
    _write_csv(PROMPT_LEAKAGE_CSV, rows)
    return rows


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("input", {}).get("candidate_item", {}).get("candidate_item_id", "")).strip()


def _history_signature(row: dict[str, Any]) -> str:
    hist = row.get("input", {}).get("user_history", [])
    ids = [str(x.get("item_id", "")) for x in hist]
    raw = "|".join(ids)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def split_overlap_check() -> list[dict[str, Any]]:
    split_rows = {split: _read_jsonl(LORA_ROOT / f"{split}_pointwise.jsonl") for split in ["train", "valid", "test"]}
    keysets: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for split, rows in split_rows.items():
        keysets["candidate_item_id"][split] = {_candidate_id(row) for row in rows}
        keysets["history_signature_candidate"][split] = {_history_signature(row) + "::" + _candidate_id(row) for row in rows}
        keysets["sample_id"][split] = {str(row.get("sample_id", "")) for row in rows}

    out = []
    examples = []
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for overlap_type, by_split in keysets.items():
        for a, b in pairs:
            overlap = sorted(by_split[a] & by_split[b])
            denom = max(1, min(len(by_split[a]), len(by_split[b])))
            status = "pass"
            if overlap_type in {"history_signature_candidate", "sample_id"} and overlap:
                status = "needs_review"
            out.append(
                {
                    "overlap_type": f"{overlap_type}:{a}_vs_{b}",
                    "num_overlap": len(overlap),
                    "rate": len(overlap) / denom,
                    "examples_path": str(OVERLAP_EXAMPLES) if overlap else "",
                    "status": status,
                }
            )
            for value in overlap[:20]:
                examples.append({"overlap_type": overlap_type, "split_a": a, "split_b": b, "value": value})
    out.append(
        {
            "overlap_type": "user_id_candidate_item_id",
            "num_overlap": "NA",
            "rate": "NA",
            "examples_path": "",
            "status": "not_applicable_user_id_missing_in_data_done_lora_pointwise",
        }
    )
    _write_jsonl(OVERLAP_EXAMPLES, examples)
    _write_csv(OVERLAP_CSV, out)
    return out


def _group_rows(rows: list[dict[str, Any]], pool_size: int = 6) -> list[list[dict[str, Any]]]:
    return [rows[i : i + pool_size] for i in range(0, len(rows), pool_size) if len(rows[i : i + pool_size]) == pool_size]


def candidate_order_bias() -> list[dict[str, Any]]:
    out = []
    for split in ["train", "valid", "test"]:
        groups = _group_rows(_read_jsonl(LORA_ROOT / f"{split}_pointwise.jsonl"))
        counts = Counter()
        for group in groups:
            for idx, row in enumerate(group, 1):
                if int(row.get("output", {}).get("relevance_label", 0)) == 1:
                    counts[idx] += 1
                    break
        total = sum(counts.values())
        for pos in range(1, 7):
            out.append(
                {
                    "split": split,
                    "position": pos,
                    "positive_count": counts[pos],
                    "positive_rate": counts[pos] / total if total else 0.0,
                }
            )
    _write_csv(ORDER_BIAS_CSV, out)
    return out


def write_eval_logic_audit(order_rows: list[dict[str, Any]]) -> None:
    test_pos1 = next((r for r in order_rows if r["split"] == "test" and int(r["position"]) == 1), {})
    report = f"""# Framework-Day9.5 Pointwise Eval Logic Audit

## Score Source

The Day9 pointwise evaluator uses `relevance_score` parsed from model output if present; otherwise it converts parsed `relevance_label` into score `1.0` or `0.0`.

## Parse Failure Fallback

In the original Day9 evaluator, `parse_relevance()` returns score `0.0` on parse failure. This means failed candidates are demoted, not promoted.

## Tie-Breaker

The original Day9 evaluator sorts by `(-score, local_idx)`, where `local_idx` is the candidate's original order within the 6-candidate group. This is unsafe if candidate order is label-biased.

## Candidate Order

The pointwise files are grouped in candidate-pool order. Current local audit shows the test positive-at-position-1 rate is `{test_pos1.get('positive_rate', 'NA')}`. If this is high, a tie or parse-failure-heavy evaluator that preserves original order can produce inflated HR@1/NDCG.

## Label Leakage Status

Static code inspection found no direct use of `target_label` during ranking score construction, but the original tie-breaker can leak through order bias if positives are consistently first. Day9.5 independent safe eval therefore replaces original-order tie-break with lexical/random-neutral tie-break and recomputes metrics from prediction outputs when server predictions are available.
"""
    EVAL_LOGIC_MD.write_text(report, encoding="utf-8")


def prediction_distribution(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not preds:
        rows = [
            {
                "true_label": "missing_server_artifact",
                "predicted_label": "missing_server_artifact",
                "count": 0,
                "rate": 0,
                "parse_success_rate": 0,
                "mean_score_if_available": "NA",
            }
        ]
        _write_csv(PRED_DIST_CSV, rows)
        return rows
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        true_label = str(row.get("target_label", "NA"))
        pred_label = str(row.get("parsed_relevance_label", "NA"))
        buckets[(true_label, pred_label)].append(row)
    rows = []
    total = len(preds)
    for (true_label, pred_label), bucket in sorted(buckets.items()):
        scores = [float(x.get("relevance_score", 0.0)) for x in bucket if str(x.get("relevance_score", "")) != ""]
        rows.append(
            {
                "true_label": true_label,
                "predicted_label": pred_label,
                "count": len(bucket),
                "rate": len(bucket) / total if total else 0.0,
                "parse_success_rate": sum(1 for x in bucket if x.get("parse_success")) / len(bucket) if bucket else 0,
                "mean_score_if_available": mean(scores) if scores else "NA",
            }
        )
    _write_csv(PRED_DIST_CSV, rows)
    return rows


def _metrics_from_ranks(ranks: list[int], pool_size: int = 6) -> dict[str, float]:
    if not ranks:
        return {"NDCG@10": 0.0, "MRR": 0.0, "HR@1": 0.0, "HR@3": 0.0, "NDCG@3": 0.0, "NDCG@5": 0.0, "HR@10": 0.0}
    rows = []
    for rank in ranks:
        rows.append(
            {
                "MRR": 1.0 / rank,
                "HR@1": 1.0 if rank <= 1 else 0.0,
                "HR@3": 1.0 if rank <= 3 else 0.0,
                "HR@10": 1.0 if rank <= 10 else 0.0,
                "NDCG@3": 1.0 / math.log2(rank + 1) if rank <= 3 else 0.0,
                "NDCG@5": 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0,
                "NDCG@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
            }
        )
    return {key: mean([row[key] for row in rows]) for key in rows[0]}


def independent_safe_eval(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_groups = _group_rows(_read_jsonl(LORA_ROOT / "test_pointwise.jsonl"))[:512]
    if not preds:
        rng = random.Random(42)
        random_ranks = []
        oracle_ranks = []
        for group in test_groups:
            pool = [_candidate_id(row) for row in group]
            positives = [_candidate_id(row) for row in group if int(row.get("output", {}).get("relevance_label", 0)) == 1]
            target = positives[0] if positives else (pool[0] if pool else "")
            shuffled = pool[:]
            rng.shuffle(shuffled)
            random_ranks.append(shuffled.index(target) + 1 if target in shuffled else 7)
            oracle_ranks.append(1)
        rows = [
            {
                "method": "original_day9_pointwise_eval",
                "status": "missing_server_artifact",
                "NDCG@10": "NA",
                "MRR": "NA",
                "HR@1": "NA",
                "HR@3": "NA",
                "NDCG@3": "NA",
                "NDCG@5": "NA",
                "notes": f"Need sync: {PRED_PATH}",
            },
            {
                "method": "independent_safe_eval",
                "status": "missing_server_artifact",
                "NDCG@10": "NA",
                "MRR": "NA",
                "HR@1": "NA",
                "HR@3": "NA",
                "NDCG@3": "NA",
                "NDCG@5": "NA",
                "notes": f"Need sync: {PRED_PATH}",
            },
            {
                "method": "random_same_users",
                "status": "computed_without_predictions",
                **_metrics_from_ranks(random_ranks),
                "notes": "Deterministic seed=42 random ranking on same local test users.",
            },
            {
                "method": "oracle",
                "status": "computed_without_predictions",
                **_metrics_from_ranks(oracle_ranks),
                "notes": "Label-based upper bound only.",
            },
        ]
        _write_csv(INDEPENDENT_EVAL_CSV, rows)
        return rows

    rng = random.Random(42)
    safe_ranks = []
    random_ranks = []
    oracle_ranks = []
    for group_idx, group in enumerate(test_groups):
        group_preds = preds[group_idx * 6 : group_idx * 6 + 6]
        scored = []
        target = ""
        for row, pred in zip(group, group_preds):
            cid = _candidate_id(row)
            if int(row.get("output", {}).get("relevance_label", 0)) == 1:
                target = cid
            parsed = parse_relevance(str(pred.get("raw_response", "")))
            score = float(parsed["score"]) if parsed["schema_valid"] else 0.5
            scored.append((cid, score, cid))
        scored.sort(key=lambda x: (-x[1], x[2]))
        ranking = [x[0] for x in scored]
        safe_ranks.append(ranking.index(target) + 1 if target in ranking else 7)
        random_pool = [x[0] for x in scored]
        rng.shuffle(random_pool)
        random_ranks.append(random_pool.index(target) + 1 if target in random_pool else 7)
        oracle_ranks.append(1)
    rows = []
    original_rows = _read_csv_dicts(POINTWISE_SUMMARY)
    original = next((r for r in original_rows if r.get("method") == "day9_pointwise_v1_aggregated_ranking"), None)
    if original:
        rows.append({"method": "original_day9_pointwise_eval", "status": "from_server_summary", **{k: original.get(k, "NA") for k in ["NDCG@10", "MRR", "HR@1", "HR@3", "NDCG@3", "NDCG@5"]}, "notes": "Original Day9 evaluator result."})
    else:
        rows.append({"method": "original_day9_pointwise_eval", "status": "missing_summary", "NDCG@10": "NA", "MRR": "NA", "HR@1": "NA", "HR@3": "NA", "NDCG@3": "NA", "NDCG@5": "NA", "notes": f"Need sync: {POINTWISE_SUMMARY}"})
    for method, ranks, notes in [
        ("independent_safe_eval", safe_ranks, "score from parsed model output; parse failure score=0.5; lexical candidate_id tie-break."),
        ("random_same_users", random_ranks, "deterministic seed=42 random ranking on same users."),
        ("oracle", oracle_ranks, "label-based upper bound only."),
    ]:
        metrics = _metrics_from_ranks(ranks)
        rows.append({"method": method, "status": "computed", **metrics, "notes": notes})
    _write_csv(INDEPENDENT_EVAL_CSV, rows)
    return rows


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def server_ingest_report() -> None:
    missing = []
    for path in [PRED_PATH, POINTWISE_SUMMARY]:
        if not path.exists():
            missing.append(str(path))
    comparison_status = "present" if COMPARISON_CSV.exists() else "missing"
    report = f"""# Framework-Day9 Server Result Ingest Report

## Local Artifact Status

- pointwise prediction JSONL: `{'present' if PRED_PATH.exists() else 'missing_server_artifact'}`
- pointwise eval summary CSV: `{'present' if POINTWISE_SUMMARY.exists() else 'missing_server_artifact'}`
- formulation comparison CSV: `{comparison_status}`

Missing files:

```text
{chr(10).join(missing) if missing else 'none'}
```

## Interpretation Boundary

The reported Day9 pointwise-v1 server result is suspiciously near-oracle and requires leakage/evaluation audit before it can be used as baseline evidence. If local Codex cannot see the server prediction JSONL and summary, it must not infer success from copied console numbers.

## Required Sync If Missing

Please sync:

```text
output-repaired/framework/day9_qwen_lora_beauty_pointwise_predictions.jsonl
data_done/framework_day9_pointwise_eval_summary.csv
data_done/framework_day9_qwen_lora_baseline_formulation_comparison.csv
```
"""
    INGEST_REPORT_MD.write_text(report, encoding="utf-8")


def final_report(
    prompt_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    order_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    indep_rows: list[dict[str, Any]],
) -> None:
    prompt_status = "pass" if all(r["status"] == "pass" for r in prompt_rows) else "needs_review"
    pos1_test = next((r for r in order_rows if r["split"] == "test" and int(r["position"]) == 1), {})
    missing_pred = not PRED_PATH.exists()
    conclusion = "missing_server_artifact"
    if not missing_pred:
        safe = next((r for r in indep_rows if r.get("method") == "independent_safe_eval"), {})
        try:
            ndcg = float(safe.get("NDCG@10", 0))
            conclusion = "safe_to_use_pointwise_baseline" if ndcg > 0.9 else "eval_bug_or_order_bias_possible"
        except Exception:
            conclusion = "needs_review"
    report = f"""# Framework-Day9.5 Pointwise Leakage / Evaluation Audit Report

## 1. Why Audit

The server-reported pointwise-v1 aggregated ranking was near oracle. That is suspicious for a 300-step small Qwen-LoRA baseline and must be audited before any success claim or Day10 scaling.

## 2. Prompt/Input Leakage Check

- status: `{prompt_status}`
- output: `data_done/framework_day95_pointwise_prompt_leakage_check.csv`
- suspicious examples: `data_done/framework_day95_pointwise_suspicious_examples.jsonl`

This checks `input` and `metadata`; `output.relevance_label` is allowed because it is the supervised training target, not model input.

## 3. Split Overlap Check

- output: `data_done/framework_day95_pointwise_split_overlap_check.csv`
- note: `data_done_lora` pointwise samples do not preserve explicit `user_id`, so user_id+candidate overlap cannot be fully checked from this artifact alone.

## 4. Eval Logic Audit

- score source: parsed model `relevance_score` or parsed `relevance_label`
- parse failure fallback in original evaluator: score `0.0`
- original tie-breaker: original candidate order via `local_idx`
- concern: if positives are first in candidate order, ties/failures can inflate metrics

See `data_done/framework_day95_pointwise_eval_logic_audit.md`.

## 5. Candidate Order Bias

- test positive-at-position-1 rate: `{pos1_test.get('positive_rate', 'NA')}`
- output: `data_done/framework_day95_candidate_order_bias.csv`

If this rate is high, the original evaluator's order-preserving tie-break is unsafe.

## 6. Prediction Distribution

- prediction artifact present: `{not missing_pred}`
- output: `data_done/framework_day95_pointwise_prediction_distribution.csv`

If prediction artifact is missing, sync `output-repaired/framework/day9_qwen_lora_beauty_pointwise_predictions.jsonl` from the server and rerun this audit.

## 7. Independent Safe Eval

- output: `data_done/framework_day95_pointwise_independent_eval.csv`
- rule: scores only from parsed model output; parse failure score `0.5`; tie-break by lexical candidate_item_id, not label or original order.

## 8. Conclusion

Current audit status: `{conclusion}`.

Do not write "pointwise baseline succeeded" until the independent safe eval is computed from the server prediction JSONL and reviewed.

## 9. Day10 Recommendation

First sync the missing Day9 server artifacts if needed, rerun:

```bash
python main_framework_day95_pointwise_leakage_audit.py
```

Only if independent safe eval remains strong and no leakage/order-bias issue is found should Day10 scale the selected baseline. Do not enter confidence/evidence/CEP framework yet.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day9.5 pointwise leakage/evaluation audit.")
    parser.parse_args()
    server_ingest_report()
    prompt_rows = prompt_leakage_check()
    overlap_rows = split_overlap_check()
    order_rows = candidate_order_bias()
    write_eval_logic_audit(order_rows)
    preds = _read_jsonl(PRED_PATH)
    pred_rows = prediction_distribution(preds)
    indep_rows = independent_safe_eval(preds)
    final_report(prompt_rows, overlap_rows, order_rows, pred_rows, indep_rows)


if __name__ == "__main__":
    main()
