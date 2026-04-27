from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_day6_eval_qwen_lora_listwise import (
    _candidate_ids,
    _complete_ranking,
    _generate_lora_predictions,
    _oracle_rankings,
    _random_rankings,
    _read_jsonl,
    _rank_metrics_for_sample,
    _target_id,
    evaluate_rankings,
    parse_ranking,
)
from main_framework_day7_qwen_lora_eval_diagnosis import _run_base_comparison


DAY7_PRED = Path("output-repaired/framework/day7_qwen_lora_beauty_listwise_eval512_predictions.jsonl")
DAY6_PRED = Path("output-repaired/framework/day6_qwen_lora_beauty_listwise_predictions.jsonl")
DAY8_PRED = Path("output-repaired/framework/day8_qwen_lora_beauty_listwise_eval512_repaired_predictions.jsonl")
TAXONOMY_CSV = Path("data_done/framework_day8_parse_failure_taxonomy.csv")
EXAMPLES_REVIEW = Path("data_done/framework_day8_parse_failure_examples_review.md")
BEFORE_AFTER_CSV = Path("data_done/framework_day8_parser_repair_before_after.csv")
REPAIRED_SUMMARY_CSV = Path("data_done/framework_day8_repaired_eval512_summary.csv")
BASE_VS_LORA_CSV = Path("data_done/framework_day8_base_vs_lora_strict_prompt_comparison.csv")
REPORT = Path("data_done/framework_day8_qwen_lora_output_repair_report.md")


def _find_predictions() -> Path | None:
    if DAY7_PRED.exists():
        return DAY7_PRED
    if DAY6_PRED.exists():
        return DAY6_PRED
    root = Path("output-repaired/framework")
    if not root.exists():
        return None
    candidates = sorted(root.glob("*qwen*lora*beauty*listwise*prediction*.jsonl"))
    return candidates[0] if candidates else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _find_predictions() -> Path | None:
    if DAY7_PRED.exists():
        return DAY7_PRED
    if DAY6_PRED.exists():
        return DAY6_PRED
    root = Path("output-repaired/framework")
    if not root.exists():
        return None
    candidates = sorted(root.glob("*qwen_lora*beauty*listwise*prediction*.jsonl"))
    return candidates[0] if candidates else None


def _raw_rankings_from_preds(preds: list[dict[str, Any]]) -> list[list[str]]:
    rankings = []
    for pred in preds:
        pool = [str(x) for x in pred.get("candidate_pool", [])]
        ranked = [str(x) for x in pred.get("ranked_item_ids", [])]
        rankings.append(_complete_ranking(ranked, pool) if pred.get("parse_success") else [])
    return rankings


def _repaired_preds(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for pred in preds:
        pool = [str(x) for x in pred.get("candidate_pool", [])]
        parsed = parse_ranking(str(pred.get("raw_response", "")), pool)
        row = dict(pred)
        row.update(
            {
                "ranked_item_ids": parsed["ranked_item_ids"],
                "raw_ranked_item_ids": parsed.get("raw_ranked_item_ids", []),
                "parse_success_raw": bool(pred.get("parse_success", False)),
                "schema_valid_raw": bool(pred.get("schema_valid", False)),
                "parse_success_repaired": parsed.get("parse_success_repaired", parsed["parse_success"]),
                "schema_valid": parsed["schema_valid"],
                "parse_success": parsed["parse_success"],
                "invalid_item_count": parsed["invalid_item_count"],
                "duplicate_item_count": parsed["duplicate_item_count"],
                "repaired_item_count": parsed.get("repaired_item_count", 0),
                "parser_repair_used": parsed.get("parser_repair_used", False),
                "ranking_key_used": parsed.get("ranking_key_used", ""),
                "parse_error": parsed["parse_error"],
            }
        )
        out.append(row)
    return out


def _sample_from_pred(pred: dict[str, Any]) -> dict[str, Any]:
    pool = [{"candidate_item_id": iid} for iid in pred.get("candidate_pool", [])]
    return {
        "sample_id": pred.get("sample_id", ""),
        "input": {"candidate_pool": pool},
        "output": {"target_item_id": pred.get("target_item_id", "")},
    }


def _metrics_from_preds(method: str, preds: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [_sample_from_pred(p) for p in preds]
    rankings = _raw_rankings_from_preds(preds)
    return evaluate_rankings(method, samples, rankings, preds)


def _taxonomy_flags(pred: dict[str, Any]) -> dict[str, Any]:
    raw = str(pred.get("raw_response", ""))
    parsed_json = None
    block = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if block:
        parsed_json = block.group(1)
    elif "{" in raw and "}" in raw:
        parsed_json = raw[raw.find("{") : raw.rfind("}") + 1]
    flags = {
        "non_json": bool(raw.strip()) and parsed_json is None,
        "extra_explanatory_text": bool(parsed_json) and raw.strip() != parsed_json.strip() and not raw.strip().startswith("```"),
        "missing_ranked_item_ids": False,
        "alternative_key_name": False,
        "incomplete_list": False,
        "repeated_item_ids": int(pred.get("duplicate_item_count", 0)) > 0,
        "malformed_quotes_or_brackets": False,
        "uses_item_titles_instead_of_ids": False,
    }
    if parsed_json:
        try:
            obj = json.loads(parsed_json)
            keys = set(obj)
            aliases = {"ranked_items", "recommendations", "recommended_items", "item_ids", "ranking"}
            flags["missing_ranked_item_ids"] = "ranked_item_ids" not in keys
            flags["alternative_key_name"] = bool(keys & aliases)
            ranked = obj.get("ranked_item_ids")
            for alias in aliases:
                if ranked is None and alias in obj:
                    ranked = obj[alias]
            pool = set(str(x) for x in pred.get("candidate_pool", []))
            if isinstance(ranked, list):
                flags["incomplete_list"] = len(ranked) < len(pool)
                for item in ranked:
                    value = item
                    if isinstance(item, dict):
                        value = item.get("candidate_item_id") or item.get("item_id") or item.get("id") or ""
                    if str(value) and str(value) not in pool:
                        flags["uses_item_titles_instead_of_ids"] = True
            else:
                flags["incomplete_list"] = True
        except Exception:
            flags["malformed_quotes_or_brackets"] = True
    return flags


def build_taxonomy(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = [_taxonomy_flags(p) for p in preds]
    n = len(flags)
    row = {"num_samples": n}
    for key in [
        "non_json",
        "extra_explanatory_text",
        "missing_ranked_item_ids",
        "alternative_key_name",
        "incomplete_list",
        "repeated_item_ids",
        "malformed_quotes_or_brackets",
        "uses_item_titles_instead_of_ids",
    ]:
        row[f"{key}_rate"] = mean([1.0 if f[key] else 0.0 for f in flags]) if n else 0.0
        row[f"{key}_count"] = sum(1 for f in flags if f[key])
    return [row]


def write_examples_review(preds: list[dict[str, Any]]) -> None:
    rows = []
    for pred in preds:
        if not pred.get("parse_success") or not pred.get("schema_valid"):
            flags = _taxonomy_flags(pred)
            reasons = [k for k, v in flags.items() if v]
            rows.append(
                {
                    "sample_id": pred.get("sample_id", ""),
                    "target_item_id": pred.get("target_item_id", ""),
                    "candidate_pool": pred.get("candidate_pool", []),
                    "reasons": reasons,
                    "raw_response": str(pred.get("raw_response", ""))[:1200],
                }
            )
        if len(rows) >= 20:
            break
    lines = ["# Framework-Day8 Parse Failure Examples Review", ""]
    for i, row in enumerate(rows, 1):
        lines.extend(
            [
                f"## Example {i}",
                "",
                f"- sample_id: `{row['sample_id']}`",
                f"- target_item_id: `{row['target_item_id']}`",
                f"- reasons: `{', '.join(row['reasons']) if row['reasons'] else 'unknown'}`",
                f"- candidate_pool: `{row['candidate_pool']}`",
                "",
                "```text",
                row["raw_response"],
                "```",
                "",
            ]
        )
    EXAMPLES_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_REVIEW.write_text("\n".join(lines), encoding="utf-8")


def build_before_after(eval_set: str, original_preds: list[dict[str, Any]], repaired_preds: list[dict[str, Any]]) -> dict[str, Any]:
    before = _metrics_from_preds("before", original_preds)
    after = _metrics_from_preds("after", repaired_preds)
    total_items = sum(len(p.get("raw_ranked_item_ids", p.get("ranked_item_ids", []))) for p in repaired_preds)
    repaired_items = sum(int(p.get("repaired_item_count", 0)) for p in repaired_preds)
    return {
        "eval_set": eval_set,
        "num_samples": len(original_preds),
        "raw_parse_success_rate": before["parse_success_rate"],
        "repaired_parse_success_rate": after["parse_success_rate"],
        "raw_schema_valid_rate": before["schema_valid_rate"],
        "repaired_schema_valid_rate": after["schema_valid_rate"],
        "invalid_item_rate": after["invalid_item_rate"],
        "duplicate_item_rate": after["duplicate_item_rate"],
        "repaired_item_rate": repaired_items / total_items if total_items else 0.0,
        "NDCG@10_before": before["NDCG@10"],
        "NDCG@10_after": after["NDCG@10"],
        "MRR_before": before["MRR"],
        "MRR_after": after["MRR"],
        "HR@1_before": before["HR@1"],
        "HR@1_after": after["HR@1"],
        "HR@3_before": before["HR@3"],
        "HR@3_after": after["HR@3"],
        "NDCG@3_before": before["NDCG@3"],
        "NDCG@3_after": after["NDCG@3"],
        "NDCG@5_before": before["NDCG@5"],
        "NDCG@5_after": after["NDCG@5"],
    }


def _run_strict_eval(config_path: Path, adapter_path: str, eval_samples: int) -> list[dict[str, Any]]:
    cfg = _read_config(config_path)
    cfg["prompt_template"] = "prompts/framework/qwen_candidate_ranking_baseline_json_strict.txt"
    samples = _read_jsonl(Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_listwise.jsonl")), limit=eval_samples)
    pred_rows, rankings = _generate_lora_predictions(cfg, samples, adapter_path, max_new_tokens=128)
    _write_jsonl(DAY8_PRED, pred_rows)
    summary = evaluate_rankings("strict_prompt_repaired_parser", samples, rankings, pred_rows)
    random_row = evaluate_rankings("random_ranking_same_samples", samples, _random_rankings(samples), None)
    return [summary, random_row]


def _write_report(taxonomy: list[dict[str, Any]], before_after: list[dict[str, Any]], strict_rows: list[dict[str, Any]] | None, base_status: str) -> None:
    tax = taxonomy[0] if taxonomy else {}
    ba = before_after[0] if before_after else {}
    strict = strict_rows[0] if strict_rows else {}
    report = f"""# Framework-Day8 Qwen-LoRA Output Repair Report

## 1. Day7 Recap

Day7 showed Qwen-LoRA has some signal on 512 samples, but parse/schema stability remains weak. Day8 repairs output handling before any longer training or CEP integration.

## 2. Failure Taxonomy

- non-JSON rate: `{tax.get('non_json_rate', 'NA')}`
- extra explanatory text rate: `{tax.get('extra_explanatory_text_rate', 'NA')}`
- missing ranked_item_ids rate: `{tax.get('missing_ranked_item_ids_rate', 'NA')}`
- alternative key rate: `{tax.get('alternative_key_name_rate', 'NA')}`
- incomplete list rate: `{tax.get('incomplete_list_rate', 'NA')}`
- title/non-ID output rate: `{tax.get('uses_item_titles_instead_of_ids_rate', 'NA')}`

## 3. Parser-Only Repair

- raw parse success: `{ba.get('raw_parse_success_rate', 'NA')}`
- repaired parse success: `{ba.get('repaired_parse_success_rate', 'NA')}`
- raw NDCG@10: `{ba.get('NDCG@10_before', 'NA')}`
- repaired NDCG@10: `{ba.get('NDCG@10_after', 'NA')}`
- raw MRR: `{ba.get('MRR_before', 'NA')}`
- repaired MRR: `{ba.get('MRR_after', 'NA')}`

## 4. Generation Config Repair

Generation is deterministic: `do_sample=false`; no `temperature`, `top_p`, or `top_k`; `max_new_tokens=128`; EOS/PAD IDs are set from the tokenizer.

## 5. Strict Prompt Re-Eval

- strict prompt status: `{'completed' if strict_rows else 'skipped'}`
- strict NDCG@10: `{strict.get('NDCG@10', 'NA')}`
- strict MRR: `{strict.get('MRR', 'NA')}`
- strict parse success: `{strict.get('parse_success_rate', 'NA')}`
- strict schema valid: `{strict.get('schema_valid_rate', 'NA')}`

## 6. Base Qwen Comparison

Status: `{base_status}`.

## 7. Day9 Recommendation

If parse success reaches at least `0.95` and strict prompt metrics clearly beat random, proceed to longer Beauty listwise training. If parse improves but metrics remain weak, revise training length/target before entering CEP. If parse remains poor, further prompt/schema repair is needed.

## 8. Boundary

Day8 performs parser/generation/prompt repair only. It does not train, call APIs, or implement confidence/evidence/CEP framework.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day8 Qwen-LoRA output parser/prompt repair.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_small.yaml")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_listwise_day6_small")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--run_strict_eval", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--run_base_comparison", action="store_true")
    parser.add_argument("--base_eval_samples", type=int, default=128)
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else None
    if pred_path is None or not pred_path.exists():
        pred_path = DAY7_PRED if DAY7_PRED.exists() else _find_predictions()

    taxonomy: list[dict[str, Any]]
    before_after: list[dict[str, Any]]
    if pred_path and pred_path.exists():
        original_preds = _read_jsonl(pred_path)
        repaired_preds = _repaired_preds(original_preds)
        taxonomy = build_taxonomy(original_preds)
        before_after = [build_before_after(pred_path.name, original_preds, repaired_preds)]
        write_examples_review(original_preds)
    else:
        taxonomy = [{"num_samples": 0, "notes": "prediction JSONL not found locally; run on server"}]
        before_after = [{"eval_set": "missing_predictions", "num_samples": 0, "notes": "run on server with Day7 predictions"}]
        EXAMPLES_REVIEW.parent.mkdir(parents=True, exist_ok=True)
        EXAMPLES_REVIEW.write_text("# Framework-Day8 Parse Failure Examples Review\n\nPrediction JSONL not found locally. Run this script on the server.\n", encoding="utf-8")

    _write_csv(TAXONOMY_CSV, taxonomy)
    _write_csv(BEFORE_AFTER_CSV, before_after)

    strict_rows = None
    if args.run_strict_eval:
        strict_rows = _run_strict_eval(Path(args.config), args.adapter_path, args.eval_samples)
        _write_csv(REPAIRED_SUMMARY_CSV, strict_rows)
    else:
        _write_csv(REPAIRED_SUMMARY_CSV, [{"status": "skipped", "reason": "run with --run_strict_eval on server"}])

    base_status = "skipped_due_to_runtime"
    if args.run_base_comparison:
        base_rows = _run_base_comparison(Path(args.config), args.base_eval_samples, pred_path)
        _write_csv(BASE_VS_LORA_CSV, base_rows)
        base_status = "completed"
    else:
        _write_csv(BASE_VS_LORA_CSV, [{"method": "base_qwen_strict_prompt", "status": "skipped_due_to_runtime"}])

    _write_report(taxonomy, before_after, strict_rows, base_status)
    print(json.dumps({"taxonomy": taxonomy, "before_after": before_after, "strict": strict_rows, "base_status": base_status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
