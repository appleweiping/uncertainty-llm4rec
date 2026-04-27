from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
)
from src.framework.prompt_formatters import format_sample


DAY6_PRED_DEFAULT = Path("output-repaired/framework/day6_qwen_lora_beauty_listwise_predictions.jsonl")
DAY7_PRED_512 = Path("output-repaired/framework/day7_qwen_lora_beauty_listwise_eval512_predictions.jsonl")
PARSE_DIAG_CSV = Path("data_done/framework_day7_parse_failure_diagnostics.csv")
PARSE_EXAMPLES = Path("data_done/framework_day7_parse_failure_examples.jsonl")
CASE_STUDY_CSV = Path("data_done/framework_day7_ranking_case_study.csv")
EVAL512_CSV = Path("data_done/framework_day7_eval_512_summary.csv")
REPAIR_PLAN = Path("data_done/framework_day7_prompt_parser_repair_plan.md")
BASE_VS_LORA = Path("data_done/framework_day7_base_vs_lora_comparison.csv")
REPORT = Path("data_done/framework_day7_qwen_lora_eval_diagnosis_report.md")


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


def _find_day6_predictions() -> Path | None:
    if DAY6_PRED_DEFAULT.exists():
        return DAY6_PRED_DEFAULT
    root = Path("output-repaired/framework")
    if not root.exists():
        return None
    candidates = sorted(root.glob("*day6*qwen*lora*beauty*listwise*prediction*.jsonl"))
    return candidates[0] if candidates else None


def _extract_json_text(text: str) -> str | None:
    if not text:
        return None
    block = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


def _classify_output(pred: dict[str, Any]) -> dict[str, Any]:
    raw = str(pred.get("raw_response", ""))
    candidate_pool = [str(x) for x in pred.get("candidate_pool", [])]
    parse_error = str(pred.get("parse_error", ""))
    text = _extract_json_text(raw)
    empty = not raw.strip()
    code_block = bool(re.search(r"```(?:json)?", raw, flags=re.IGNORECASE))
    non_json = (text is None) and not empty
    extra_text = False
    missing_key = False
    key_not_list = False
    valid_json_schema_invalid = False
    too_few_items = False
    item_not_in_pool = int(pred.get("invalid_item_count", 0)) > 0
    duplicate = int(pred.get("duplicate_item_count", 0)) > 0
    if text is not None:
        stripped = raw.strip()
        extra_text = not (stripped == text or stripped.startswith("```"))
        try:
            obj = json.loads(text)
            ranked = obj.get("ranked_item_ids")
            if "ranked_item_ids" not in obj:
                missing_key = True
            elif not isinstance(ranked, list):
                key_not_list = True
            else:
                too_few_items = len(ranked) < len(candidate_pool)
        except Exception:
            pass
    schema_valid = bool(pred.get("schema_valid", False))
    parse_success = bool(pred.get("parse_success", False))
    valid_json_schema_invalid = parse_success and not schema_valid
    if parse_error == "missing_ranked_item_ids":
        missing_key = True
    reason = parse_error or ""
    if empty:
        reason = "empty_output"
    elif non_json:
        reason = "non_json"
    elif missing_key:
        reason = "missing_ranked_item_ids"
    elif key_not_list:
        reason = "ranked_item_ids_not_list"
    elif item_not_in_pool:
        reason = "item_not_in_candidate_pool"
    elif duplicate:
        reason = "duplicate_item_id"
    elif too_few_items and not schema_valid:
        reason = "too_few_items"
    elif valid_json_schema_invalid:
        reason = "valid_json_but_schema_invalid"
    return {
        "empty_output": empty,
        "non_json": non_json,
        "missing_ranked_item_ids": missing_key,
        "ranked_item_ids_not_list": key_not_list,
        "item_not_in_candidate_pool": item_not_in_pool,
        "duplicate_item": duplicate,
        "too_few_items": too_few_items,
        "extra_text": extra_text,
        "code_block_json": code_block,
        "valid_json_but_schema_invalid": valid_json_schema_invalid,
        "failure_reason": reason,
    }


def _load_samples_by_id(test_file: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(test_file)
    return {str(row.get("sample_id", "")): row for row in rows}


def _target_rank(pred: dict[str, Any]) -> int:
    target = str(pred.get("target_item_id", ""))
    pool = [str(x) for x in pred.get("candidate_pool", [])]
    ranked = [str(x) for x in pred.get("ranked_item_ids", [])]
    full = _complete_ranking(ranked, pool) if bool(pred.get("parse_success", False)) else pool
    return full.index(target) + 1 if target in full else len(pool) + 1


def _per_sample_metric(pred: dict[str, Any]) -> dict[str, float]:
    target = str(pred.get("target_item_id", ""))
    pool = [str(x) for x in pred.get("candidate_pool", [])]
    ranked = [str(x) for x in pred.get("ranked_item_ids", [])]
    full = _complete_ranking(ranked, pool) if bool(pred.get("parse_success", False)) else pool
    return _rank_metrics_for_sample(full, target, len(pool))


def build_parse_diagnostics(preds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flags = [_classify_output(p) for p in preds]
    n = len(preds)
    row = {
        "num_samples": n,
        "parse_success_rate": mean([1.0 if p.get("parse_success") else 0.0 for p in preds]) if n else 0.0,
        "schema_valid_rate": mean([1.0 if p.get("schema_valid") else 0.0 for p in preds]) if n else 0.0,
    }
    for key in [
        "empty_output",
        "non_json",
        "missing_ranked_item_ids",
        "ranked_item_ids_not_list",
        "item_not_in_candidate_pool",
        "duplicate_item",
        "too_few_items",
        "extra_text",
        "code_block_json",
        "valid_json_but_schema_invalid",
    ]:
        row[f"{key}_rate"] = mean([1.0 if f[key] else 0.0 for f in flags]) if n else 0.0
    examples = []
    for pred, flag in zip(preds, flags):
        if not pred.get("parse_success") or not pred.get("schema_valid"):
            examples.append(
                {
                    "sample_id": pred.get("sample_id", ""),
                    "raw_output": pred.get("raw_response", ""),
                    "candidate_pool_item_ids": pred.get("candidate_pool", []),
                    "target_item_id": pred.get("target_item_id", ""),
                    "failure_reason": flag["failure_reason"],
                }
            )
        if len(examples) >= 20:
            break
    return [row], examples


def build_case_study(preds: list[dict[str, Any]], sample_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"hit1": [], "rank2_3": [], "rank_gt3": [], "parse_failure": []}
    for pred in preds:
        rank = _target_rank(pred)
        if not pred.get("parse_success") or not pred.get("schema_valid"):
            buckets["parse_failure"].append(pred)
        elif rank == 1:
            buckets["hit1"].append(pred)
        elif 2 <= rank <= 3:
            buckets["rank2_3"].append(pred)
        else:
            buckets["rank_gt3"].append(pred)
    rows = []
    for bucket, bucket_preds in buckets.items():
        for pred in bucket_preds[:20]:
            sample = sample_map.get(str(pred.get("sample_id", "")), {})
            hist = sample.get("input", {}).get("user_history", [])[:3]
            metrics = _per_sample_metric(pred)
            rows.append(
                {
                    "case_type": bucket,
                    "sample_id": pred.get("sample_id", ""),
                    "user_history_truncated": json.dumps(hist, ensure_ascii=False)[:500],
                    "candidate_pool": json.dumps(pred.get("candidate_pool", []), ensure_ascii=False),
                    "target_item_id": pred.get("target_item_id", ""),
                    "predicted_ranked_item_ids": json.dumps(pred.get("ranked_item_ids", []), ensure_ascii=False),
                    "target_rank": _target_rank(pred),
                    "NDCG": metrics["NDCG@10"],
                    "MRR": metrics["MRR"],
                    "parse_success": pred.get("parse_success", False),
                    "failure_reason": _classify_output(pred)["failure_reason"],
                    "raw_output_truncated": str(pred.get("raw_response", ""))[:500],
                }
            )
    return rows


def _write_repair_plan(parse_row: dict[str, Any]) -> None:
    text = f"""# Framework-Day7 Prompt / Parser Repair Plan

## Diagnosis Inputs

- parse success rate: `{parse_row.get('parse_success_rate')}`
- schema valid rate: `{parse_row.get('schema_valid_rate')}`
- empty output rate: `{parse_row.get('empty_output_rate')}`
- non-JSON rate: `{parse_row.get('non_json_rate')}`
- missing `ranked_item_ids` rate: `{parse_row.get('missing_ranked_item_ids_rate')}`
- too-few-items rate: `{parse_row.get('too_few_items_rate')}`
- extra-text rate: `{parse_row.get('extra_text_rate')}`

## Prompt Repair Candidates

1. Make JSON-only instruction stricter and explicitly say: output exactly one JSON object and no explanation.
2. Require `ranked_item_ids` length to equal the candidate pool size.
3. Restate that values must be candidate_item_id strings only, not titles.
4. Keep deterministic generation: `do_sample=false`; avoid temperature/top-p unless needed.
5. Keep max_new_tokens large enough for six IDs; 128 is sufficient for 5neg.

## Parser Repair Candidates

Parser-only compatibility is low-risk if raw outputs commonly use alternate keys. Accept possible aliases such as `ranked_items`, `recommendations`, `ranking`, or `item_ids`, then re-validate against the candidate pool. This should be reported as parser repair, not method improvement.

## Training Boundary

Do not change training data format or enter CEP/confidence/evidence framework during Day7. First determine whether the weak Day6 result is parser/prompt/generation instability or insufficient LoRA training.
"""
    REPAIR_PLAN.parent.mkdir(parents=True, exist_ok=True)
    REPAIR_PLAN.write_text(text, encoding="utf-8")


def _run_eval512(config_path: Path, adapter_path: str, max_eval_samples: int) -> list[dict[str, Any]]:
    cfg = _read_config(config_path)
    samples = _read_jsonl(Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_listwise.jsonl")), limit=max_eval_samples)
    pred_rows, lora_rankings = _generate_lora_predictions(cfg, samples, adapter_path, max_new_tokens=128)
    _write_jsonl(DAY7_PRED_512, pred_rows)
    rows = [
        evaluate_rankings("qwen_lora_day6_small_eval512", samples, lora_rankings, pred_rows),
        evaluate_rankings("random_ranking_same_samples", samples, _random_rankings(samples), None),
        evaluate_rankings("oracle_positive_upper_bound", samples, _oracle_rankings(samples), None),
    ]
    _write_csv(EVAL512_CSV, rows)
    return rows


def _rankings_from_prediction_rows(preds: list[dict[str, Any]]) -> list[list[str]]:
    rankings = []
    for pred in preds:
        pool = [str(x) for x in pred.get("candidate_pool", [])]
        ranked = [str(x) for x in pred.get("ranked_item_ids", [])]
        rankings.append(_complete_ranking(ranked, pool) if pred.get("parse_success") else [])
    return rankings


def _run_base_comparison(config_path: Path, max_eval_samples: int, lora_pred_path: Path | None) -> list[dict[str, Any]]:
    # Optional and intentionally explicit: base inference can be slow.
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    cfg = _read_config(config_path)
    samples = _read_jsonl(Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_listwise.jsonl")), limit=max_eval_samples)
    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(cfg.get("bf16", "true")).lower() == "true" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for sampling_key in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, sampling_key):
                setattr(model.generation_config, sampling_key, None)
    model = model.cuda().eval()
    pred_rows, rankings = [], []
    for idx, sample in enumerate(samples):
        prompt, _ = format_sample(sample, task_type=str(cfg["task_type"]), template_path=cfg["prompt_template"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg.get("max_seq_len", 2048))).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pool = _candidate_ids(sample)
        # Reuse the Day6 parser behavior via a tiny local import to avoid changing task schema.
        from main_framework_day6_eval_qwen_lora_listwise import parse_ranking

        parsed = parse_ranking(raw_text, pool)
        rankings.append(_complete_ranking(parsed["ranked_item_ids"], pool) if parsed["parse_success"] else [])
        pred_rows.append(
            {
                "sample_id": sample.get("sample_id", f"sample_{idx}"),
                "target_item_id": _target_id(sample),
                "candidate_pool": pool,
                "raw_response": raw_text,
                "ranked_item_ids": parsed["ranked_item_ids"],
                "parse_success": parsed["parse_success"],
                "schema_valid": parsed["schema_valid"],
                "invalid_item_count": parsed["invalid_item_count"],
                "duplicate_item_count": parsed["duplicate_item_count"],
            }
        )
    rows = [evaluate_rankings("base_qwen_untrained_adapter_free", samples, rankings, pred_rows)]
    if lora_pred_path and lora_pred_path.exists():
        lora_preds = _read_jsonl(lora_pred_path, limit=max_eval_samples)
        rows.insert(
            0,
            evaluate_rankings(
                "qwen_lora_day6_small_same_subset",
                samples[: len(lora_preds)],
                _rankings_from_prediction_rows(lora_preds),
                lora_preds,
            ),
        )
    _write_csv(BASE_VS_LORA, rows)
    return rows


def _write_report(parse_rows: list[dict[str, Any]], eval512_rows: list[dict[str, Any]] | None, base_rows: list[dict[str, Any]] | None, pred_path: Path | None) -> None:
    parse_row = parse_rows[0] if parse_rows else {}
    eval_status = "completed" if eval512_rows else "skipped"
    base_status = "completed" if base_rows else "skipped_due_to_runtime_or_not_requested"
    report = f"""# Framework-Day7 Qwen-LoRA Eval Diagnosis Report

## 1. Day6 Recap

Day6 completed train -> save adapter -> infer -> parse -> evaluate. However, performance was weak: Qwen-LoRA was only slightly above random on NDCG/MRR, while HR@3/NDCG@5 were lower than random. Parse success was `0.8672`.

## 2. Parse Failure Diagnosis

- prediction source: `{pred_path}`
- parse success rate: `{parse_row.get('parse_success_rate', 'NA')}`
- schema valid rate: `{parse_row.get('schema_valid_rate', 'NA')}`
- non-JSON rate: `{parse_row.get('non_json_rate', 'NA')}`
- missing key rate: `{parse_row.get('missing_ranked_item_ids_rate', 'NA')}`
- too-few-items rate: `{parse_row.get('too_few_items_rate', 'NA')}`

## 3. Ranking Case Study

See `data_done/framework_day7_ranking_case_study.csv` for hit@1, rank 2-3, rank >3, and parse failure examples.

## 4. Eval Sample Sensitivity

512/large-subset eval status: `{eval_status}`. If completed, see `data_done/framework_day7_eval_512_summary.csv`.

## 5. Base vs LoRA

Base comparison status: `{base_status}`. If skipped, this is due to runtime control; run with `--run_base_comparison` on the server if needed.

## 6. Prompt / Parser Repair Plan

See `data_done/framework_day7_prompt_parser_repair_plan.md`.

## 7. Day8 Recommendation

If parse failures are mainly schema/key/extra-text issues, first do prompt/parser repair and re-evaluate the same raw outputs. If parse improves but ranking remains weak, run a 500-1000 step Beauty listwise train. If base and LoRA look similar, 100 steps is likely insufficient. Do not enter CEP/confidence/evidence framework until the Qwen-LoRA baseline is stable.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day7 Qwen-LoRA eval/parser diagnosis.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_small.yaml")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_listwise_day6_small")
    parser.add_argument("--run_eval512", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--run_base_comparison", action="store_true")
    parser.add_argument("--base_eval_samples", type=int, default=128)
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else _find_day6_predictions()
    parse_rows: list[dict[str, Any]] = []
    if pred_path and pred_path.exists():
        preds = _read_jsonl(pred_path)
        parse_rows, examples = build_parse_diagnostics(preds)
        _write_csv(PARSE_DIAG_CSV, parse_rows)
        _write_jsonl(PARSE_EXAMPLES, examples)
        cfg = _read_config(args.config)
        test_file = Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_listwise.jsonl"))
        sample_map = _load_samples_by_id(test_file)
        _write_csv(CASE_STUDY_CSV, build_case_study(preds, sample_map))
        _write_repair_plan(parse_rows[0])
    else:
        parse_rows = [{"num_samples": 0, "parse_success_rate": "NA", "schema_valid_rate": "NA", "notes": "Day6 predictions not found locally. Run this script on the server."}]
        _write_csv(PARSE_DIAG_CSV, parse_rows)
        _write_jsonl(PARSE_EXAMPLES, [])
        _write_csv(CASE_STUDY_CSV, [])
        _write_repair_plan(parse_rows[0])

    eval512_rows = None
    if args.run_eval512:
        eval512_rows = _run_eval512(Path(args.config), args.adapter_path, args.eval_samples)
    else:
        _write_csv(EVAL512_CSV, [{"status": "skipped", "reason": "run with --run_eval512 on server to generate 512-sample eval"}])

    base_rows = None
    if args.run_base_comparison:
        base_rows = _run_base_comparison(Path(args.config), args.base_eval_samples, pred_path)
    else:
        _write_csv(BASE_VS_LORA, [{"method": "base_qwen_untrained_adapter_free", "status": "skipped_due_to_runtime", "notes": "Run with --run_base_comparison on server if needed."}])

    _write_report(parse_rows, eval512_rows, base_rows, pred_path)
    print(json.dumps({"parse": parse_rows, "eval512": eval512_rows, "base": base_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
