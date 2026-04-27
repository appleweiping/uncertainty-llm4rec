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
from src.framework.prompt_formatters import format_sample


PRED_PATH = Path("output-repaired/framework/day6_qwen_lora_beauty_listwise_predictions.jsonl")
SUMMARY_CSV = Path("data_done/framework_day6_beauty_listwise_eval_summary.csv")
BASELINE_CSV = Path("data_done/framework_day6_beauty_listwise_baseline_comparison.csv")
REPORT_PATH = Path("data_done/framework_day6_beauty_listwise_eval_report.md")
FINAL_REPORT_PATH = Path("data_done/framework_day6_qwen_lora_small_train_eval_report.md")


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_ids(sample: dict[str, Any]) -> list[str]:
    return [str(x.get("candidate_item_id", "")).strip() for x in sample.get("input", {}).get("candidate_pool", [])]


def _target_id(sample: dict[str, Any]) -> str:
    out = sample.get("output", {})
    if out.get("target_item_id"):
        return str(out["target_item_id"]).strip()
    ranked = out.get("ranked_item_ids", [])
    return str(ranked[0]).strip() if ranked else ""


RANKING_KEYS = [
    "ranked_item_ids",
    "ranked_items",
    "recommendations",
    "recommended_items",
    "item_ids",
    "ranking",
]


def _extract_json_text(text: str) -> str | None:
    if not text:
        return None
    block = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_ranked_list(obj: dict[str, Any]) -> tuple[Any, str]:
    for key in RANKING_KEYS:
        if key in obj:
            return obj[key], key
    return None, ""


def _coerce_item_id(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ["candidate_item_id", "item_id", "id"]:
            if key in entry:
                return str(entry[key]).strip()
    return str(entry).strip()


def parse_ranking(raw_text: str, candidate_pool: list[str]) -> dict[str, Any]:
    parsed_text = _extract_json_text(raw_text)
    base = {
        "parse_success_raw": False,
        "schema_valid_raw": False,
        "parse_success_repaired": False,
        "parser_repair_used": False,
        "repaired_item_count": 0,
        "raw_ranked_item_ids": [],
        "ranking_key_used": "",
    }
    if parsed_text is None:
        return {**base, **{
            "parse_success": False,
            "schema_valid": False,
            "ranked_item_ids": [],
            "invalid_item_count": 0,
            "duplicate_item_count": 0,
            "parse_error": "no_json_object",
        }}
    try:
        obj = json.loads(parsed_text)
    except Exception as exc:
        return {**base, **{
            "parse_success": False,
            "schema_valid": False,
            "ranked_item_ids": [],
            "invalid_item_count": 0,
            "duplicate_item_count": 0,
            "parse_error": f"json_error:{type(exc).__name__}",
        }}
    ranked, key_used = _extract_ranked_list(obj)
    raw_ranked = obj.get("ranked_item_ids")
    raw_success = isinstance(raw_ranked, list)
    if not isinstance(ranked, list):
        return {**base, **{
            "parse_success_raw": raw_success,
            "schema_valid_raw": raw_success,
            "parse_success": True,
            "schema_valid": False,
            "ranked_item_ids": [],
            "invalid_item_count": 0,
            "duplicate_item_count": 0,
            "parse_error": "missing_ranked_item_ids",
        }}
    pool = set(candidate_pool)
    cleaned: list[str] = []
    invalid = 0
    duplicates = 0
    seen: set[str] = set()
    for item in ranked:
        iid = _coerce_item_id(item)
        if iid not in pool:
            invalid += 1
            continue
        if iid in seen:
            duplicates += 1
            continue
        seen.add(iid)
        cleaned.append(iid)
    complete = _complete_ranking(cleaned, candidate_pool)
    repaired_item_count = len(complete) - len(cleaned)
    repair_used = key_used != "ranked_item_ids" or invalid > 0 or duplicates > 0 or repaired_item_count > 0
    raw_ids = [_coerce_item_id(x) for x in raw_ranked] if isinstance(raw_ranked, list) else []
    return {
        "parse_success_raw": raw_success,
        "schema_valid_raw": raw_success,
        "parse_success_repaired": True,
        "parser_repair_used": repair_used,
        "repaired_item_count": repaired_item_count,
        "raw_ranked_item_ids": raw_ids,
        "ranking_key_used": key_used,
        "parse_success": True,
        "schema_valid": True,
        "ranked_item_ids": complete,
        "invalid_item_count": invalid,
        "duplicate_item_count": duplicates,
        "parse_error": "",
    }


def _complete_ranking(ranked: list[str], candidate_pool: list[str]) -> list[str]:
    seen = set(ranked)
    return ranked + [iid for iid in candidate_pool if iid not in seen]


def _rank_metrics_for_sample(ranking: list[str], target: str, pool_size: int) -> dict[str, float]:
    if target in ranking:
        rank = ranking.index(target) + 1
    else:
        rank = pool_size + 1
    out: dict[str, float] = {
        "MRR": 1.0 / rank if rank <= pool_size else 0.0,
        "HR@1": 1.0 if rank <= 1 else 0.0,
        "HR@3": 1.0 if rank <= 3 else 0.0,
        "HR@10": 1.0 if rank <= 10 else 0.0,
        "NDCG@3": 1.0 / math.log2(rank + 1) if rank <= 3 else 0.0,
        "NDCG@5": 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0,
        "NDCG@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
    }
    return out


def evaluate_rankings(method: str, samples: list[dict[str, Any]], rankings: list[list[str]], parse_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics = []
    pool_sizes = []
    for sample, ranking in zip(samples, rankings):
        pool = _candidate_ids(sample)
        target = _target_id(sample)
        pool_sizes.append(len(pool))
        metrics.append(_rank_metrics_for_sample(ranking, target, len(pool)))
    row: dict[str, Any] = {
        "method": method,
        "num_samples": len(samples),
        "candidate_pool_size_mean": mean(pool_sizes) if pool_sizes else 0,
        "hr10_trivial_flag": (max(pool_sizes) <= 10 or mean(pool_sizes) <= 10) if pool_sizes else True,
    }
    for key in ["NDCG@10", "MRR", "HR@1", "HR@3", "NDCG@3", "NDCG@5", "HR@10"]:
        row[key] = mean([m[key] for m in metrics]) if metrics else 0.0
    if parse_rows is not None:
        row["parse_success_rate"] = mean([1.0 if r["parse_success"] else 0.0 for r in parse_rows]) if parse_rows else 0.0
        row["schema_valid_rate"] = mean([1.0 if r["schema_valid"] else 0.0 for r in parse_rows]) if parse_rows else 0.0
        total_output_items = sum(len(r.get("raw_ranked_item_ids", [])) for r in parse_rows)
        row["invalid_item_rate"] = (
            sum(int(r["invalid_item_count"]) for r in parse_rows) / total_output_items if total_output_items else 0.0
        )
        row["duplicate_item_rate"] = (
            sum(int(r["duplicate_item_count"]) for r in parse_rows) / total_output_items if total_output_items else 0.0
        )
    else:
        row["parse_success_rate"] = 1.0
        row["schema_valid_rate"] = 1.0
        row["invalid_item_rate"] = 0.0
        row["duplicate_item_rate"] = 0.0
    return row


def _generate_lora_predictions(cfg: dict[str, Any], samples: list[dict[str, Any]], adapter_path: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], list[list[str]]]:
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(cfg.get("bf16", "true")).lower() == "true" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for sampling_key in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, sampling_key):
                setattr(model.generation_config, sampling_key, None)
    model.cuda()
    model.eval()

    def _deterministic_generate(model: Any, inputs: Any) -> Any:
        # Keep generation deterministic and avoid passing sampling-only flags
        # such as temperature/top_p/top_k when do_sample=False.
        kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        return model.generate(**kwargs)

    pred_rows: list[dict[str, Any]] = []
    rankings: list[list[str]] = []
    for idx, sample in enumerate(samples):
        prompt, _ = format_sample(sample, task_type=str(cfg["task_type"]), template_path=cfg["prompt_template"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg.get("max_seq_len", 2048))).to("cuda")
        with torch.no_grad():
            output_ids = _deterministic_generate(model, inputs)
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pool = _candidate_ids(sample)
        parsed = parse_ranking(raw_text, pool)
        complete = _complete_ranking(parsed["ranked_item_ids"], pool) if parsed["parse_success"] else []
        rankings.append(complete)
        pred_rows.append(
            {
                "sample_id": sample.get("sample_id", f"sample_{idx}"),
                "user_index": idx,
                "target_item_id": _target_id(sample),
                "candidate_pool": pool,
                "raw_response": raw_text,
                "ranked_item_ids": parsed["ranked_item_ids"],
                "raw_ranked_item_ids": parsed.get("raw_ranked_item_ids", parsed["ranked_item_ids"]),
                "parse_success": parsed["parse_success"],
                "schema_valid": parsed["schema_valid"],
                "parse_success_raw": parsed.get("parse_success_raw", parsed["parse_success"]),
                "schema_valid_raw": parsed.get("schema_valid_raw", parsed["schema_valid"]),
                "parse_success_repaired": parsed.get("parse_success_repaired", parsed["parse_success"]),
                "parser_repair_used": parsed.get("parser_repair_used", False),
                "repaired_item_count": parsed.get("repaired_item_count", 0),
                "ranking_key_used": parsed.get("ranking_key_used", "ranked_item_ids"),
                "invalid_item_count": parsed["invalid_item_count"],
                "duplicate_item_count": parsed["duplicate_item_count"],
                "parse_error": parsed["parse_error"],
            }
        )
    return pred_rows, rankings


def _random_rankings(samples: list[dict[str, Any]]) -> list[list[str]]:
    rng = random.Random(42)
    rankings = []
    for sample in samples:
        pool = _candidate_ids(sample)
        pool = pool[:]
        rng.shuffle(pool)
        rankings.append(pool)
    return rankings


def _oracle_rankings(samples: list[dict[str, Any]]) -> list[list[str]]:
    rankings = []
    for sample in samples:
        pool = _candidate_ids(sample)
        target = _target_id(sample)
        rankings.append([target] + [iid for iid in pool if iid != target])
    return rankings


def _write_reports(summary_rows: list[dict[str, Any]], adapter_path: str, config_path: str) -> None:
    lora = next((r for r in summary_rows if r["method"] == "qwen_lora_day6_small"), {})
    report = f"""# Framework-Day6 Beauty Listwise Eval Report

## Scope

This is an adapter inference / parsing / ranking smoke for the Day6 Beauty listwise Qwen-LoRA baseline. It does not use confidence, evidence, CEP fusion, or API calls.

## Adapter

- config: `{config_path}`
- adapter path: `{adapter_path}`
- adapter commit policy: ignored artifact, do not commit

## Parser

- parse success rate: `{lora.get('parse_success_rate', 'NA')}`
- schema valid rate: `{lora.get('schema_valid_rate', 'NA')}`
- invalid item rate: `{lora.get('invalid_item_rate', 'NA')}`
- duplicate item rate: `{lora.get('duplicate_item_rate', 'NA')}`

## Ranking Metrics

- NDCG@10: `{lora.get('NDCG@10', 'NA')}`
- MRR: `{lora.get('MRR', 'NA')}`
- HR@1: `{lora.get('HR@1', 'NA')}`
- HR@3: `{lora.get('HR@3', 'NA')}`
- NDCG@3: `{lora.get('NDCG@3', 'NA')}`
- NDCG@5: `{lora.get('NDCG@5', 'NA')}`
- HR@10: `{lora.get('HR@10', 'NA')}` with `hr10_trivial_flag={lora.get('hr10_trivial_flag', 'NA')}`

HR@10 is trivial for Beauty 5neg candidate pools and should not be used as a main claim.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    final = f"""# Framework-Day6 Qwen-LoRA Small Train + Eval Report

## 1. Day5 Tiny Train Recap

Day5 confirmed real server-side Qwen3-8B LoRA optimizer steps: loss decreased from `0.4933` to `0.0090`, no NaN, peak GPU memory `20.2543 GB`, runtime `32.62s`.

## 2. Day6 Small Train Setup

Day6 uses Beauty listwise closed-candidate ranking with `max_train_samples=512`, `max_steps=100`, `batch_size=1`, `gradient_accumulation_steps=4`, LoRA rank `8`, alpha `16`, and max sequence length `2048`.

## 3. Adapter Save Path

`{adapter_path}`. This is an ignored artifact and must not be committed.

## 4. Inference / Parser Result

See `data_done/framework_day6_beauty_listwise_eval_summary.csv`.

## 5. Ranking Metrics

See `data_done/framework_day6_beauty_listwise_baseline_comparison.csv`. Main metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5. HR@10 is marked trivial because candidate pool size is 6.

## 6. Random / Oracle Comparison

The eval script always includes random ranking and oracle positive upper bound as sanity checks. Oracle is not a real method.

## 7. Day7 Readiness

If training status is success and eval parse quality is acceptable, Day7 can move to a larger Beauty listwise baseline train. Do not add CEP/confidence/evidence fusion until the local baseline train/eval loop is stable.

## 8. Limitations

This is small train and small eval only. It is not full-domain training, not a framework result, and not a CEP/calibration/evidence experiment.
"""
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH.write_text(final, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day6 Qwen-LoRA listwise adapter eval smoke.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_small.yaml")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_listwise_day6_small")
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    test_file = Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_listwise.jsonl"))
    samples = _read_jsonl(test_file, limit=args.max_eval_samples)
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    pred_rows, lora_rankings = _generate_lora_predictions(cfg, samples, args.adapter_path, args.max_new_tokens)
    _write_jsonl(PRED_PATH, pred_rows)
    summary_rows = [
        evaluate_rankings("qwen_lora_day6_small", samples, lora_rankings, pred_rows),
        evaluate_rankings("random_ranking", samples, _random_rankings(samples), None),
        evaluate_rankings("oracle_positive_upper_bound", samples, _oracle_rankings(samples), None),
    ]
    _write_csv(SUMMARY_CSV, summary_rows)
    comparison_rows = [
        {
            "method": r["method"],
            "parse_success_rate": r["parse_success_rate"],
            "NDCG@10": r["NDCG@10"],
            "MRR": r["MRR"],
            "HR@1": r["HR@1"],
            "HR@3": r["HR@3"],
            "NDCG@3": r["NDCG@3"],
            "NDCG@5": r["NDCG@5"],
            "notes": "HR@10 trivial for 5neg candidate pool" if r["hr10_trivial_flag"] else "",
        }
        for r in summary_rows
    ]
    _write_csv(BASELINE_CSV, comparison_rows)
    _write_reports(summary_rows, adapter_path=args.adapter_path, config_path=args.config)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
