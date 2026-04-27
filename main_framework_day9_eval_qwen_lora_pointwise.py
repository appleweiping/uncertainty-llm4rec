from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_day6_eval_qwen_lora_listwise import (
    _oracle_rankings,
    _random_rankings,
    _rank_metrics_for_sample,
    _write_jsonl,
    evaluate_rankings,
)
from src.framework.prompt_formatters import format_sample


PRED_PATH = Path("output-repaired/framework/day9_qwen_lora_beauty_pointwise_predictions.jsonl")
SUMMARY_CSV = Path("data_done/framework_day9_pointwise_eval_summary.csv")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def parse_relevance(raw_text: str) -> dict[str, Any]:
    text = _extract_json_text(raw_text)
    if text is None:
        return {"parse_success": False, "schema_valid": False, "score": 0.0, "label": None, "parse_error": "no_json_object"}
    try:
        obj = json.loads(text)
    except Exception as exc:
        return {"parse_success": False, "schema_valid": False, "score": 0.0, "label": None, "parse_error": f"json_error:{type(exc).__name__}"}

    label = obj.get("relevance_label")
    score_value = obj.get("relevance_score")
    if isinstance(label, bool):
        parsed_label = 1 if label else 0
    elif isinstance(label, (int, float)) and int(label) in {0, 1}:
        parsed_label = int(label)
    elif isinstance(label, str) and label.strip().lower() in {"0", "1", "true", "false", "yes", "no"}:
        normalized = label.strip().lower()
        parsed_label = 1 if normalized in {"1", "true", "yes"} else 0
    else:
        parsed_label = None
    try:
        score = float(score_value) if score_value is not None else float(parsed_label or 0)
    except Exception:
        score = float(parsed_label or 0)
    score = max(0.0, min(1.0, score))
    return {
        "parse_success": True,
        "schema_valid": parsed_label is not None or score_value is not None,
        "score": score,
        "label": parsed_label,
        "parse_error": "" if parsed_label is not None or score_value is not None else "missing_relevance_label",
    }


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("input", {}).get("candidate_item", {}).get("candidate_item_id", "")).strip()


def _is_positive(row: dict[str, Any]) -> bool:
    return int(row.get("output", {}).get("relevance_label", 0)) == 1


def _group_pointwise_rows(rows: list[dict[str, Any]], pool_size: int = 6) -> list[list[dict[str, Any]]]:
    groups = []
    for start in range(0, len(rows), pool_size):
        group = rows[start : start + pool_size]
        if len(group) == pool_size:
            groups.append(group)
    return groups


def _listwise_samples_from_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    samples = []
    for idx, group in enumerate(groups):
        pool = [{"candidate_item_id": _candidate_id(row)} for row in group]
        positives = [_candidate_id(row) for row in group if _is_positive(row)]
        target = positives[0] if positives else _candidate_id(group[0])
        samples.append(
            {
                "sample_id": f"pointwise_group_{idx}",
                "input": {"candidate_pool": pool},
                "output": {"target_item_id": target},
            }
        )
    return samples


def _generate_pointwise_predictions(cfg: dict[str, Any], rows: list[dict[str, Any]], adapter_path: str, max_new_tokens: int) -> list[dict[str, Any]]:
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

    pred_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt, _ = format_sample(row, task_type=str(cfg["task_type"]), template_path=cfg["prompt_template"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg.get("max_seq_len", 1536))).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = parse_relevance(raw_text)
        pred_rows.append(
            {
                "sample_id": row.get("sample_id", f"pointwise_{idx}"),
                "candidate_item_id": _candidate_id(row),
                "target_label": int(row.get("output", {}).get("relevance_label", 0)),
                "raw_response": raw_text,
                "relevance_score": parsed["score"],
                "parsed_relevance_label": parsed["label"],
                "parse_success": parsed["parse_success"],
                "schema_valid": parsed["schema_valid"],
                "parse_error": parsed["parse_error"],
            }
        )
    return pred_rows


def _rankings_from_scores(groups: list[list[dict[str, Any]]], pred_rows: list[dict[str, Any]]) -> list[list[str]]:
    rankings: list[list[str]] = []
    offset = 0
    for group in groups:
        scored = []
        for local_idx, row in enumerate(group):
            pred = pred_rows[offset + local_idx]
            scored.append((_candidate_id(row), float(pred.get("relevance_score", 0.0)), local_idx))
        offset += len(group)
        scored.sort(key=lambda x: (-x[1], x[2]))
        rankings.append([item_id for item_id, _, _ in scored])
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day9 pointwise-v1 adapter by aggregating candidate scores.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_pointwise_small.yaml")
    parser.add_argument("--adapter_path", default="artifacts/lora/qwen3_8b_beauty_pointwise_day9_small")
    parser.add_argument("--eval_users", type=int, default=512)
    parser.add_argument("--candidate_pool_size", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    test_file = Path(str(cfg.get("test_file") or "data_done_lora/beauty/test_pointwise.jsonl"))
    all_rows = _read_jsonl(test_file)
    groups = _group_pointwise_rows(all_rows, pool_size=args.candidate_pool_size)[: args.eval_users]
    pointwise_rows = [row for group in groups for row in group]
    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    pred_rows = _generate_pointwise_predictions(cfg, pointwise_rows, args.adapter_path, args.max_new_tokens)
    _write_jsonl(PRED_PATH, pred_rows)
    samples = _listwise_samples_from_groups(groups)
    rankings = _rankings_from_scores(groups, pred_rows)
    summary = evaluate_rankings("day9_pointwise_v1_aggregated_ranking", samples, rankings, None)
    summary["num_pointwise_rows"] = len(pred_rows)
    summary["parse_success_rate"] = sum(1 for p in pred_rows if p["parse_success"]) / len(pred_rows) if pred_rows else 0
    summary["schema_valid_rate"] = sum(1 for p in pred_rows if p["schema_valid"]) / len(pred_rows) if pred_rows else 0
    rows = [
        summary,
        evaluate_rankings("random_ranking_same_samples", samples, _random_rankings(samples), None),
        evaluate_rankings("oracle_positive_upper_bound", samples, _oracle_rankings(samples), None),
    ]
    _write_csv(SUMMARY_CSV, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
