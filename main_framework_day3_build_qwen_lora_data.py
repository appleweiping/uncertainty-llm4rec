from __future__ import annotations

import json
import random
import csv
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 42
NEGATIVES_PER_TRAIN_POSITIVE = 5
SAMPLE_USERS_LARGE_DOMAIN = 1000
DOMAINS = ["beauty", "books", "electronics", "movies"]
DATA_ROOT = Path("data_done")
OUT_ROOT = Path("data_done_lora")
PROMPT_DIR = Path("prompts/framework")
CONFIG_DIR = Path("configs/framework")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    _mkdir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _item_from_history(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": _safe_str(x.get("item_id")),
        "title": _safe_str(x.get("title")),
        "text": _safe_str(x.get("text")),
        "text_missing": bool(x.get("text_missing", False)),
        "text_fallback_used": bool(x.get("text_fallback_used", False)),
        "text_source": _safe_str(x.get("text_source", "metadata")),
    }


def _candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_item_id": _safe_str(row.get("candidate_item_id")),
        "title": _safe_str(row.get("candidate_title")),
        "text": _safe_str(row.get("candidate_text")),
        "candidate_text_missing": bool(row.get("candidate_text_missing", False)),
        "candidate_text_fallback_used": bool(row.get("candidate_text_fallback_used", False)),
        "candidate_text_source": _safe_str(row.get("candidate_text_source", "metadata")),
    }


def _instruction(task: str) -> str:
    if task == "candidate_ranking":
        return (
            "You are a recommendation model. Rank only the candidate items provided in the candidate pool. "
            "Do not invent item IDs outside the candidate pool. Return JSON with ranked_item_ids."
        )
    return (
        "You are a recommendation model. Decide whether the candidate item matches the user's history. "
        "Return JSON with relevance_label. This label is not a calibrated probability."
    )


def _group_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["user_id"], []).append(row)
    return grouped


def _sample_users(users: list[str], domain: str) -> list[str]:
    users = sorted(users)
    if domain == "beauty" or len(users) <= SAMPLE_USERS_LARGE_DOMAIN:
        return users
    rng = random.Random(SEED)
    return sorted(rng.sample(users, SAMPLE_USERS_LARGE_DOMAIN))


def _listwise_from_candidate_group(domain: str, split: str, uid: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if int(r.get("label", 0)) == 1]
    target = _safe_str(pos[0]["candidate_item_id"]) if pos else ""
    rows = sorted(rows, key=lambda r: _safe_str(r.get("candidate_item_id")))
    random.Random(f"{SEED}_{domain}_{split}_{uid}").shuffle(rows)
    candidate_pool = [_candidate_from_row(r) for r in rows]
    text_missing_rate = sum(1 for c in candidate_pool if c["candidate_text_missing"] or c["candidate_text_fallback_used"]) / len(candidate_pool)
    return {
        "sample_id": f"{domain}_{split}_{uid}_listwise",
        "domain": domain,
        "task": "candidate_ranking",
        "instruction": _instruction("candidate_ranking"),
        "input": {
            "user_history": [_item_from_history(x) for x in rows[0].get("history", [])],
            "candidate_pool": candidate_pool,
        },
        "output": {
            "ranked_item_ids": [target] + [c["candidate_item_id"] for c in candidate_pool if c["candidate_item_id"] != target],
            "target_item_id": target,
        },
        "metadata": {
            "candidate_pool_setting": rows[0].get("candidate_pool_setting", "5neg"),
            "text_missing_rate": text_missing_rate,
            "source_split": split,
            "closed_catalog": True,
        },
    }


def _pointwise_from_row(domain: str, split: str, idx: int, row: dict[str, Any]) -> dict[str, Any]:
    cand = _candidate_from_row(row)
    return {
        "sample_id": f"{domain}_{split}_{idx}_pointwise",
        "domain": domain,
        "task": "candidate_relevance",
        "instruction": _instruction("candidate_relevance"),
        "input": {
            "user_history": [_item_from_history(x) for x in row.get("history", [])],
            "candidate_item": cand,
        },
        "output": {
            "relevance_label": int(row.get("label", 0)),
        },
        "metadata": {
            "candidate_pool_setting": row.get("candidate_pool_setting", "5neg"),
            "text_missing": bool(cand["candidate_text_missing"]),
            "text_fallback_used": bool(cand["candidate_text_fallback_used"]),
            "source_split": split,
            "not_calibrated_probability": True,
        },
    }


def _load_train_vocab(domain: str) -> list[str]:
    vocab = []
    for row in _read_jsonl(DATA_ROOT / domain / "train.jsonl"):
        for x in row.get("history", []):
            iid = _safe_str(x.get("item_id"))
            if iid:
                vocab.append(iid)
    return sorted(set(vocab))


def _load_item_map(domain: str) -> dict[str, dict[str, Any]]:
    path = DATA_ROOT / domain / "items.csv"
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = _safe_str(row.get("item_id"))
            if not item_id:
                continue
            text = _safe_str(row.get("candidate_text"))
            title = _safe_str(row.get("title"))
            fallback = row.get("text_fallback_used")
            fallback_bool = str(fallback).lower() == "true" or text.startswith("Item ID:") or title.startswith("Unknown item")
            out[item_id] = {
                "title": title or f"Unknown item {item_id}",
                "text": text or f"Item ID: {item_id}",
                "candidate_text_missing": fallback_bool,
                "candidate_text_fallback_used": fallback_bool,
                "candidate_text_source": "missing_metadata_fallback" if fallback_bool else "metadata",
            }
    return out


def _train_candidate_rows(domain: str, selected_users: set[str]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    train_rows = _read_jsonl(DATA_ROOT / domain / "train.jsonl")
    train_vocab = _load_train_vocab(domain)
    item_map = _load_item_map(domain)
    candidate_rows = []
    for row in train_rows:
        uid = row["user_id"]
        if uid not in selected_users:
            continue
        hist = row.get("history", [])
        if len(hist) < 2:
            continue
        prefix = hist[:-1]
        target = hist[-1]
        seen = {_safe_str(x.get("item_id")) for x in hist}
        pool = [i for i in train_vocab if i not in seen]
        if len(pool) >= NEGATIVES_PER_TRAIN_POSITIVE:
            negs = rng.sample(pool, NEGATIVES_PER_TRAIN_POSITIVE)
        else:
            negs = pool
        pos_row = {
            "user_id": uid,
            "history": prefix,
            "candidate_item_id": target["item_id"],
            "candidate_title": target.get("title", ""),
            "candidate_text": target.get("text", ""),
            "candidate_text_missing": target.get("text_missing", False),
            "candidate_text_fallback_used": target.get("text_fallback_used", False),
            "candidate_text_source": target.get("text_source", "metadata"),
            "label": 1,
            "candidate_pool_setting": "5neg",
        }
        candidate_rows.append(pos_row)
        for neg in negs:
            meta = item_map.get(neg, {})
            candidate_rows.append(
                {
                    "user_id": uid,
                    "history": prefix,
                    "candidate_item_id": neg,
                    "candidate_title": meta.get("title") or f"Unknown item {neg}",
                    "candidate_text": meta.get("text") or f"Item ID: {neg}",
                    "candidate_text_missing": bool(meta.get("candidate_text_missing", True)),
                    "candidate_text_fallback_used": bool(meta.get("candidate_text_fallback_used", True)),
                    "candidate_text_source": meta.get("candidate_text_source", "missing_metadata_fallback"),
                    "label": 0,
                    "candidate_pool_setting": "5neg",
                }
            )
    return candidate_rows


def _build_domain(domain: str) -> dict[str, Any]:
    out = OUT_ROOT / domain
    _mkdir(out)
    valid_rows_all = _read_jsonl(DATA_ROOT / domain / "valid.jsonl")
    test_rows_all = _read_jsonl(DATA_ROOT / domain / "test.jsonl")
    valid_users = _sample_users(list(_group_candidate_rows(valid_rows_all)), domain)
    test_users = set(valid_users)
    selected_users = set(valid_users)
    train_candidate_rows = _train_candidate_rows(domain, selected_users)
    split_rows = {
        "train": train_candidate_rows,
        "valid": [r for r in valid_rows_all if r["user_id"] in selected_users],
        "test": [r for r in test_rows_all if r["user_id"] in test_users],
    }
    stats = {
        "domain": domain,
        "seed": SEED,
        "mode": "full" if domain == "beauty" else f"sample_{SAMPLE_USERS_LARGE_DOMAIN}",
        "selected_users": len(selected_users),
    }
    for split, rows in split_rows.items():
        grouped = _group_candidate_rows(rows)
        listwise = [_listwise_from_candidate_group(domain, split, uid, group) for uid, group in grouped.items()]
        pointwise = [_pointwise_from_row(domain, split, i, row) for i, row in enumerate(rows)]
        _write_jsonl(out / f"{split}_listwise.jsonl", listwise)
        _write_jsonl(out / f"{split}_pointwise.jsonl", pointwise)
        stats[f"{split}_listwise_rows"] = len(listwise)
        stats[f"{split}_pointwise_rows"] = len(pointwise)
        stats[f"{split}_users"] = len(grouped)
        stats[f"{split}_positive_rows"] = sum(1 for r in rows if int(r.get("label", 0)) == 1)
        stats[f"{split}_negative_rows"] = sum(1 for r in rows if int(r.get("label", 0)) == 0)
    examples = {
        "listwise": json.loads((out / "valid_listwise.jsonl").open(encoding="utf-8").readline()),
        "pointwise": json.loads((out / "valid_pointwise.jsonl").open(encoding="utf-8").readline()),
    }
    _write_json(out / "prompt_examples.json", examples)
    schema = {
        "has_sample_id": True,
        "has_instruction": True,
        "has_input": True,
        "has_output": True,
        "listwise_closed_catalog": True,
        "pointwise_label_values": [0, 1],
        "calibrated_probability_used_as_label": False,
        "notes": "LoRA baseline learns raw relevance/ranking labels. CEP calibration remains a framework-stage post-processing mechanism.",
    }
    _write_json(out / "schema_validation.json", schema)
    _write_json(out / "data_stats.json", stats)
    if domain != "beauty":
        _write_json(
            out / "full_generation_manifest.json",
            {
                "domain": domain,
                "full_generation_not_written": True,
                "reason": "Framework-Day3 writes sample_1000 only for large domains; full files can be generated later with explicit confirmation.",
                "source": f"data_done/{domain}",
            },
        )
    return stats


def _write_prompts() -> None:
    _mkdir(PROMPT_DIR)
    (PROMPT_DIR / "qwen_candidate_ranking_baseline.txt").write_text(
        """You are a recommendation ranking model.

Task: rank candidate items for the user based only on the provided user history and candidate pool.

Rules:
- Only choose item IDs from the candidate_pool.
- Do not invent item IDs outside the candidate_pool.
- Treat text marked text_fallback_used=true as missing metadata, not as semantic description.
- Return valid JSON only.

Output JSON schema:
{
  "ranked_item_ids": ["<candidate_item_id>", "..."]
}
""",
        encoding="utf-8",
    )
    (PROMPT_DIR / "qwen_candidate_relevance_baseline.txt").write_text(
        """You are a recommendation relevance model.

Task: decide whether the candidate item matches the user's history.

Rules:
- Use only the provided user_history and candidate_item.
- Treat text marked text_fallback_used=true as missing metadata, not as semantic description.
- Return a raw relevance judgment, not a calibrated probability.
- Return valid JSON only.

Output JSON schema:
{
  "relevance_label": 0,
  "relevance_score": 0.0
}
""",
        encoding="utf-8",
    )
    (PROMPT_DIR / "qwen_evidence_output_schema_future.txt").write_text(
        """Future CEP/evidence generator schema. Not used for Framework-Day3 baseline training.

{
  "relevance_probability": 0.0,
  "positive_evidence": [],
  "negative_evidence": [],
  "ambiguity": 0.0,
  "missing_information": 0.0,
  "evidence_risk": 0.0
}

Important: calibrated_relevance_probability is produced later by valid-set calibration. It is not a raw LoRA training label.
""",
        encoding="utf-8",
    )


def _write_configs() -> None:
    _mkdir(CONFIG_DIR)
    base_model = "TODO:/home/ajifang/models/Qwen/Qwen3-8B"
    common = {
        "model_name_or_path": base_model,
        "tokenizer_name_or_path": base_model,
        "max_seq_len": 4096,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": "2e-4",
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "bf16": True,
        "seed": SEED,
    }
    for task, train_file, valid_file, prompt, eval_mode in [
        (
            "listwise",
            "data_done_lora/beauty/train_listwise.jsonl",
            "data_done_lora/beauty/valid_listwise.jsonl",
            "prompts/framework/qwen_candidate_ranking_baseline.txt",
            "closed_candidate_ranking",
        ),
        (
            "pointwise",
            "data_done_lora/beauty/train_pointwise.jsonl",
            "data_done_lora/beauty/valid_pointwise.jsonl",
            "prompts/framework/qwen_candidate_relevance_baseline.txt",
            "binary_relevance",
        ),
    ]:
        cfg = dict(common)
        cfg.update(
            {
                "train_file": train_file,
                "valid_file": valid_file,
                "output_dir": f"outputs/framework/qwen3_8b_lora_baseline_beauty_{task}",
                "task_type": f"qwen_recommendation_baseline_{task}",
                "prompt_template": prompt,
                "evaluation_mode": eval_mode,
                "notes": "Scaffold only. Framework-Day3 does not start training. Verify server model path before use.",
            }
        )
        lines = [f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()]
        (CONFIG_DIR / f"qwen3_8b_lora_baseline_beauty_{task}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(stats_by_domain: dict[str, dict[str, Any]]) -> None:
    (DATA_ROOT / "framework_day3_decodingmatters_audit.md").write_text(
        """# Framework-Day3 DecodingMatters Audit

## Summary

DecodingMatters is useful as an implementation reference for local LLM recommendation training and constrained decoding, but it should not be copied as our method. Our contribution remains CEP / calibrated relevance posterior / uncertainty-aware decision support, not LoRA itself.

## Findings

1. `train.py` expects CSV `train_file` and `eval_file`. The dataset rows include columns such as `history_item_title`, `history_item_id`, `item_title`, and `item_id`.
2. `D3Dataset` converts each row into an instruction prompt with user history titles and a target item title as the response. Labels are causal-LM tokens for the target response.
3. The code references `candidate_file=train_file` in `train.py`, but the visible `D3Dataset` constructor does not consume a separate candidate file. Candidate/item catalog constraints mainly appear during evaluation through `info_file`.
4. `category` maps dataset names to natural-language category words. `K` is passed through but is not materially used in the visible dataset prompt path. `version` appears in `train.py` signature but is not used.
5. `base_model` is loaded with `AutoModelForCausalLM.from_pretrained`; tokenizer is loaded separately.
6. The visible training code uses HuggingFace `Trainer` full model training. It does not instantiate PEFT/LoRA adapters in `train.py`.
7. Training output is a saved model directory via `model.save_pretrained(output_dir)`.
8. Evaluation generates item titles with beam search and maps generated titles back to item IDs via `info_file`; metrics are computed by exact title match in `calc.py`.
9. Decoding/framework intervention is in `evaluate.py` and `LogitProcesser.py`: constrained decoding limits next tokens to catalog item-title prefixes, and optional CF logits alter token scores.
10. We can reference its dataset/trainer/evaluation separation and constrained closed-catalog idea. We should not copy its title-generation target, exact-title metric, or CF-logit decoding as our contribution.
""",
        encoding="utf-8",
    )
    (DATA_ROOT / "framework_day3_qwen_lora_task_design.md").write_text(
        """# Framework-Day3 Qwen-LoRA Baseline Task Design

## Boundary

LoRA trains a local Qwen3-8B recommendation baseline. CEP is the framework layer that calibrates relevance posterior and applies evidence-risk decision support on top of model/backbone outputs.

## Format A: Listwise Ranking Instruction

One sample contains user history plus a closed candidate pool. The target is the positive candidate ID / ranked candidate IDs. This is closest to recommender ranking and makes catalog constraints explicit.

Pros: natural ranking baseline, easy to evaluate with NDCG/MRR/HR@1/HR@3, aligns with closed-candidate evaluation, avoids catalog-free hallucination. Cons: fewer training samples and longer context.

## Format B: Pointwise Relevance Instruction

One sample contains user history plus one candidate item. The target is a raw relevance label. This is closer to CEP evidence/relevance posterior, but the LoRA target is still raw relevance, not calibrated probability.

Pros: many samples, simple binary objective, easy to connect to CEP later. Cons: less like a standalone recommender, needs candidate aggregation for ranking, and calibration must remain separate.

## Recommendation

Use listwise as the first Qwen-LoRA recommendation baseline. Keep pointwise as a bridge to CEP/evidence generator design. Do not train LoRA directly on calibrated relevance probabilities; calibration is a valid-set framework operation.
""",
        encoding="utf-8",
    )
    stats_lines = ["# data_done_lora Stats", "", "| domain | mode | train listwise | train pointwise | valid listwise | valid pointwise | test listwise | test pointwise |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for domain, s in stats_by_domain.items():
        stats_lines.append(
            f"| {domain} | {s['mode']} | {s['train_listwise_rows']} | {s['train_pointwise_rows']} | {s['valid_listwise_rows']} | {s['valid_pointwise_rows']} | {s['test_listwise_rows']} | {s['test_pointwise_rows']} |"
        )
    (DATA_ROOT / "framework_day3_lora_training_plan.md").write_text(
        """# Framework-Day3 LoRA Training Plan

1. Reference DecodingMatters for a minimal train/eval entry point and dataset-to-causal-LM wrapping.
2. Implement our own Dataset class because `data_done_lora` uses closed-candidate JSONL and must preserve CEP metadata/fallback flags.
3. First train Qwen-LoRA as a recommendation baseline, not as the CEP method.
4. After a trained baseline exists, compare:
   - Qwen-LoRA baseline
   - Qwen-LoRA + CEP calibrated posterior
   - Qwen-LoRA + evidence risk
   - Qwen-LoRA + full CEP framework
5. Before training, Framework-Day4 should run tokenizer/parser/inference smoke and verify the server-side Qwen3-8B path.
""",
        encoding="utf-8",
    )
    (DATA_ROOT / "framework_day3_qwen_lora_readiness_report.md").write_text(
        """# Framework-Day3 Qwen-LoRA Readiness Report

## 1. Why LoRA and Framework Are Separate

LoRA is a way to train a local Qwen3-8B recommendation baseline. The framework contribution is CEP: calibrated relevance posterior, evidence risk, and uncertainty-aware decision mechanisms on top of model/backbone outputs.

## 2. DecodingMatters Reference Points

DecodingMatters shows a practical local LLM recommender pipeline: CSV samples, dataset-to-instruction conversion, HuggingFace Trainer, closed-catalog constrained decoding, and optional logit processing. Its visible training script does not use PEFT/LoRA, so it is an engineering reference rather than the method to copy.

## 3. Our Qwen-LoRA Baseline Task

We generate both listwise closed-candidate ranking and pointwise relevance JSONL. Listwise is the preferred baseline recommender format; pointwise is kept as a bridge to CEP/evidence generator work.

## 4. data_done_lora Outputs

"""
        + "\n".join(stats_lines)
        + """

## 5. Prompt Templates

Prompts are under `prompts/framework/`. The baseline prompts require JSON output and prohibit candidate IDs outside the candidate pool. The future evidence schema prompt is only a reference and is not used for first-stage baseline training.

## 6. Config Scaffold

`configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml` and `configs/framework/qwen3_8b_lora_baseline_beauty_pointwise.yaml` are scaffolds only. The model path is marked TODO and must be verified on the server.

## 7. Next Step

Framework-Day4 should run Qwen tokenizer/inference/parser smoke or implement the LoRA Dataset/Trainer. Do not start large-scale training yet.
""",
        encoding="utf-8",
    )


def main() -> None:
    _write_prompts()
    _write_configs()
    stats_by_domain = {domain: _build_domain(domain) for domain in DOMAINS}
    _write_reports(stats_by_domain)
    print("Framework-Day3 Qwen-LoRA readiness artifacts generated.")


if __name__ == "__main__":
    main()
