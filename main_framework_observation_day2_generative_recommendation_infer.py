from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_observation_day1_local_confidence_infer import (
    _compact_json,
    _extract_json_text,
    _read_jsonl,
    _truncate_text,
)


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_base_generative_candidate_grounded.yaml"


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_by_user(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("user_id", ""))].append(row)
    return dict(grouped)


def _stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _select_user_pools(cfg: dict[str, Any], split: str, max_users: int | None) -> list[list[dict[str, Any]]]:
    rows = _read_jsonl(Path(str(cfg[f"{split}_file"])))
    grouped = _group_by_user(rows)
    complete_user_ids = sorted(user_id for user_id, pool in grouped.items() if len(pool) >= 2)
    limit = max_users if max_users is not None else int(cfg.get("max_users", 100))
    seed = int(cfg.get("seed", 42))
    if len(complete_user_ids) > limit:
        rng = random.Random(seed)
        selected = sorted(rng.sample(complete_user_ids, limit))
    else:
        selected = complete_user_ids
    pools: list[list[dict[str, Any]]] = []
    for user_id in selected:
        pool = list(grouped[user_id])
        pool.sort(key=lambda row: str(row.get("candidate_item_id", "")))
        if bool(cfg.get("shuffle_candidates", True)):
            rng = random.Random(_stable_seed(seed, split, user_id, "day2_candidate_shuffle"))
            rng.shuffle(pool)
        pools.append(pool)
    return pools


def _target_row(pool: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in pool if int(row.get("label", 0)) == 1]
    return positives[0] if positives else pool[0]


def _history_payload(row: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    history = list(row.get("history", []))
    max_history_items = int(cfg.get("max_history_items", 8))
    if max_history_items > 0:
        history = history[-max_history_items:]
    return [
        {
            "title": _truncate_text(item.get("title", ""), int(cfg.get("max_title_chars", 180))),
            "text": _truncate_text(item.get("text", ""), int(cfg.get("max_history_text_chars", 240))),
        }
        for item in history
    ]


def _candidate_payload(pool: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [
        {
            "candidate_label": labels[idx],
            "candidate_number": idx + 1,
            "title": _truncate_text(row.get("candidate_title", ""), int(cfg.get("max_title_chars", 180))),
            "text": _truncate_text(row.get("candidate_text", ""), int(cfg.get("max_candidate_text_chars", 260))),
        }
        for idx, row in enumerate(pool)
    ]


def format_prompt(pool: list[dict[str, Any]], cfg: dict[str, Any], setting: str) -> str:
    if setting == "candidate_grounded":
        template_path = Path(str(cfg["prompt_template_candidate_grounded"]))
    elif setting == "open_title":
        template_path = Path(str(cfg["prompt_template_open_title"]))
    else:
        raise ValueError(f"Unsupported setting: {setting}")
    template = template_path.read_text(encoding="utf-8").strip()
    target = _target_row(pool)
    payload: dict[str, Any] = {"user_history": _history_payload(target, cfg)}
    if setting == "candidate_grounded":
        payload["candidate_pool"] = _candidate_payload(pool, cfg)
    return f"{template}\n\nInput JSON:\n{_compact_json(payload)}\n\nOutput JSON:\n"


def parse_generation(raw_text: str) -> dict[str, Any]:
    json_text = _extract_json_text(raw_text)
    if json_text is None:
        return {
            "parse_success": False,
            "schema_valid": False,
            "recommended_title": "",
            "confidence": None,
            "parse_error": "no_json_object",
        }
    try:
        obj = json.loads(json_text)
    except Exception as exc:
        return {
            "parse_success": False,
            "schema_valid": False,
            "recommended_title": "",
            "confidence": None,
            "parse_error": f"json_error:{type(exc).__name__}",
        }
    title = str(obj.get("recommended_title", "")).strip()
    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    schema_valid = bool(title) and confidence is not None
    return {
        "parse_success": True,
        "schema_valid": schema_valid,
        "recommended_title": title,
        "confidence": confidence,
        "parse_error": "" if schema_valid else "missing_or_invalid_title_confidence",
    }


def _sample_id(split: str, pool: list[dict[str, Any]]) -> str:
    target = _target_row(pool)
    return f"{split}_{target.get('user_id', '')}"


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in _read_jsonl(path)}


def _prediction_row(
    split: str,
    pool: list[dict[str, Any]],
    prompt: str,
    raw_text: str,
    parsed: dict[str, Any],
    setting: str,
    model_variant: str,
    backend: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    target = _target_row(pool)
    candidate_pool = [
        {
            "candidate_item_id": str(row.get("candidate_item_id", "")),
            "candidate_title": str(row.get("candidate_title", "")),
            "label": int(row.get("label", 0)),
            "input_position": idx + 1,
        }
        for idx, row in enumerate(pool)
    ]
    return {
        "sample_id": _sample_id(split, pool),
        "domain": target.get("domain", cfg.get("domain", "beauty")),
        "split": split,
        "setting": setting,
        "model_variant": model_variant,
        "backend": backend,
        "user_id": str(target.get("user_id", "")),
        "history": target.get("history", []),
        "candidate_pool": candidate_pool,
        "target_item_id": str(target.get("candidate_item_id", "")),
        "target_title": str(target.get("candidate_title", "")),
        "raw_response": raw_text,
        "recommended_title": parsed["recommended_title"],
        "confidence": parsed["confidence"],
        "parse_success": parsed["parse_success"],
        "schema_valid": parsed["schema_valid"],
        "parse_error": parsed["parse_error"],
        "prompt_token_proxy_chars": len(prompt),
        "candidate_order_seeded_shuffle": bool(cfg.get("shuffle_candidates", True)),
    }


def _load_vllm(cfg: dict[str, Any], model_variant: str):
    from vllm import LLM  # type: ignore

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    enable_lora = model_variant == "lora" and bool(cfg.get("vllm_enable_lora", True))
    kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "dtype": "bfloat16" if str(cfg.get("bf16", "true")).lower() == "true" else "float16",
        "tensor_parallel_size": int(cfg.get("vllm_tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(cfg.get("vllm_gpu_memory_utilization", 0.85)),
        "max_model_len": int(cfg.get("vllm_max_model_len", cfg.get("max_seq_len", 4096))),
        "enable_lora": enable_lora,
        "disable_log_stats": True,
    }
    if enable_lora:
        kwargs["max_loras"] = int(cfg.get("vllm_max_loras", 1))
        kwargs["max_lora_rank"] = int(cfg.get("vllm_max_lora_rank", 8))
    return LLM(**kwargs)


def _lora_request(cfg: dict[str, Any], model_variant: str):
    if model_variant != "lora":
        return None
    from vllm.lora.request import LoRARequest  # type: ignore

    adapter_path = str(cfg.get("adapter_path") or "")
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
    return LoRARequest("qwen_day2_generative", 1, adapter_path)


def run_vllm(
    cfg: dict[str, Any],
    split: str,
    setting: str,
    model_variant: str,
    max_users: int | None,
    resume: bool,
) -> Path:
    from vllm import SamplingParams  # type: ignore

    output_path = Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    pools = _select_user_pools(cfg, split, max_users)
    prompt_rows = []
    for pool in pools:
        sid = _sample_id(split, pool)
        if sid in finished:
            continue
        prompt_rows.append((pool, format_prompt(pool, cfg, setting)))

    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    sampling_params = SamplingParams(
        max_tokens=int(cfg.get("max_new_tokens", 96)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        seed=int(cfg.get("seed", 42)),
    )
    batch_size = int(cfg.get("vllm_batch_size", 12))
    for start in range(0, len(prompt_rows), batch_size):
        batch = prompt_rows[start : start + batch_size]
        outputs = llm.generate([prompt for _, prompt in batch], sampling_params, lora_request=lora_request)
        rows = []
        for (pool, prompt), output in zip(batch, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed = parse_generation(raw_text)
            rows.append(_prediction_row(split, pool, prompt, raw_text, parsed, setting, model_variant, "vllm", cfg))
        _append_jsonl(output_path, rows)
    return output_path


def _load_transformers(cfg: dict[str, Any], model_variant: str):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(cfg.get("bf16", "true")).lower() == "true" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    if model_variant == "lora":
        from peft import PeftModel  # type: ignore

        adapter_path = str(cfg.get("adapter_path") or "")
        if not adapter_path or not Path(adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.cuda()
    model.eval()
    return model, tokenizer


def run_transformers(
    cfg: dict[str, Any],
    split: str,
    setting: str,
    model_variant: str,
    max_users: int | None,
    resume: bool,
) -> Path:
    import torch  # type: ignore

    output_path = Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    pools = _select_user_pools(cfg, split, max_users)
    prompt_rows = []
    for pool in pools:
        sid = _sample_id(split, pool)
        if sid in finished:
            continue
        prompt_rows.append((pool, format_prompt(pool, cfg, setting)))

    model, tokenizer = _load_transformers(cfg, model_variant)
    batch_size = int(cfg.get("transformers_batch_size", 1))
    for start in range(0, len(prompt_rows), batch_size):
        batch = prompt_rows[start : start + batch_size]
        prompts = [prompt for _, prompt in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        do_sample = bool(cfg.get("do_sample", False))
        gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 96)),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(cfg.get("temperature", 1.0))
            gen_kwargs["top_p"] = float(cfg.get("top_p", 1.0))
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        rows = []
        for i, (pool, prompt) in enumerate(batch):
            prompt_len = int(inputs["attention_mask"][i].sum().item())
            raw_text = tokenizer.decode(generated[i][prompt_len:], skip_special_tokens=True)
            parsed = parse_generation(raw_text)
            rows.append(_prediction_row(split, pool, prompt, raw_text, parsed, setting, model_variant, "transformers", cfg))
        _append_jsonl(output_path, rows)
    return output_path


def run_inference(
    cfg: dict[str, Any],
    split: str,
    setting: str,
    model_variant: str,
    backend: str,
    max_users: int | None,
    resume: bool,
) -> Path:
    if backend == "vllm":
        return run_vllm(cfg, split, setting, model_variant, max_users, resume)
    if backend == "transformers":
        return run_transformers(cfg, split, setting, model_variant, max_users, resume)
    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--setting", choices=["candidate_grounded", "open_title"], default=None)
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default=None)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _read_config(args.config)
    setting = args.setting or str(cfg.get("setting", "candidate_grounded"))
    model_variant = args.model_variant or str(cfg.get("model_variant", "base"))
    backend = args.backend or str(cfg.get("backend", cfg.get("inference_backend", "vllm")))
    path = run_inference(cfg, args.split, setting, model_variant, backend, args.max_users, args.resume)
    print(
        json.dumps(
            {
                "output_path": str(path),
                "split": args.split,
                "setting": setting,
                "model_variant": model_variant,
                "backend": backend,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
