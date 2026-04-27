from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_lora_confidence.yaml"


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


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_confidence_prompt(
    sample: dict[str, Any],
    template_path: str | Path,
    max_history_items: int = 20,
    max_title_chars: int = 160,
    max_history_text_chars: int = 360,
    max_candidate_text_chars: int = 520,
) -> str:
    template = Path(template_path).read_text(encoding="utf-8").strip()
    history = sample.get("history", [])
    if max_history_items > 0:
        history = history[-max_history_items:]
    payload = {
        "user_history": [
            {
                "item_id": str(item.get("item_id", "")),
                "title": _truncate_text(item.get("title", ""), max_title_chars),
                "text": _truncate_text(item.get("text", ""), max_history_text_chars),
                "text_missing": bool(item.get("text_missing", False)),
                "text_fallback_used": bool(item.get("text_fallback_used", False)),
            }
            for item in history
        ],
        "candidate_item": {
            "candidate_item_id": str(sample.get("candidate_item_id", "")),
            "title": _truncate_text(sample.get("candidate_title", ""), max_title_chars),
            "text": _truncate_text(sample.get("candidate_text", ""), max_candidate_text_chars),
            "candidate_text_missing": bool(sample.get("candidate_text_missing", False)),
            "candidate_text_fallback_used": bool(sample.get("candidate_text_fallback_used", False)),
        },
    }
    return f"{template}\n\nInput JSON:\n{_compact_json(payload)}\n\nOutput JSON:\n"


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


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and int(value) in {0, 1}:
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1", "recommend", "recommended"}:
            return True
        if v in {"false", "no", "0", "not_recommend", "not recommended"}:
            return False
    return None


def parse_confidence_response(raw_text: str) -> dict[str, Any]:
    text = _extract_json_text(raw_text)
    if text is None:
        return {
            "parse_success": False,
            "schema_valid": False,
            "recommend": None,
            "confidence": None,
            "decision_basis": "",
            "decision_basis_valid": False,
            "reason": "",
            "parse_error": "no_json_object",
        }
    try:
        obj = json.loads(text)
    except Exception as exc:
        return {
            "parse_success": False,
            "schema_valid": False,
            "recommend": None,
            "confidence": None,
            "decision_basis": "",
            "decision_basis_valid": False,
            "reason": "",
            "parse_error": f"json_error:{type(exc).__name__}",
        }
    recommend = _parse_bool(obj.get("recommend"))
    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    confidence_level_raw = obj.get("confidence_level")
    confidence_level = str(confidence_level_raw).strip().lower() if confidence_level_raw is not None else ""
    confidence_level_valid = confidence_level in {"low", "medium", "high", "very_high"}
    if not confidence_level_valid:
        confidence_level = ""
    decision_basis_raw = obj.get("decision_basis")
    decision_basis = str(decision_basis_raw).strip().lower() if decision_basis_raw is not None else ""
    decision_basis_valid = decision_basis in {
        "strong_match",
        "weak_match",
        "unrelated",
        "insufficient_information",
    }
    if not decision_basis_valid:
        decision_basis = ""
    reason = str(obj.get("reason", "")).strip()
    schema_valid = recommend is not None and confidence is not None
    return {
        "parse_success": True,
        "schema_valid": schema_valid,
        "recommend": recommend,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "confidence_level_valid": confidence_level_valid,
        "decision_basis": decision_basis,
        "decision_basis_valid": decision_basis_valid,
        "reason": reason,
        "parse_error": "" if schema_valid else "missing_or_invalid_recommend_confidence",
    }


def _existing_ids(output_path: Path) -> set[str]:
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    for row in _read_jsonl(output_path):
        ids.add(str(row.get("sample_id", "")))
    return ids


def _sample_id(split: str, idx: int, sample: dict[str, Any]) -> str:
    return f"{split}_{idx}_{sample.get('user_id', '')}_{sample.get('candidate_item_id', '')}"


def _load_model(cfg: dict[str, Any], model_variant: str):
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
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for sampling_key in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, sampling_key):
                setattr(model.generation_config, sampling_key, None)
    model.cuda()
    model.eval()
    return model, tokenizer


def _prediction_row(
    sample: dict[str, Any],
    sid: str,
    split: str,
    raw_text: str,
    parsed: dict[str, Any],
    model_variant: str,
    adapter_path: str,
    inference_backend: str,
) -> dict[str, Any]:
    return {
        "sample_id": sid,
        "domain": sample.get("domain", "beauty"),
        "split": split,
        "user_id": sample.get("user_id", ""),
        "candidate_item_id": sample.get("candidate_item_id", ""),
        "label": int(sample.get("label", 0)),
        "raw_response": raw_text,
        "recommend": parsed["recommend"],
        "confidence": parsed["confidence"],
        "confidence_level": parsed.get("confidence_level", ""),
        "confidence_level_valid": bool(parsed.get("confidence_level_valid", False)),
        "decision_basis": parsed.get("decision_basis", ""),
        "decision_basis_valid": bool(parsed.get("decision_basis_valid", False)),
        "reason": parsed["reason"],
        "parse_success": parsed["parse_success"],
        "schema_valid": parsed["schema_valid"],
        "parse_error": parsed["parse_error"],
        "model_variant": model_variant,
        "adapter_path": adapter_path,
        "inference_backend": inference_backend,
    }


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_transformers_inference(
    cfg: dict[str, Any],
    split: str,
    model_variant: str,
    max_samples: int | None,
    resume: bool,
) -> Path:
    import torch  # type: ignore

    split_file = Path(str(cfg[f"{split}_file"]))
    output_dir = Path(str(cfg["output_dir"]))
    output_path = output_dir / f"{split}_raw.jsonl"
    rows = _read_jsonl(split_file, limit=max_samples)
    finished = _existing_ids(output_path) if resume else set()
    model, tokenizer = _load_model(cfg, model_variant=model_variant)
    pending_out: list[dict[str, Any]] = []
    max_history_items = int(cfg.get("max_history_items", 20))
    max_title_chars = int(cfg.get("max_title_chars", 160))
    max_history_text_chars = int(cfg.get("max_history_text_chars", 360))
    max_candidate_text_chars = int(cfg.get("max_candidate_text_chars", 520))
    adapter_path = str(cfg.get("adapter_path", "")) if model_variant == "lora" else ""
    for idx, sample in enumerate(rows):
        sid = _sample_id(split, idx, sample)
        if sid in finished:
            continue
        prompt = format_confidence_prompt(
            sample,
            cfg["prompt_template"],
            max_history_items=max_history_items,
            max_title_chars=max_title_chars,
            max_history_text_chars=max_history_text_chars,
            max_candidate_text_chars=max_candidate_text_chars,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg.get("max_seq_len", 2048))).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg.get("max_new_tokens", 96)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = parse_confidence_response(raw_text)
        pending_out.append(
            _prediction_row(sample, sid, split, raw_text, parsed, model_variant, adapter_path, "transformers")
        )
        if len(pending_out) % 50 == 0:
            _append_jsonl(output_path, pending_out)
            pending_out = []
    _append_jsonl(output_path, pending_out)
    return output_path


def _load_vllm(cfg: dict[str, Any], model_variant: str):
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on server
        raise RuntimeError(
            "vLLM backend requested but vllm is not importable. Install vllm on the server or use inference_backend=transformers."
        ) from exc

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    dtype = "bfloat16" if str(cfg.get("bf16", "true")).lower() == "true" else "float16"
    enable_lora = model_variant == "lora" and bool(cfg.get("vllm_enable_lora", True))
    llm_kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "dtype": dtype,
        "tensor_parallel_size": int(cfg.get("vllm_tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(cfg.get("vllm_gpu_memory_utilization", 0.85)),
        "max_model_len": int(cfg.get("vllm_max_model_len", cfg.get("max_seq_len", 2048))),
        "enable_lora": enable_lora,
    }
    if enable_lora:
        llm_kwargs["max_loras"] = int(cfg.get("vllm_max_loras", 1))
        llm_kwargs["max_lora_rank"] = int(cfg.get("vllm_max_lora_rank", 8))
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(cfg.get("max_new_tokens", 96)),
    )
    return llm, sampling_params


def _vllm_lora_request(cfg: dict[str, Any], model_variant: str):
    if model_variant != "lora":
        return None
    adapter_path = str(cfg.get("adapter_path") or "")
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
    try:
        from vllm.lora.request import LoRARequest  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on server
        raise RuntimeError("vLLM LoRA requested but vllm.lora.request.LoRARequest is not importable.") from exc
    return LoRARequest("qwen_lora_confidence", 1, adapter_path)


def _run_vllm_inference(
    cfg: dict[str, Any],
    split: str,
    model_variant: str,
    max_samples: int | None,
    resume: bool,
) -> Path:
    split_file = Path(str(cfg[f"{split}_file"]))
    output_dir = Path(str(cfg["output_dir"]))
    output_path = output_dir / f"{split}_raw.jsonl"
    rows = _read_jsonl(split_file, limit=max_samples)
    finished = _existing_ids(output_path) if resume else set()
    llm, sampling_params = _load_vllm(cfg, model_variant=model_variant)
    lora_request = _vllm_lora_request(cfg, model_variant=model_variant)
    max_history_items = int(cfg.get("max_history_items", 20))
    max_title_chars = int(cfg.get("max_title_chars", 160))
    max_history_text_chars = int(cfg.get("max_history_text_chars", 360))
    max_candidate_text_chars = int(cfg.get("max_candidate_text_chars", 520))
    batch_size = int(cfg.get("vllm_batch_size", 16))
    adapter_path = str(cfg.get("adapter_path", "")) if model_variant == "lora" else ""
    batch: list[tuple[int, dict[str, Any], str, str]] = []

    def flush_batch(items: list[tuple[int, dict[str, Any], str, str]]) -> None:
        if not items:
            return
        prompts = [x[3] for x in items]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        out_rows: list[dict[str, Any]] = []
        for (_, sample, sid, _), output in zip(items, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed = parse_confidence_response(raw_text)
            out_rows.append(_prediction_row(sample, sid, split, raw_text, parsed, model_variant, adapter_path, "vllm"))
        _append_jsonl(output_path, out_rows)

    for idx, sample in enumerate(rows):
        sid = _sample_id(split, idx, sample)
        if sid in finished:
            continue
        prompt = format_confidence_prompt(
            sample,
            cfg["prompt_template"],
            max_history_items=max_history_items,
            max_title_chars=max_title_chars,
            max_history_text_chars=max_history_text_chars,
            max_candidate_text_chars=max_candidate_text_chars,
        )
        batch.append((idx, sample, sid, prompt))
        if len(batch) >= batch_size:
            flush_batch(batch)
            batch = []
    flush_batch(batch)
    return output_path


def run_inference(
    cfg: dict[str, Any],
    split: str,
    model_variant: str,
    max_samples: int | None,
    resume: bool,
    inference_backend: str | None = None,
) -> Path:
    backend = (inference_backend or str(cfg.get("inference_backend", "transformers"))).strip().lower()
    if backend == "transformers":
        return _run_transformers_inference(cfg, split, model_variant, max_samples, resume)
    if backend == "vllm":
        return _run_vllm_inference(cfg, split, model_variant, max_samples, resume)
    raise ValueError(f"Unsupported inference_backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1 local Qwen confidence inference.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--inference_backend", choices=["transformers", "vllm"], default=None)
    parser.add_argument("--output_dir", default=None, help="Optional prediction output directory override.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = _read_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    variant = args.model_variant or str(cfg.get("model_variant", "lora"))
    path = run_inference(
        cfg,
        split=args.split,
        model_variant=variant,
        max_samples=args.max_samples,
        resume=args.resume,
        inference_backend=args.inference_backend,
    )
    backend = args.inference_backend or str(cfg.get("inference_backend", "transformers"))
    print(json.dumps({"output_path": str(path), "split": args.split, "model_variant": variant, "inference_backend": backend}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
