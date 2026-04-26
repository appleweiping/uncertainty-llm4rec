from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(".")
SUMMARY_MD = Path("data_done/framework_day45_lora_data_readiness_check.md")
SUMMARY_JSON = Path("data_done/framework_day45_lora_data_readiness_check.json")
DEFAULT_CONFIG = Path("configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml")


def _read_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                cfg[key.strip()] = value.lower() == "true"
            else:
                cfg[key.strip()] = value
        return cfg


def _line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _import_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_available() -> bool | str:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception as exc:
        return f"unknown:{type(exc).__name__}"


def _is_todo_model_path(path: str) -> bool:
    return (not path) or path.startswith("TODO")


def build_readiness() -> dict[str, Any]:
    source_train = Path("data_done/beauty/train.jsonl")
    lora_files = [
        Path("data_done_lora/beauty/train_listwise.jsonl"),
        Path("data_done_lora/beauty/valid_listwise.jsonl"),
        Path("data_done_lora/beauty/test_listwise.jsonl"),
        Path("data_done_lora/beauty/train_pointwise.jsonl"),
        Path("data_done_lora/beauty/valid_pointwise.jsonl"),
        Path("data_done_lora/beauty/test_pointwise.jsonl"),
    ]
    config_exists = DEFAULT_CONFIG.exists()
    cfg = _read_config(DEFAULT_CONFIG) if config_exists else {}
    model_path = str(cfg.get("model_name_or_path", ""))
    tokenizer_path = str(cfg.get("tokenizer_name_or_path", ""))
    train_file = Path(str(cfg.get("train_file", ""))) if cfg.get("train_file") else None
    valid_file = Path(str(cfg.get("valid_file", ""))) if cfg.get("valid_file") else None
    result = {
        "source_data": {
            "data_done_beauty_train_exists": source_train.exists(),
            "data_done_beauty_train_rows": _line_count(source_train),
        },
        "lora_data": {
            str(path): {"exists": path.exists(), "rows": _line_count(path)} for path in lora_files
        },
        "config": {
            "config_path": str(DEFAULT_CONFIG),
            "exists": config_exists,
            "train_file": str(train_file) if train_file else "",
            "train_file_exists": train_file.exists() if train_file else False,
            "valid_file": str(valid_file) if valid_file else "",
            "valid_file_exists": valid_file.exists() if valid_file else False,
            "model_name_or_path": model_path,
            "model_path_is_todo": _is_todo_model_path(model_path),
            "model_path_exists": Path(model_path).exists() if not _is_todo_model_path(model_path) else False,
            "tokenizer_name_or_path": tokenizer_path,
            "tokenizer_path_is_todo": _is_todo_model_path(tokenizer_path),
            "tokenizer_path_exists": Path(tokenizer_path).exists() if not _is_todo_model_path(tokenizer_path) else False,
        },
        "environment": {
            "torch_importable": _import_available("torch"),
            "transformers_importable": _import_available("transformers"),
            "peft_importable": _import_available("peft"),
            "accelerate_importable": _import_available("accelerate"),
            "bitsandbytes_importable": _import_available("bitsandbytes"),
            "cuda_available": _cuda_available(),
        },
    }
    missing_lora = [p for p, meta in result["lora_data"].items() if not meta["exists"]]
    zero_lora = [p for p, meta in result["lora_data"].items() if meta["exists"] and not meta["rows"]]
    blocking = []
    if not result["source_data"]["data_done_beauty_train_exists"]:
        blocking.append("missing_data_done_beauty_train")
    if missing_lora:
        blocking.append("missing_data_done_lora_jsonl")
    if zero_lora:
        blocking.append("empty_data_done_lora_jsonl")
    if not result["config"]["train_file_exists"] or not result["config"]["valid_file_exists"]:
        blocking.append("config_points_to_missing_lora_files")
    if result["config"]["model_path_is_todo"] or not result["config"]["model_path_exists"]:
        blocking.append("model_path_missing_or_todo")
    if not result["environment"]["peft_importable"]:
        blocking.append("dependency_missing_peft")
    if result["environment"]["cuda_available"] is not True:
        blocking.append("cuda_unavailable")
    result["status"] = "ready_for_day5_forward_smoke" if not blocking else "blocked"
    result["blocking_reasons"] = blocking
    return result


def write_report(result: dict[str, Any]) -> None:
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lora_lines = []
    for path, meta in result["lora_data"].items():
        lora_lines.append(f"- `{path}`: exists=`{meta['exists']}`, rows=`{meta['rows']}`")
    md = f"""# Framework-Day4.5 LoRA Data Readiness Check

## Status

- Status: `{result['status']}`
- Blocking reasons: `{', '.join(result['blocking_reasons']) if result['blocking_reasons'] else 'none'}`

## Source data_done

- `data_done/beauty/train.jsonl` exists: `{result['source_data']['data_done_beauty_train_exists']}`
- rows: `{result['source_data']['data_done_beauty_train_rows']}`

## data_done_lora Files

{chr(10).join(lora_lines)}

## Config

- config: `{result['config']['config_path']}`
- train_file exists: `{result['config']['train_file_exists']}` (`{result['config']['train_file']}`)
- valid_file exists: `{result['config']['valid_file_exists']}` (`{result['config']['valid_file']}`)
- model path TODO: `{result['config']['model_path_is_todo']}`
- model path exists: `{result['config']['model_path_exists']}`

## Environment

- torch importable: `{result['environment']['torch_importable']}`
- transformers importable: `{result['environment']['transformers_importable']}`
- peft importable: `{result['environment']['peft_importable']}`
- accelerate importable: `{result['environment']['accelerate_importable']}`
- bitsandbytes importable: `{result['environment']['bitsandbytes_importable']}`
- cuda available: `{result['environment']['cuda_available']}`
"""
    SUMMARY_MD.write_text(md, encoding="utf-8")


def main() -> None:
    result = build_readiness()
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
