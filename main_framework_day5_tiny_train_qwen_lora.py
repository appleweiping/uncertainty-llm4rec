from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from src.framework.lora_dataset import QwenRecommendationDataset


METRICS_PATH = Path("data_done/framework_day5_lora_tiny_train_metrics.json")
REPORT_PATH = Path("data_done/framework_day5_lora_tiny_train_report.md")


def _is_todo_path(path: str) -> bool:
    return (not path) or path.startswith("TODO")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _target_modules(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "q_proj,v_proj").split(",") if x.strip()]


def _check_blockers(cfg: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    model_path = str(cfg.get("model_name_or_path", ""))
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    if _is_todo_path(model_path) or not Path(model_path).exists():
        blockers.append("model_path_missing")
    if _is_todo_path(tokenizer_path) or not Path(tokenizer_path).exists():
        blockers.append("tokenizer_path_missing")
    if not Path(str(cfg.get("train_file", ""))).exists():
        blockers.append("train_file_missing")
    if not Path(str(cfg.get("valid_file", ""))).exists():
        blockers.append("valid_file_missing")
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            blockers.append("gpu_missing")
    except Exception:
        blockers.append("dependency_missing:torch")
    for dep in ["transformers", "peft"]:
        try:
            __import__(dep)
        except Exception:
            blockers.append(f"dependency_missing:{dep}")
    return blockers


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_report(
    metrics: dict[str, Any],
    report_path: Path = REPORT_PATH,
    title: str = "Framework-Day5 Beauty Listwise Qwen-LoRA Tiny Train Report",
    scope: str = "This is a tiny LoRA training smoke for the Beauty listwise baseline only. It does not implement confidence, evidence, CEP fusion, API calls, or formal long training.",
) -> None:
    losses = metrics.get("losses", [])
    report = f"""# {title}

## Scope

{scope}

## Status

- Status: `{metrics.get('status')}`
- Blocked reasons: `{', '.join(metrics.get('blocked_reasons', [])) if metrics.get('blocked_reasons') else 'none'}`
- OOM status: `{metrics.get('oom_status')}`
- Ready for Day6 real train: `{metrics.get('ready_for_day6_real_train')}`

## Config

- train samples: `{metrics.get('train_samples')}`
- eval samples: `{metrics.get('eval_samples')}`
- max steps: `{metrics.get('max_steps')}`
- batch size: `{metrics.get('batch_size')}`
- gradient accumulation steps: `{metrics.get('gradient_accumulation_steps')}`
- max seq len: `{metrics.get('max_seq_len')}`
- LoRA rank/alpha/dropout: `{metrics.get('lora_rank')}` / `{metrics.get('lora_alpha')}` / `{metrics.get('lora_dropout')}`
- adapter output dir: `{metrics.get('adapter_output_dir')}`

## Loss

- loss first: `{metrics.get('loss_first')}`
- loss last: `{metrics.get('loss_last')}`
- loss NaN count: `{metrics.get('loss_nan_count')}`
- recorded losses: `{losses[:10]}`

## GPU

- peak GPU memory GB: `{metrics.get('peak_gpu_memory_gb')}`

## Interpretation

Passing this tiny train means the Qwen3-8B LoRA baseline infrastructure can perform optimizer steps on server data. It is not a performance result and should not be used as CEP/framework evidence.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def _collate(batch: list[dict[str, Any]], tokenizer: Any):
    import torch  # type: ignore

    pad_id = tokenizer.pad_token_id
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def run_tiny_train(
    config_path: str | Path,
    metrics_path: Path = METRICS_PATH,
    report_path: Path = REPORT_PATH,
    report_title: str = "Framework-Day5 Beauty Listwise Qwen-LoRA Tiny Train Report",
    report_scope: str = "This is a tiny LoRA training smoke for the Beauty listwise baseline only. It does not implement confidence, evidence, CEP fusion, API calls, or formal long training.",
) -> dict[str, Any]:
    cfg = _read_config(config_path)
    blockers = _check_blockers(cfg)
    metrics: dict[str, Any] = {
        "config_path": str(config_path),
        "status": "blocked" if blockers else "running",
        "blocked_reasons": blockers,
        "oom_status": False,
        "ready_for_day6_real_train": False,
        "adapter_output_dir": str(cfg.get("output_dir", "")),
        "train_samples": int(cfg.get("max_train_samples", 0) or 0),
        "eval_samples": int(cfg.get("max_eval_samples", 0) or 0),
        "max_steps": int(cfg.get("max_steps", 0) or 0),
        "batch_size": int(cfg.get("batch_size", 0) or 0),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 0) or 0),
        "max_seq_len": int(cfg.get("max_seq_len", 0) or 0),
        "lora_rank": int(cfg.get("lora_rank", 0) or 0),
        "lora_alpha": int(cfg.get("lora_alpha", 0) or 0),
        "lora_dropout": float(cfg.get("lora_dropout", 0.0) or 0.0),
        "losses": [],
        "loss_first": None,
        "loss_last": None,
        "loss_nan_count": None,
        "peak_gpu_memory_gb": None,
    }
    if blockers:
        _write_json(metrics_path, metrics)
        _write_report(metrics, report_path=report_path, title=report_title, scope=report_scope)
        return metrics

    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = int(cfg.get("max_seq_len", 2048))
    train_ds = QwenRecommendationDataset(
        cfg["train_file"],
        task_type=str(cfg["task_type"]),
        prompt_template=cfg["prompt_template"],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=int(cfg.get("max_train_samples", 64)),
    )
    valid_ds = QwenRecommendationDataset(
        cfg["valid_file"],
        task_type=str(cfg["task_type"]),
        prompt_template=cfg["prompt_template"],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=int(cfg.get("max_eval_samples", 64)),
    )

    dtype = torch.bfloat16 if _as_bool(cfg.get("bf16"), True) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False
    if _as_bool(cfg.get("gradient_checkpointing", True), True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=_target_modules(cfg.get("target_modules")),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.cuda()
    model.train()

    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    max_steps = int(cfg.get("max_steps", 10))
    learning_rate = float(cfg.get("learning_rate", 2e-4))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: _collate(b, tokenizer))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    losses: list[float] = []
    loss_nan_count = 0
    step = 0
    micro_step = 0
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        while step < max_steps:
            for batch in loader:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / grad_accum
                raw_loss = float(out.loss.detach().cpu().item())
                if math.isnan(raw_loss) or math.isinf(raw_loss):
                    loss_nan_count += 1
                loss.backward()
                micro_step += 1
                if micro_step % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    losses.append(raw_loss)
                    step += 1
                    print(json.dumps({"step": step, "loss": raw_loss}, ensure_ascii=False), flush=True)
                    if step >= max_steps:
                        break
    except torch.cuda.OutOfMemoryError:
        metrics["oom_status"] = True
        metrics["status"] = "oom"
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    if metrics.get("status") not in {"oom", "failed"}:
        metrics["status"] = "success" if losses and loss_nan_count == 0 else "failed"

    save_strategy = str(cfg.get("save_strategy", "final")).lower()
    output_dir = Path(str(cfg.get("output_dir", "artifacts/lora_smoke/qwen3_8b_beauty_listwise_day5_tiny")))
    if metrics["status"] == "success" and save_strategy in {"final", "save_final", "adapter"}:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    metrics.update(
        {
            "train_samples": len(train_ds),
            "eval_samples": len(valid_ds),
            "max_steps": max_steps,
            "completed_steps": step,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "max_seq_len": max_seq_len,
            "lora_rank": int(cfg.get("lora_rank", 8)),
            "lora_alpha": int(cfg.get("lora_alpha", 16)),
            "lora_dropout": float(cfg.get("lora_dropout", 0.05)),
            "losses": losses,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_nan_count": loss_nan_count,
            "peak_gpu_memory_gb": round(peak_gb, 4) if peak_gb is not None else None,
            "runtime_seconds": round(time.time() - start, 2),
            "ready_for_day6_real_train": metrics["status"] == "success" and step >= max_steps and loss_nan_count == 0,
            "adapter_output_dir": str(output_dir) if save_strategy in {"final", "save_final", "adapter"} else "",
            "save_strategy": save_strategy,
        }
    )
    _write_json(metrics_path, metrics)
    _write_report(metrics, report_path=report_path, title=report_title, scope=report_scope)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day5 tiny Qwen-LoRA train smoke.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_tiny.yaml")
    args = parser.parse_args()
    result = run_tiny_train(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
