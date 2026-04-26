from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from src.framework.lora_dataset import QwenRecommendationDataset


SUMMARY_DIR = Path("data_done")


def _read_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data or {}
    except Exception:
        cfg: dict[str, Any] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
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


def _write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _is_todo_path(path: str) -> bool:
    return (not path) or path.startswith("TODO") or path.startswith("TODO_MODEL_PATH")


def _dependency_status() -> dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "peft": importlib.util.find_spec("peft") is not None,
    }


def _gpu_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_tokenizer_if_possible(cfg: dict[str, Any], blocked: list[str]):
    model_path = str(cfg.get("tokenizer_name_or_path") or cfg.get("model_name_or_path") or "")
    if _is_todo_path(model_path) or not Path(model_path).exists():
        blocked.append("model_path_missing")
        return None
    if importlib.util.find_spec("transformers") is None:
        blocked.append("dependency_missing:transformers")
        return None
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _maybe_model_forward_smoke(cfg: dict[str, Any], dataset: QwenRecommendationDataset, blocked: list[str]) -> dict[str, Any]:
    if "model_path_missing" in blocked or any(x.startswith("dependency_missing") for x in blocked):
        return {"attempted": False, "status": "skipped", "reason": ",".join(blocked)}
    if not _gpu_available():
        blocked.append("gpu_missing")
        return {"attempted": False, "status": "skipped", "reason": "gpu_missing"}
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore

        model_path = str(cfg["model_name_or_path"])
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        target_modules = [x.strip() for x in str(cfg.get("target_modules", "q_proj,v_proj")).split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_rank", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg).cuda()
        batch = dataset[0]
        inputs = {
            "input_ids": torch.tensor([batch["input_ids"]], dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor([batch["attention_mask"]], dtype=torch.long, device="cuda"),
            "labels": torch.tensor([batch["labels"]], dtype=torch.long, device="cuda"),
        }
        with torch.no_grad():
            out = model(**inputs)
        return {"attempted": True, "status": "success", "loss": float(out.loss.detach().cpu().item())}
    except Exception as exc:
        blocked.append(f"forward_smoke_failed:{type(exc).__name__}")
        return {"attempted": True, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def run_dry_run(config_path: str | Path, sample_count: int = 5) -> dict[str, Any]:
    cfg = _read_config(config_path)
    blocked: list[str] = []
    deps = _dependency_status()
    for dep in ["torch", "transformers", "peft"]:
        if not deps[dep]:
            blocked.append(f"dependency_missing:{dep}")
    tokenizer = _load_tokenizer_if_possible(cfg, blocked)
    task_type = str(cfg.get("task_type", ""))
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    max_train_samples = int(cfg.get("max_train_samples", sample_count) or sample_count)
    max_eval_samples = int(cfg.get("max_eval_samples", sample_count) or sample_count)
    train_ds = QwenRecommendationDataset(
        cfg["train_file"],
        task_type=task_type,
        prompt_template=cfg["prompt_template"],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=max_train_samples,
    )
    valid_ds = QwenRecommendationDataset(
        cfg["valid_file"],
        task_type=task_type,
        prompt_template=cfg["prompt_template"],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=max_eval_samples,
    )
    sample_prompts = {
        "config_path": str(config_path),
        "task_type": task_type,
        "train_samples": train_ds.sample_prompts(sample_count),
        "valid_samples": valid_ds.sample_prompts(sample_count),
    }
    _write_json(SUMMARY_DIR / "framework_day4_lora_sample_prompts.json", sample_prompts)
    stats_rows = [
        train_ds.stats(split="train", tokenizer_available=tokenizer is not None),
        valid_ds.stats(split="valid", tokenizer_available=tokenizer is not None),
    ]
    _write_csv(SUMMARY_DIR / "framework_day4_lora_dataset_stats.csv", stats_rows)
    if tokenizer is None:
        label_mask_check = {"status": "skipped", "reason": "tokenizer_not_loaded"}
    else:
        encoded0 = train_ds[0]
        labels = encoded0.get("labels", [])
        label_mask_check = {
            "status": "success",
            "input_len": len(encoded0.get("input_ids", [])),
            "label_len": len(labels),
            "masked_label_count": sum(1 for x in labels if x == -100),
            "supervised_label_count": sum(1 for x in labels if x != -100),
        }
    forward = _maybe_model_forward_smoke(cfg, train_ds, blocked)
    status = "ready_for_server_smoke" if not blocked else "blocked"
    return {
        "config_path": str(config_path),
        "dry_run": True,
        "status": status,
        "blocked_reasons": sorted(set(blocked)),
        "dependencies": deps,
        "gpu_available": _gpu_available(),
        "tokenizer_loaded": tokenizer is not None,
        "train_samples_checked": len(train_ds),
        "valid_samples_checked": len(valid_ds),
        "label_mask_check": label_mask_check,
        "forward_smoke": forward,
    }


def _write_reports(result: dict[str, Any]) -> None:
    blocked = result.get("blocked_reasons", [])
    report = f"""# Framework-Day4 LoRA Dry-Run Report

## Result

- Status: `{result['status']}`
- Config: `{result['config_path']}`
- Blocked reasons: `{', '.join(blocked) if blocked else 'none'}`
- GPU available: `{result['gpu_available']}`
- Tokenizer loaded: `{result['tokenizer_loaded']}`
- Train samples checked: `{result['train_samples_checked']}`
- Valid samples checked: `{result['valid_samples_checked']}`
- Label mask check: `{result['label_mask_check']['status']}`
- Forward smoke: `{result['forward_smoke']['status']}`

This is not a training run. No API was called, no LoRA training was started, and no CEP/confidence/evidence fusion was implemented.
"""
    (SUMMARY_DIR / "framework_day4_lora_dry_run_report.md").write_text(report, encoding="utf-8")
    (SUMMARY_DIR / "framework_day4_future_confidence_evidence_framework_note.md").write_text(
        """# Future Confidence + Evidence Framework Note

The future CEP framework combines two observation lines.

1. Confidence line: Week1-Week4 verbalized confidence, calibration, robustness, and multi-model observation from `Paper/version1`, `Paper/version2`, and older week1-week4 outputs.
2. Evidence line: Day9+ evidence decomposition, calibrated relevance posterior, evidence risk, external backbone plug-in, and robustness from `output-repaired`.

Qwen-LoRA is a fixed local recommendation baseline. It should learn relevance/ranking ability, not calibrated probability or final CEP risk. Confidence calibration, evidence risk, and CEP fusion are decision-stage framework modules fitted or selected on valid data and fixed for test evaluation.

Framework-Day4 intentionally does not implement confidence module, evidence module, or CEP fusion. It only verifies that the Qwen-LoRA baseline training pipeline is clean.
""",
        encoding="utf-8",
    )
    (SUMMARY_DIR / "framework_day4_qwen_lora_baseline_pipeline_report.md").write_text(
        f"""# Framework-Day4 Qwen-LoRA Baseline Pipeline Report

## 1. Day3 Recap

Framework-Day3 prepared `data_done_lora`, baseline prompts, and Qwen3-8B LoRA config scaffolds for Beauty listwise and pointwise tasks.

## 2. LoRA Baseline vs Framework Boundary

LoRA is only the local Qwen3-8B recommendation baseline. CEP/calibrated posterior/evidence risk are not trained into this baseline and remain later decision-stage framework modules.

## 3. Future Confidence + Evidence Direction

The future framework will combine calibrated confidence uncertainty from the week1-week4 confidence line and evidence risk / calibrated relevance posterior from the Day9+ evidence line. Day4 does not implement that fusion.

## 4. Dataset Loader

`src/framework/lora_dataset.py` reads listwise and pointwise JSONL, formats samples, masks labels, preserves metadata, and excludes calibrated probability / CEP fields from targets.

## 5. Prompt Formatter

`src/framework/prompt_formatters.py` provides stable Qwen baseline formatting for closed-candidate ranking and pointwise relevance.

## 6. Training Entrypoint

`main_framework_day4_train_qwen_lora_baseline.py` reads config, checks dependencies/model path/GPU, builds datasets, supports dry-run, and can run a forward smoke only when model path and GPU are available.

## 7. Dry-Run Result

- Status: `{result['status']}`
- Blocked reasons: `{', '.join(blocked) if blocked else 'none'}`
- Label mask check: `{result['label_mask_check']['status']}`
- Dataset stats: `data_done/framework_day4_lora_dataset_stats.csv`
- Sample prompts: `data_done/framework_day4_lora_sample_prompts.json`

## 8. Current Ready / Blocked State

The local dataset/prompt/config pipeline is ready. Full model smoke/training is blocked locally if `model_path_missing`, `gpu_missing`, or dependency issues are listed above. This is environment readiness, not method failure.

## 9. Day5 Recommendation

If the server model path and GPU are available, run a Beauty listwise tiny LoRA train. If not, first fix the server environment. Do not enter confidence/evidence framework fusion before baseline training smoke is stable.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sample_count", type=int, default=5)
    args = parser.parse_args()
    result = run_dry_run(args.config, sample_count=args.sample_count)
    _write_reports(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
