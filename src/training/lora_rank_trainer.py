from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.training.rank_dataset import (
    RankSupervisedExample,
    build_rank_supervised_examples,
    load_rank_samples,
    summarize_rank_samples,
)
from src.training.framework_artifacts import update_framework_manifest, utc_now_iso
from src.training.training_log import append_training_status, save_training_summary
from src.utils.exp_io import load_yaml


@dataclass(frozen=True)
class TrainingRunContext:
    run_name: str
    domain: str
    task: str
    base_model_config: Path
    prompt_path: Path
    train_input_path: Path
    valid_input_path: Path
    eval_input_path: Path
    adapter_output_dir: Path
    framework_output_dir: Path
    logs_dir: Path
    output_root: Path
    topk: int
    seed: int
    method_family: str
    method_variant: str
    model_name: str
    train_status_path: Path
    framework_compare_path: Path
    training_cfg: dict[str, Any]
    evaluation_cfg: dict[str, Any]
    support_signals: dict[str, Any]
    model_cfg: dict[str, Any]
    startup_check_path: Path
    dataset_preview_path: Path
    training_summary_path: Path
    framework_manifest_path: Path
    compare_markdown_path: Path


def build_training_run_context(config: dict[str, Any]) -> TrainingRunContext:
    training_cfg = config.get("training", {}) or {}
    evaluation_cfg = config.get("evaluation", {}) or {}
    summary_cfg = config.get("summary", {}) or {}
    model_cfg = config.get("model", {}) or {}

    return TrainingRunContext(
        run_name=str(config["run_name"]),
        domain=str(config.get("domain", "beauty")),
        task=str(config.get("task", "candidate_ranking")),
        base_model_config=Path(str(config["base_model_config"])),
        prompt_path=Path(str(config["prompt_path"])),
        train_input_path=Path(str(config["train_input_path"])),
        valid_input_path=Path(str(config["valid_input_path"])),
        eval_input_path=Path(str(config["eval_input_path"])),
        adapter_output_dir=Path(str(config["adapter_output_dir"])),
        framework_output_dir=Path(str(config["framework_output_dir"])),
        logs_dir=Path(str(config["logs_dir"])),
        output_root=Path(str(config.get("output_root", "outputs"))),
        topk=int(config.get("topk", 10)),
        seed=int(config.get("seed", 42)),
        method_family=str(config.get("method_family", "trainable_lora_framework")),
        method_variant=str(config.get("method_variant", config["run_name"])),
        model_name=str(config.get("model_name", "qwen3_8b_local")),
        train_status_path=Path(str(summary_cfg.get("train_status_path", "outputs/summary/week7_5_train_status.csv"))),
        framework_compare_path=Path(
            str(summary_cfg.get("framework_compare_path", "outputs/summary/week7_5_framework_compare.csv"))
        ),
        training_cfg=training_cfg,
        evaluation_cfg=evaluation_cfg,
        support_signals=config.get("support_signals", {}) or {},
        model_cfg=model_cfg,
        startup_check_path=Path(str(summary_cfg.get("startup_check_path", "outputs/summary/week7_5_startup_check.json"))),
        dataset_preview_path=Path(str(summary_cfg.get("dataset_preview_path", "outputs/summary/week7_5_dataset_preview.csv"))),
        training_summary_path=Path(str(summary_cfg.get("training_summary_path", "artifacts/logs/qwen3_rank_beauty_framework_v1/training_summary.csv"))),
        framework_manifest_path=Path(
            str(summary_cfg.get("framework_manifest_path", "outputs/beauty_qwen3_rank_framework_v1/framework_run_manifest.json"))
        ),
        compare_markdown_path=Path(
            str(summary_cfg.get("framework_compare_markdown_path", "outputs/summary/week7_5_framework_compare.md"))
        ),
    )


def _ensure_run_dirs(ctx: TrainingRunContext) -> None:
    for path in [
        ctx.adapter_output_dir,
        ctx.framework_output_dir,
        ctx.framework_output_dir / "predictions",
        ctx.framework_output_dir / "tables",
        ctx.logs_dir,
        ctx.train_status_path.parent,
        ctx.framework_compare_path.parent,
        ctx.startup_check_path.parent,
        ctx.dataset_preview_path.parent,
        ctx.framework_manifest_path.parent,
        ctx.compare_markdown_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _examples_to_records(examples: list[RankSupervisedExample]) -> list[dict[str, Any]]:
    return [
        {
            "source_event_id": example.source_event_id,
            "user_id": example.user_id,
            "prompt": example.prompt,
            "target_text": example.target_text,
            "positive_item_id": example.positive_item_id,
            "candidate_item_ids": example.candidate_item_ids,
        }
        for example in examples
    ]


def _write_example_preview(examples: list[RankSupervisedExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in _examples_to_records(examples):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_training_dataset(
    examples: list[RankSupervisedExample],
    *,
    tokenizer,
    max_seq_length: int,
):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("datasets is required for LoRA ranking training.") from exc

    eos_token = tokenizer.eos_token or ""
    records = []
    for example in examples:
        prompt = example.prompt.strip()
        response = example.target_text.strip()
        text = f"{prompt}\n{response}{eos_token}"
        prompt_only = f"{prompt}\n"
        prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )["input_ids"]
        labels = full_ids[:]
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        records.append({"input_ids": full_ids, "labels": labels})

    dataset = Dataset.from_list(records)

    def _with_attention_mask(batch):
        batch["attention_mask"] = [[1] * len(ids) for ids in batch["input_ids"]]
        return batch

    dataset = dataset.map(_with_attention_mask, batched=True)
    # Keep plain Python lists here. Some server images have a torchvision build
    # that breaks Hugging Face Datasets' torch formatter, while the Transformers
    # data collator can tensorize these text-only records directly.
    return dataset


def _build_training_arguments(training_arguments_cls, ctx: TrainingRunContext):
    eval_strategy = str(ctx.training_cfg.get("eval_strategy", "epoch"))
    requested_args = {
        "output_dir": str(ctx.logs_dir / "trainer_outputs"),
        "overwrite_output_dir": True,
        "num_train_epochs": float(ctx.training_cfg.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(ctx.training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(ctx.training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(ctx.training_cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(ctx.training_cfg.get("learning_rate", 2e-4)),
        "warmup_ratio": float(ctx.training_cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(ctx.training_cfg.get("logging_steps", 10)),
        "save_strategy": str(ctx.training_cfg.get("save_strategy", "no")),
        "evaluation_strategy": eval_strategy,
        "eval_strategy": eval_strategy,
        "bf16": bool(ctx.training_cfg.get("bf16", True)),
        "fp16": bool(ctx.training_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(ctx.training_cfg.get("gradient_checkpointing", True)),
        "report_to": [],
        "seed": ctx.seed,
    }
    supported_args = set(inspect.signature(training_arguments_cls.__init__).parameters)
    if "eval_strategy" in supported_args and "evaluation_strategy" not in supported_args:
        requested_args.pop("evaluation_strategy", None)
    elif "evaluation_strategy" in supported_args and "eval_strategy" not in supported_args:
        requested_args.pop("eval_strategy", None)
    return training_arguments_cls(
        **{key: value for key, value in requested_args.items() if key in supported_args}
    )


def _write_adapter_manifest(ctx: TrainingRunContext, *, train_count: int, valid_count: int) -> None:
    manifest = {
        "run_name": ctx.run_name,
        "domain": ctx.domain,
        "task": ctx.task,
        "method_family": ctx.method_family,
        "method_variant": ctx.method_variant,
        "base_model_config": str(ctx.base_model_config),
        "adapter_output_dir": str(ctx.adapter_output_dir),
        "framework_output_dir": str(ctx.framework_output_dir),
        "train_count": train_count,
        "valid_count": valid_count,
        "topk": ctx.topk,
        "seed": ctx.seed,
        "training": ctx.training_cfg,
        "evaluation": ctx.evaluation_cfg,
        "support_signals": ctx.support_signals,
        "created_at": utc_now_iso(),
    }
    (ctx.adapter_output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_dataset_preview(
    *,
    ctx: TrainingRunContext,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
) -> None:
    preview_rows = [
        {"split": "train", **summarize_rank_samples(train_rows)},
        {"split": "valid", **summarize_rank_samples(valid_rows)},
    ]
    header = list(preview_rows[0].keys())
    with ctx.dataset_preview_path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in preview_rows:
            writer.writerow(row)


def _run_startup_check(
    *,
    ctx: TrainingRunContext,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    train_examples: list[RankSupervisedExample],
    valid_examples: list[RankSupervisedExample],
) -> dict[str, Any]:
    model_name_or_path = str(
        ctx.model_cfg.get("model_name_or_path")
        or ctx.model_cfg.get("tokenizer_name_or_path")
        or ""
    ).strip()
    tokenizer_name_or_path = str(
        ctx.model_cfg.get("tokenizer_name_or_path")
        or ctx.model_cfg.get("model_name_or_path")
        or ""
    ).strip()
    target_modules = list(
        ctx.training_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    )
    check = {
        "run_name": ctx.run_name,
        "domain": ctx.domain,
        "task": ctx.task,
        "train_input_exists": ctx.train_input_path.exists(),
        "valid_input_exists": ctx.valid_input_path.exists(),
        "eval_input_exists": ctx.eval_input_path.exists(),
        "prompt_path_exists": ctx.prompt_path.exists(),
        "base_model_config_exists": ctx.base_model_config.exists(),
        "model_name_or_path": model_name_or_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "uses_local_model_path": bool(model_name_or_path.startswith("/") or ":" in model_name_or_path),
        "target_module_count": len(target_modules),
        "adapter_output_dir": str(ctx.adapter_output_dir),
        "framework_output_dir": str(ctx.framework_output_dir),
        "logs_dir": str(ctx.logs_dir),
        "train_sample_count": len(train_rows),
        "valid_sample_count": len(valid_rows),
        "train_example_count": len(train_examples),
        "valid_example_count": len(valid_examples),
        "avg_train_prompt_chars": round(
            sum(len(example.prompt) for example in train_examples) / max(len(train_examples), 1),
            2,
        ),
        "avg_valid_prompt_chars": round(
            sum(len(example.prompt) for example in valid_examples) / max(len(valid_examples), 1),
            2,
        ),
        "dry_run": bool(ctx.training_cfg.get("dry_run", False)),
        "created_at": utc_now_iso(),
    }
    ctx.startup_check_path.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    return check


def _run_actual_training(
    *,
    ctx: TrainingRunContext,
    train_examples: list[RankSupervisedExample],
    valid_examples: list[RankSupervisedExample],
) -> None:
    try:
        from peft import LoraConfig, get_peft_model
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "LoRA ranking training requires transformers and peft in the current environment."
        ) from exc

    model_name_or_path = str(
        ctx.model_cfg.get("model_name_or_path")
        or ctx.model_cfg.get("tokenizer_name_or_path")
        or ""
    ).strip()
    tokenizer_name_or_path = str(
        ctx.model_cfg.get("tokenizer_name_or_path")
        or ctx.model_cfg.get("model_name_or_path")
        or ""
    ).strip()
    if not model_name_or_path:
        raise ValueError("model.model_name_or_path is required in the LoRA framework config.")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=bool(ctx.model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(ctx.model_cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = str(ctx.model_cfg.get("dtype", "bfloat16")).strip().lower()
    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=ctx.model_cfg.get("device_map", "auto"),
        trust_remote_code=bool(ctx.model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(ctx.model_cfg.get("local_files_only", True)),
    )

    lora_cfg = LoraConfig(
        r=int(ctx.training_cfg.get("lora_r", 16)),
        lora_alpha=int(ctx.training_cfg.get("lora_alpha", 32)),
        lora_dropout=float(ctx.training_cfg.get("lora_dropout", 0.05)),
        bias=str(ctx.training_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(
            ctx.training_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    model = get_peft_model(model, lora_cfg)

    max_seq_length = int(ctx.training_cfg.get("max_seq_length", 1024))
    train_dataset = _build_training_dataset(train_examples, tokenizer=tokenizer, max_seq_length=max_seq_length)
    valid_dataset = _build_training_dataset(valid_examples, tokenizer=tokenizer, max_seq_length=max_seq_length)

    training_args = _build_training_arguments(TrainingArguments, ctx)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    )
    trainer.train()
    trainer.save_model(str(ctx.adapter_output_dir))
    tokenizer.save_pretrained(str(ctx.adapter_output_dir))


def run_lora_rank_training(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    startup_check: bool = False,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    ctx = build_training_run_context(config)
    _ensure_run_dirs(ctx)

    training_cfg = ctx.training_cfg
    resolved_dry_run = bool(dry_run or training_cfg.get("dry_run", False))
    resolved_train_samples = max_train_samples or training_cfg.get("max_train_samples")
    resolved_valid_samples = max_valid_samples or training_cfg.get("max_valid_samples")
    include_reason = bool(training_cfg.get("include_reason", False))

    train_rows = load_rank_samples(ctx.train_input_path, max_samples=resolved_train_samples)
    valid_rows = load_rank_samples(ctx.valid_input_path, max_samples=resolved_valid_samples)

    train_examples = build_rank_supervised_examples(
        train_rows,
        prompt_path=ctx.prompt_path,
        topk=ctx.topk,
        include_reason=include_reason,
    )
    valid_examples = build_rank_supervised_examples(
        valid_rows,
        prompt_path=ctx.prompt_path,
        topk=ctx.topk,
        include_reason=include_reason,
    )

    _write_example_preview(train_examples, ctx.logs_dir / "train_supervised_examples.jsonl")
    _write_example_preview(valid_examples, ctx.logs_dir / "valid_supervised_examples.jsonl")
    _write_adapter_manifest(ctx, train_count=len(train_examples), valid_count=len(valid_examples))
    _write_dataset_preview(ctx=ctx, train_rows=train_rows, valid_rows=valid_rows)
    startup_report = _run_startup_check(
        ctx=ctx,
        train_rows=train_rows,
        valid_rows=valid_rows,
        train_examples=train_examples,
        valid_examples=valid_examples,
    )

    if startup_check:
        update_framework_manifest(
            path=ctx.framework_manifest_path,
            run_name=ctx.run_name,
            domain=ctx.domain,
            model=ctx.model_name,
            method_family=ctx.method_family,
            method_variant=ctx.method_variant,
            adapter_output_dir=str(ctx.adapter_output_dir),
            framework_output_dir=str(ctx.framework_output_dir),
            compare_csv_path=str(ctx.framework_compare_path),
            compare_markdown_path=str(ctx.compare_markdown_path),
            training_summary_path=str(ctx.training_summary_path),
            startup_check_path=str(ctx.startup_check_path),
            dataset_preview_path=str(ctx.dataset_preview_path),
            latest_stage="startup_check",
            latest_status="startup_ready",
            extra_fields={
                "train_samples": len(train_examples),
                "valid_samples": len(valid_examples),
                "startup_report": startup_report,
                "adapter_ready": False,
                "framework_metrics_ready": False,
            },
        )
        append_training_status(
            {
                "run_name": ctx.run_name,
                "domain": ctx.domain,
                "task": ctx.task,
                "method_family": ctx.method_family,
                "method_variant": ctx.method_variant,
                "model": ctx.model_name,
                "stage": "startup_check",
                "status": "startup_ready",
                "dry_run": True,
                "startup_check_only": True,
                "train_samples": len(train_examples),
                "valid_samples": len(valid_examples),
                "adapter_output_dir": str(ctx.adapter_output_dir),
                "framework_output_dir": str(ctx.framework_output_dir),
                "logs_dir": str(ctx.logs_dir),
                "startup_check_path": str(ctx.startup_check_path),
                "dataset_preview_path": str(ctx.dataset_preview_path),
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "notes": "Startup-only validation for the Week7.5 ranking LoRA framework.",
            },
            ctx.train_status_path,
        )
        return {
            "run_name": ctx.run_name,
            "adapter_output_dir": str(ctx.adapter_output_dir),
            "framework_output_dir": str(ctx.framework_output_dir),
            "logs_dir": str(ctx.logs_dir),
            "startup_check_path": str(ctx.startup_check_path),
            "dataset_preview_path": str(ctx.dataset_preview_path),
            "startup_report": startup_report,
        }

    if not resolved_dry_run:
        _run_actual_training(ctx=ctx, train_examples=train_examples, valid_examples=valid_examples)

    summary = {
        "run_name": ctx.run_name,
        "domain": ctx.domain,
        "task": ctx.task,
        "method_family": ctx.method_family,
        "method_variant": ctx.method_variant,
        "model": ctx.model_name,
        "base_model_config": str(ctx.base_model_config),
        "train_input_path": str(ctx.train_input_path),
        "valid_input_path": str(ctx.valid_input_path),
        "eval_input_path": str(ctx.eval_input_path),
        "adapter_output_dir": str(ctx.adapter_output_dir),
        "framework_output_dir": str(ctx.framework_output_dir),
        "logs_dir": str(ctx.logs_dir),
        "topk": ctx.topk,
        "seed": ctx.seed,
        "train_samples": len(train_examples),
        "valid_samples": len(valid_examples),
        "dry_run": resolved_dry_run,
        "run_inference_after_train": bool(ctx.evaluation_cfg.get("run_inference_after_train", True)),
        "strongest_handcrafted_baseline": str(ctx.support_signals.get("structured_risk_exp_name", "")),
        "direct_ranking_baseline": str(ctx.support_signals.get("direct_ranking_exp_name", "")),
        "pointwise_signal_path": str(ctx.support_signals.get("pointwise_uncertainty_path", "")),
        "created_at": utc_now_iso(),
    }
    save_training_summary(summary, ctx.training_summary_path)
    update_framework_manifest(
        path=ctx.framework_manifest_path,
        run_name=ctx.run_name,
        domain=ctx.domain,
        model=ctx.model_name,
        method_family=ctx.method_family,
        method_variant=ctx.method_variant,
        adapter_output_dir=str(ctx.adapter_output_dir),
        framework_output_dir=str(ctx.framework_output_dir),
        compare_csv_path=str(ctx.framework_compare_path),
        compare_markdown_path=str(ctx.compare_markdown_path),
        training_summary_path=str(ctx.training_summary_path),
        startup_check_path=str(ctx.startup_check_path),
        dataset_preview_path=str(ctx.dataset_preview_path),
        latest_stage="lora_train_rank",
        latest_status="dry_run_ready" if resolved_dry_run else "artifact_ready",
        extra_fields={
            "train_samples": len(train_examples),
            "valid_samples": len(valid_examples),
            "startup_report": startup_report,
            "adapter_ready": not resolved_dry_run,
            "framework_metrics_ready": False,
        },
    )

    append_training_status(
        {
            "run_name": ctx.run_name,
            "domain": ctx.domain,
            "task": ctx.task,
            "method_family": ctx.method_family,
            "method_variant": ctx.method_variant,
            "model": ctx.model_name,
            "stage": "lora_train_rank",
            "status": "dry_run_ready" if resolved_dry_run else "artifact_ready",
            "dry_run": resolved_dry_run,
            "startup_check_only": False,
            "train_samples": len(train_examples),
            "valid_samples": len(valid_examples),
            "adapter_output_dir": str(ctx.adapter_output_dir),
            "framework_output_dir": str(ctx.framework_output_dir),
            "logs_dir": str(ctx.logs_dir),
            "startup_check_path": str(ctx.startup_check_path),
            "dataset_preview_path": str(ctx.dataset_preview_path),
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "notes": "Week7.5 Day1 ranking-only LoRA framework skeleton. Pointwise and pairwise remain supporting layers.",
        },
        ctx.train_status_path,
    )

    return summary
