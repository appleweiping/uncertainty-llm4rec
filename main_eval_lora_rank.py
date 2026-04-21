from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
)
from src.llm import build_backend_from_dict, load_model_config
from src.llm.inference import run_candidate_ranking_inference
from src.llm.prompt_builder import PromptBuilder
from src.training.framework_artifacts import append_stage_status, update_framework_manifest, utc_now_iso
from src.utils.exp_io import load_jsonl, load_yaml, save_jsonl
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def _save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_summary_dict(summary: dict[str, Any], path: str | Path) -> None:
    _save_table(pd.DataFrame([summary]), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="LoRA framework config path.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional manual adapter path override.")
    parser.add_argument("--input_path", type=str, default=None, help="Optional manual ranking input override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite predictions if they already exist.")
    parser.add_argument("--skip_inference", action="store_true", help="Reuse an existing framework prediction file.")
    parser.add_argument("--resume_partial", action="store_true", help="Resume from an existing partial ranking prediction file when possible.")
    parser.add_argument("--checkpoint_every_batches", type=int, default=1, help="Save partial ranking predictions every N batches during inference.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional eval sample override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for ranking evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    eval_cfg = config.get("evaluation", {}) or {}
    framework_exp_name = str(eval_cfg.get("framework_exp_name", config.get("run_name", "week7_5_framework")))
    output_root = str(config.get("output_root", "outputs"))
    paths = ensure_exp_dirs(framework_exp_name, output_root)
    set_global_seed(args.seed if args.seed is not None else config.get("seed"))
    summary_cfg = config.get("summary", {}) or {}
    train_status_path = Path(str(summary_cfg.get("train_status_path", "outputs/summary/week7_5_train_status.csv")))
    framework_manifest_path = Path(
        str(summary_cfg.get("framework_manifest_path", f"{paths.root}/framework_run_manifest.json"))
    )
    compare_csv_path = str(summary_cfg.get("framework_compare_path", "outputs/summary/week7_5_framework_compare.csv"))
    compare_markdown_path = str(
        summary_cfg.get("framework_compare_markdown_path", "outputs/summary/week7_5_framework_compare.md")
    )
    training_summary_path = str(
        summary_cfg.get("training_summary_path", "artifacts/logs/qwen3_rank_beauty_framework_v1/training_summary.csv")
    )
    startup_check_path = str(summary_cfg.get("startup_check_path", "outputs/summary/week7_5_startup_check.json"))
    dataset_preview_path = str(summary_cfg.get("dataset_preview_path", "outputs/summary/week7_5_dataset_preview.csv"))

    input_path = Path(args.input_path or config.get("eval_input_path"))
    prediction_path = paths.predictions_dir / "rank_predictions.jsonl"
    if prediction_path.exists() and not args.overwrite and not args.skip_inference and not args.resume_partial:
        raise FileExistsError(
            f"Framework ranking predictions already exist: {prediction_path}. "
            "Use --overwrite to regenerate, --resume_partial to continue, or --skip_inference to evaluate the existing file."
        )

    if not args.skip_inference:
        base_model_cfg = load_model_config(config["base_model_config"])
        runtime_cfg = base_model_cfg.get("runtime", {}) or {}
        runtime_cfg["adapter_path"] = args.adapter_path or config.get("adapter_output_dir")
        runtime_cfg["batch_size"] = eval_cfg.get("batch_size", runtime_cfg.get("batch_size", 1))
        base_model_cfg["runtime"] = runtime_cfg
        generation_cfg = base_model_cfg.get("generation", {}) or {}
        generation_cfg["max_new_tokens"] = eval_cfg.get("max_new_tokens", generation_cfg.get("max_new_tokens", 300))
        base_model_cfg["generation"] = generation_cfg

        llm_backend = build_backend_from_dict(base_model_cfg)
        prompt_builder = PromptBuilder(template_path=str(config["prompt_path"]))
        samples = load_jsonl(input_path, max_samples=args.max_samples or eval_cfg.get("max_eval_samples"))
        existing_predictions: list[dict[str, Any]] = []
        if prediction_path.exists() and args.resume_partial:
            existing_predictions = load_jsonl(prediction_path)
            resume_count = min(len(existing_predictions), len(samples))
            if resume_count:
                print(f"[{framework_exp_name}] Resuming from existing partial output: {resume_count} rows already saved.")
                samples = samples[resume_count:]
            if not samples:
                print(f"[{framework_exp_name}] Partial output already covers all requested samples.")
                args.skip_inference = True
        if not args.skip_inference:
            results = run_candidate_ranking_inference(
                samples=samples,
                llm_backend=llm_backend,
                prompt_builder=prompt_builder,
                topk=int(config.get("topk", 10)),
                max_new_tokens=eval_cfg.get("max_new_tokens", generation_cfg.get("max_new_tokens", 300)),
                checkpoint_path=prediction_path,
                checkpoint_every_batches=args.checkpoint_every_batches,
                existing_records=existing_predictions,
            )
            save_jsonl(results, prediction_path)

    print(f"[{framework_exp_name}] Loading ranking predictions from: {prediction_path}")
    raw_records = load_jsonl(prediction_path)
    raw_df = pd.DataFrame(raw_records)
    eval_df = build_ranking_eval_frame(raw_df)

    print(f"[{framework_exp_name}] Loaded {len(eval_df)} ranking samples.")
    metrics = compute_ranking_task_metrics(eval_df, k=args.k)
    exposure_df = compute_ranking_exposure_distribution(eval_df, k=args.k)

    _save_summary_dict(metrics, paths.tables_dir / "ranking_metrics.csv")
    _save_table(exposure_df, paths.tables_dir / "ranking_exposure_distribution.csv")
    _save_table(eval_df, paths.tables_dir / "ranking_eval_records.csv")
    eval_summary = {
        "run_name": framework_exp_name,
        "domain": str(config.get("domain", "beauty")),
        "task": "candidate_ranking",
        "method_family": str(config.get("method_family", "trainable_lora_framework")),
        "method_variant": str(config.get("method_variant", framework_exp_name)),
        "model": str(config.get("model_name", "qwen3_8b_local")),
        "adapter_path": str(args.adapter_path or config.get("adapter_output_dir")),
        "prediction_path": str(prediction_path),
        "metrics_path": str(paths.tables_dir / "ranking_metrics.csv"),
        "sample_count": len(eval_df),
        "HR@10": metrics.get("HR@10"),
        "NDCG@10": metrics.get("NDCG@10"),
        "MRR": metrics.get("MRR"),
        "parse_success_rate": metrics.get("parse_success_rate"),
        "coverage@10": metrics.get("coverage@10"),
        "head_exposure_ratio@10": metrics.get("head_exposure_ratio@10"),
        "longtail_coverage@10": metrics.get("longtail_coverage@10"),
        "created_at": utc_now_iso(),
    }
    _save_summary_dict(eval_summary, paths.tables_dir / "framework_eval_summary.csv")
    append_stage_status(
        {
            "run_name": framework_exp_name,
            "domain": str(config.get("domain", "beauty")),
            "task": "candidate_ranking",
            "method_family": str(config.get("method_family", "trainable_lora_framework")),
            "method_variant": str(config.get("method_variant", framework_exp_name)),
            "model": str(config.get("model_name", "qwen3_8b_local")),
            "stage": "framework_eval",
            "status": "artifact_ready",
            "dry_run": False,
            "startup_check_only": False,
            "adapter_output_dir": str(args.adapter_path or config.get("adapter_output_dir")),
            "framework_output_dir": str(paths.root),
            "prediction_path": str(prediction_path),
            "metrics_path": str(paths.tables_dir / "ranking_metrics.csv"),
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "notes": "Framework ranking evaluation completed and aligned to the standard ranking metrics schema.",
        },
        train_status_path,
    )
    update_framework_manifest(
        path=framework_manifest_path,
        run_name=framework_exp_name,
        domain=str(config.get("domain", "beauty")),
        model=str(config.get("model_name", "qwen3_8b_local")),
        method_family=str(config.get("method_family", "trainable_lora_framework")),
        method_variant=str(config.get("method_variant", framework_exp_name)),
        adapter_output_dir=str(args.adapter_path or config.get("adapter_output_dir")),
        framework_output_dir=str(paths.root),
        compare_csv_path=compare_csv_path,
        compare_markdown_path=compare_markdown_path,
        training_summary_path=training_summary_path,
        startup_check_path=startup_check_path,
        dataset_preview_path=dataset_preview_path,
        latest_stage="framework_eval",
        latest_status="artifact_ready",
        extra_fields={
            "framework_prediction_path": str(prediction_path),
            "framework_metrics_path": str(paths.tables_dir / "ranking_metrics.csv"),
            "framework_eval_summary_path": str(paths.tables_dir / "framework_eval_summary.csv"),
            "framework_metrics_ready": True,
        },
    )

    print(f"[{framework_exp_name}] LoRA ranking evaluation done.")
    print(f"[{framework_exp_name}] Predictions saved to: {prediction_path}")
    print(f"[{framework_exp_name}] Tables saved to: {paths.tables_dir}")


if __name__ == "__main__":
    main()
