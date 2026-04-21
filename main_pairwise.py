from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.llm import build_backend_from_config
from src.llm.inference import run_pairwise_preference_inference
from src.utils.exp_io import get_prompt_builder, load_jsonl, load_yaml, save_jsonl
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Experiment config path.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Input pairwise jsonl path.")
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional explicit experiment output directory.")
    parser.add_argument("--prompt_path", type=str, default="prompts/pairwise_preference.txt", help="Prompt template path.")
    parser.add_argument("--model_config", type=str, default=None, help="Model config path.")
    parser.add_argument("--task_type", type=str, default="pairwise_preference", help="Task type, fixed to pairwise_preference.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--output_path", type=str, default=None, help="Prediction output jsonl path.")
    parser.add_argument("--resume_partial", action="store_true", help="Resume from an existing partial pairwise prediction file when possible.")
    parser.add_argument("--checkpoint_every_batches", type=int, default=1, help="Save partial pairwise predictions every N batches during inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    return {
        "exp_name": args.exp_name if args.exp_name is not None else cfg.get("exp_name", "pairwise"),
        "input_path": args.input_path if args.input_path is not None else cfg.get("input_path"),
        "output_root": args.output_root if args.output_root is not None else cfg.get("output_root", "outputs"),
        "output_dir": args.output_dir if args.output_dir is not None else cfg.get("output_dir"),
        "prompt_path": args.prompt_path if args.prompt_path is not None else cfg.get("prompt_path", "prompts/pairwise_preference.txt"),
        "model_config": args.model_config if args.model_config is not None else cfg.get("model_config"),
        "task_type": args.task_type if args.task_type is not None else cfg.get("task_type", "pairwise_preference"),
        "max_samples": args.max_samples if args.max_samples is not None else cfg.get("max_samples"),
        "output_path": args.output_path if args.output_path is not None else cfg.get("output_path"),
        "resume_partial": bool(args.resume_partial or cfg.get("resume_partial", False)),
        "checkpoint_every_batches": args.checkpoint_every_batches if args.checkpoint_every_batches is not None else cfg.get("checkpoint_every_batches", 1),
        "overwrite": bool(args.overwrite or cfg.get("overwrite", False)),
        "seed": args.seed if args.seed is not None else cfg.get("seed"),
    }


def resolve_predictions_output_dir(
    *,
    exp_name: str,
    output_root: str | Path,
    output_dir: str | Path | None,
) -> Path:
    if output_dir:
        predictions_dir = Path(output_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        return predictions_dir
    return ensure_exp_dirs(exp_name, output_root).predictions_dir


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    exp_name = str(cfg["exp_name"])
    input_path = cfg["input_path"]
    output_root = cfg["output_root"]
    output_dir = cfg["output_dir"]
    prompt_path = cfg["prompt_path"]
    model_config = cfg["model_config"]
    task_type = str(cfg["task_type"]).strip().lower()
    max_samples = cfg["max_samples"]
    output_path = cfg["output_path"]
    resume_partial = bool(cfg["resume_partial"])
    checkpoint_every_batches = int(cfg["checkpoint_every_batches"]) if cfg["checkpoint_every_batches"] is not None else 1
    overwrite = cfg["overwrite"]
    seed = cfg["seed"]

    if task_type != "pairwise_preference":
        raise ValueError(f"main_pairwise.py only supports task_type=pairwise_preference, got: {task_type}")

    set_global_seed(seed)

    if model_config is None:
        raise ValueError("model_config must be provided via config or CLI.")
    if input_path is None:
        raise ValueError("input_path must be provided via config or CLI.")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"[{exp_name}] Input file not found: {input_path}")

    predictions_dir = resolve_predictions_output_dir(
        exp_name=exp_name,
        output_root=output_root,
        output_dir=output_dir,
    )
    output_path = Path(output_path) if output_path is not None else predictions_dir / "pairwise_predictions.jsonl"

    print(f"[{exp_name}] Input path: {input_path}")
    print(f"[{exp_name}] Output path: {output_path}")
    print(f"[{exp_name}] Model config: {model_config}")
    if seed is not None:
        print(f"[{exp_name}] Seed: {seed}")

    if output_path.exists() and not overwrite and not resume_partial:
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config or --overwrite to rerun.")
        return

    samples = load_jsonl(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples.")
    existing_predictions: list[dict[str, Any]] = []
    if output_path.exists() and resume_partial:
        existing_predictions = load_jsonl(output_path)
        resume_count = min(len(existing_predictions), len(samples))
        if resume_count:
            print(f"[{exp_name}] Resuming from existing partial output: {resume_count} rows already saved.")
            samples = samples[resume_count:]
        if not samples:
            print(f"[{exp_name}] Partial output already covers all requested samples.")
            return

    llm_backend = build_backend_from_config(model_config)
    prompt_builder = get_prompt_builder(prompt_path)

    predictions = run_pairwise_preference_inference(
        samples=samples,
        llm_backend=llm_backend,
        prompt_builder=prompt_builder,
        checkpoint_path=output_path,
        checkpoint_every_batches=checkpoint_every_batches,
        existing_records=existing_predictions,
    )

    save_jsonl(predictions, output_path)
    print(f"[{exp_name}] Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
