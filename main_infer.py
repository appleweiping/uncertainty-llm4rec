from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.llm import build_backend_from_config
from src.llm.inference import run_pointwise_inference
from src.utils.exp_io import get_prompt_builder, load_jsonl, load_yaml, save_jsonl
from src.utils.paths import default_input_path_for_exp, ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def infer_prediction_filename(input_path: str | Path) -> str:
    stem = Path(input_path).stem.lower()
    if stem.startswith("valid"):
        return "valid_raw.jsonl"
    if stem.startswith("test"):
        return "test_raw.jsonl"
    if stem.startswith("train"):
        return "train_raw.jsonl"
    return f"{stem}_raw.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Experiment config path.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Input pointwise jsonl path.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional default data directory.")
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--prompt_path", type=str, default="prompts/pointwise_yesno.txt", help="Prompt template path.")
    parser.add_argument("--model_config", type=str, default=None, help="Model config path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--output_path", type=str, default=None, help="Prediction output jsonl path.")
    parser.add_argument("--split_name", type=str, default=None, help="Optional split name, e.g. train/valid/test.")
    parser.add_argument("--resume_partial", action="store_true", help="Resume from an existing partial pointwise prediction file when possible.")
    parser.add_argument("--checkpoint_every_batches", type=int, default=1, help="Save partial pointwise predictions every N batches during inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    merged = {
        "exp_name": args.exp_name if args.exp_name is not None else cfg.get("exp_name", "clean"),
        "input_path": args.input_path if args.input_path is not None else cfg.get("input_path"),
        "data_root": args.data_root if args.data_root is not None else cfg.get("data_root"),
        "output_root": args.output_root if args.output_root is not None else cfg.get("output_root", "outputs"),
        "prompt_path": args.prompt_path if args.prompt_path is not None else cfg.get("prompt_path", "prompts/pointwise_yesno.txt"),
        "model_config": args.model_config if args.model_config is not None else cfg.get("model_config"),
        "max_samples": args.max_samples if args.max_samples is not None else cfg.get("max_samples"),
        "output_path": args.output_path if args.output_path is not None else cfg.get("output_path"),
        "split_name": args.split_name if args.split_name is not None else cfg.get("split_name"),
        "resume_partial": bool(args.resume_partial or cfg.get("resume_partial", False)),
        "checkpoint_every_batches": args.checkpoint_every_batches if args.checkpoint_every_batches is not None else cfg.get("checkpoint_every_batches", 1),
        "overwrite": bool(args.overwrite or cfg.get("overwrite", False)),
        "seed": args.seed if args.seed is not None else cfg.get("seed"),
    }
    return merged


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    exp_name = str(cfg["exp_name"])
    output_root = cfg["output_root"]
    input_path = cfg["input_path"]
    data_root = cfg["data_root"]
    prompt_path = cfg["prompt_path"]
    model_config = cfg["model_config"]
    max_samples = cfg["max_samples"]
    output_path = cfg["output_path"]
    split_name = cfg["split_name"]
    resume_partial = bool(cfg["resume_partial"])
    checkpoint_every_batches = int(cfg["checkpoint_every_batches"]) if cfg["checkpoint_every_batches"] is not None else 1
    overwrite = cfg["overwrite"]
    seed = cfg["seed"]

    set_global_seed(seed)

    if model_config is None:
        raise ValueError("model_config must be provided via config or CLI.")

    paths = ensure_exp_dirs(exp_name, output_root)

    if input_path is None:
        if data_root is None:
            raise ValueError("input_path or data_root must be provided via config or CLI.")
        input_path = default_input_path_for_exp(exp_name, data_root)

    input_path = Path(input_path)
    if output_path is not None:
        output_path = Path(output_path)
    elif split_name is not None:
        output_path = paths.predictions_dir / f"{str(split_name).strip().lower()}_raw.jsonl"
    else:
        output_path = paths.predictions_dir / infer_prediction_filename(input_path)

    print(f"[{exp_name}] Input path: {input_path}")
    print(f"[{exp_name}] Output path: {output_path}")
    print(f"[{exp_name}] Model config: {model_config}")
    if seed is not None:
        print(f"[{exp_name}] Seed: {seed}")

    if not input_path.exists():
        raise FileNotFoundError(f"[{exp_name}] Input file not found: {input_path}")

    if output_path.exists() and not overwrite and not resume_partial:
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config, --overwrite, or --resume_partial to continue.")
        return

    samples = load_jsonl(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples.")
    existing_predictions: list[dict[str, Any]] = []
    if output_path.exists() and resume_partial:
        existing_predictions = load_jsonl(output_path)
        resume_count = min(len(existing_predictions), len(samples))
        if resume_count:
            print(f"[{exp_name}] Resuming from existing partial output: {resume_count} rows already saved.")
        if resume_count >= len(samples):
            print(f"[{exp_name}] Partial output already covers all requested samples.")
            return

    llm_backend = build_backend_from_config(model_config)
    prompt_builder = get_prompt_builder(prompt_path)

    predictions = run_pointwise_inference(
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
