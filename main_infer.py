from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.llm import build_backend_from_config
from src.llm.inference import run_pointwise_inference, run_pointwise_inference_concurrent
from src.utils.paths import default_input_path_for_exp, ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict]:
    df = pd.read_json(path, lines=True)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(path, orient="records", lines=True, force_ascii=False)

class FunctionPromptBuilderAdapter:
    def __init__(self, fn, template_path: str | Path):
        self.fn = fn
        self.template_path = str(template_path)

    def build_pointwise_prompt(self, sample: dict, candidate: dict) -> str:
        try:
            return self.fn(sample, candidate, template_path=self.template_path)
        except TypeError:
            return self.fn(sample, candidate)


def get_prompt_builder(prompt_path: str | Path):
    try:
        from src.llm.prompt_builder import PromptBuilder

        return PromptBuilder(template_path=str(prompt_path))
    except Exception:
        try:
            from src.llm.prompt_builder import build_pointwise_prompt

            return FunctionPromptBuilderAdapter(build_pointwise_prompt, template_path=prompt_path)
        except Exception as e:
            raise ImportError("Cannot find a usable prompt builder in src/llm/prompt_builder.py") from e


def infer_prediction_filename(input_path: str | Path) -> str:
    stem = Path(input_path).stem.lower()
    if stem.startswith("valid"):
        return "valid_raw.jsonl"
    if stem.startswith("test"):
        return "test_raw.jsonl"
    if stem.startswith("train"):
        return "train_raw.jsonl"
    return f"{stem}_raw.jsonl"


def resolve_split_input_path(cfg: dict[str, Any], split_name: str | None) -> str | None:
    if not split_name:
        return None
    split_input_paths = cfg.get("split_input_paths") or {}
    if not isinstance(split_input_paths, dict):
        return None
    return split_input_paths.get(str(split_name).strip().lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Experiment config path.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Input pointwise jsonl path.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional default data directory.")
    parser.add_argument("--output_root", type=str, default=None, help="Output root directory.")
    parser.add_argument("--prompt_path", type=str, default=None, help="Prompt template path.")
    parser.add_argument("--model_config", type=str, default=None, help="Model config path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--output_path", type=str, default=None, help="Prediction output jsonl path.")
    parser.add_argument("--split_name", type=str, default=None, help="Optional split name, e.g. train/valid/test.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument("--output_schema", type=str, default=None, help="Optional parser/output schema.")
    parser.add_argument("--method_variant", type=str, default=None, help="Optional method variant tag.")
    parser.add_argument("--concurrent", action="store_true", help="Use concurrent API inference with checkpoint/resume.")
    parser.add_argument("--max_workers", type=int, default=None, help="Concurrent inference worker count.")
    parser.add_argument("--requests_per_minute", type=int, default=None, help="Optional global concurrent request rate limit.")
    parser.add_argument("--max_retries", type=int, default=None, help="Per-sample retry count for concurrent inference.")
    parser.add_argument("--retry_backoff_seconds", type=float, default=None, help="Initial retry backoff for concurrent inference.")
    parser.add_argument("--timeout_seconds", type=float, default=None, help="Optional per-request timeout for API backends.")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Checkpoint frequency for concurrent inference.")
    parser.add_argument("--resume", action="store_true", help="Resume concurrent inference from existing checkpoint output.")
    parser.add_argument("--no_resume", action="store_true", help="Disable concurrent inference resume even if config enables it.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    merged = {
        "exp_name": args.exp_name if args.exp_name is not None else cfg.get("exp_name", "clean"),
        "input_path": args.input_path
        if args.input_path is not None
        else resolve_split_input_path(cfg, args.split_name if args.split_name is not None else cfg.get("split_name"))
        or cfg.get("input_path"),
        "data_root": args.data_root if args.data_root is not None else cfg.get("data_root"),
        "output_root": args.output_root if args.output_root is not None else cfg.get("output_root", "outputs"),
        "prompt_path": args.prompt_path if args.prompt_path is not None else cfg.get("prompt_path", "prompts/pointwise_yesno.txt"),
        "model_config": args.model_config if args.model_config is not None else cfg.get("model_config"),
        "max_samples": args.max_samples if args.max_samples is not None else cfg.get("max_samples"),
        "output_path": args.output_path if args.output_path is not None else cfg.get("output_path"),
        "split_name": args.split_name if args.split_name is not None else cfg.get("split_name"),
        "overwrite": bool(args.overwrite or cfg.get("overwrite", False)),
        "seed": args.seed if args.seed is not None else cfg.get("seed"),
        "output_schema": args.output_schema if args.output_schema is not None else cfg.get("output_schema"),
        "method_variant": args.method_variant if args.method_variant is not None else cfg.get("method_variant"),
        "concurrent": bool(args.concurrent or cfg.get("concurrent", cfg.get("enable_concurrent", False))),
        "max_workers": args.max_workers if args.max_workers is not None else cfg.get("max_workers", 4),
        "requests_per_minute": args.requests_per_minute if args.requests_per_minute is not None else cfg.get("requests_per_minute"),
        "max_retries": args.max_retries if args.max_retries is not None else cfg.get("max_retries", 3),
        "retry_backoff_seconds": args.retry_backoff_seconds if args.retry_backoff_seconds is not None else cfg.get("retry_backoff_seconds", 2.0),
        "timeout_seconds": args.timeout_seconds if args.timeout_seconds is not None else cfg.get("timeout_seconds"),
        "checkpoint_every": args.checkpoint_every if args.checkpoint_every is not None else cfg.get("checkpoint_every", 1),
        "resume": bool((args.resume or cfg.get("resume", True)) and not args.no_resume),
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
    overwrite = cfg["overwrite"]
    seed = cfg["seed"]
    output_schema = cfg["output_schema"]
    method_variant = cfg["method_variant"]
    concurrent = cfg["concurrent"]
    max_workers = int(cfg["max_workers"])
    requests_per_minute = cfg["requests_per_minute"]
    max_retries = int(cfg["max_retries"])
    retry_backoff_seconds = float(cfg["retry_backoff_seconds"])
    timeout_seconds = cfg["timeout_seconds"]
    checkpoint_every = int(cfg["checkpoint_every"])
    resume = bool(cfg["resume"])

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
    if output_schema:
        print(f"[{exp_name}] Output schema: {output_schema}")
    if method_variant:
        print(f"[{exp_name}] Method variant: {method_variant}")
    if concurrent:
        print(
            f"[{exp_name}] Concurrent inference enabled: max_workers={max_workers}, "
            f"requests_per_minute={requests_per_minute or 'unlimited'}, resume={resume}"
        )
    if timeout_seconds is not None:
        print(f"[{exp_name}] Timeout seconds: {timeout_seconds}")
    if seed is not None:
        print(f"[{exp_name}] Seed: {seed}")

    if not input_path.exists():
        raise FileNotFoundError(f"[{exp_name}] Input file not found: {input_path}")

    if output_path.exists() and not overwrite and not (concurrent and resume):
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config or --overwrite to rerun.")
        return

    samples = load_jsonl(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples.")

    llm_backend = build_backend_from_config(model_config)
    prompt_builder = get_prompt_builder(prompt_path)

    if concurrent:
        predictions = run_pointwise_inference_concurrent(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
            output_path=output_path,
            output_schema=output_schema,
            method_variant=method_variant,
            split_name=split_name,
            max_workers=max_workers,
            requests_per_minute=int(requests_per_minute) if requests_per_minute else None,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
            checkpoint_every=checkpoint_every,
            resume=resume,
            overwrite=overwrite,
        )
    else:
        predictions = run_pointwise_inference(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
            output_schema=output_schema,
            method_variant=method_variant,
        )
        save_jsonl(predictions, output_path)
    print(f"[{exp_name}] Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
