from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.llm.inference import run_pointwise_inference
from src.utils.paths import default_input_path_for_exp, ensure_exp_dirs


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


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


def build_backend_from_config(model_cfg_path: str | Path):
    model_cfg = load_yaml(model_cfg_path)
    backend_name = str(model_cfg.get("backend_name", "")).strip().lower()
    if not backend_name:
        raise ValueError(f"backend_name is required in model config: {model_cfg_path}")

    generation_cfg = model_cfg.get("generation", {}) or {}
    connection_cfg = model_cfg.get("connection", {}) or {}

    if backend_name == "deepseek":
        from src.llm.deepseek_backend import DeepSeekBackend

        model_name = _first_present(model_cfg.get("model_name"), generation_cfg.get("model_name"))
        if not model_name:
            raise ValueError(f"model_name is required in model config: {model_cfg_path}")

        return DeepSeekBackend(
            model_name=str(model_name),
            temperature=float(_first_present(generation_cfg.get("temperature"), model_cfg.get("temperature"), 0.0)),
            max_tokens=int(_first_present(generation_cfg.get("max_tokens"), model_cfg.get("max_tokens"), 300)),
            base_url=str(_first_present(connection_cfg.get("base_url"), model_cfg.get("base_url"), "https://api.deepseek.com")),
            api_key_env=str(_first_present(connection_cfg.get("api_key_env"), model_cfg.get("api_key_env"), "DEEPSEEK_API_KEY")),
        )

    raise ValueError(f"Unsupported backend_name: {backend_name}")


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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction file.")
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
        "overwrite": bool(args.overwrite or cfg.get("overwrite", False)),
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

    if not input_path.exists():
        raise FileNotFoundError(f"[{exp_name}] Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config or --overwrite to rerun.")
        return

    samples = load_jsonl(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples.")

    llm_backend = build_backend_from_config(model_config)
    prompt_builder = get_prompt_builder(prompt_path)

    predictions = run_pointwise_inference(
        samples=samples,
        llm_backend=llm_backend,
        prompt_builder=prompt_builder,
    )

    save_jsonl(predictions, output_path)
    print(f"[{exp_name}] Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
