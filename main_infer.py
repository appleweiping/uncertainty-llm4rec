# main_infer.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.llm.inference import run_pointwise_inference
from src.utils.paths import default_input_path_for_exp, ensure_exp_dirs

# ===== backend 兼容导入 =====
BackendClass = None

try:
    from src.llm.openai_backend import DeepSeekBackend as BackendClass
except Exception:
    try:
        from src.llm.deepseek_backend import DeepSeekBackend as BackendClass
    except Exception:
        try:
            from src.llm.openai_backend import OpenAIBackend as BackendClass
        except Exception:
            try:
                from src.llm.deepseek_backend import OpenAIBackend as BackendClass
            except Exception:
                BackendClass = None

if BackendClass is None:
    raise ImportError(
        "Cannot find a usable LLM backend under src/llm/. "
        "Please check your backend filename/class."
    )

# ===== prompt builder 兼容导入 =====
PromptBuilderClass = None
build_prompt_fn = None

try:
    from src.llm.prompt_builder import PromptBuilder as PromptBuilderClass
except Exception:
    PromptBuilderClass = None

if PromptBuilderClass is None:
    try:
        from src.llm.prompt_builder import build_pointwise_prompt as build_prompt_fn
    except Exception:
        try:
            from src.llm.prompt_builder import build_prompt as build_prompt_fn
        except Exception:
            build_prompt_fn = None

if PromptBuilderClass is None and build_prompt_fn is None:
    raise ImportError(
        "Cannot find a usable prompt builder under src/llm/prompt_builder.py. "
        "Please check whether you have PromptBuilder / build_pointwise_prompt / build_prompt."
    )


def load_jsonl(path: str | Path) -> list[dict]:
    df = pd.read_json(path, lines=True)
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
    if PromptBuilderClass is not None:
        try:
            return PromptBuilderClass(template_path=str(prompt_path))
        except TypeError:
            return PromptBuilderClass()

    return FunctionPromptBuilderAdapter(build_prompt_fn, template_path=prompt_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="clean",
        help="Experiment name, e.g. clean / noisy."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional explicit input jsonl path. If omitted, infer from exp_name."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed",
        help="Directory for processed datasets."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/pointwise_yesno.txt",
        help="Prompt template path."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists."
    )
    args = parser.parse_args()

    paths = ensure_exp_dirs(args.exp_name, args.output_root)

    input_path = (
        Path(args.input_path)
        if args.input_path is not None
        else default_input_path_for_exp(args.exp_name, args.data_root)
    )
    output_path = paths.predictions_dir / "test_raw.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists at {output_path}. Skip inference.")
        return

    samples = load_jsonl(input_path)
    print(f"Loaded {len(samples)} samples from {input_path}")

    llm_backend = BackendClass()
    prompt_builder = get_prompt_builder(args.prompt_path)

    predictions = run_pointwise_inference(
        samples=samples,
        llm_backend=llm_backend,
        prompt_builder=prompt_builder,
    )

    save_jsonl(predictions, output_path)
    print(f"[{args.exp_name}] Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()