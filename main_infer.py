# main_infer.py

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

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

from src.llm.inference import run_pointwise_inference


def load_jsonl(path: str | Path) -> list[dict]:
    df = pd.read_json(path, lines=True)
    return df.to_dict(orient="records")


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(path, orient="records", lines=True, force_ascii=False)


def get_prompt_builder():
    """
    兼容类式 / 函数式 prompt builder
    """
    if PromptBuilderClass is not None:
        try:
            return PromptBuilderClass()
        except Exception:
            # 有些项目需要模板路径
            try:
                return PromptBuilderClass(template_path="prompts/pointwise_yesno.txt")
            except Exception:
                raise

    return build_prompt_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to input jsonl file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/predictions/test_raw.jsonl",
        help="Path to output prediction jsonl file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists at {output_path}. Skip inference.")
        return

    samples = load_jsonl(input_path)
    print(f"Loaded {len(samples)} samples from {input_path}")

    llm_backend = BackendClass()
    prompt_builder = get_prompt_builder()

    predictions = run_pointwise_inference(
        samples=samples,
        llm_backend=llm_backend,
        prompt_builder=prompt_builder,
    )

    save_jsonl(predictions, output_path)
    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()