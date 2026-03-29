# main_infer.py
from __future__ import annotations

from pathlib import Path

from src.data.dataset import load_samples
from src.llm.deepseek_backend import DeepSeekBackend
from src.llm.inference import run_pointwise_inference
from src.llm.prompt_builder import PromptBuilder
from src.utils.io import save_jsonl


def main() -> None:
    sample_path = "data/processed/test.jsonl"
    output_path = "outputs/predictions/test_raw.jsonl"
    prompt_path = "prompts/pointwise_yesno.txt"

    if Path(output_path).exists():
        print(f"Output already exists at {output_path}. Skip inference.")
        return

    samples = load_samples(sample_path)
    print(f"Loaded {len(samples)} samples from {sample_path}")

    llm = DeepSeekBackend(
        model_name="deepseek-chat",
        temperature=0.0,
        max_tokens=300,
        base_url="https://api.deepseek.com",
    )

    prompt_builder = PromptBuilder(prompt_path)

    predictions = run_pointwise_inference(
        samples=samples,
        llm_backend=llm,
        prompt_builder=prompt_builder,
    )

    save_jsonl(predictions, output_path)
    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()