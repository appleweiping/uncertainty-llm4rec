from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.llm import build_backend_from_config
from src.llm.prompt_builder import PromptBuilder
from src.llm.self_consistency import run_self_consistency
from src.uncertainty.consistency_confidence import compute_consistency_summary
from src.utils.io import save_jsonl
from src.utils.paths import ensure_exp_dirs


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl_records(path: str | Path, max_samples: int | None = None) -> list[dict]:
    df = pd.read_json(path, lines=True)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


def _normalize_candidate_from_pointwise_sample(sample: dict[str, Any]) -> dict[str, Any]:
    candidate_item_id = (
        sample.get("candidate_item_id")
        or sample.get("item_id")
        or sample.get("target_item_id")
        or ""
    )

    candidate_title = sample.get("candidate_title") or sample.get("title") or ""
    candidate_meta = (
        sample.get("candidate_meta")
        or sample.get("candidate_description")
        or sample.get("candidate_text")
        or sample.get("text")
        or ""
    )

    return {
        "item_id": candidate_item_id,
        "title": str(candidate_title).strip(),
        "meta": str(candidate_meta).strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    parser.add_argument("--input_path", type=str, default=None, help="Optional explicit input jsonl path.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional explicit output path.")
    parser.add_argument("--split_name", type=str, default="test", help="Split name, e.g. test.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of repeated generations per sample.")
    parser.add_argument("--sleep_time", type=float, default=0.0, help="Optional sleep between repeated generations.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_name = str(cfg.get("exp_name", "")).strip()
    if not exp_name:
        raise ValueError("exp_name is required in experiment config.")

    model_config = cfg.get("model_config")
    prompt_path = cfg.get("prompt_path", "prompts/pointwise_yesno.txt")
    output_root = cfg.get("output_root", "outputs")
    input_path = Path(args.input_path or cfg.get("input_path"))
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples")
    overwrite = bool(args.overwrite or cfg.get("overwrite", False))

    paths = ensure_exp_dirs(exp_name, output_root)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else paths.root / "self_consistency" / f"{str(args.split_name).strip().lower()}_self_consistency.jsonl"
    )

    if output_path.exists() and not overwrite:
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config or --overwrite to rerun.")
        return

    samples = load_jsonl_records(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples for self-consistency from: {input_path}")

    llm_backend = build_backend_from_config(model_config)
    prompt_builder = PromptBuilder(template_path=str(prompt_path))

    rows: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running self-consistency"):
        candidate = _normalize_candidate_from_pointwise_sample(sample)
        prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
        runs = run_self_consistency(
            llm_backend=llm_backend,
            prompt=prompt,
            num_samples=args.num_samples,
            sleep_time=args.sleep_time,
        )
        summary = compute_consistency_summary(runs)

        row = {
            "user_id": sample.get("user_id", ""),
            "target_item_id": sample.get("target_item_id", candidate.get("item_id", "")),
            "candidate_item_id": candidate.get("item_id", ""),
            "label": int(sample.get("label", 0)),
            "target_popularity_group": sample.get("target_popularity_group", "unknown"),
            "prompt": prompt,
            "consistency_runs": runs,
        }
        row.update(summary)
        rows.append(row)

    save_jsonl(rows, output_path)
    print(f"[{exp_name}] Saved {len(rows)} self-consistency rows to: {output_path}")


if __name__ == "__main__":
    main()
