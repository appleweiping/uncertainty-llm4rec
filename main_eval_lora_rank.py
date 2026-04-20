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

    input_path = Path(args.input_path or config.get("eval_input_path"))
    prediction_path = paths.predictions_dir / "rank_predictions.jsonl"
    if prediction_path.exists() and not args.overwrite and not args.skip_inference:
        raise FileExistsError(
            f"Framework ranking predictions already exist: {prediction_path}. "
            "Use --overwrite to regenerate or --skip_inference to evaluate the existing file."
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
        results = run_candidate_ranking_inference(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
            topk=int(config.get("topk", 10)),
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

    print(f"[{framework_exp_name}] LoRA ranking evaluation done.")
    print(f"[{framework_exp_name}] Predictions saved to: {prediction_path}")
    print(f"[{framework_exp_name}] Tables saved to: {paths.tables_dir}")


if __name__ == "__main__":
    main()
