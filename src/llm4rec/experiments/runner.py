"""Deterministic smoke experiment runner for Phase 1 skeleton validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_dataset
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.experiments.config import load_config, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_seed
from llm4rec.io.artifacts import create_run_dir, read_jsonl, write_environment, write_jsonl


def run_preprocess_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return preprocess_dataset(config)


def run_experiment(config_path: str | Path, *, preprocess: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    return run_experiment_config(config, preprocess=preprocess)


def run_experiment_config(config: dict[str, Any], *, preprocess: bool = False) -> dict[str, Any]:
    seed = set_seed(int(config.get("seed") or 0))
    method_name = _method_name(config)
    run_name = _run_name(config, method_name)
    run_id = f"{run_name}_seed{seed}"
    run_dir = create_run_dir(_output_dir(config), run_id)
    logger = RunLogger(run_dir / "logs.txt")
    logger.log(f"start run_id={run_id} method={method_name}")
    if preprocess:
        dataset_config = load_config(_required_path(config, "dataset", "config_path"))
        preprocess_manifest = preprocess_dataset(dataset_config)
        logger.log(f"preprocessed dataset to {preprocess_manifest['processed_dir']}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_environment(run_dir)
    predictions = _build_predictions(config)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    logger.log(f"wrote predictions count={len(predictions)}")
    metrics = evaluate_predictions(
        predictions_jsonl=predictions_path,
        output_dir=run_dir,
        top_k=[int(k) for k in config.get("top_k", [1, 5, 10])],
    )
    logger.log("wrote metrics.json and metrics.csv")
    logger.log("finish")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }


def run_all(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return run_experiment_config(config, preprocess=True)


def _build_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
    method = _method_name(config)
    if method != "skeleton":
        raise ValueError("Phase 1 runner only supports method: skeleton")
    processed_dir = Path(_required_path(config, "dataset", "processed_dir"))
    split = str(config.get("split") or "test")
    examples = [
        row for row in read_jsonl(processed_dir / "examples.jsonl")
        if str(row.get("split")) == split
    ]
    if not examples:
        raise ValueError(f"no examples found for split={split} in {processed_dir}")
    predictions = []
    for row in examples:
        target = str(row["target"])
        candidates = [str(item) for item in row.get("candidates", [])]
        ordered = [target] + sorted(item for item in candidates if item != target)
        scores = [1.0 / (rank + 1) for rank in range(len(ordered))]
        predictions.append(
            {
                "user_id": str(row["user_id"]),
                "target_item": target,
                "candidate_items": candidates,
                "predicted_items": ordered,
                "scores": scores,
                "method": "skeleton",
                "domain": str(row.get("domain") or config.get("domain") or "tiny"),
                "raw_output": None,
                "metadata": {
                    "example_id": row.get("example_id"),
                    "split": row.get("split"),
                    "phase": "phase1_reproducible_skeleton",
                    "not_a_formal_baseline": True,
                },
            }
        )
    return predictions


def _method_name(config: dict[str, Any]) -> str:
    method = config.get("method")
    if isinstance(method, dict):
        return str(method.get("name") or method.get("type") or "")
    return str(method or "")


def _run_name(config: dict[str, Any], method_name: str) -> str:
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    if output.get("run_name"):
        return str(output["run_name"])
    if config.get("run_name"):
        return str(config["run_name"])
    return f"smoke_{method_name}"


def _output_dir(config: dict[str, Any]) -> str:
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    return str(output.get("output_dir") or config.get("output_dir") or "outputs/runs")


def _required_path(config: dict[str, Any], section: str, key: str) -> str:
    value = (config.get(section) or {}).get(key)
    if not value:
        raise ValueError(f"config.{section}.{key} is required")
    return str(value)
