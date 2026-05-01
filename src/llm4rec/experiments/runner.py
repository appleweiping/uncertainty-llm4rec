"""Deterministic smoke experiment runner for skeleton and minimal baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_dataset
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.experiments.config import load_config, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_seed
from llm4rec.io.artifacts import create_run_dir, read_jsonl, write_environment, write_json, write_jsonl
from llm4rec.llm.hf_provider import HFLocalProvider
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.rankers.base import BaseRanker
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.llm_generative import LLMConfidenceObservationRanker, LLMGenerativeRanker
from llm4rec.rankers.llm_reranker import LLMReranker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker
from llm4rec.trainers.traditional import TraditionalRankerTrainer


def run_preprocess_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return preprocess_dataset(config)


def run_experiment(config_path: str | Path, *, preprocess: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    return run_experiment_config(config, preprocess=preprocess)


def run_experiment_config(config: dict[str, Any], *, preprocess: bool = False) -> dict[str, Any]:
    config = _resolve_runtime_config(config)
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
    cost_latency = dict(metrics.get("aggregate", {}).get("efficiency", {}))
    write_json(run_dir / "cost_latency.json", cost_latency)
    logger.log("wrote cost_latency.json")
    logger.log("finish")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }


def run_all(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    baselines = config.get("baselines")
    if not baselines:
        return run_experiment_config(config, preprocess=True)
    if not isinstance(baselines, list) or not all(isinstance(item, str) for item in baselines):
        raise ValueError("config.baselines must be an inline list of baseline names")
    dataset_config = load_config(_required_path(config, "dataset", "config_path"))
    preprocess_manifest = preprocess_dataset(dataset_config)
    runs = []
    for baseline in baselines:
        child_config = _config_for_baseline(config, baseline)
        result = run_experiment_config(child_config, preprocess=False)
        result["preprocess_manifest"] = preprocess_manifest
        runs.append(result)
    return {
        "run_count": len(runs),
        "baseline_methods": baselines,
        "preprocess_manifest": preprocess_manifest,
        "runs": runs,
    }


def _build_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
    method = _method_name(config)
    if method == "skeleton":
        return _build_skeleton_predictions(config)
    return _build_ranker_predictions(config)


def _build_skeleton_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
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


def _build_ranker_predictions(config: dict[str, Any]) -> list[dict[str, Any]]:
    processed_dir = Path(_required_path(config, "dataset", "processed_dir"))
    split = str(config.get("split") or "test")
    examples = read_jsonl(processed_dir / "examples.jsonl")
    item_catalog = _read_csv(processed_dir / "items.csv")
    interactions = _read_csv(processed_dir / "interactions.csv")
    train_examples = [row for row in examples if str(row.get("split")) == "train"]
    eval_examples = [row for row in examples if str(row.get("split")) == split]
    if not eval_examples:
        raise ValueError(f"no examples found for split={split} in {processed_dir}")
    ranker = _build_ranker(config)
    trainer = TraditionalRankerTrainer(
        ranker,
        train_examples=train_examples,
        item_catalog=item_catalog,
        interactions=interactions,
    )
    trainer_result = trainer.train()
    predictions = []
    for example in eval_examples:
        candidates = [str(item_id) for item_id in example.get("candidates", [])]
        result = ranker.rank(example, candidates)
        record = result.to_prediction_record()
        record["metadata"].update(
            {
                "trainer": trainer_result.metadata,
                "not_ours_method": True,
            }
        )
        record["metadata"].setdefault("phase", "phase2_minimal_baseline")
        predictions.append(record)
    return predictions


def _build_ranker(config: dict[str, Any]) -> BaseRanker:
    method = _method_name(config)
    method_config = _method_config(config)
    params = dict(method_config.get("params") or {})
    seed = int(method_config.get("seed") or config.get("seed") or 0)
    if method == "random":
        return RandomRanker(seed=seed)
    if method == "popularity":
        return PopularityRanker()
    if method == "bm25":
        return BM25Ranker(
            text_policy=str(params.get("text_policy") or "title"),
            k1=float(params.get("k1") or 1.5),
            b=float(params.get("b") or 0.75),
        )
    if method in {"mf", "matrix_factorization"}:
        return MatrixFactorizationRanker(
            seed=seed,
            factors=int(params.get("factors") or 8),
            epochs=int(params.get("epochs") or 25),
            learning_rate=float(params.get("learning_rate") or 0.05),
            regularization=float(params.get("regularization") or 0.001),
        )
    if method == "llm_generative":
        return LLMGenerativeRanker(
            provider=_build_llm_provider(config),
            method_name=str(params.get("prediction_method") or "llm_generative_mock"),
            text_policy=str(params.get("text_policy") or "title"),
        )
    if method == "llm_rerank":
        return LLMReranker(
            provider=_build_llm_provider(config),
            method_name=str(params.get("prediction_method") or "llm_rerank_mock"),
            text_policy=str(params.get("text_policy") or "title"),
        )
    if method == "llm_confidence_observation":
        return LLMConfidenceObservationRanker(
            provider=_build_llm_provider(config),
            method_name=str(params.get("prediction_method") or "llm_confidence_observation_mock"),
            text_policy=str(params.get("text_policy") or "title"),
        )
    raise ValueError(f"unknown experiment method: {method}")


def _build_llm_provider(config: dict[str, Any]) -> Any:
    llm_config = _llm_config(config)
    provider_type = str(llm_config.get("provider") or llm_config.get("type") or "")
    if provider_type == "mock":
        return MockLLMProvider(
            response_mode=str(llm_config.get("response_mode") or "generative_correct"),
            seed=int(llm_config.get("seed") or config.get("seed") or 0),
        )
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=_required_llm_text(llm_config, "model"),
            api_key_env=_required_llm_text(llm_config, "api_key_env"),
            base_url=_required_llm_text(llm_config, "base_url"),
            timeout_seconds=float(llm_config.get("timeout_seconds") or 60.0),
            max_retries=int(llm_config.get("max_retries") or 2),
        )
    if provider_type == "hf_local":
        return HFLocalProvider(
            model_name_or_path=_required_llm_text(llm_config, "model_name_or_path"),
            device=str(llm_config.get("device") or "auto"),
        )
    raise ValueError(f"unknown llm provider type: {provider_type}")


def _llm_config(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("config.llm is required for LLM methods")
    if llm.get("config_path"):
        loaded = load_config(str(llm["config_path"]))
        merged = dict(loaded)
        merged.update({key: value for key, value in llm.items() if key != "config_path"})
        return merged
    return llm


def _resolve_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    llm = resolved.get("llm")
    if isinstance(llm, dict) and llm.get("config_path"):
        source_path = str(llm["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in llm.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["llm"] = merged
    return resolved


def _required_llm_text(llm_config: dict[str, Any], key: str) -> str:
    value = llm_config.get(key)
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null"}:
        raise ValueError(f"config.llm.{key} is required for provider={llm_config.get('provider')}")
    return text


def _method_config(config: dict[str, Any]) -> dict[str, Any]:
    method = config.get("method")
    return method if isinstance(method, dict) else {"name": method}


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


def _config_for_baseline(config: dict[str, Any], baseline: str) -> dict[str, Any]:
    child = json.loads(json.dumps(config))
    child.pop("baselines", None)
    child["method"] = {
        "name": baseline,
        "type": "llm" if baseline.startswith("llm_") else "ranker",
        "seed": int(config.get("seed") or 0),
        "params": _default_params_for(baseline),
    }
    output = child.get("output") if isinstance(child.get("output"), dict) else {}
    output["run_name"] = _default_run_name_for(baseline)
    output["output_dir"] = str(output.get("output_dir") or child.get("output_dir") or "outputs/runs")
    child["output"] = output
    child["run_name"] = output["run_name"]
    child["output_dir"] = output["output_dir"]
    return child


def _default_params_for(method: str) -> dict[str, Any]:
    if method == "bm25":
        return {"text_policy": "title", "k1": 1.5, "b": 0.75}
    if method == "mf":
        return {"factors": 4, "epochs": 20, "learning_rate": 0.05, "regularization": 0.001}
    if method == "llm_generative":
        return {"prediction_method": "llm_generative_mock", "text_policy": "title"}
    if method == "llm_rerank":
        return {"prediction_method": "llm_rerank_mock", "text_policy": "title"}
    if method == "llm_confidence_observation":
        return {"prediction_method": "llm_confidence_observation_mock", "text_policy": "title"}
    return {}


def _default_run_name_for(method: str) -> str:
    if method == "llm_generative":
        return "smoke_llm_generative"
    if method == "llm_rerank":
        return "smoke_llm_rerank"
    if method == "llm_confidence_observation":
        return "smoke_llm_confidence"
    return f"smoke_all_{method}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_decode_csv_row(row) for row in csv.DictReader(handle)]


def _decode_csv_row(row: dict[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            decoded[key] = value
            continue
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                decoded[key] = json.loads(stripped)
                continue
            except json.JSONDecodeError:
                pass
        decoded[key] = value
    return decoded


def _required_path(config: dict[str, Any], section: str, key: str) -> str:
    value = (config.get(section) or {}).get(key)
    if not value:
        raise ValueError(f"config.{section}.{key} is required")
    return str(value)
