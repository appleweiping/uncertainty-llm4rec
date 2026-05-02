"""Deterministic smoke experiment runner for skeleton and minimal baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_dataset
from llm4rec.evaluation.aggregation import aggregate_run_metrics
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.evaluation.table_export import export_phase5_tables
from llm4rec.experiments.config import load_config, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_seed
from llm4rec.io.artifacts import create_run_dir, read_jsonl, write_environment, write_json, write_jsonl
from llm4rec.llm.hf_provider import HFLocalProvider
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.methods.ours_method import OursMethodRanker
from llm4rec.rankers.base import BaseRanker
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.llm_generative import LLMConfidenceObservationRanker, LLMGenerativeRanker
from llm4rec.rankers.llm_reranker import LLMReranker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker
from llm4rec.rankers.sequential import MarkovSequentialRanker, SasrecInterfaceRanker, SequentialLastItemRanker
from llm4rec.trainers.lora import LoraTrainer
from llm4rec.trainers.sequential import SequentialTrainer
from llm4rec.trainers.traditional import TraditionalRankerTrainer


SEQUENTIAL_METHODS = {"sequential_last_item", "sequential_markov", "sasrec_interface", "sequential_interface"}


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
    predictions = _build_predictions(config, run_dir=run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    logger.log(f"wrote predictions count={len(predictions)}")
    eval_context = _evaluation_context_for_config(config)
    metrics = evaluate_predictions(
        predictions_jsonl=predictions_path,
        output_dir=run_dir,
        top_k=[int(k) for k in config.get("top_k", [1, 5, 10])],
        item_catalog=eval_context["item_catalog"],
        train_examples=eval_context["train_examples"],
        all_examples=eval_context["all_examples"],
        evaluation_config=config,
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
    seeds = _seeds_for_config(config)
    for seed in seeds:
        for baseline in baselines:
            child_config = _config_for_baseline(config, baseline, seed=seed)
            result = run_experiment_config(child_config, preprocess=False)
            result["preprocess_manifest"] = preprocess_manifest
            runs.append(result)
    postprocess = _postprocess_run_all(config, runs)
    return {
        "run_count": len(runs),
        "baseline_methods": baselines,
        "seeds": seeds,
        "preprocess_manifest": preprocess_manifest,
        "runs": runs,
        "postprocess": postprocess,
    }


def _build_predictions(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    method = _method_name(config)
    if method == "skeleton":
        return _build_skeleton_predictions(config)
    if method == "lora_dry_run":
        return _build_lora_dry_run(config, run_dir=run_dir)
    return _build_ranker_predictions(config, run_dir=run_dir)


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


def _build_ranker_predictions(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
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
    if _method_name(config) in SEQUENTIAL_METHODS:
        training = _training_config(config)
        checkpoint_dir = Path(str(training.get("checkpoint_dir") or (run_dir / "checkpoints")))
        trainer = SequentialTrainer(
            ranker,
            train_examples=train_examples,
            item_catalog=item_catalog,
            interactions=interactions,
            checkpoint_dir=checkpoint_dir,
            config=config,
            seed=int(config.get("seed") or 0),
            eval_only=bool(training.get("eval_only", False)),
        )
        trainer.train()
        return trainer.predict(eval_examples)
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


def _build_lora_dry_run(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    training = _training_config(config)
    output_dir = str(training.get("output_dir") or (run_dir / "artifacts" / "lora_dry_run"))
    trainer = LoraTrainer({**training, "output_dir": output_dir}, dry_run=bool(training.get("dry_run", True)))
    result = trainer.train()
    write_json(run_dir / "artifacts" / "lora_dry_run_manifest.json", result.metadata)
    return []


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
    if method == "sequential_last_item":
        return SequentialLastItemRanker(max_history_length=int(params.get("max_history_length") or 50))
    if method in {"sequential_markov", "sequential_interface"}:
        return MarkovSequentialRanker(max_history_length=int(params.get("max_history_length") or 50))
    if method == "sasrec_interface":
        return SasrecInterfaceRanker(max_history_length=int(params.get("max_history_length") or 50))
    if method in {"llm_generative", "llm_generative_mock"}:
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
    if _is_ours_method(method, method_config):
        return OursMethodRanker(
            provider=_build_llm_provider(config),
            method_config=method_config,
            seed=seed,
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
    method = resolved.get("method")
    if isinstance(method, dict) and method.get("config_path"):
        source_path = str(method["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in method.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["method"] = merged
    llm = resolved.get("llm")
    if isinstance(llm, dict) and llm.get("config_path"):
        source_path = str(llm["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in llm.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["llm"] = merged
    training = resolved.get("training")
    if isinstance(training, dict) and training.get("config_path"):
        source_path = str(training["config_path"])
        merged = load_config(source_path)
        merged.update({key: value for key, value in training.items() if key != "config_path"})
        merged["config_path"] = source_path
        resolved["training"] = merged
    return resolved


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training")
    if not isinstance(training, dict):
        return {}
    if training.get("config_path"):
        loaded = load_config(str(training["config_path"]))
        loaded.update({key: value for key, value in training.items() if key != "config_path"})
        return loaded
    return training


def _required_llm_text(llm_config: dict[str, Any], key: str) -> str:
    value = llm_config.get(key)
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null"}:
        raise ValueError(f"config.llm.{key} is required for provider={llm_config.get('provider')}")
    return text


def _method_config(config: dict[str, Any]) -> dict[str, Any]:
    method = config.get("method")
    if isinstance(method, dict) and method.get("config_path"):
        loaded = load_config(str(method["config_path"]))
        loaded.update({key: value for key, value in method.items() if key != "config_path"})
        loaded["config_path"] = str(method["config_path"])
        return loaded
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


def _config_for_baseline(config: dict[str, Any], baseline: str, *, seed: int | None = None) -> dict[str, Any]:
    child = json.loads(json.dumps(config))
    child.pop("baselines", None)
    child.pop("seeds", None)
    resolved_seed = int(seed if seed is not None else config.get("seed") or 0)
    child["seed"] = resolved_seed
    method_config = _named_method_config(baseline)
    if method_config is None:
        method_config = {
            "name": baseline,
            "type": _default_method_type_for(baseline),
            "seed": resolved_seed,
            "params": _default_params_for(baseline),
        }
    method_config["seed"] = resolved_seed
    child["method"] = method_config
    output = child.get("output") if isinstance(child.get("output"), dict) else {}
    output["run_name"] = _run_name_for_child(config, baseline)
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
    if method in {"sequential_last_item", "sequential_markov", "sasrec_interface", "sequential_interface"}:
        return {"max_history_length": 5}
    if method in {"llm_generative", "llm_generative_mock"}:
        return {"prediction_method": "llm_generative_mock", "text_policy": "title"}
    if method == "llm_rerank":
        return {"prediction_method": "llm_rerank_mock", "text_policy": "title"}
    if method == "llm_confidence_observation":
        return {"prediction_method": "llm_confidence_observation_mock", "text_policy": "title"}
    return {}


def _default_method_type_for(method: str) -> str:
    if method.startswith("ours_"):
        return "ours_method"
    if method.startswith("llm_"):
        return "llm"
    if method in SEQUENTIAL_METHODS:
        return "sequential"
    if method == "lora_dry_run":
        return "training"
    return "ranker"


def _default_run_name_for(method: str) -> str:
    if method == "sequential_last_item":
        return "smoke_sequential_last_item"
    if method == "sequential_markov":
        return "smoke_sequential_markov"
    if method == "sasrec_interface":
        return "smoke_sasrec_interface"
    if method == "lora_dry_run":
        return "smoke_lora_dry_run"
    if method in {"llm_generative", "llm_generative_mock"}:
        return "smoke_llm_generative"
    if method == "llm_rerank":
        return "smoke_llm_rerank"
    if method == "llm_confidence_observation":
        return "smoke_llm_confidence"
    return f"smoke_all_{method}"


def _is_ours_method(method: str, method_config: dict[str, Any]) -> bool:
    return (
        str(method_config.get("type") or "") == "ours_method"
        or method == "ours_uncertainty_guided"
        or method.startswith("ours_ablation_")
        or method == "ours_fallback_only"
    )


def _named_method_config(name: str) -> dict[str, Any] | None:
    path = Path("configs") / "methods" / f"{name}.yaml"
    if not path.exists():
        return None
    return load_config(path)


def _run_name_for_child(config: dict[str, Any], method: str) -> str:
    parent_name = str(
        ((config.get("output") or {}) if isinstance(config.get("output"), dict) else {}).get("run_name")
        or config.get("run_name")
        or ""
    )
    suffix = _child_run_suffix(method)
    if parent_name == "smoke_ours_ablation":
        return f"smoke_ours_ablation_{suffix}"
    if parent_name == "smoke_phase6_all":
        return f"smoke_phase6_{suffix}"
    if parent_name.startswith(("pilot_", "real_", "r2_")):
        return f"{parent_name}_{suffix}"
    return _default_run_name_for(method)


def _child_run_suffix(method: str) -> str:
    mapping = {
        "ours_uncertainty_guided": "ours_full",
        "ours_ablation_no_uncertainty": "ours_no_uncertainty",
        "ours_ablation_no_grounding": "ours_no_grounding",
        "ours_ablation_no_candidate_normalized_confidence": "ours_no_candidate_normalized_confidence",
        "ours_ablation_no_popularity_adjustment": "ours_no_popularity_adjustment",
        "ours_ablation_no_echo_guard": "ours_no_echo_guard",
        "ours_fallback_only": "ours_fallback_only",
        "llm_generative": "llm_generative",
        "llm_rerank": "llm_rerank",
        "bm25": "bm25",
        "popularity": "popularity",
        "mf": "mf",
        "sequential_markov": "sequential_markov",
    }
    return mapping.get(method, method)


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


def _evaluation_context_for_config(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(_required_path(config, "dataset", "processed_dir"))
    examples = read_jsonl(processed_dir / "examples.jsonl")
    return {
        "item_catalog": _read_csv(processed_dir / "items.csv"),
        "all_examples": examples,
        "train_examples": [row for row in examples if str(row.get("split")) == "train"],
    }


def _seeds_for_config(config: dict[str, Any]) -> list[int]:
    seeds = config.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(seed) for seed in seeds]
    return [int(config.get("seed") or 0)]


def _postprocess_run_all(config: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    del runs
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    runs_dir = Path(str(output.get("output_dir") or config.get("output_dir") or "outputs/runs"))
    tables_config = config.get("tables") if isinstance(config.get("tables"), dict) else {}
    aggregate_config = config.get("aggregate") if isinstance(config.get("aggregate"), dict) else {}
    table_output_dir = Path(str(tables_config.get("output_dir") or aggregate_config.get("output_dir") or "outputs/tables"))
    postprocess: dict[str, Any] = {}
    if bool(aggregate_config.get("enabled", False)):
        postprocess["aggregation"] = aggregate_run_metrics(runs_dir, output_dir=table_output_dir)
    if bool(tables_config.get("enabled", False)):
        postprocess["tables"] = export_phase5_tables(runs_dir, output_dir=table_output_dir)
    return postprocess


def _required_path(config: dict[str, Any], section: str, key: str) -> str:
    value = (config.get(section) or {}).get(key)
    if not value:
        raise ValueError(f"config.{section}.{key} is required")
    return str(value)
