"""Shared evaluator for reproducible prediction artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm4rec.evaluation.export import export_metrics
from llm4rec.evaluation.prediction_schema import validate_prediction
from llm4rec.evaluation.slices import enrich_predictions_for_slices
from llm4rec.io.artifacts import read_jsonl
from llm4rec.metrics.calibration import calibration_metrics
from llm4rec.metrics.confidence import confidence_metrics
from llm4rec.metrics.coverage import coverage_metrics
from llm4rec.metrics.diversity import diversity_metrics
from llm4rec.metrics.efficiency import efficiency_metrics
from llm4rec.metrics.long_tail import long_tail_metrics
from llm4rec.metrics.novelty import novelty_metrics, train_item_popularity
from llm4rec.metrics.ranking import ranking_metrics
from llm4rec.metrics.slicing import slice_predictions
from llm4rec.metrics.validity import validity_metrics


def evaluate_predictions(
    *,
    predictions_jsonl: str | Path,
    output_dir: str | Path,
    top_k: list[int],
    item_catalog: list[dict[str, Any]] | None = None,
    train_examples: list[dict[str, Any]] | None = None,
    all_examples: list[dict[str, Any]] | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_rows = read_jsonl(predictions_jsonl)
    predictions = [
        validate_prediction(row, row_number=index + 1)
        for index, row in enumerate(raw_rows)
    ]
    context = _evaluation_context(
        output_dir=output_dir,
        item_catalog=item_catalog,
        train_examples=train_examples,
        all_examples=all_examples,
        evaluation_config=evaluation_config,
    )
    catalog_items = _catalog_items(context["item_catalog"], predictions)
    train_popularity = train_item_popularity(context["train_examples"])
    predictions = enrich_predictions_for_slices(
        predictions,
        examples=context["all_examples"],
        item_catalog=context["item_catalog"],
        train_popularity=train_popularity,
        catalog_items=catalog_items,
    )
    aggregate = _compute(
        predictions,
        top_k=top_k,
        item_catalog=context["item_catalog"],
        catalog_items=catalog_items,
        train_popularity=train_popularity,
    )
    by_domain = _slice_metrics(
        slice_predictions(predictions, key="domain"),
        top_k=top_k,
        item_catalog=context["item_catalog"],
        catalog_items=catalog_items,
        train_popularity=train_popularity,
    )
    by_user_history_bucket = _slice_metrics(
        slice_predictions(predictions, key="metadata.user_history_bucket"),
        top_k=top_k,
        item_catalog=context["item_catalog"],
        catalog_items=catalog_items,
        train_popularity=train_popularity,
    )
    by_item_popularity_bucket = _slice_metrics(
        slice_predictions(predictions, key="metadata.target_item_popularity_bucket"),
        top_k=top_k,
        item_catalog=context["item_catalog"],
        catalog_items=catalog_items,
        train_popularity=train_popularity,
    )
    by_method = _slice_metrics(
        slice_predictions(predictions, key="method"),
        top_k=top_k,
        item_catalog=context["item_catalog"],
        catalog_items=catalog_items,
        train_popularity=train_popularity,
    )
    metadata = _metrics_metadata(
        predictions,
        output_dir=output_dir,
        config=context["config"],
        train_popularity=train_popularity,
    )
    metrics = {
        "count": len(predictions),
        "top_k": top_k,
        "aggregate": aggregate,
        "by_domain": by_domain,
        "per_domain": by_domain,
        "by_user_history_bucket": by_user_history_bucket,
        "by_item_popularity_bucket": by_item_popularity_bucket,
        "by_method": by_method,
        "metadata": metadata,
        "schema_version": "llm4rec_prediction_v1",
        "is_experiment_result": False,
        "note": _note_for_methods(predictions),
    }
    export_metrics(metrics, output_dir=output_dir)
    return metrics


def _compute(
    predictions: list[dict[str, Any]],
    *,
    top_k: list[int],
    item_catalog: list[dict[str, Any]],
    catalog_items: list[str],
    train_popularity: dict[str, int],
) -> dict[str, Any]:
    ranking = ranking_metrics(predictions, top_k=top_k)
    validity = validity_metrics(predictions)
    return {
        **ranking,
        **validity,
        "coverage_metrics": coverage_metrics(predictions, top_k=top_k, catalog_items=catalog_items),
        "diversity": diversity_metrics(predictions, item_catalog=item_catalog, top_k=top_k),
        "novelty": novelty_metrics(
            predictions,
            train_popularity=train_popularity,
            catalog_items=catalog_items,
            top_k=top_k,
        ),
        "long_tail": long_tail_metrics(
            predictions,
            train_popularity=train_popularity,
            catalog_items=catalog_items,
            top_k=top_k,
        ),
        "confidence": confidence_metrics(predictions),
        "calibration": calibration_metrics(predictions),
        "efficiency": efficiency_metrics(predictions),
    }


def _slice_metrics(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    top_k: list[int],
    item_catalog: list[dict[str, Any]],
    catalog_items: list[str],
    train_popularity: dict[str, int],
) -> dict[str, Any]:
    return {
        str(name): _compute(
            rows,
            top_k=top_k,
            item_catalog=item_catalog,
            catalog_items=catalog_items,
            train_popularity=train_popularity,
        )
        for name, rows in sorted(grouped.items())
    }


def _evaluation_context(
    *,
    output_dir: str | Path,
    item_catalog: list[dict[str, Any]] | None,
    train_examples: list[dict[str, Any]] | None,
    all_examples: list[dict[str, Any]] | None,
    evaluation_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = evaluation_config or _load_resolved_config(output_dir)
    loaded_item_catalog = item_catalog
    loaded_examples = all_examples
    loaded_train_examples = train_examples
    processed_dir = ((config.get("dataset") or {}) if isinstance(config.get("dataset"), dict) else {}).get("processed_dir")
    if processed_dir and (loaded_item_catalog is None or loaded_examples is None or loaded_train_examples is None):
        processed_path = Path(str(processed_dir))
        if loaded_item_catalog is None and (processed_path / "items.csv").exists():
            loaded_item_catalog = _read_csv(processed_path / "items.csv")
        if (loaded_examples is None or loaded_train_examples is None) and (processed_path / "examples.jsonl").exists():
            examples = read_jsonl(processed_path / "examples.jsonl")
            loaded_examples = loaded_examples or examples
            loaded_train_examples = loaded_train_examples or [
                row for row in examples if str(row.get("split")) == "train"
            ]
    return {
        "config": config,
        "item_catalog": loaded_item_catalog or [],
        "train_examples": loaded_train_examples or [],
        "all_examples": loaded_examples or [],
    }


def _load_resolved_config(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "resolved_config.yaml"
    if not path.exists():
        return {}
    try:
        return _load_yaml_subset(path)
    except Exception:
        return {}


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(raw_value.strip())
    return root


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _catalog_items(item_catalog: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[str]:
    items = [str(row["item_id"]) for row in item_catalog if row.get("item_id") is not None]
    if items:
        return sorted(set(items))
    return sorted({str(item) for row in predictions for item in row.get("candidate_items", [])})


def _metrics_metadata(
    predictions: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    train_popularity: dict[str, int],
) -> dict[str, Any]:
    methods = sorted({str(row.get("method") or "unknown") for row in predictions})
    dataset = "unknown"
    if isinstance(config.get("dataset"), dict):
        dataset = str(config["dataset"].get("name") or "unknown")
    seed = config.get("seed", "unknown")
    return {
        "run_id": Path(output_dir).name,
        "run_dir": str(output_dir),
        "method": methods[0] if len(methods) == 1 else "mixed",
        "methods": methods,
        "dataset": dataset,
        "seed": seed,
        "popularity_source": "train_examples_only",
        "train_popularity_item_count": len(train_popularity),
    }


def _note_for_methods(predictions: list[dict[str, Any]]) -> str:
    methods = {str(row.get("method") or "") for row in predictions}
    if any(method.startswith("ours_") for method in methods):
        return (
            "Phase 6 OursMethod mock smoke metrics only; not a paper result, "
            "calibration claim, or effectiveness claim."
        )
    if methods == {"skeleton"}:
        return "Phase 1 skeleton smoke metrics only; not a formal baseline or paper result."
    if any(method.startswith("llm_") for method in methods):
        return (
            "Phase 3 mock LLM baseline and uncertainty-observation smoke metrics only; "
            "not a paper result or OursMethod claim."
        )
    if any(method.startswith("sequential_") or method in {"sasrec_interface"} for method in methods):
        return (
            "Phase 4 lightweight sequential baseline smoke metrics only; "
            "not a strong SASRec result, paper result, or OursMethod claim."
        )
    return (
        "Phase 2 minimal baseline smoke metrics only; use for infrastructure "
        "validation, not as paper-level experimental evidence."
    )
