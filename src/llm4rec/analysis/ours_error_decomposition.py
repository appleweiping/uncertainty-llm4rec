"""Offline R3 OursMethod error decomposition from saved artifacts.

The functions in this module intentionally read only existing run artifacts.
They do not instantiate LLM providers or mutate evaluator definitions.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from llm4rec.metrics.calibration import brier_score, expected_calibration_error
from llm4rec.metrics.confidence import _confidence_rows
from llm4rec.metrics.long_tail import assign_popularity_buckets
from llm4rec.metrics.novelty import train_item_popularity
from llm4rec.metrics.ranking import dedupe

R3_PREFIX = "r3_movielens_1m_real_llm_full_candidate500"
DEFAULT_SEEDS = (13, 21, 42)
OURS_METHOD = "ours_uncertainty_guided_real"
FALLBACK_METHOD = "ours_fallback_only"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def method_run_dir(runs_dir: str | Path, method: str, seed: int) -> Path:
    path = Path(runs_dir) / f"{R3_PREFIX}_{method}_seed{seed}"
    if path.exists():
        return path
    # Historical R3 used shortened suffixes for Ours ablation run directories.
    aliases = {
        "ours_ablation_no_uncertainty": "ours_no_uncertainty",
        "ours_ablation_no_grounding": "ours_no_grounding",
        "ours_ablation_no_popularity_adjustment": "ours_no_popularity_adjustment",
        "ours_ablation_no_echo_guard": "ours_no_echo_guard",
    }
    alias = aliases.get(method)
    if alias:
        alias_path = Path(runs_dir) / f"{R3_PREFIX}_{alias}_seed{seed}"
        if alias_path.exists():
            return alias_path
    return path


def required_prediction_paths(
    runs_dir: str | Path,
    *,
    methods: Iterable[str] = (OURS_METHOD, FALLBACK_METHOD, "llm_rerank_real"),
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> list[Path]:
    paths: list[Path] = []
    for method in methods:
        for seed in seeds:
            paths.append(method_run_dir(runs_dir, method, int(seed)) / "predictions.jsonl")
    return paths


def missing_required_artifacts(
    runs_dir: str | Path,
    *,
    methods: Iterable[str] = (OURS_METHOD, FALLBACK_METHOD, "llm_rerank_real"),
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> list[str]:
    return [str(path) for path in required_prediction_paths(runs_dir, methods=methods, seeds=seeds) if not path.exists()]


def load_method_predictions(runs_dir: str | Path, method: str, seed: int) -> list[dict[str, Any]]:
    path = method_run_dir(runs_dir, method, seed) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    for row in rows:
        row.setdefault("metadata", {})
        row["metadata"].setdefault("seed", seed)
    return rows


def load_dataset_context(processed_dir: str | Path | None = None) -> dict[str, Any]:
    processed_path = Path(processed_dir or "data/processed/movielens_1m/r2_full_single_dataset")
    items = _read_csv(processed_path / "items.csv") if (processed_path / "items.csv").exists() else []
    examples = read_jsonl(processed_path / "examples.jsonl") if (processed_path / "examples.jsonl").exists() else []
    train_examples = [row for row in examples if str(row.get("split")) == "train"]
    catalog_items = [str(row.get("item_id")) for row in items if row.get("item_id") is not None]
    popularity = train_item_popularity(train_examples)
    item_buckets = assign_popularity_buckets(popularity, catalog_items=catalog_items)
    item_titles = {str(row.get("item_id")): str(row.get("title", "")) for row in items if row.get("item_id") is not None}
    examples_by_id = {str(row.get("example_id")): row for row in examples if row.get("example_id") is not None}
    return {
        "items": items,
        "examples": examples,
        "train_examples": train_examples,
        "catalog_items": catalog_items,
        "train_popularity": popularity,
        "item_buckets": item_buckets,
        "item_titles": item_titles,
        "examples_by_id": examples_by_id,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def enrich_rows(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach target popularity metadata used by evaluator slices."""
    item_buckets = context.get("item_buckets") or {}
    train_popularity = context.get("train_popularity") or {}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        metadata = dict(copy.get("metadata") or {})
        target = str(copy.get("target_item") or "")
        metadata.setdefault("target_item_popularity_bucket", item_buckets.get(target, "unknown"))
        metadata.setdefault("target_train_popularity", int(train_popularity.get(target, 0)))
        copy["metadata"] = metadata
        enriched.append(copy)
    return enriched


def hit_at_k(row: dict[str, Any], k: int = 10) -> int:
    target = str(row.get("target_item"))
    return int(target in _dedupe_limit((str(item) for item in row.get("predicted_items", [])), k))


def mrr_at_k(row: dict[str, Any], k: int = 10) -> float:
    target = str(row.get("target_item"))
    for rank, item in enumerate(_dedupe_limit((str(value) for value in row.get("predicted_items", [])), k), start=1):
        if item == target:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(row: dict[str, Any], k: int = 10) -> float:
    target = str(row.get("target_item"))
    for rank, item in enumerate(_dedupe_limit((str(value) for value in row.get("predicted_items", [])), k), start=1):
        if item == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def _dedupe_limit(items: Iterable[str], k: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= k:
            break
    return output


def compute_group_metrics(rows: list[dict[str, Any]], *, context: dict[str, Any] | None = None, k: int = 10) -> dict[str, Any]:
    context = context or {}
    enriched = enrich_rows(rows, context) if context else rows
    ranking = _ranking_summary(enriched, k=k)
    validity = _validity_summary(enriched)
    confidence_rows, missing_confidence = _confidence_rows(enriched)
    target_buckets = Counter(str((row.get("metadata") or {}).get("target_item_popularity_bucket", "unknown")) for row in enriched)
    predicted_buckets: Counter[str] = Counter()
    item_buckets = context.get("item_buckets") or {}
    for row in enriched:
        metadata = row.get("metadata") or {}
        grounded = metadata.get("grounded_item_id")
        if grounded is not None:
            predicted_buckets[item_buckets.get(str(grounded), str(metadata.get("popularity_bucket", "unknown")))] += 1
        else:
            for item in _dedupe_limit((str(value) for value in row.get("predicted_items", [])), k):
                predicted_buckets[item_buckets.get(item, "unknown")] += 1
    confidences = [float(row["confidence"]) for row in confidence_rows]
    correct = [bool(row["correct"]) for row in confidence_rows]
    high_conf_wrong = sum(1 for row in confidence_rows if row["confidence"] >= 0.85 and not row["correct"])
    high_conf_correct = sum(1 for row in confidence_rows if row["confidence"] >= 0.85 and row["correct"])
    low_conf_correct = sum(1 for row in confidence_rows if row["confidence"] < 0.5 and row["correct"])
    return {
        "count": len(enriched),
        f"recall@{k}": ranking.get(f"recall@{k}", 0.0),
        f"ndcg@{k}": ranking.get(f"ndcg@{k}", 0.0),
        f"mrr@{k}": ranking.get(f"mrr@{k}", 0.0),
        f"hit_rate@{k}": ranking.get(f"hit_rate@{k}", 0.0),
        "validity_rate": validity.get("validity_rate", 0.0),
        "hallucination_rate": validity.get("hallucination_rate", 0.0),
        "parse_success_rate": _mean_bool(enriched, "parse_success"),
        "grounding_success_rate": _mean_bool(enriched, "grounding_success"),
        "mean_confidence": mean(confidences) if confidences else None,
        "confidence_count": len(confidence_rows),
        "count_missing_confidence": missing_confidence,
        "ece": expected_calibration_error(confidence_rows),
        "brier": brier_score(confidence_rows),
        "confidence_accuracy": (sum(correct) / len(correct)) if correct else None,
        "high_confidence_wrong_count": high_conf_wrong,
        "high_confidence_correct_count": high_conf_correct,
        "low_confidence_correct_count": low_conf_correct,
        "target_popularity_bucket_distribution": dict(sorted(target_buckets.items())),
        "predicted_item_popularity_bucket_distribution": dict(sorted(predicted_buckets.items())),
    }


def _ranking_summary(rows: list[dict[str, Any]], *, k: int) -> dict[str, float]:
    if not rows:
        return {f"recall@{k}": 0.0, f"hit_rate@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}
    hits = [hit_at_k(row, k) for row in rows]
    mrrs = [mrr_at_k(row, k) for row in rows]
    ndcgs = [ndcg_at_k(row, k) for row in rows]
    return {
        f"recall@{k}": sum(hits) / len(rows),
        f"hit_rate@{k}": sum(hits) / len(rows),
        f"mrr@{k}": sum(mrrs) / len(rows),
        f"ndcg@{k}": sum(ndcgs) / len(rows),
    }


def _mean_bool(rows: list[dict[str, Any]], metadata_key: str) -> float | None:
    values = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if isinstance(metadata.get(metadata_key), bool):
            values.append(float(metadata[metadata_key]))
    return sum(values) / len(values) if values else None


def _validity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "validity_rate": 0.0,
            "hallucination_rate": 0.0,
            "valid_prediction_count": 0,
            "hallucinated_prediction_count": 0,
        }
    valid_count = 0
    hallucinated_count = 0
    for row in rows:
        metadata = row.get("metadata") or {}
        if "_analysis_is_valid_prediction" in metadata:
            is_valid = bool(metadata["_analysis_is_valid_prediction"])
            hallucinated = bool(metadata.get("_analysis_is_hallucinated_prediction", not is_valid))
        else:
            is_valid, hallucinated = row_validity_flags(row)
        valid_count += int(is_valid)
        hallucinated_count += int(hallucinated)
    return {
        "validity_rate": valid_count / len(rows),
        "hallucination_rate": hallucinated_count / len(rows),
        "valid_prediction_count": valid_count,
        "hallucinated_prediction_count": hallucinated_count,
    }


def row_validity_flags(row: dict[str, Any]) -> tuple[bool, bool]:
    predicted = [str(item) for item in row.get("predicted_items", [])]
    if not predicted:
        return False, True
    candidates = {str(item) for item in row.get("candidate_items", [])}
    is_valid = all(item in candidates for item in predicted)
    return is_valid, not is_valid


def decision_attribution(
    runs_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> dict[str, list[dict[str, Any]]]:
    context = load_dataset_context(processed_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    delta_rows: list[dict[str, Any]] = []
    all_ours: list[dict[str, Any]] = []
    all_fallback: list[dict[str, Any]] = []
    for seed in seeds:
        ours = load_method_predictions(runs_dir, OURS_METHOD, int(seed))
        fallback = load_method_predictions(runs_dir, FALLBACK_METHOD, int(seed))
        fallback_by_key = {_row_key(row): row for row in fallback}
        all_ours.extend(ours)
        all_fallback.extend(fallback)
        for row in ours:
            metadata = row.get("metadata") or {}
            decision = str(metadata.get("uncertainty_decision") or "unknown")
            grouped[decision].append(row)
            fallback_row = fallback_by_key.get(_row_key(row))
            if fallback_row is None:
                continue
            ours_hit = hit_at_k(row)
            fallback_hit = hit_at_k(fallback_row)
            top10_same = dedupe(str(item) for item in row.get("predicted_items", []))[:10] == dedupe(
                str(item) for item in fallback_row.get("predicted_items", [])
            )[:10]
            delta_rows.append(
                {
                    "seed": seed,
                    "user_id": row.get("user_id"),
                    "example_id": metadata.get("example_id"),
                    "decision": decision,
                    "target_item": row.get("target_item"),
                    "ours_hit@10": ours_hit,
                    "fallback_hit@10": fallback_hit,
                    "delta_hit@10": ours_hit - fallback_hit,
                    "ours_mrr@10": mrr_at_k(row),
                    "fallback_mrr@10": mrr_at_k(fallback_row),
                    "delta_mrr@10": mrr_at_k(row) - mrr_at_k(fallback_row),
                    "ours_ndcg@10": ndcg_at_k(row),
                    "fallback_ndcg@10": ndcg_at_k(fallback_row),
                    "delta_ndcg@10": ndcg_at_k(row) - ndcg_at_k(fallback_row),
                    "top10_same_as_fallback": top10_same,
                    "confidence": metadata.get("confidence"),
                    "grounding_score": metadata.get("grounding_score"),
                    "candidate_adherent": metadata.get("candidate_adherent"),
                    "grounded_item_id": metadata.get("grounded_item_id"),
                    "generated_title": metadata.get("generated_title"),
                    "target_popularity_bucket": context["item_buckets"].get(str(row.get("target_item")), "unknown"),
                    "grounded_popularity_bucket": context["item_buckets"].get(str(metadata.get("grounded_item_id")), "unknown"),
                    "prompt_hash": metadata.get("prompt_hash"),
                    "cache_key": ((metadata.get("provider_metadata") or {}).get("cache_key")),
                }
            )
    attribution_rows: list[dict[str, Any]] = []
    for decision in ["accept", "fallback", "rerank", "abstain", "unknown"]:
        rows = grouped.get(decision, [])
        if not rows and decision == "unknown":
            continue
        metrics = compute_group_metrics(rows, context=context, k=10)
        decision_deltas = [row for row in delta_rows if row["decision"] == decision]
        metrics.update(_delta_summary(decision_deltas))
        metrics["decision"] = decision
        attribution_rows.append(metrics)
    all_metrics = compute_group_metrics(all_ours, context=context, k=10)
    all_metrics.update(_delta_summary(delta_rows))
    all_metrics["decision"] = "__all__"
    attribution_rows.append(all_metrics)
    fallback_metrics = compute_group_metrics(all_fallback, context=context, k=10)
    fallback_metrics["decision"] = "__fallback_only__"
    attribution_rows.append(fallback_metrics)
    if output_dir:
        output = Path(output_dir)
        write_csv(output / "r3_ours_decision_attribution.csv", attribution_rows)
        write_csv(output / "r3_ours_vs_fallback_deltas.csv", delta_rows)
    return {"decision_attribution": attribution_rows, "deltas": delta_rows}


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    return (str(metadata.get("example_id") or ""), str(row.get("user_id")), str(row.get("target_item")))


def _delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "differs_from_fallback_count": 0,
            "top10_same_as_fallback_rate": None,
            "delta_hit@10_sum": 0,
            "delta_hit@10_mean": None,
            "help_count": 0,
            "hurt_count": 0,
            "neutral_count": 0,
        }
    same = sum(1 for row in rows if row.get("top10_same_as_fallback"))
    deltas = [int(row.get("delta_hit@10") or 0) for row in rows]
    return {
        "differs_from_fallback_count": len(rows) - same,
        "top10_same_as_fallback_rate": same / len(rows),
        "delta_hit@10_sum": sum(deltas),
        "delta_hit@10_mean": sum(deltas) / len(deltas),
        "help_count": sum(1 for value in deltas if value > 0),
        "hurt_count": sum(1 for value in deltas if value < 0),
        "neutral_count": sum(1 for value in deltas if value == 0),
    }


def rerank_audit(
    runs_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    context = load_dataset_context(processed_dir)
    rows_out: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows = load_method_predictions(runs_dir, "llm_rerank_real", int(seed))
        aggregate_rows.extend(rows)
        rows_out.append(_rerank_audit_row(rows, seed=seed, context=context))
    rows_out.append(_rerank_audit_row(aggregate_rows, seed="all", context=context))
    if output_dir:
        write_csv(Path(output_dir) / "r3_llm_rerank_audit.csv", rows_out)
    return rows_out


def _rerank_audit_row(rows: list[dict[str, Any]], *, seed: int | str, context: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"seed": seed, "count": 0}
    parse_success = sum(1 for row in rows if bool((row.get("metadata") or {}).get("parse_success")))
    candidate_adherent = sum(1 for row in rows if bool((row.get("metadata") or {}).get("candidate_adherent")))
    grounding_success = sum(1 for row in rows if bool((row.get("metadata") or {}).get("grounding_success")))
    raw_ranked_grounded = sum(1 for row in rows if (row.get("metadata") or {}).get("raw_ranked_grounding"))
    empty_predictions = sum(1 for row in rows if not row.get("predicted_items"))
    invalid_rows = sum(1 for row in rows if not row_validity_flags(row)[0])
    all_zero_scores = sum(
        1
        for row in rows
        if row.get("scores") and all(float(score) == 0.0 for score in row.get("scores", []))
    )
    input_order_rows = sum(
        1
        for row in rows
        if dedupe(str(item) for item in row.get("predicted_items", []))[:10]
        == dedupe(str(item) for item in row.get("candidate_items", []))[:10]
    )
    metrics = compute_group_metrics(rows, context=context, k=10)
    interpretation = (
        "parser_or_truncated_json_failure"
        if parse_success / len(rows) < 0.5 and all_zero_scores / len(rows) > 0.9
        else "model_behavior_or_valid_low_signal"
    )
    return {
        "seed": seed,
        "count": len(rows),
        "recall@10": metrics["recall@10"],
        "ndcg@10": metrics["ndcg@10"],
        "mrr@10": metrics["mrr@10"],
        "parse_success_count": parse_success,
        "parse_success_rate": parse_success / len(rows),
        "candidate_adherent_count": candidate_adherent,
        "candidate_adherent_rate": candidate_adherent / len(rows),
        "grounding_success_count": grounding_success,
        "grounding_success_rate": grounding_success / len(rows),
        "raw_ranked_grounding_nonempty_count": raw_ranked_grounded,
        "empty_prediction_rows": empty_predictions,
        "invalid_prediction_rows": invalid_rows,
        "all_zero_score_rows": all_zero_scores,
        "input_order_top10_rows": input_order_rows,
        "input_order_top10_rate": input_order_rows / len(rows),
        "interpretation": interpretation,
    }


def summary_from_outputs(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    attribution = _read_csv_if_exists(output / "r3_ours_decision_attribution.csv")
    sweep = _read_csv_if_exists(output / "r3_ours_policy_sweep.csv")
    rerank = _read_csv_if_exists(output / "r3_llm_rerank_audit.csv")
    return {
        "decision_attribution": attribution,
        "policy_sweep": sweep,
        "rerank_audit": rerank,
    }


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
