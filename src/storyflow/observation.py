"""Utilities for Phase 2A generative observation inputs and mock runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from storyflow.generation import (
    CATALOG_CONSTRAINED_JSON_TEMPLATE,
    build_prompt,
    compute_prompt_hash,
)
from storyflow.grounding import TitleGrounder
from storyflow.metrics import (
    brier_score,
    cbu_tau,
    expected_calibration_error,
    ground_hit_rate,
    tail_underconfidence_gap,
    wbc_tau,
)
from storyflow.providers import MockProvider
from storyflow.schemas import ItemCatalogRecord, PopularityBucket


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_catalog_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "item_id": str(row["item_id"]),
                    "title": str(row["title"]),
                    "popularity": int(float(row.get("popularity") or 0)),
                    "popularity_bucket": str(row.get("popularity_bucket") or "tail"),
                }
            )
    return rows


def catalog_records(catalog_rows: Iterable[dict[str, Any]]) -> list[ItemCatalogRecord]:
    return [
        ItemCatalogRecord(
            item_id=str(row["item_id"]),
            title=str(row["title"]),
            popularity=float(row.get("popularity") or 0),
            metadata={
                "popularity_bucket": row.get("popularity_bucket", "tail"),
                "genres": row.get("genres", ""),
            },
        )
        for row in catalog_rows
    ]


def processed_dataset_dir(
    *,
    dataset: str,
    processed_suffix: str,
    root: str | Path = ".",
) -> Path:
    return Path(root) / "data" / "processed" / dataset / processed_suffix


def _sorted_examples(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(example) for example in examples),
        key=lambda row: (
            int(row.get("target_timestamp") or 0),
            str(row.get("user_id", "")),
            str(row.get("example_id", "")),
        ),
    )


def _select_examples(
    examples: list[dict[str, Any]],
    *,
    max_examples: int | None,
    stratify_by_popularity: bool,
) -> list[dict[str, Any]]:
    if max_examples is None or max_examples >= len(examples):
        return examples
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1 when provided")
    if not stratify_by_popularity:
        return examples[:max_examples]

    grouped = {bucket.value: [] for bucket in PopularityBucket}
    for example in examples:
        bucket = str(example.get("target_popularity_bucket") or "tail")
        grouped.setdefault(bucket, []).append(example)

    selected: list[dict[str, Any]] = []
    while len(selected) < max_examples:
        made_progress = False
        for bucket in ("head", "mid", "tail"):
            if grouped.get(bucket):
                selected.append(grouped[bucket].pop(0))
                made_progress = True
                if len(selected) >= max_examples:
                    break
        if not made_progress:
            break
    return selected


def _catalog_candidate_records(
    example: dict[str, Any],
    *,
    catalog_rows: list[dict[str, Any]],
    candidate_count: int,
    allow_target_in_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic catalog candidate list for diagnostic prompts.

    The default policy excludes the target item so this cannot leak the answer.
    It is meant for grounding diagnostics, not for correctness evaluation.
    """

    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")

    history_ids = {str(item_id) for item_id in example.get("history_item_ids", [])}
    target_item_id = str(example["target_item_id"])
    excluded_ids = set(history_ids)
    if not allow_target_in_candidates:
        excluded_ids.add(target_item_id)

    grouped: dict[str, list[dict[str, Any]]] = {bucket.value: [] for bucket in PopularityBucket}
    for row in sorted(
        catalog_rows,
        key=lambda catalog_row: (
            -int(catalog_row.get("popularity") or 0),
            str(catalog_row.get("title") or ""),
            str(catalog_row.get("item_id") or ""),
        ),
    ):
        item_id = str(row["item_id"])
        if item_id in excluded_ids:
            continue
        bucket = str(row.get("popularity_bucket") or "tail")
        grouped.setdefault(bucket, []).append(row)

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    while len(candidates) < candidate_count:
        made_progress = False
        for bucket in ("head", "mid", "tail"):
            bucket_rows = grouped.get(bucket, [])
            while bucket_rows:
                candidate = bucket_rows.pop(0)
                item_id = str(candidate["item_id"])
                if item_id not in seen_ids:
                    candidates.append(candidate)
                    seen_ids.add(item_id)
                    made_progress = True
                    break
            if len(candidates) >= candidate_count:
                break
        if not made_progress:
            break

    fallback_history_count = 0
    if len(candidates) < candidate_count:
        catalog_by_id = {str(row["item_id"]): row for row in catalog_rows}
        for item_id in example.get("history_item_ids", []):
            item_id = str(item_id)
            if item_id in seen_ids or item_id == target_item_id or item_id not in catalog_by_id:
                continue
            candidates.append(catalog_by_id[item_id])
            seen_ids.add(item_id)
            fallback_history_count += 1
            if len(candidates) >= candidate_count:
                break

    candidate_ids = {str(candidate["item_id"]) for candidate in candidates}
    policy = {
        "name": (
            "round_robin_popularity_allow_target"
            if allow_target_in_candidates
            else "round_robin_popularity_no_target"
        ),
        "candidate_count_requested": candidate_count,
        "candidate_count_actual": len(candidates),
        "allow_target_in_candidates": allow_target_in_candidates,
        "target_in_candidates": target_item_id in candidate_ids,
        "history_item_count_in_candidates": len(history_ids & candidate_ids),
        "history_fallback_count": fallback_history_count,
        "uses_target_title": target_item_id in candidate_ids,
        "is_diagnostic_grounding_gate": True,
        "correctness_not_interpretable_without_unbiased_candidate_generation": True,
        "note": (
            "Diagnostic catalog-constrained prompt candidate set. Default policy "
            "excludes target item to prevent leakage."
        ),
    }
    return candidates, policy


def _enrich_example_from_catalog(
    example: dict[str, Any],
    *,
    catalog_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = dict(example)
    missing_history = [item_id for item_id in row["history_item_ids"] if item_id not in catalog_by_id]
    if missing_history:
        raise ValueError(f"history item ids missing from catalog: {missing_history[:5]}")
    target_item_id = str(row["target_item_id"])
    if target_item_id not in catalog_by_id:
        raise ValueError(f"target item id missing from catalog: {target_item_id}")
    target = catalog_by_id[target_item_id]
    row["history_item_titles"] = row.get("history_item_titles") or [
        catalog_by_id[item_id]["title"] for item_id in row["history_item_ids"]
    ]
    row["target_title"] = row.get("target_title") or row.get("target_item_title") or target["title"]
    row["target_item_title"] = row.get("target_item_title") or row["target_title"]
    row["target_item_popularity"] = int(
        row.get("target_item_popularity") or target.get("popularity") or 0
    )
    row["target_popularity_bucket"] = str(
        row.get("target_popularity_bucket") or target.get("popularity_bucket") or "tail"
    )
    return row


def build_observation_input_records(
    *,
    dataset: str,
    processed_suffix: str,
    split: str,
    processed_dir: str | Path,
    max_examples: int | None = None,
    stratify_by_popularity: bool = False,
    prompt_template: str = "forced_json",
    candidate_count: int | None = None,
    allow_target_in_candidates: bool = False,
) -> list[dict[str, Any]]:
    processed_dir = Path(processed_dir)
    catalog_csv = processed_dir / "item_catalog.csv"
    examples_jsonl = processed_dir / "observation_examples.jsonl"
    catalog = load_catalog_rows(catalog_csv)
    catalog_by_id = {row["item_id"]: row for row in catalog}
    examples = [
        _enrich_example_from_catalog(example, catalog_by_id=catalog_by_id)
        for example in read_jsonl(examples_jsonl)
        if str(example.get("split")) == split
    ]
    examples = _select_examples(
        _sorted_examples(examples),
        max_examples=max_examples,
        stratify_by_popularity=stratify_by_popularity,
    )
    records: list[dict[str, Any]] = []
    for example in examples:
        candidate_records: list[dict[str, Any]] = []
        candidate_policy: dict[str, Any] | None = None
        if prompt_template == CATALOG_CONSTRAINED_JSON_TEMPLATE.name:
            candidate_records, candidate_policy = _catalog_candidate_records(
                example,
                catalog_rows=catalog,
                candidate_count=candidate_count or 20,
                allow_target_in_candidates=allow_target_in_candidates,
            )
            if not candidate_records:
                raise ValueError(
                    "catalog_constrained_json requires at least one catalog candidate"
                )
            prompt = build_prompt(
                example["history_item_titles"],
                template=prompt_template,
                candidate_titles=[candidate["title"] for candidate in candidate_records],
            )
        else:
            prompt = build_prompt(example["history_item_titles"], template=prompt_template)
        prompt_hash = compute_prompt_hash(prompt)
        input_id = f"{dataset}:{processed_suffix}:{example['example_id']}:{prompt_hash[:12]}"
        record = {
            "input_id": input_id,
            "dataset": dataset,
            "processed_suffix": processed_suffix,
            "example_id": example["example_id"],
            "user_id": example["user_id"],
            "split": example["split"],
            "history_item_ids": example["history_item_ids"],
            "history_item_titles": example["history_item_titles"],
            "history_timestamps": example.get("history_timestamps", []),
            "history_length": int(example["history_length"]),
            "target_item_id": example["target_item_id"],
            "target_title": example["target_title"],
            "target_timestamp": int(example["target_timestamp"]),
            "target_popularity": int(example["target_item_popularity"]),
            "target_popularity_bucket": example["target_popularity_bucket"],
            "prompt_template": prompt_template,
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "source": {
                "processed_dir": str(processed_dir),
                "catalog_csv": str(catalog_csv),
                "observation_examples": str(examples_jsonl),
            },
        }
        if candidate_records:
            record.update(
                {
                    "catalog_candidate_item_ids": [
                        str(candidate["item_id"]) for candidate in candidate_records
                    ],
                    "catalog_candidate_titles": [
                        str(candidate["title"]) for candidate in candidate_records
                    ],
                    "catalog_candidate_popularity_buckets": [
                        str(candidate.get("popularity_bucket") or "tail")
                        for candidate in candidate_records
                    ],
                    "candidate_policy": candidate_policy,
                }
            )
        records.append(record)
    return records


def default_observation_input_path(
    *,
    dataset: str,
    processed_suffix: str,
    split: str,
    prompt_template: str,
    candidate_count: int | None = None,
    root: str | Path = ".",
) -> Path:
    stem = f"{split}_{prompt_template}"
    if prompt_template == CATALOG_CONSTRAINED_JSON_TEMPLATE.name:
        stem = f"{stem}_c{candidate_count or 20}"
    return (
        Path(root)
        / "outputs"
        / "observation_inputs"
        / dataset
        / processed_suffix
        / f"{stem}.jsonl"
    )


def write_observation_inputs(
    records: list[dict[str, Any]],
    *,
    output_jsonl: str | Path,
    dataset: str,
    processed_suffix: str,
    split: str,
    prompt_template: str,
    stratify_by_popularity: bool,
    candidate_count: int | None = None,
    allow_target_in_candidates: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_jsonl)
    write_jsonl(output_path, records)
    bucket_counts: dict[str, int] = {}
    for record in records:
        bucket = str(record["target_popularity_bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    manifest = {
        "created_at_utc": utc_now_iso(),
        "dataset": dataset,
        "processed_suffix": processed_suffix,
        "split": split,
        "prompt_template": prompt_template,
        "stratify_by_popularity": stratify_by_popularity,
        "candidate_count": candidate_count,
        "allow_target_in_candidates": allow_target_in_candidates,
        "input_count": len(records),
        "bucket_counts": bucket_counts,
        "output_jsonl": str(output_path),
        "is_experiment_result": False,
        "note": "Observation inputs only; no API or model has been run.",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def default_mock_output_dir(
    *,
    input_jsonl: str | Path,
    provider_mode: str,
    root: str | Path = ".",
) -> Path:
    input_path = Path(input_jsonl)
    parts = input_path.parts
    dataset = parts[-3] if len(parts) >= 3 else "dataset"
    processed_suffix = parts[-2] if len(parts) >= 2 else "processed"
    run_name = f"{input_path.stem}_{provider_mode}"
    return Path(root) / "outputs" / "observations" / "mock" / dataset / processed_suffix / run_name


def _completed_input_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["input_id"] for row in read_jsonl(path)}


def _candidate_dicts(candidates: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def compute_observation_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("cannot compute metrics for empty observation rows")
    probabilities = [float(row["confidence"]) for row in records]
    labels = [int(row["correctness"]) for row in records]
    buckets = [str(row["target_popularity_bucket"]) for row in records]
    grounded = [bool(row.get("grounded_item_id")) for row in records]

    bucket_metrics: dict[str, dict[str, Any]] = {}
    for bucket in ("head", "mid", "tail"):
        bucket_rows = [
            row for row in records if str(row["target_popularity_bucket"]) == bucket
        ]
        if not bucket_rows:
            bucket_metrics[bucket] = {
                "count": 0,
                "mean_confidence": None,
                "correctness_rate": None,
                "ground_hit_rate": None,
            }
            continue
        bucket_metrics[bucket] = {
            "count": len(bucket_rows),
            "mean_confidence": sum(float(row["confidence"]) for row in bucket_rows)
            / len(bucket_rows),
            "correctness_rate": sum(int(row["correctness"]) for row in bucket_rows)
            / len(bucket_rows),
            "ground_hit_rate": sum(bool(row.get("grounded_item_id")) for row in bucket_rows)
            / len(bucket_rows),
        }

    metrics = {
        "provider": "mock",
        "count": len(records),
        "ground_hit": ground_hit_rate(grounded),
        "correctness": sum(labels) / len(labels),
        "ece": expected_calibration_error(probabilities, labels, n_bins=10),
        "brier": brier_score(probabilities, labels),
        "cbu_tau": _finite_or_none(cbu_tau(probabilities, labels, tau=low_confidence_tau)),
        "wbc_tau": _finite_or_none(wbc_tau(probabilities, labels, tau=high_confidence_tau)),
        "low_confidence_tau": low_confidence_tau,
        "high_confidence_tau": high_confidence_tau,
        "tail_underconfidence_gap": _finite_or_none(
            tail_underconfidence_gap(probabilities, labels, buckets)
        ),
        "wrong_high_confidence_count": sum(
            label == 0 and prob > high_confidence_tau
            for prob, label in zip(probabilities, labels)
        ),
        "correct_low_confidence_count": sum(
            label == 1 and prob < low_confidence_tau
            for prob, label in zip(probabilities, labels)
        ),
        "bucket_metrics": bucket_metrics,
        "is_experiment_result": False,
        "note": "Mock provider sanity metrics only; not a real API/model result.",
    }
    return metrics


def observation_metrics_markdown(metrics: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "This report is generated from `provider=mock`. It is a pipeline sanity check, not an API pilot and not a paper result.",
        "",
        "## Summary",
        "",
        f"- Count: {metrics['count']}",
        f"- GroundHit: {metrics['ground_hit']:.4f}",
        f"- Correctness: {metrics['correctness']:.4f}",
        f"- ECE: {metrics['ece']:.4f}",
        f"- Brier: {metrics['brier']:.4f}",
        f"- CBU_tau: {metrics['cbu_tau']}",
        f"- WBC_tau: {metrics['wbc_tau']}",
        f"- Tail Underconfidence Gap: {metrics['tail_underconfidence_gap']}",
        f"- Wrong-high-confidence count: {metrics['wrong_high_confidence_count']}",
        f"- Correct-low-confidence count: {metrics['correct_low_confidence_count']}",
        "",
        "## Bucket Metrics",
        "",
        "| bucket | count | mean confidence | correctness | GroundHit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for bucket, bucket_metrics in metrics["bucket_metrics"].items():
        lines.append(
            "| {bucket} | {count} | {conf} | {corr} | {ground} |".format(
                bucket=bucket,
                count=bucket_metrics["count"],
                conf=bucket_metrics["mean_confidence"],
                corr=bucket_metrics["correctness_rate"],
                ground=bucket_metrics["ground_hit_rate"],
            )
        )
    return "\n".join(lines) + "\n"


def run_mock_observation(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    provider_mode: str = "popularity_biased",
    max_examples: int | None = None,
    resume: bool = True,
    seed: int = 13,
    low_confidence_tau: float = 0.5,
    high_confidence_tau: float = 0.7,
) -> dict[str, Any]:
    input_path = Path(input_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    grounded_path = output_dir / "grounded_predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"

    inputs = read_jsonl(input_path)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    if not inputs:
        raise ValueError("input_jsonl contains no records to process")

    catalog_csv = inputs[0]["source"]["catalog_csv"]
    catalog_rows = load_catalog_rows(catalog_csv)
    provider = MockProvider(catalog_rows, mode=provider_mode, seed=seed)
    grounder = TitleGrounder(catalog_records(catalog_rows))

    completed = _completed_input_ids(grounded_path) if resume else set()
    if not resume:
        raw_path.write_text("", encoding="utf-8")
        grounded_path.write_text("", encoding="utf-8")

    raw_rows: list[dict[str, Any]] = []
    grounded_rows: list[dict[str, Any]] = []
    for input_record in inputs:
        if input_record["input_id"] in completed:
            continue
        output = provider.generate(input_record)
        prediction_id = f"mock:{input_record['input_id']}"
        grounded = grounder.ground(
            output.parsed.generated_title,
            prediction_id=prediction_id,
        )
        correctness = int(grounded.is_grounded and grounded.item_id == input_record["target_item_id"])
        raw_rows.append(
            {
                "input_id": input_record["input_id"],
                "example_id": input_record["example_id"],
                "provider": "mock",
                "provider_mode": provider_mode,
                "raw_text": output.raw_text,
                "parsed": asdict(output.parsed),
                "created_at_utc": utc_now_iso(),
            }
        )
        grounded_rows.append(
            {
                "input_id": input_record["input_id"],
                "example_id": input_record["example_id"],
                "user_id": input_record["user_id"],
                "split": input_record["split"],
                "prompt_hash": input_record["prompt_hash"],
                "provider": "mock",
                "provider_mode": provider_mode,
                "generated_title": output.parsed.generated_title,
                "confidence": output.parsed.confidence,
                "is_likely_correct": output.parsed.is_likely_correct,
                "target_item_id": input_record["target_item_id"],
                "target_title": input_record["target_title"],
                "target_popularity": input_record["target_popularity"],
                "target_popularity_bucket": input_record["target_popularity_bucket"],
                "grounded_item_id": grounded.item_id,
                "grounding_status": grounded.status.value,
                "grounding_score": grounded.score,
                "grounding_ambiguity": grounded.ambiguity,
                "grounding_second_score": grounded.second_score,
                "grounding_candidates": _candidate_dicts(grounded.candidates),
                "correctness": correctness,
                "is_experiment_result": False,
            }
        )
    if raw_rows:
        write_jsonl(raw_path, raw_rows, append=resume and raw_path.exists())
    if grounded_rows:
        write_jsonl(grounded_path, grounded_rows, append=resume and grounded_path.exists())

    all_grounded = [
        row
        for row in read_jsonl(grounded_path)
        if row["input_id"] in {input_record["input_id"] for input_record in inputs}
    ]
    metrics = compute_observation_metrics(
        all_grounded,
        low_confidence_tau=low_confidence_tau,
        high_confidence_tau=high_confidence_tau,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        observation_metrics_markdown(metrics, title="Mock Observation Sanity Report"),
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        "provider": "mock",
        "provider_mode": provider_mode,
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "raw_responses": str(raw_path),
        "grounded_predictions": str(grounded_path),
        "metrics": str(metrics_path),
        "report": str(report_path),
        "requested_input_count": len(inputs),
        "newly_processed_count": len(grounded_rows),
        "total_scored_count": len(all_grounded),
        "resume": resume,
        "is_experiment_result": False,
        "note": "Mock provider run only. No external API, model training, or paper result.",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
