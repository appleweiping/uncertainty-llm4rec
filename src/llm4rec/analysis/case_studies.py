"""Case-study exports for R3 OursMethod artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from llm4rec.analysis.ours_error_decomposition import (
    DEFAULT_SEEDS,
    FALLBACK_METHOD,
    OURS_METHOD,
    hit_at_k,
    load_dataset_context,
    load_method_predictions,
    method_run_dir,
    write_csv,
)

CASE_FILES = {
    "high_conf_wrong": "r3_case_studies_high_conf_wrong.csv",
    "accept_hurts": "r3_case_studies_accept_hurts.csv",
    "accept_helps": "r3_case_studies_accept_helps.csv",
    "fallback_saves": "r3_case_studies_fallback_saves.csv",
    "tail_underconfident": "r3_case_studies_tail_underconfident.csv",
}

CASE_COLUMNS = [
    "seed",
    "user_id",
    "example_id",
    "history_titles",
    "target_title",
    "generated_title",
    "grounded_item_id",
    "grounded_title",
    "confidence",
    "decision",
    "fallback_top10",
    "ours_top10",
    "fallback_hit@10",
    "ours_hit@10",
    "popularity_bucket",
    "raw_output_reference",
    "cache_key",
    "prompt_hash",
]


def export_case_studies(
    runs_dir: str | Path,
    *,
    output_dir: str | Path,
    processed_dir: str | Path | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    max_rows: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    context = load_dataset_context(processed_dir)
    cases: dict[str, list[dict[str, Any]]] = {key: [] for key in CASE_FILES}
    for seed in seeds:
        ours = load_method_predictions(runs_dir, OURS_METHOD, int(seed))
        fallback = load_method_predictions(runs_dir, FALLBACK_METHOD, int(seed))
        fallback_by_key = {_row_key(row): row for row in fallback}
        raw_ref = str(method_run_dir(runs_dir, OURS_METHOD, int(seed)) / "raw_llm_outputs.jsonl")
        for row in ours:
            fallback_row = fallback_by_key.get(_row_key(row))
            if fallback_row is None:
                continue
            case = _case_row(row, fallback_row, seed=int(seed), context=context, raw_ref=raw_ref)
            decision = case["decision"]
            confidence = float(case["confidence"] or 0.0)
            ours_hit = int(case["ours_hit@10"])
            fallback_hit = int(case["fallback_hit@10"])
            bucket = str(case["popularity_bucket"])
            grounded_id = str(case["grounded_item_id"] or "")
            target_id = str(row.get("target_item") or "")
            if confidence >= 0.85 and grounded_id != target_id:
                _append_limited(cases["high_conf_wrong"], case, max_rows)
            if decision == "accept" and fallback_hit == 1 and ours_hit == 0:
                _append_limited(cases["accept_hurts"], case, max_rows)
            if decision == "accept" and fallback_hit == 0 and ours_hit == 1:
                _append_limited(cases["accept_helps"], case, max_rows)
            if decision == "fallback" and fallback_hit == 1 and grounded_id != target_id:
                _append_limited(cases["fallback_saves"], case, max_rows)
            if bucket == "tail" and confidence < 0.7:
                _append_limited(cases["tail_underconfident"], case, max_rows)
    output = Path(output_dir)
    for key, filename in CASE_FILES.items():
        write_csv(output / filename, cases[key], fieldnames=CASE_COLUMNS)
    return cases


def _append_limited(rows: list[dict[str, Any]], row: dict[str, Any], max_rows: int) -> None:
    if len(rows) < max_rows:
        rows.append(row)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    return (str(metadata.get("example_id") or ""), str(row.get("user_id")), str(row.get("target_item")))


def _case_row(
    row: dict[str, Any],
    fallback_row: dict[str, Any],
    *,
    seed: int,
    context: dict[str, Any],
    raw_ref: str,
) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    item_titles = context.get("item_titles") or {}
    item_buckets = context.get("item_buckets") or {}
    example = (context.get("examples_by_id") or {}).get(str(metadata.get("example_id") or ""), {})
    history_ids = metadata.get("history_item_ids") or example.get("history") or []
    history_titles = metadata.get("history_titles") or [item_titles.get(str(item), str(item)) for item in history_ids]
    target = str(row.get("target_item") or "")
    grounded = metadata.get("grounded_item_id")
    provider_metadata = metadata.get("provider_metadata") or {}
    return {
        "seed": seed,
        "user_id": row.get("user_id"),
        "example_id": metadata.get("example_id"),
        "history_titles": " | ".join(str(title) for title in history_titles[:25]),
        "target_title": item_titles.get(target, target),
        "generated_title": metadata.get("generated_title"),
        "grounded_item_id": grounded,
        "grounded_title": metadata.get("grounded_title") or item_titles.get(str(grounded), ""),
        "confidence": metadata.get("confidence"),
        "decision": metadata.get("uncertainty_decision"),
        "fallback_top10": " | ".join(item_titles.get(str(item), str(item)) for item in fallback_row.get("predicted_items", [])[:10]),
        "ours_top10": " | ".join(item_titles.get(str(item), str(item)) for item in row.get("predicted_items", [])[:10]),
        "fallback_hit@10": hit_at_k(fallback_row),
        "ours_hit@10": hit_at_k(row),
        "popularity_bucket": item_buckets.get(target, "unknown"),
        "raw_output_reference": raw_ref,
        "cache_key": provider_metadata.get("cache_key"),
        "prompt_hash": metadata.get("prompt_hash"),
    }

