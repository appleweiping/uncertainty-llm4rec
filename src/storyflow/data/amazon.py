"""Amazon Reviews 2023 readiness and preprocessing helpers."""

from __future__ import annotations

import json
import gzip
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from storyflow.data.preprocessing import (
    attach_catalog_fields_to_examples,
    attach_popularity_buckets,
    build_user_sequences,
    chronological_sort,
    clean_title,
    compute_item_popularity,
    filter_users_by_interaction_count,
    k_core_filter,
    make_leave_last_splits,
    make_rolling_examples,
    truncate_items_to_interactions,
    write_csv_rows,
    write_jsonl,
)
from storyflow.grounding import normalize_title


REVIEW_FIELD_KEYS = ("user_id_field", "item_id_field", "rating_field", "timestamp_field", "review_text_field")
METADATA_FIELD_KEYS = ("metadata_join_key", "title_field")


@dataclass(frozen=True, slots=True)
class AmazonPrepareSummary:
    dataset: str
    output_dir: Path
    item_count: int
    interaction_count: int
    user_count: int
    example_count: int
    split_counts: dict[str, int]


def amazon_review_to_interaction(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert one Amazon review row to canonical interaction fields."""

    user_field = str(config.get("user_id_field") or "user_id")
    item_field = str(config.get("item_id_field") or "parent_asin")
    rating_field = str(config.get("rating_field") or "rating")
    timestamp_field = str(config.get("timestamp_field") or "timestamp")
    return {
        "user_id": str(row.get(user_field) or "").strip(),
        "item_id": str(row.get(item_field) or "").strip(),
        "rating": float(row.get(rating_field) or 0.0),
        "timestamp": int(float(row.get(timestamp_field) or 0)),
    }


def amazon_metadata_to_item(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert one Amazon metadata row to canonical item catalog fields."""

    item_field = str(config.get("metadata_join_key") or config.get("item_id_field") or "parent_asin")
    title_field = str(config.get("title_field") or "title")
    title = clean_title(str(row.get(title_field) or ""))
    metadata = {
        "category": row.get("categories"),
        "brand": row.get("store"),
    }
    return {
        "item_id": str(row.get(item_field) or "").strip(),
        "title": title,
        "title_normalized": normalize_title(title) if title else "",
        "metadata": metadata,
    }


def _candidate_paths(config: dict[str, Any], key: str) -> list[Path]:
    paths: list[Path] = []
    for candidate_key in (key, f"{key}_gz"):
        value = config.get(candidate_key)
        if value:
            paths.append(Path(str(value)))
    return paths


def resolve_existing_raw_path(config: dict[str, Any], key: str) -> Path:
    """Resolve the configured raw path, preferring existing JSONL over gzip."""

    candidates = _candidate_paths(config, key)
    for path in candidates:
        if path.exists():
            return path
    if candidates:
        return candidates[0]
    raise ValueError(f"no configured path for {key}")


def _path_status(config: dict[str, Any], key: str) -> dict[str, Any]:
    candidates = _candidate_paths(config, key)
    existing = [path for path in candidates if path.exists()]
    selected = resolve_existing_raw_path(config, key) if candidates else None
    return {
        "configured": [str(path) for path in candidates],
        "selected": str(selected) if selected else None,
        "exists": bool(existing),
        "existing": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "compressed": path.suffix == ".gz",
            }
            for path in existing
        ],
    }


def _open_text(path: str | Path):
    input_path = Path(path)
    if input_path.suffix == ".gz":
        return gzip.open(input_path, "rt", encoding="utf-8")
    return input_path.open("r", encoding="utf-8")


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    count = 0
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def _metadata_items_for_interactions(
    *,
    metadata_jsonl: str | Path,
    config: dict[str, Any],
    needed_item_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only metadata rows needed by the current interaction sample."""

    if not needed_item_ids:
        return [], {
            "needed_item_count": 0,
            "matched_item_count": 0,
            "missing_item_count": 0,
            "metadata_rows_scanned": 0,
        }
    item_field = str(config.get("metadata_join_key") or config.get("item_id_field") or "parent_asin")
    items_by_id: dict[str, dict[str, Any]] = {}
    rows_scanned = 0
    for raw in iter_jsonl(metadata_jsonl, limit=None):
        rows_scanned += 1
        raw_item_id = str(raw.get(item_field) or "").strip()
        if raw_item_id not in needed_item_ids or raw_item_id in items_by_id:
            continue
        item = amazon_metadata_to_item(raw, config)
        if item["item_id"] and item["title"]:
            items_by_id[item["item_id"]] = item
        if len(items_by_id) >= len(needed_item_ids):
            break
    metadata_stats = {
        "needed_item_count": len(needed_item_ids),
        "matched_item_count": len(items_by_id),
        "missing_item_count": max(0, len(needed_item_ids) - len(items_by_id)),
        "metadata_rows_scanned": rows_scanned,
    }
    return list(items_by_id.values()), metadata_stats


def _sample_schema(
    path: Path,
    *,
    expected_fields: Iterable[str],
    sample_records: int,
) -> dict[str, Any]:
    seen_fields: set[str] = set()
    missing_counts = {field: 0 for field in expected_fields if field}
    row_count = 0
    for row in iter_jsonl(path, limit=sample_records):
        row_count += 1
        seen_fields.update(row.keys())
        for field in missing_counts:
            if row.get(field) in (None, ""):
                missing_counts[field] += 1
    return {
        "path": str(path),
        "sample_records_requested": sample_records,
        "sample_records_read": row_count,
        "fields_seen": sorted(seen_fields),
        "missing_counts": missing_counts,
    }


def inspect_amazon_config(
    config: dict[str, Any],
    *,
    check_online: bool = False,
    sample_records: int = 0,
) -> dict[str, Any]:
    """Return an availability/readiness manifest without downloading full data."""

    manifest: dict[str, Any] = {
        "dataset": config.get("name"),
        "category_name": config.get("category_name"),
        "hf_dataset": config.get("hf_dataset"),
        "hf_review_config": config.get("hf_review_config"),
        "hf_meta_config": config.get("hf_meta_config"),
        "source_url": config.get("source_url"),
        "raw_reviews_path": config.get("raw_reviews_path"),
        "raw_reviews_path_gz": config.get("raw_reviews_path_gz"),
        "raw_metadata_path": config.get("raw_metadata_path"),
        "raw_metadata_path_gz": config.get("raw_metadata_path_gz"),
        "sample_records": sample_records,
        "full_download_attempted": False,
        "full_processed": False,
        "status": "dry_run_ready",
        "warnings": [],
        "resume_command": f"python scripts/inspect_amazon_reviews_2023.py --dataset {config.get('name')} --check-online",
        "full_mode_command": config.get("full_mode_command_template"),
    }
    reviews_status = _path_status(config, "raw_reviews_path")
    metadata_status = _path_status(config, "raw_metadata_path")
    manifest["raw_reviews"] = reviews_status
    manifest["raw_metadata"] = metadata_status
    manifest["raw_reviews_path_exists"] = reviews_status["exists"]
    manifest["raw_metadata_path_exists"] = metadata_status["exists"]
    if reviews_status["exists"] and metadata_status["exists"]:
        manifest["status"] = "local_raw_available"
    else:
        manifest["status"] = "dry_run_ready"
        manifest["warnings"].append("Local raw review or metadata JSONL is missing.")
    if sample_records > 0:
        if reviews_status["exists"]:
            review_path = Path(str(reviews_status["selected"]))
            expected_review_fields = [
                str(config.get(key))
                for key in REVIEW_FIELD_KEYS
                if config.get(key)
            ]
            manifest["review_schema_sample"] = _sample_schema(
                review_path,
                expected_fields=expected_review_fields,
                sample_records=sample_records,
            )
        if metadata_status["exists"]:
            metadata_path = Path(str(metadata_status["selected"]))
            expected_metadata_fields = [
                str(config.get(key))
                for key in METADATA_FIELD_KEYS
                if config.get(key)
            ]
            manifest["metadata_schema_sample"] = _sample_schema(
                metadata_path,
                expected_fields=expected_metadata_fields,
                sample_records=sample_records,
            )
    if check_online:
        url = (
            "https://datasets-server.huggingface.co/splits?dataset="
            + str(config.get("hf_dataset"))
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            manifest["online_check"] = {
                "status": "ok",
                "url": url,
                "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            manifest["status"] = "online_check_failed"
            manifest["online_check"] = {
                "status": "failed",
                "url": url,
                "error": str(exc),
            }
            manifest["warnings"].append("Hugging Face availability check failed; no success was claimed.")
    return manifest


def write_amazon_readiness_report_legacy_mojibake(manifest: dict[str, Any], path: str | Path) -> None:
    lines = [
        f"# Amazon Reviews 2023 readiness: {manifest.get('dataset')}",
        "",
        "本报告只说明入口和可恢复状态，不表示已经下载或处理 full data。",
        "",
        f"- 数据集: {manifest.get('hf_dataset')}",
        f"- 类别: {manifest.get('category_name')}",
        f"- 状态: {manifest.get('status')}",
        f"- full download attempted: {manifest.get('full_download_attempted')}",
        f"- full processed: {manifest.get('full_processed')}",
        f"- reviews path exists: {manifest.get('raw_reviews_path_exists')}",
        f"- metadata path exists: {manifest.get('raw_metadata_path_exists')}",
        f"- 恢复/检查命令: `{manifest.get('resume_command')}`",
        f"- full mode 命令模板: `{manifest.get('full_mode_command')}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = manifest.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings or ["None"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_amazon_readiness_report(manifest: dict[str, Any], path: str | Path) -> None:
    lines = [
        f"# Amazon Reviews 2023 readiness: {manifest.get('dataset')}",
        "",
        "本报告只说明入口和可恢复状态，不表示已经下载或处理 full data。",
        "",
        f"- 数据集: {manifest.get('hf_dataset')}",
        f"- 类别: {manifest.get('category_name')}",
        f"- 状态: {manifest.get('status')}",
        f"- full download attempted: {manifest.get('full_download_attempted')}",
        f"- full processed: {manifest.get('full_processed')}",
        f"- reviews path exists: {manifest.get('raw_reviews_path_exists')}",
        f"- metadata path exists: {manifest.get('raw_metadata_path_exists')}",
        f"- 恢复/检查命令: `{manifest.get('resume_command')}`",
        f"- full mode 命令模板: `{manifest.get('full_mode_command')}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = manifest.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings or ["None"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_amazon_from_jsonl(
    *,
    config: dict[str, Any],
    reviews_jsonl: str | Path,
    metadata_jsonl: str | Path,
    output_suffix: str,
    max_records: int | None = None,
) -> AmazonPrepareSummary:
    """Prepare Amazon JSONL files into the Storyflow processed schema."""

    min_user_interactions = int(config.get("preprocess_min_user_interactions") or 4)
    user_k_core = int(config.get("preprocess_user_k_core") or 5)
    item_k_core = int(config.get("preprocess_item_k_core") or 5)
    min_history = int(config.get("preprocess_min_history") or 3)
    max_history = int(config.get("preprocess_max_history") or 50)
    split_policy = str(config.get("preprocess_split_policy") or "global_chronological")

    interactions = [
        row
        for row in (
            amazon_review_to_interaction(raw, config)
            for raw in iter_jsonl(reviews_jsonl, limit=max_records)
        )
        if row["user_id"] and row["item_id"] and row["timestamp"] > 0
    ]
    interactions = filter_users_by_interaction_count(
        interactions,
        min_interactions=min_user_interactions,
    )
    interactions = k_core_filter(
        interactions,
        user_k=user_k_core,
        item_k=item_k_core,
    )
    interactions = chronological_sort(interactions)
    needed_item_ids = {str(row["item_id"]) for row in interactions}
    metadata_items, metadata_stats = _metadata_items_for_interactions(
        metadata_jsonl=metadata_jsonl,
        config=config,
        needed_item_ids=needed_item_ids,
    )
    available_item_ids = {str(item["item_id"]) for item in metadata_items}
    interactions = [
        row
        for row in interactions
        if str(row["item_id"]) in available_item_ids
    ]
    interactions = chronological_sort(interactions)
    items = truncate_items_to_interactions(metadata_items, interactions)
    popularity = compute_item_popularity(interactions)
    items = attach_popularity_buckets(
        items,
        popularity,
        head_fraction=float(config.get("head_fraction") or 0.2),
        tail_fraction=float(config.get("tail_fraction") or 0.2),
    )
    sequences = build_user_sequences(interactions)
    if split_policy == "global_chronological":
        examples = make_rolling_examples(
            sequences,
            min_history=min_history,
            max_history=max_history,
        )
        from storyflow.data.preprocessing import assign_global_chronological_splits

        examples = assign_global_chronological_splits(examples)
    else:
        examples = make_leave_last_splits(
            sequences,
            min_history=min_history,
            max_history=max_history,
            leave_last_n=2,
        )
    examples = attach_catalog_fields_to_examples(examples, items)
    output_dir = Path(str(config["processed_dir"])) / output_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "item_catalog.csv",
        sorted(items, key=lambda row: row["item_id"]),
        ["item_id", "title", "title_normalized", "popularity", "popularity_bucket"],
    )
    write_csv_rows(
        output_dir / "interactions.csv",
        interactions,
        ["user_id", "item_id", "rating", "timestamp"],
    )
    write_csv_rows(
        output_dir / "item_popularity.csv",
        [
            {"item_id": item_id, "popularity": count}
            for item_id, count in sorted(popularity.items())
        ],
        ["item_id", "popularity"],
    )
    write_jsonl(output_dir / "user_sequences.jsonl", sequences)
    write_jsonl(output_dir / "observation_examples.jsonl", examples)
    split_counts = dict(Counter(example["split"] for example in examples))
    is_sample_result = max_records is not None
    manifest = {
        "dataset": config.get("name"),
        "category_name": config.get("category_name"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_suffix": output_suffix,
        "split_policy": split_policy,
        "min_user_interactions": min_user_interactions,
        "user_k_core": user_k_core,
        "item_k_core": item_k_core,
        "min_history": min_history,
        "max_history": max_history,
        "item_count": len(items),
        "interaction_count": len(interactions),
        "user_count": len(sequences),
        "example_count": len(examples),
        "split_counts": split_counts,
        "source_reviews_jsonl": str(reviews_jsonl),
        "source_metadata_jsonl": str(metadata_jsonl),
        "max_records": max_records,
        "is_full_result": max_records is None,
        "is_sample_result": is_sample_result,
        "is_experiment_result": False,
        "result_scope": "local_sample" if is_sample_result else "full_prepare_entry",
        "claim_note": (
            "Local sample/sanity output only; not a paper result."
            if is_sample_result
            else "Full prepare output requires explicit run approval and manifest review before any claim."
        ),
        "metadata_stats": metadata_stats,
        "config_snapshot": {
            "name": config.get("name"),
            "category_name": config.get("category_name"),
            "source_name": config.get("source_name"),
            "source_url": config.get("source_url"),
            "raw_reviews_path": config.get("raw_reviews_path"),
            "raw_metadata_path": config.get("raw_metadata_path"),
            "processed_dir": config.get("processed_dir"),
            "preprocess_min_user_interactions": config.get("preprocess_min_user_interactions"),
            "preprocess_user_k_core": config.get("preprocess_user_k_core"),
            "preprocess_item_k_core": config.get("preprocess_item_k_core"),
            "preprocess_min_history": config.get("preprocess_min_history"),
            "preprocess_max_history": config.get("preprocess_max_history"),
            "preprocess_split_policy": config.get("preprocess_split_policy"),
            "head_fraction": config.get("head_fraction"),
            "tail_fraction": config.get("tail_fraction"),
        },
        "outputs": {
            "item_catalog": str(output_dir / "item_catalog.csv"),
            "interactions": str(output_dir / "interactions.csv"),
            "item_popularity": str(output_dir / "item_popularity.csv"),
            "user_sequences": str(output_dir / "user_sequences.jsonl"),
            "observation_examples": str(output_dir / "observation_examples.jsonl"),
        },
    }
    (output_dir / "preprocess_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return AmazonPrepareSummary(
        dataset=str(config.get("name")),
        output_dir=output_dir,
        item_count=len(items),
        interaction_count=len(interactions),
        user_count=len(sequences),
        example_count=len(examples),
        split_counts=split_counts,
    )
