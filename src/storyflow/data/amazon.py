"""Amazon Reviews 2023 readiness and preprocessing helpers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
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


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


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
        "raw_metadata_path": config.get("raw_metadata_path"),
        "sample_records": sample_records,
        "full_download_attempted": False,
        "full_processed": False,
        "status": "dry_run_ready",
        "warnings": [],
        "resume_command": f"python scripts/inspect_amazon_reviews_2023.py --dataset {config.get('name')} --check-online",
        "full_mode_command": config.get("full_mode_command_template"),
    }
    for path_key in ("raw_reviews_path", "raw_metadata_path"):
        path = Path(str(config.get(path_key) or ""))
        manifest[f"{path_key}_exists"] = path.exists()
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

    interactions = [
        row
        for row in (
            amazon_review_to_interaction(raw, config)
            for raw in iter_jsonl(reviews_jsonl, limit=max_records)
        )
        if row["user_id"] and row["item_id"] and row["timestamp"] > 0
    ]
    metadata_items = [
        item
        for item in (
            amazon_metadata_to_item(raw, config)
            for raw in iter_jsonl(metadata_jsonl, limit=None)
        )
        if item["item_id"] and item["title"]
    ]
    interactions = filter_users_by_interaction_count(
        interactions,
        min_interactions=int(config.get("preprocess_min_user_interactions") or 4),
    )
    interactions = k_core_filter(
        interactions,
        user_k=int(config.get("preprocess_user_k_core") or 5),
        item_k=int(config.get("preprocess_item_k_core") or 5),
    )
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
    min_history = int(config.get("preprocess_min_history") or 3)
    max_history = int(config.get("preprocess_max_history") or 50)
    split_policy = str(config.get("preprocess_split_policy") or "global_chronological")
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
    write_jsonl(output_dir / "user_sequences.jsonl", sequences)
    write_jsonl(output_dir / "observation_examples.jsonl", examples)
    split_counts = dict(Counter(example["split"] for example in examples))
    manifest = {
        "dataset": config.get("name"),
        "category_name": config.get("category_name"),
        "split_policy": split_policy,
        "item_count": len(items),
        "interaction_count": len(interactions),
        "user_count": len(sequences),
        "example_count": len(examples),
        "split_counts": split_counts,
        "source_reviews_jsonl": str(reviews_jsonl),
        "source_metadata_jsonl": str(metadata_jsonl),
        "max_records": max_records,
        "is_full_result": max_records is None,
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
