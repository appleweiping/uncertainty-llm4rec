"""Dataset preprocessing utilities for Storyflow / TRUCE-Rec."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from storyflow.grounding import normalize_title
from storyflow.metrics import assign_popularity_buckets
from storyflow.schemas import PopularityBucket

_SPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class DatasetPreparationError(RuntimeError):
    """Raised when a dataset cannot be prepared from available local files."""


@dataclass(frozen=True, slots=True)
class PrepareResult:
    dataset: str
    output_dir: Path
    item_count: int
    interaction_count: int
    user_count: int
    example_count: int
    split_counts: dict[str, int]


def clean_title(title: str) -> str:
    """Clean title text while preserving human-readable casing."""

    title = _CONTROL_RE.sub(" ", str(title))
    return _SPACE_RE.sub(" ", title).strip()


def read_interactions_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical interaction CSV rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "user_id": str(row["user_id"]),
                    "item_id": str(row["item_id"]),
                    "rating": float(row.get("rating") or 0.0),
                    "timestamp": int(float(row.get("timestamp") or 0)),
                }
            )
    return rows


def read_item_catalog_csv(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = clean_title(row["title"])
            rows.append(
                {
                    "item_id": str(row["item_id"]),
                    "title": title,
                    "title_normalized": normalize_title(title),
                    "genres": row.get("genres", ""),
                }
            )
    return rows


def read_movielens_1m(extracted_dir: str | Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Read MovieLens 1M ratings.dat and movies.dat from an extracted archive."""

    root = Path(extracted_dir)
    ratings_path = root / "ratings.dat"
    movies_path = root / "movies.dat"
    if not ratings_path.exists() or not movies_path.exists():
        raise DatasetPreparationError(
            f"MovieLens 1M files not found under {root}. Expected ratings.dat "
            "and movies.dat. Run scripts/download_datasets.py first."
        )

    items: list[dict[str, Any]] = []
    with movies_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            movie_id, title, genres = line.rstrip("\n").split("::", 2)
            cleaned = clean_title(title)
            items.append(
                {
                    "item_id": movie_id,
                    "title": cleaned,
                    "title_normalized": normalize_title(cleaned),
                    "genres": genres,
                }
            )

    interactions: list[dict[str, Any]] = []
    with ratings_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            user_id, movie_id, rating, timestamp = line.rstrip("\n").split("::")
            interactions.append(
                {
                    "user_id": user_id,
                    "item_id": movie_id,
                    "rating": float(rating),
                    "timestamp": int(timestamp),
                }
            )
    return items, interactions


def filter_users_by_interaction_count(
    interactions: Iterable[dict[str, Any]],
    *,
    min_interactions: int,
) -> list[dict[str, Any]]:
    """Keep users with at least min_interactions records."""

    rows = list(interactions)
    counts = Counter(row["user_id"] for row in rows)
    return [
        row
        for row in rows
        if counts[row["user_id"]] >= min_interactions
    ]


def k_core_filter(
    interactions: Iterable[dict[str, Any]],
    *,
    user_k: int,
    item_k: int,
    max_iterations: int = 100,
) -> list[dict[str, Any]]:
    """Iteratively keep users/items that satisfy k-core thresholds."""

    rows = list(interactions)
    if user_k < 1 or item_k < 1:
        raise ValueError("user_k and item_k must be >= 1")
    for _ in range(max_iterations):
        user_counts = Counter(row["user_id"] for row in rows)
        item_counts = Counter(row["item_id"] for row in rows)
        filtered = [
            row
            for row in rows
            if user_counts[row["user_id"]] >= user_k
            and item_counts[row["item_id"]] >= item_k
        ]
        if len(filtered) == len(rows):
            return filtered
        rows = filtered
    raise ValueError("k-core filtering did not converge")


def chronological_sort(
    interactions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        interactions,
        key=lambda row: (
            str(row["user_id"]),
            int(row["timestamp"]),
            str(row["item_id"]),
        ),
    )


def compute_item_popularity(
    interactions: Iterable[dict[str, Any]],
) -> dict[str, int]:
    return dict(Counter(row["item_id"] for row in interactions))


def attach_popularity_buckets(
    items: Iterable[dict[str, Any]],
    popularity: dict[str, int],
    *,
    head_fraction: float,
    tail_fraction: float,
) -> list[dict[str, Any]]:
    if not popularity:
        return [dict(item, popularity=0, popularity_bucket="tail") for item in items]
    buckets = assign_popularity_buckets(
        popularity,
        head_fraction=head_fraction,
        tail_fraction=tail_fraction,
    )
    enriched = []
    for item in items:
        item_id = item["item_id"]
        bucket = buckets.get(item_id, PopularityBucket.TAIL)
        enriched.append(
            {
                **item,
                "popularity": popularity.get(item_id, 0),
                "popularity_bucket": bucket.value,
            }
        )
    return enriched


def build_user_sequences(
    interactions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chronological_sort(interactions):
        grouped[row["user_id"]].append(row)

    sequences = []
    for user_id in sorted(grouped):
        rows = grouped[user_id]
        sequences.append(
            {
                "user_id": user_id,
                "item_ids": [row["item_id"] for row in rows],
                "timestamps": [int(row["timestamp"]) for row in rows],
                "ratings": [float(row.get("rating", 0.0)) for row in rows],
            }
        )
    return sequences


def _history_for(
    sequence: dict[str, Any],
    target_index: int,
    *,
    max_history: int | None,
) -> list[str]:
    start = _history_start_index(target_index, max_history=max_history)
    return sequence["item_ids"][start:target_index]


def _history_start_index(target_index: int, *, max_history: int | None) -> int:
    return 0 if max_history is None else max(0, target_index - max_history)


def make_rolling_examples(
    sequences: Iterable[dict[str, Any]],
    *,
    min_history: int,
    max_history: int | None,
) -> list[dict[str, Any]]:
    """Create examples with target index k in [min_history, n - 1]."""

    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    examples: list[dict[str, Any]] = []
    for sequence in sequences:
        item_ids = sequence["item_ids"]
        for target_index in range(min_history, len(item_ids)):
            examples.append(
                _example_from_sequence(sequence, target_index, max_history, "unassigned")
            )
    return examples


def make_leave_last_splits(
    sequences: Iterable[dict[str, Any]],
    *,
    min_history: int,
    max_history: int | None,
    leave_last_n: int,
) -> list[dict[str, Any]]:
    """Create per-user train/validation/test splits."""

    if leave_last_n not in {1, 2}:
        raise ValueError("leave_last_n must be 1 or 2")
    examples: list[dict[str, Any]] = []
    for sequence in sequences:
        item_ids = sequence["item_ids"]
        if len(item_ids) <= min_history:
            continue
        n_items = len(item_ids)
        holdout_start = max(min_history, n_items - leave_last_n)
        for target_index in range(min_history, holdout_start):
            split = "train"
            examples.append(_example_from_sequence(sequence, target_index, max_history, split))
        if leave_last_n == 2 and n_items >= min_history + 2:
            examples.append(
                _example_from_sequence(sequence, n_items - 2, max_history, "val")
            )
        examples.append(
            _example_from_sequence(sequence, n_items - 1, max_history, "test")
        )
    return examples


def _example_from_sequence(
    sequence: dict[str, Any],
    target_index: int,
    max_history: int | None,
    split: str,
) -> dict[str, Any]:
    history_start = _history_start_index(target_index, max_history=max_history)
    history_item_ids = sequence["item_ids"][history_start:target_index]
    history_timestamps = sequence["timestamps"][history_start:target_index]
    return {
        "example_id": f"{sequence['user_id']}:{target_index}",
        "user_id": sequence["user_id"],
        "history_item_ids": history_item_ids,
        "history_timestamps": history_timestamps,
        "target_item_id": sequence["item_ids"][target_index],
        "target_timestamp": sequence["timestamps"][target_index],
        "target_index": target_index,
        "history_start_index": history_start,
        "history_end_index": target_index - 1,
        "history_length": len(history_item_ids),
        "split": split,
    }


def attach_catalog_fields_to_examples(
    examples: Iterable[dict[str, Any]],
    items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach title and popularity fields needed by observation pipelines."""

    item_by_id = {str(item["item_id"]): item for item in items}
    enriched: list[dict[str, Any]] = []
    for example in examples:
        target = item_by_id.get(str(example["target_item_id"]))
        history_items = [
            item_by_id[item_id]
            for item_id in example["history_item_ids"]
            if item_id in item_by_id
        ]
        row = dict(example)
        row["history_item_titles"] = [item["title"] for item in history_items]
        if target is not None:
            row["target_item_title"] = target["title"]
            row["target_title"] = target["title"]
            row["target_item_popularity"] = int(target.get("popularity", 0) or 0)
            row["target_popularity_bucket"] = str(
                target.get("popularity_bucket", PopularityBucket.TAIL.value)
            )
            row["item_popularity"] = row["target_item_popularity"]
            row["popularity_bucket"] = row["target_popularity_bucket"]
        else:
            row["target_item_title"] = ""
            row["target_title"] = ""
            row["target_item_popularity"] = 0
            row["target_popularity_bucket"] = PopularityBucket.TAIL.value
            row["item_popularity"] = 0
            row["popularity_bucket"] = PopularityBucket.TAIL.value
        enriched.append(row)
    return enriched


def assign_global_chronological_splits(
    examples: Iterable[dict[str, Any]],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> list[dict[str, Any]]:
    rows = sorted(
        (dict(example) for example in examples),
        key=lambda row: (int(row["target_timestamp"]), row["user_id"], row["example_id"]),
    )
    if not rows:
        return []
    train_cut = int(len(rows) * train_fraction)
    val_cut = int(len(rows) * (train_fraction + val_fraction))
    for index, row in enumerate(rows):
        if index < train_cut:
            row["split"] = "train"
        elif index < val_cut:
            row["split"] = "val"
        else:
            row["split"] = "test"
    return rows


def truncate_items_to_interactions(
    items: Iterable[dict[str, Any]],
    interactions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    used_item_ids = {row["item_id"] for row in interactions}
    return [item for item in items if item["item_id"] in used_item_ids]


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_movielens_1m(
    config: dict[str, Any],
    *,
    split_policy: str | None = None,
    min_user_interactions: int | None = None,
    user_k_core: int | None = None,
    item_k_core: int | None = None,
    min_history: int | None = None,
    max_history: int | None = None,
    max_users: int | None = None,
    output_suffix: str | None = None,
) -> PrepareResult:
    """Prepare MovieLens 1M into catalog, interactions, sequences, examples."""

    dataset = str(config["name"])
    raw_root = Path(str(config["raw_dir"]))
    extracted_dir = raw_root / str(config.get("expected_archive_dir") or "ml-1m")
    items, interactions = read_movielens_1m(extracted_dir)

    min_user_interactions = int(
        min_user_interactions or config.get("preprocess_min_user_interactions") or 4
    )
    user_k_core = int(user_k_core or config.get("preprocess_user_k_core") or 1)
    item_k_core = int(item_k_core or config.get("preprocess_item_k_core") or 1)
    min_history = int(min_history or config.get("preprocess_min_history") or 3)
    max_history = int(max_history or config.get("preprocess_max_history") or 50)
    split_policy = str(split_policy or config.get("preprocess_split_policy") or "leave_last_two")

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
    if max_users is not None:
        keep_users = {
            user_id
            for user_id in sorted({row["user_id"] for row in interactions})[:max_users]
        }
        interactions = [row for row in interactions if row["user_id"] in keep_users]
        interactions = k_core_filter(
            interactions,
            user_k=max(1, min(user_k_core, min_user_interactions)),
            item_k=1,
        )
        interactions = chronological_sort(interactions)

    items = truncate_items_to_interactions(items, interactions)
    popularity = compute_item_popularity(interactions)
    items = attach_popularity_buckets(
        items,
        popularity,
        head_fraction=float(config.get("head_fraction") or 0.2),
        tail_fraction=float(config.get("tail_fraction") or 0.2),
    )
    sequences = build_user_sequences(interactions)
    if split_policy == "leave_last_one":
        examples = make_leave_last_splits(
            sequences,
            min_history=min_history,
            max_history=max_history,
            leave_last_n=1,
        )
    elif split_policy == "leave_last_two":
        examples = make_leave_last_splits(
            sequences,
            min_history=min_history,
            max_history=max_history,
            leave_last_n=2,
        )
    elif split_policy == "global_chronological":
        rolling = make_rolling_examples(
            sequences,
            min_history=min_history,
            max_history=max_history,
        )
        examples = assign_global_chronological_splits(rolling)
    else:
        raise ValueError(f"unknown split_policy: {split_policy}")
    examples = attach_catalog_fields_to_examples(examples, items)

    processed_root = Path(str(config["processed_dir"]))
    run_name = output_suffix or split_policy
    output_dir = processed_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv_rows(
        output_dir / "item_catalog.csv",
        sorted(items, key=lambda row: row["item_id"]),
        [
            "item_id",
            "title",
            "title_normalized",
            "genres",
            "popularity",
            "popularity_bucket",
        ],
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
    manifest = {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_policy": split_policy,
        "min_user_interactions": min_user_interactions,
        "user_k_core": user_k_core,
        "item_k_core": item_k_core,
        "min_history": min_history,
        "max_history": max_history,
        "max_users": max_users,
        "item_count": len(items),
        "interaction_count": len(interactions),
        "user_count": len(sequences),
        "example_count": len(examples),
        "split_counts": split_counts,
        "config_snapshot": {
            "name": config.get("name"),
            "type": config.get("type"),
            "source_name": config.get("source_name"),
            "source_url": config.get("source_url"),
            "raw_dir": config.get("raw_dir"),
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
    return PrepareResult(
        dataset=dataset,
        output_dir=output_dir,
        item_count=len(items),
        interaction_count=len(interactions),
        user_count=len(sequences),
        example_count=len(examples),
        split_counts=split_counts,
    )
