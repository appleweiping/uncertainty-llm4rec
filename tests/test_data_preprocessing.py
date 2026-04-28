from __future__ import annotations

from pathlib import Path

from storyflow.data import (
    assign_global_chronological_splits,
    build_user_sequences,
    chronological_sort,
    compute_item_popularity,
    filter_users_by_interaction_count,
    k_core_filter,
    make_leave_last_splits,
    make_rolling_examples,
    read_interactions_csv,
    read_item_catalog_csv,
)
from storyflow.data.preprocessing import attach_popularity_buckets
from storyflow.schemas import PopularityBucket

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_read_tiny_csv_and_clean_titles() -> None:
    items = read_item_catalog_csv(FIXTURES / "tiny_items.csv")
    interactions = read_interactions_csv(FIXTURES / "tiny_interactions.csv")

    assert items[0]["title"] == "Alpha Movie"
    assert items[0]["title_normalized"] == "alpha movie"
    assert len(interactions) == 14


def test_filter_users_by_interaction_count_keeps_users_with_more_than_three() -> None:
    interactions = read_interactions_csv(FIXTURES / "tiny_interactions.csv")
    filtered = filter_users_by_interaction_count(interactions, min_interactions=4)

    assert {row["user_id"] for row in filtered} == {"u1", "u2", "u3"}
    assert all(row["user_id"] != "u4" for row in filtered)


def test_k_core_filter_iterates_users_and_items() -> None:
    interactions = read_interactions_csv(FIXTURES / "tiny_interactions.csv")
    filtered = k_core_filter(interactions, user_k=3, item_k=3)

    assert {row["user_id"] for row in filtered} == {"u1", "u2", "u3"}
    assert {row["item_id"] for row in filtered} == {"i1", "i2", "i3", "i4"}


def test_chronological_sort_orders_within_each_user() -> None:
    interactions = read_interactions_csv(FIXTURES / "tiny_interactions.csv")
    sorted_rows = chronological_sort(interactions)
    u1_rows = [row for row in sorted_rows if row["user_id"] == "u1"]

    assert [row["item_id"] for row in u1_rows] == ["i2", "i1", "i3", "i4"]
    assert [row["timestamp"] for row in u1_rows] == [90, 100, 110, 120]


def test_leave_last_two_split_builds_train_val_test_examples() -> None:
    interactions = chronological_sort(
        filter_users_by_interaction_count(
            read_interactions_csv(FIXTURES / "tiny_interactions.csv"),
            min_interactions=4,
        )
    )
    sequences = build_user_sequences(interactions)
    examples = make_leave_last_splits(
        sequences,
        min_history=1,
        max_history=3,
        leave_last_n=2,
    )

    split_counts = {split: 0 for split in ("train", "val", "test")}
    for example in examples:
        split_counts[example["split"]] += 1
    assert split_counts == {"train": 3, "val": 3, "test": 3}
    assert all(len(example["history_item_ids"]) >= 1 for example in examples)


def test_rolling_examples_and_global_chronological_split() -> None:
    interactions = chronological_sort(
        filter_users_by_interaction_count(
            read_interactions_csv(FIXTURES / "tiny_interactions.csv"),
            min_interactions=4,
        )
    )
    sequences = build_user_sequences(interactions)
    rolling = make_rolling_examples(sequences, min_history=2, max_history=2)
    split = assign_global_chronological_splits(
        rolling,
        train_fraction=0.5,
        val_fraction=0.25,
    )

    assert len(rolling) == 6
    assert all(len(example["history_item_ids"]) == 2 for example in rolling)
    assert [example["split"] for example in split].count("train") == 3
    assert [example["split"] for example in split].count("val") == 1
    assert [example["split"] for example in split].count("test") == 2


def test_popularity_bucket_attachment_uses_processed_popularity() -> None:
    items = read_item_catalog_csv(FIXTURES / "tiny_items.csv")
    interactions = read_interactions_csv(FIXTURES / "tiny_interactions.csv")
    popularity = compute_item_popularity(interactions)
    enriched = attach_popularity_buckets(
        items,
        popularity,
        head_fraction=0.2,
        tail_fraction=0.2,
    )
    by_id = {item["item_id"]: item for item in enriched}

    assert by_id["i1"]["popularity_bucket"] == PopularityBucket.HEAD.value
    assert by_id["i5"]["popularity_bucket"] == PopularityBucket.TAIL.value
