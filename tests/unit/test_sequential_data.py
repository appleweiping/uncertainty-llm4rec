from __future__ import annotations

from llm4rec.data.base import Interaction
from llm4rec.data.sequential import (
    build_item_index,
    pad_history_indices,
    sequential_examples_from_interactions,
    truncate_history,
)


def test_sequential_examples_sort_by_timestamp_and_exclude_target() -> None:
    interactions = [
        Interaction(user_id="u1", item_id="i3", timestamp=3),
        Interaction(user_id="u1", item_id="i1", timestamp=1),
        Interaction(user_id="u1", item_id="i2", timestamp=2),
    ]
    item_index = build_item_index(["i1", "i2", "i3"])
    examples = sequential_examples_from_interactions(
        interactions,
        item_index=item_index,
        max_history_length=2,
        min_history=1,
        domain="tiny",
    )
    assert [example.target for example in examples] == ["i2", "i3"]
    assert examples[0].history == ["i1"]
    assert examples[1].history == ["i1", "i2"]
    assert examples[1].target not in examples[1].history
    assert examples[1].metadata["timestamp_sorted"] is True


def test_sequential_max_length_padding_and_stable_indexing() -> None:
    item_index = build_item_index(["i2", "i1", "i3"])
    assert item_index == {"<PAD>": 0, "i1": 1, "i2": 2, "i3": 3}
    assert truncate_history(["i1", "i2", "i3"], max_history_length=2) == ["i2", "i3"]
    assert pad_history_indices(["i1", "i3"], item_index, max_history_length=4) == [0, 0, 1, 3]


def test_repeated_target_item_can_be_filtered_from_history() -> None:
    interactions = [
        Interaction(user_id="u1", item_id="i1", timestamp=1),
        Interaction(user_id="u1", item_id="i2", timestamp=2),
        Interaction(user_id="u1", item_id="i1", timestamp=3),
    ]
    examples = sequential_examples_from_interactions(
        interactions,
        item_index=build_item_index(["i1", "i2"]),
        max_history_length=5,
        min_history=1,
        filter_target_from_history=True,
    )
    assert examples[-1].target == "i1"
    assert "i1" not in examples[-1].history
