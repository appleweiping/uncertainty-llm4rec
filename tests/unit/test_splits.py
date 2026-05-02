from __future__ import annotations

from llm4rec.data.base import Interaction
from llm4rec.data.splits import leave_one_out_split, temporal_split


def _interactions() -> list[Interaction]:
    return [
        Interaction("u1", "i1", 3),
        Interaction("u1", "i2", 1),
        Interaction("u1", "i3", 2),
        Interaction("u1", "i4", 4),
        Interaction("u2", "i1", 1),
        Interaction("u2", "i2", 2),
        Interaction("u2", "i3", 3),
    ]


def test_leave_one_out_is_chronological_and_deterministic() -> None:
    examples = leave_one_out_split(_interactions(), min_history=1, domain="tiny")
    rows = [(example.user_id, example.split, example.history, example.target) for example in examples]
    assert rows == [
        ("u1", "train", ["i2"], "i3"),
        ("u1", "valid", ["i2", "i3"], "i1"),
        ("u1", "test", ["i2", "i3", "i1"], "i4"),
        ("u2", "valid", ["i1"], "i2"),
        ("u2", "test", ["i1", "i2"], "i3"),
    ]


def test_leave_one_out_can_limit_train_examples_per_user() -> None:
    examples = leave_one_out_split(
        [
            Interaction("u1", "i1", 1),
            Interaction("u1", "i2", 2),
            Interaction("u1", "i3", 3),
            Interaction("u1", "i4", 4),
            Interaction("u1", "i5", 5),
            Interaction("u1", "i6", 6),
        ],
        min_history=1,
        train_examples_per_user=1,
        domain="tiny",
    )
    rows = [(example.split, example.history, example.target) for example in examples]
    assert rows == [
        ("train", ["i1", "i2", "i3"], "i4"),
        ("valid", ["i1", "i2", "i3", "i4"], "i5"),
        ("test", ["i1", "i2", "i3", "i4", "i5"], "i6"),
    ]


def test_temporal_split_simple_interface() -> None:
    examples = temporal_split(_interactions(), min_history=1, train_fraction=0.5, valid_fraction=0.25)
    assert {example.split for example in examples} <= {"train", "valid", "test"}
    assert examples
