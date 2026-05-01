"""Sequential recommendation data helpers with leakage safeguards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm4rec.data.base import Interaction, UserExample
from llm4rec.data.splits import chronological_user_sequences


PAD_TOKEN = "<PAD>"


@dataclass(frozen=True, slots=True)
class SequentialExample:
    example_id: str
    user_id: str
    history: list[str]
    target: str
    history_indices: list[int]
    target_index: int
    candidates: list[str] | None = None
    candidate_indices: list[int] | None = None
    split: str = "train"
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_item_index(item_ids: list[str], *, pad_token: str = PAD_TOKEN) -> dict[str, int]:
    """Build a deterministic item-id to index map with 0 reserved for padding."""

    if not pad_token:
        raise ValueError("pad_token must be non-empty")
    unique = sorted({str(item_id) for item_id in item_ids if str(item_id)})
    if pad_token in unique:
        raise ValueError("pad_token must not collide with item ids")
    return {pad_token: 0, **{item_id: index + 1 for index, item_id in enumerate(unique)}}


def sequential_examples_from_interactions(
    interactions: list[Interaction],
    *,
    item_index: dict[str, int] | None = None,
    max_history_length: int = 50,
    min_history: int = 1,
    split: str = "train",
    domain: str | None = None,
    filter_target_from_history: bool = True,
) -> list[SequentialExample]:
    """Create rolling next-item sequence examples from timestamp-sorted interactions."""

    if max_history_length < 1:
        raise ValueError("max_history_length must be >= 1")
    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if item_index is None:
        item_index = build_item_index([row.item_id for row in interactions])
    output: list[SequentialExample] = []
    for user_id, rows in chronological_user_sequences(interactions).items():
        item_ids = [str(row.item_id) for row in rows]
        for target_pos in range(min_history, len(item_ids)):
            target = item_ids[target_pos]
            history = item_ids[:target_pos]
            if filter_target_from_history:
                history = [item_id for item_id in history if item_id != target]
            history = truncate_history(history, max_history_length=max_history_length)
            output.append(
                _sequential_example(
                    example_id=f"{user_id}:{target_pos}",
                    user_id=user_id,
                    history=history,
                    target=target,
                    item_index=item_index,
                    max_history_length=max_history_length,
                    split=split,
                    domain=domain,
                    metadata={
                        "target_position": target_pos,
                        "timestamp_sorted": True,
                        "target_filtered_from_history": filter_target_from_history,
                    },
                )
            )
    return output


def sequential_example_from_user_example(
    example: dict[str, Any] | UserExample,
    *,
    item_index: dict[str, int],
    max_history_length: int = 50,
    filter_target_from_history: bool = True,
) -> SequentialExample:
    row = _as_dict(example)
    target = str(row["target"])
    history = [str(item_id) for item_id in row.get("history", [])]
    if filter_target_from_history:
        history = [item_id for item_id in history if item_id != target]
    history = truncate_history(history, max_history_length=max_history_length)
    candidates = [str(item_id) for item_id in row.get("candidates", [])] if row.get("candidates") else None
    return _sequential_example(
        example_id=str(row.get("example_id") or f"{row['user_id']}:unknown"),
        user_id=str(row["user_id"]),
        history=history,
        target=target,
        item_index=item_index,
        max_history_length=max_history_length,
        candidates=candidates,
        split=str(row.get("split") or "test"),
        domain=row.get("domain"),
        metadata={
            **dict(row.get("metadata") or {}),
            "target_filtered_from_history": filter_target_from_history,
        },
    )


def truncate_history(history: list[str], *, max_history_length: int) -> list[str]:
    if max_history_length < 1:
        raise ValueError("max_history_length must be >= 1")
    return [str(item_id) for item_id in history][-max_history_length:]


def pad_history_indices(history: list[str], item_index: dict[str, int], *, max_history_length: int) -> list[int]:
    truncated = truncate_history(history, max_history_length=max_history_length)
    indices = [int(item_index[item_id]) for item_id in truncated if item_id in item_index]
    return [0] * max(0, max_history_length - len(indices)) + indices


def _sequential_example(
    *,
    example_id: str,
    user_id: str,
    history: list[str],
    target: str,
    item_index: dict[str, int],
    max_history_length: int,
    candidates: list[str] | None = None,
    split: str,
    domain: str | None,
    metadata: dict[str, Any],
) -> SequentialExample:
    if target not in item_index:
        raise ValueError(f"target item missing from item_index: {target}")
    candidate_indices = (
        [int(item_index[item_id]) for item_id in candidates if item_id in item_index]
        if candidates is not None
        else None
    )
    return SequentialExample(
        example_id=example_id,
        user_id=user_id,
        history=history,
        target=target,
        history_indices=pad_history_indices(history, item_index, max_history_length=max_history_length),
        target_index=int(item_index[target]),
        candidates=candidates,
        candidate_indices=candidate_indices,
        split=split,
        domain=domain,
        metadata=metadata,
    )


def _as_dict(example: dict[str, Any] | UserExample) -> dict[str, Any]:
    if isinstance(example, dict):
        return example
    return {
        "example_id": example.example_id,
        "user_id": example.user_id,
        "history": example.history,
        "target": example.target,
        "candidates": example.candidates,
        "split": example.split,
        "domain": example.domain,
        "metadata": example.metadata,
    }
