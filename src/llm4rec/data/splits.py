"""Deterministic split strategies for tiny and future datasets."""

from __future__ import annotations

from collections import defaultdict

from llm4rec.data.base import Interaction, UserExample


def chronological_user_sequences(interactions: list[Interaction]) -> dict[str, list[Interaction]]:
    grouped: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        grouped[row.user_id].append(row)
    return {
        user_id: sorted(
            rows,
            key=lambda row: (
                float(row.timestamp or 0),
                str(row.item_id),
            ),
        )
        for user_id, rows in sorted(grouped.items())
    }


def leave_one_out_split(
    interactions: list[Interaction],
    *,
    min_history: int = 1,
    train_examples_per_user: int | str | None = None,
    domain: str | None = None,
) -> list[UserExample]:
    """Create train/valid/test examples per user using the last two items as holdout."""

    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    train_limit = _train_example_limit(train_examples_per_user)
    examples: list[UserExample] = []
    for user_id, rows in chronological_user_sequences(interactions).items():
        item_ids = [row.item_id for row in rows]
        if len(item_ids) <= min_history:
            continue
        train_stop = max(min_history, len(item_ids) - 2)
        train_indices = list(range(min_history, train_stop))
        if train_limit is not None:
            train_indices = train_indices[-train_limit:]
        for target_index in train_indices:
            examples.append(_example(user_id, item_ids, target_index, "train", domain))
        if len(item_ids) >= min_history + 2:
            examples.append(_example(user_id, item_ids, len(item_ids) - 2, "valid", domain))
        examples.append(_example(user_id, item_ids, len(item_ids) - 1, "test", domain))
    return examples


def temporal_split(
    interactions: list[Interaction],
    *,
    min_history: int = 1,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    domain: str | None = None,
) -> list[UserExample]:
    """Simple global chronological split over rolling next-item examples."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    rolling: list[tuple[float, UserExample]] = []
    for user_id, rows in chronological_user_sequences(interactions).items():
        item_ids = [row.item_id for row in rows]
        for target_index in range(min_history, len(rows)):
            timestamp = float(rows[target_index].timestamp or 0)
            rolling.append((timestamp, _example(user_id, item_ids, target_index, "train", domain)))
    rolling.sort(key=lambda item: (item[0], item[1].user_id, item[1].example_id))
    if not rolling:
        return []
    train_cut = int(len(rolling) * train_fraction)
    valid_cut = int(len(rolling) * (train_fraction + valid_fraction))
    output: list[UserExample] = []
    for index, (_, example) in enumerate(rolling):
        split = "train" if index < train_cut else "valid" if index < valid_cut else "test"
        output.append(
            UserExample(
                example_id=example.example_id,
                user_id=example.user_id,
                history=example.history,
                target=example.target,
                split=split,
                domain=example.domain,
                metadata=example.metadata,
            )
        )
    return output


def _example(
    user_id: str,
    item_ids: list[str],
    target_index: int,
    split: str,
    domain: str | None,
) -> UserExample:
    return UserExample(
        example_id=f"{user_id}:{target_index}",
        user_id=user_id,
        history=list(item_ids[:target_index]),
        target=item_ids[target_index],
        split=split,
        domain=domain,
        metadata={"target_index": target_index},
    )


def _train_example_limit(value: int | str | None) -> int | None:
    if value in (None, "", "all"):
        return None
    if isinstance(value, str) and value in {"last", "last_only"}:
        return 1
    limit = int(value)
    if limit < 1:
        raise ValueError("train_examples_per_user must be >= 1, 'all', or 'last_only'")
    return limit
