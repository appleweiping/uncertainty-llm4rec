"""Lightweight sequential model-state containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SequentialTransitionState:
    """Serializable Markov transition state for CPU smoke baselines."""

    transitions: dict[str, dict[str, int]] = field(default_factory=dict)
    item_popularity: dict[str, int] = field(default_factory=dict)
    item_ids: list[str] = field(default_factory=list)
    train_sequence_count: int = 0

    def score(self, previous_item: str | None, candidate_item: str) -> float:
        if previous_item:
            transition_score = self.transitions.get(previous_item, {}).get(candidate_item, 0)
            if transition_score:
                return float(transition_score)
        return float(self.item_popularity.get(candidate_item, 0)) * 1e-6

    def to_dict(self) -> dict[str, object]:
        return {
            "transitions": self.transitions,
            "item_popularity": self.item_popularity,
            "item_ids": self.item_ids,
            "train_sequence_count": self.train_sequence_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SequentialTransitionState":
        transitions = {
            str(source): {str(target): int(count) for target, count in dict(targets).items()}
            for source, targets in dict(payload.get("transitions") or {}).items()
        }
        item_popularity = {
            str(item_id): int(count)
            for item_id, count in dict(payload.get("item_popularity") or {}).items()
        }
        item_ids = [str(item_id) for item_id in payload.get("item_ids") or []]
        return cls(
            transitions=transitions,
            item_popularity=item_popularity,
            item_ids=item_ids,
            train_sequence_count=int(payload.get("train_sequence_count") or 0),
        )
