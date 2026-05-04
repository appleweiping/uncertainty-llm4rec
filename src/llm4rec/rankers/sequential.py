"""Lightweight sequential recommendation baselines for Phase 4."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm4rec.models.sequential import SequentialTransitionState
from llm4rec.rankers.base import RankingResult


class SequentialLastItemRanker:
    """Deterministic smoke baseline that scores continuity with the last history item."""

    method_name = "sequential_last_item"

    def __init__(self, *, max_history_length: int = 50) -> None:
        self.max_history_length = int(max_history_length)
        self.item_ids: list[str] = []

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_ids = sorted(str(row["item_id"]) for row in item_catalog)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        history = [str(item_id) for item_id in example.get("history", [])][-self.max_history_length :]
        last_item = history[-1] if history else None
        scores = {
            str(item_id): (1.0 if last_item and str(item_id) == last_item else 0.0)
            for item_id in candidate_items
        }
        return _prediction(
            example=example,
            candidate_items=candidate_items,
            scores=scores,
            method=self.method_name,
            metadata={
                "sequential_baseline": "last_item_continuity",
                "max_history_length": self.max_history_length,
                "last_history_item": last_item,
                "label_leakage": False,
                "uses_train_transitions": False,
            },
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "method": self.method_name,
                    "max_history_length": self.max_history_length,
                    "item_ids": self.item_ids,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("method") != self.method_name:
            raise ValueError(f"incompatible checkpoint method: {payload.get('method')}")
        self.max_history_length = int(payload.get("max_history_length") or self.max_history_length)
        self.item_ids = [str(item_id) for item_id in payload.get("item_ids") or []]


class MarkovSequentialRanker:
    """First-order train-split Markov transition baseline."""

    method_name = "sequential_markov"

    def __init__(self, *, max_history_length: int = 50) -> None:
        self.max_history_length = int(max_history_length)
        self.state = SequentialTransitionState()

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        transitions: dict[str, Counter[str]] = {}
        popularity: Counter[str] = Counter()
        for example in train_examples:
            sequence = [str(item_id) for item_id in [*example.get("history", []), example.get("target")] if item_id]
            for item_id in sequence:
                popularity[item_id] += 1
            for source, target in zip(sequence, sequence[1:]):
                transitions.setdefault(source, Counter())[target] += 1
        self.state = SequentialTransitionState(
            transitions={source: dict(counter) for source, counter in sorted(transitions.items())},
            item_popularity=dict(sorted(popularity.items())),
            item_ids=sorted(str(row["item_id"]) for row in item_catalog),
            train_sequence_count=len(train_examples),
        )

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        history = [str(item_id) for item_id in example.get("history", [])][-self.max_history_length :]
        last_item = history[-1] if history else None
        scores = {str(item_id): self.state.score(last_item, str(item_id)) for item_id in candidate_items}
        return _prediction(
            example=example,
            candidate_items=candidate_items,
            scores=scores,
            method=self.method_name,
            metadata={
                "sequential_baseline": "first_order_markov_transition_counts",
                "max_history_length": self.max_history_length,
                "last_history_item": last_item,
                "train_sequence_count": self.state.train_sequence_count,
                "label_leakage": False,
                "uses_train_transitions": True,
            },
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "method": self.method_name,
                    "max_history_length": self.max_history_length,
                    "state": self.state.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("method") != self.method_name:
            raise ValueError(f"incompatible checkpoint method: {payload.get('method')}")
        self.max_history_length = int(payload.get("max_history_length") or self.max_history_length)
        self.state = SequentialTransitionState.from_dict(dict(payload.get("state") or {}))


class SasrecInterfaceRanker(MarkovSequentialRanker):
    """SASRec-compatible interface scaffold backed by Markov smoke scoring."""

    method_name = "sasrec_interface"

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        result = super().rank(example, candidate_items)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "sequential_baseline": "sasrec_interface_markov_smoke_fallback",
                "interface_scaffold": True,
                "true_sasrec_implemented": False,
            }
        )
        return RankingResult(
            user_id=result.user_id,
            target_item=result.target_item,
            candidate_items=result.candidate_items,
            predicted_items=result.predicted_items,
            scores=result.scores,
            method=self.method_name,
            domain=result.domain,
            raw_output=result.raw_output,
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "method": self.method_name,
                    "max_history_length": self.max_history_length,
                    "state": self.state.to_dict(),
                    "true_sasrec_implemented": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("method") != self.method_name:
            raise ValueError(f"incompatible checkpoint method: {payload.get('method')}")
        self.max_history_length = int(payload.get("max_history_length") or self.max_history_length)
        self.state = SequentialTransitionState.from_dict(dict(payload.get("state") or {}))


def _prediction(
    *,
    example: dict[str, Any],
    candidate_items: list[str],
    scores: dict[str, float],
    method: str,
    metadata: dict[str, Any],
) -> RankingResult:
    ordered = sorted([str(item_id) for item_id in candidate_items], key=lambda item_id: (-scores[item_id], item_id))
    inherited_metadata = {
        key: example.get(key)
        for key in ("history_titles", "history_item_ids", "target_title")
        if example.get(key) is not None
    }
    return RankingResult(
        user_id=str(example["user_id"]),
        target_item=str(example["target"]),
        candidate_items=[str(item_id) for item_id in candidate_items],
        predicted_items=ordered,
        scores=[float(scores[item_id]) for item_id in ordered],
        method=method,
        domain=str(example.get("domain") or "tiny"),
        raw_output=None,
        metadata={
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            **inherited_metadata,
            "phase": "phase4_sequential_training_layer",
            **metadata,
        },
    )
