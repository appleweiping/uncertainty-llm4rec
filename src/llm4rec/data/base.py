"""Base data contracts for reproducible recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: float | int | None = None
    rating: float | None = None
    domain: str | None = None

    def __post_init__(self) -> None:
        if not str(self.user_id).strip():
            raise ValueError("Interaction.user_id must be non-empty")
        if not str(self.item_id).strip():
            raise ValueError("Interaction.item_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ItemRecord:
    item_id: str
    title: str
    description: str | None = None
    category: str | None = None
    brand: str | None = None
    domain: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.item_id).strip():
            raise ValueError("ItemRecord.item_id must be non-empty")
        if not str(self.title).strip():
            raise ValueError("ItemRecord.title must be non-empty")


@dataclass(frozen=True, slots=True)
class UserExample:
    example_id: str
    user_id: str
    history: list[str]
    target: str
    candidates: list[str] | None = None
    split: str = "test"
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.example_id).strip():
            raise ValueError("UserExample.example_id must be non-empty")
        if not str(self.user_id).strip():
            raise ValueError("UserExample.user_id must be non-empty")
        if not str(self.target).strip():
            raise ValueError("UserExample.target must be non-empty")
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("UserExample.split must be train, valid, or test")


class BaseDataset(Protocol):
    """Minimal dataset reader contract."""

    config: dict[str, Any]

    def load_interactions(self) -> list[Interaction]:
        """Load canonical interactions."""

    def load_items(self) -> list[ItemRecord]:
        """Load canonical item metadata."""


class BaseDataModule(Protocol):
    """Minimal data module contract used by experiment runners."""

    config: dict[str, Any]

    def prepare(self) -> dict[str, Any]:
        """Write processed artifacts and return a manifest."""

    def processed_dir(self) -> Path:
        """Return the processed artifact directory."""

    def examples(self, split: str) -> list[UserExample]:
        """Load processed examples for one split."""
