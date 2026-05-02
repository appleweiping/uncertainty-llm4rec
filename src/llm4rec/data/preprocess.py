"""Tiny CSV/JSONL preprocessing for Phase 1 smoke experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm4rec.data.base import Interaction, ItemRecord, UserExample
from llm4rec.data.candidates import attach_candidates
from llm4rec.data.registry import build_data_module, register_dataset
from llm4rec.data.splits import leave_one_out_split, temporal_split
from storyflow.data.preprocessing import clean_title


class TinyDataModule:
    """Small local data module used only for reproducible smoke tests."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def processed_dir(self) -> Path:
        return Path(str(self.config["processed_dir"]))

    def load_interactions(self) -> list[Interaction]:
        path = Path(str(self.config["interactions_path"]))
        rows = _read_rows(path)
        interactions: list[Interaction] = []
        for index, row in enumerate(rows):
            try:
                interactions.append(
                    Interaction(
                        user_id=str(row["user_id"]),
                        item_id=str(row["item_id"]),
                        timestamp=float(row["timestamp"]) if row.get("timestamp") not in (None, "") else None,
                        rating=float(row["rating"]) if row.get("rating") not in (None, "") else None,
                        domain=str(row.get("domain") or self.config.get("domain") or "tiny"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{index + 1} missing field {exc}") from exc
        return interactions

    def load_items(self) -> list[ItemRecord]:
        path = Path(str(self.config["items_path"]))
        rows = _read_rows(path)
        items: list[ItemRecord] = []
        for index, row in enumerate(rows):
            try:
                items.append(
                    ItemRecord(
                        item_id=str(row["item_id"]),
                        title=clean_title(str(row["title"])),
                        category=str(row.get("category") or row.get("genres") or "") or None,
                        domain=str(row.get("domain") or self.config.get("domain") or "tiny"),
                        raw_text=str(row.get("raw_text") or "") or None,
                        metadata={"source_row": index + 1},
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{index + 1} missing field {exc}") from exc
        return items

    def prepare(self) -> dict[str, Any]:
        interactions = self.load_interactions()
        items = self.load_items()
        split_config = dict(self.config.get("split") or {})
        strategy = str(split_config.get("strategy") or "leave_one_out")
        min_history = int(split_config.get("min_history") or 1)
        if strategy == "leave_one_out":
            examples = leave_one_out_split(
                interactions,
                min_history=min_history,
                train_examples_per_user=split_config.get("train_examples_per_user"),
                domain=str(self.config.get("domain") or "tiny"),
            )
        elif strategy == "temporal":
            examples = temporal_split(
                interactions,
                min_history=min_history,
                train_fraction=float(split_config.get("train_fraction") or 0.8),
                valid_fraction=float(split_config.get("valid_fraction") or 0.1),
                domain=str(self.config.get("domain") or "tiny"),
            )
        else:
            raise ValueError(f"unknown split strategy: {strategy}")

        candidate_config = dict(self.config.get("candidate") or {})
        examples = attach_candidates(
            examples,
            items,
            protocol=str(candidate_config.get("protocol") or "full"),
            include_history=bool(candidate_config.get("include_history", False)),
            sample_size=(
                int(candidate_config["sample_size"])
                if candidate_config.get("sample_size") not in (None, "")
                else None
            ),
            seed=int(self.config.get("seed") or 0),
        )
        return self._write_artifacts(items, interactions, examples, strategy)

    def examples(self, split: str) -> list[UserExample]:
        path = self.processed_dir() / "examples.jsonl"
        rows = _read_jsonl(path)
        return [
            UserExample(
                example_id=str(row["example_id"]),
                user_id=str(row["user_id"]),
                history=[str(item) for item in row.get("history", [])],
                target=str(row["target"]),
                candidates=[str(item) for item in row.get("candidates", [])],
                split=str(row["split"]),
                domain=row.get("domain"),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows
            if str(row.get("split")) == split
        ]

    def _write_artifacts(
        self,
        items: list[ItemRecord],
        interactions: list[Interaction],
        examples: list[UserExample],
        split_strategy: str,
    ) -> dict[str, Any]:
        output_dir = self.processed_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "items.csv", [asdict(item) for item in items])
        _write_csv(output_dir / "interactions.csv", [asdict(row) for row in interactions])
        example_rows = [asdict(example) for example in examples]
        _write_jsonl(output_dir / "examples.jsonl", example_rows)
        _write_jsonl(
            output_dir / "candidate_sets.jsonl",
            [
                {
                    "example_id": example.example_id,
                    "user_id": example.user_id,
                    "target_item": example.target,
                    "candidate_items": example.candidates or [],
                    "split": example.split,
                    "domain": example.domain,
                }
                for example in examples
            ],
        )
        split_counts: dict[str, int] = {}
        for example in examples:
            split_counts[example.split] = split_counts.get(example.split, 0) + 1
        manifest = {
            "dataset": self.config.get("name", "tiny"),
            "type": self.config.get("type", "tiny"),
            "processed_dir": str(output_dir),
            "split_strategy": split_strategy,
            "seed": int(self.config.get("seed") or 0),
            "item_count": len(items),
            "interaction_count": len(interactions),
            "example_count": len(examples),
            "split_counts": split_counts,
            "outputs": {
                "items": str(output_dir / "items.csv"),
                "interactions": str(output_dir / "interactions.csv"),
                "examples": str(output_dir / "examples.jsonl"),
                "candidate_sets": str(output_dir / "candidate_sets.jsonl"),
            },
            "is_experiment_result": False,
        }
        (output_dir / "preprocess_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return manifest


def preprocess_dataset(config: dict[str, Any]) -> dict[str, Any]:
    return build_data_module(config).prepare()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


register_dataset("tiny", TinyDataModule)
register_dataset("tiny_csv", TinyDataModule)
