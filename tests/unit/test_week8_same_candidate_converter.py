import csv
import json
from pathlib import Path

from scripts.convert_week8_same_candidate_to_truce import convert


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_week8_converter_preserves_event_alignment(tmp_path: Path) -> None:
    task = tmp_path / "books_large10000_100neg_test_same_candidate"
    task.mkdir()
    _write_jsonl(
        task / "ranking_test.jsonl",
        [
            {
                "event_id": "e1",
                "source_event_id": "src1",
                "user_id": "u1",
                "positive_item_id": "i2",
                "candidate_items": ["i2", "i3"],
                "history_item_ids": ["i1"],
            }
        ],
    )
    with (task / "item_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "category"])
        writer.writeheader()
        writer.writerow({"item_id": "i1", "title": "One", "category": "book"})
        writer.writerow({"item_id": "i2", "title": "Two", "category": "book"})
        writer.writerow({"item_id": "i3", "title": "Three", "category": "book"})
    with (task / "train_interactions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id"])
        writer.writeheader()
        writer.writerow({"user_id": "u1", "item_id": "i1"})
    out = tmp_path / "processed"
    manifest = convert(task_dir=task, output_dir=out, domain="books", split="test")
    assert manifest["example_count"] == 1
    ex = json.loads((out / "examples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ex["example_id"] == "e1"
    assert ex["metadata"]["source_event_id"] == "src1"
    assert ex["candidates"] == ["i2", "i3"]
    assert ex["target"] == "i2"
    candidate_rows = [json.loads(line) for line in (out / "candidate_sets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["label"] for row in candidate_rows] == [1, 0]
    assert (out / "items.csv").exists()
    assert (out / "preprocess_manifest.json").exists()
