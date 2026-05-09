import csv
import json
from pathlib import Path

from scripts.build_week8_observation_inputs import build_week8_observation_inputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_week8_observation_inputs_match_qwen_required_schema(tmp_path: Path) -> None:
    processed = tmp_path / "books_large10000_100neg" / "test"
    processed.mkdir(parents=True)
    _write_jsonl(
        processed / "examples.jsonl",
        [
            {
                "example_id": "evt-1",
                "user_id": "u1",
                "history": ["i1", "i2"],
                "target": "i3",
                "candidates": ["i3", "i4"],
                "split": "test",
                "domain": "books",
                "metadata": {"event_id": "evt-1", "source_event_id": "src-1"},
            }
        ],
    )
    with (processed / "items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "title", "description", "category", "brand", "domain", "raw_text"],
        )
        writer.writeheader()
        writer.writerow({"item_id": "i1", "title": "History One", "domain": "books"})
        writer.writerow({"item_id": "i2", "title": "History Two", "domain": "books"})
        writer.writerow({"item_id": "i3", "title": "Target Three", "domain": "books"})
        writer.writerow({"item_id": "i4", "title": "Candidate Four", "domain": "books"})
    with (processed / "interactions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id", "timestamp", "rating", "domain"])
        writer.writeheader()
        writer.writerow({"user_id": "u1", "item_id": "i1", "timestamp": "1", "domain": "books"})
        writer.writerow({"user_id": "u2", "item_id": "i3", "timestamp": "1", "domain": "books"})
        writer.writerow({"user_id": "u3", "item_id": "i3", "timestamp": "1", "domain": "books"})

    output = tmp_path / "inputs.jsonl"
    manifest = build_week8_observation_inputs(
        processed_dir=processed,
        dataset="books_large10000_100neg",
        domain="books",
        split="test",
        output_jsonl=output,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    assert manifest["input_count"] == 1
    assert manifest["schema_version"] == "truce_week8_observation_inputs_v1"
    assert Path(manifest["catalog_csv"]).exists()
    assert row["input_id"].startswith("week8:books_large10000_100neg:test:evt-1:")
    assert row["example_id"] == "evt-1"
    assert row["event_id"] == "evt-1"
    assert row["source_event_id"] == "src-1"
    assert row["history_item_titles"] == ["History One", "History Two"]
    assert row["target_item_id"] == "i3"
    assert row["target_title"] == "Target Three"
    assert row["target_popularity"] == 2
    assert row["target_popularity_bucket"] in {"head", "mid", "tail"}
    assert row["prompt_template"] == "forced_json"
    assert row["prompt_hash"]
    assert row["source"]["catalog_csv"] == manifest["catalog_csv"]
    assert row["source"]["same_candidate_protocol"] is True


def test_week8_observation_inputs_reject_empty_history(tmp_path: Path) -> None:
    processed = tmp_path / "movies_large10000_100neg" / "test"
    processed.mkdir(parents=True)
    _write_jsonl(
        processed / "examples.jsonl",
        [
            {
                "example_id": "evt-empty",
                "user_id": "u1",
                "history": [],
                "target": "i1",
                "candidates": ["i1"],
                "split": "test",
                "domain": "movies",
            }
        ],
    )
    with (processed / "items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title"])
        writer.writeheader()
        writer.writerow({"item_id": "i1", "title": "Target"})
    with (processed / "interactions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id"])
        writer.writeheader()
        writer.writerow({"user_id": "u2", "item_id": "i1"})

    try:
        build_week8_observation_inputs(
            processed_dir=processed,
            dataset="movies_large10000_100neg",
            domain="movies",
            split="test",
            output_jsonl=tmp_path / "inputs.jsonl",
        )
    except ValueError as exc:
        assert "empty history" in str(exc)
    else:
        raise AssertionError("expected empty history to be rejected")
