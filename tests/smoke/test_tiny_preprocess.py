from __future__ import annotations

from pathlib import Path

from llm4rec.data.preprocess import preprocess_dataset


def test_tiny_preprocess_writes_processed_artifacts(tmp_path: Path) -> None:
    config = {
        "name": "tiny",
        "type": "tiny_csv",
        "domain": "tiny",
        "seed": 13,
        "interactions_path": "tests/fixtures/tiny_interactions.csv",
        "items_path": "tests/fixtures/tiny_items.csv",
        "processed_dir": str(tmp_path / "processed"),
        "split": {"strategy": "leave_one_out", "min_history": 1},
        "candidate": {"protocol": "full", "include_history": False, "sample_size": None},
    }
    manifest = preprocess_dataset(config)
    processed = Path(manifest["processed_dir"])
    assert (processed / "items.csv").exists()
    assert (processed / "interactions.csv").exists()
    assert (processed / "examples.jsonl").exists()
    assert (processed / "candidate_sets.jsonl").exists()
    assert manifest["split_counts"]["test"] >= 1
    assert manifest["split_counts"]["valid"] >= 1
    assert manifest["split_counts"]["train"] >= 1
