import csv
import json

from scripts.import_evaluate_ours_adapter import import_and_evaluate


def test_import_evaluate_ours_adapter(tmp_path):
    out = tmp_path / "adapter_out"
    out.mkdir()
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps(
            {
                "example_id": "e1",
                "user_id": "u1",
                "target": "i2",
                "candidates": ["i1", "i2"],
                "split": "test",
                "domain": "tiny",
                "metadata": {"event_id": "e1", "source_event_id": "src1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scores = out / "candidate_scores.csv"
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "user_id", "item_id", "score"])
        writer.writeheader()
        writer.writerow({"example_id": "e1", "user_id": "u1", "item_id": "i1", "score": "0.1"})
        writer.writerow({"example_id": "e1", "user_id": "u1", "item_id": "i2", "score": "0.9"})
    manifest = {
        "controlled_baseline_name": "ours_truce_qwen_adapter_tiny",
        "output_dir": str(out),
        "seed": 13,
        "files": {"test_examples": str(examples)},
        "training": {},
    }
    result = import_and_evaluate(manifest=manifest, run_dir=tmp_path / "run")
    assert result["count"] == 1
    pred = json.loads((tmp_path / "run" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert pred["predicted_items"][0] == "i2"
    assert pred["metadata"]["source_event_id"] == "src1"
