import json
from pathlib import Path

from llm4rec.external_baselines.prediction_import import import_scored_candidates


def test_import_external_scores_to_truce_schema(tmp_path: Path) -> None:
    examples = [
        {
            "example_id": "E1",
            "user_id": "U1",
            "target": "I2",
            "candidates": ["I1", "I2"],
            "split": "test",
            "domain": "test",
        }
    ]
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text("\n".join(json.dumps(r) for r in examples) + "\n", encoding="utf-8")
    scores_path = tmp_path / "scores.csv"
    scores_path.write_text("example_id,item_id,score\nE1,I1,0.1\nE1,I2,0.9\n", encoding="utf-8")
    out_path = tmp_path / "predictions.jsonl"

    result = import_scored_candidates(
        scores_path=scores_path,
        examples_path=examples_path,
        output_path=out_path,
        method="sasrec_recbole",
        source_project="RecBole",
        model_name="SASRec",
        seed=13,
    )

    row = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert result["count"] == 1
    assert row["predicted_items"] == ["I2", "I1"]
    assert row["metadata"]["external_baseline"] is True
    assert row["metadata"]["source_project/library"] == "RecBole"


def test_import_external_scores_filters_split(tmp_path: Path) -> None:
    examples = [
        {"example_id": "E1", "user_id": "U1", "target": "I2", "candidates": ["I1", "I2"], "split": "train", "domain": "test"},
        {"example_id": "E2", "user_id": "U1", "target": "I3", "candidates": ["I2", "I3"], "split": "test", "domain": "test"},
    ]
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text("\n".join(json.dumps(r) for r in examples) + "\n", encoding="utf-8")
    scores_path = tmp_path / "scores.csv"
    scores_path.write_text("example_id,item_id,score\nE1,I2,1.0\nE2,I2,0.1\nE2,I3,0.9\n", encoding="utf-8")
    out_path = tmp_path / "predictions.jsonl"

    result = import_scored_candidates(
        scores_path=scores_path,
        examples_path=examples_path,
        output_path=out_path,
        method="tallrec_qwen_zeroshot",
        source_project="TALLRec",
        model_name="Qwen3-8B",
        seed=13,
        split="test",
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert result["count"] == 1
    assert len(rows) == 1
    assert rows[0]["target_item"] == "I3"
    assert rows[0]["predicted_items"] == ["I3", "I2"]
