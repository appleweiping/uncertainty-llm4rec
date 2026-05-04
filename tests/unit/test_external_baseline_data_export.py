import json
from pathlib import Path

from llm4rec.external_baselines.data_export import export_recbole_atomic


def test_recbole_export_preserves_examples_and_candidates(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "items.csv").write_text("item_id,title,domain\nI1,Alpha,test\nI2,Beta,test\n", encoding="utf-8")
    (processed / "interactions.csv").write_text(
        "user_id,item_id,timestamp,rating,domain\nU1,I1,123,5,test\nU1,I2,456,5,test\n",
        encoding="utf-8",
    )
    examples = [
        {"example_id": "E1", "user_id": "U1", "history": ["I1"], "target": "I2", "candidates": ["I1", "I2"], "split": "test", "domain": "test"}
    ]
    (processed / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in examples) + "\n", encoding="utf-8")
    (processed / "candidate_sets.jsonl").write_text(json.dumps({"example_id": "E1", "candidate_items": ["I1", "I2"]}) + "\n", encoding="utf-8")

    manifest = export_recbole_atomic(processed_dir=processed, output_dir=tmp_path / "out", dataset_name="tiny_recbole", seed=13)

    exported = Path(manifest["exported_dir"])
    assert (exported / "tiny_recbole.inter").exists()
    assert (exported / "tiny_recbole.item").exists()
    assert (exported / "truce_candidate_sets.jsonl").exists()
    assert manifest["split_counts"]["test"] == 1
    assert manifest["sasrec_benchmark_counts"]["test"] == 1
    assert "TRUCE evaluator" in manifest["metric_contract"]
    test_rows = (exported / "tiny_recbole.test.inter").read_text(encoding="utf-8")
    assert "456" in test_rows
    sasrec_rows = (exported / "tiny_recbole.sasrec_test.inter").read_text(encoding="utf-8")
    assert "item_id_list:token_seq" in sasrec_rows
    assert "456" in sasrec_rows
