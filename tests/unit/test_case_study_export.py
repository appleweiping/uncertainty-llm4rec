from __future__ import annotations

import csv
import json
from pathlib import Path

from llm4rec.analysis.case_studies import CASE_FILES, export_case_studies


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_case_study_export_writes_required_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "items.csv").write_text(
        "item_id,title,description,category,brand,domain,raw_text\n"
        "a,Alpha,,,,movies,Alpha\nt,Target,,,,movies,Target\n",
        encoding="utf-8",
    )
    _write_jsonl(
        processed / "examples.jsonl",
        [{"example_id": "u:1", "split": "train", "history": ["a"], "target": "t"}],
    )
    ours = {
        "user_id": "u",
        "target_item": "t",
        "candidate_items": ["a", "t"],
        "predicted_items": ["a", "t"],
        "scores": [1.0, 0.5],
        "method": "ours_uncertainty_guided_real",
        "domain": "movies",
        "raw_output": "{}",
        "metadata": {
            "example_id": "u:1",
            "uncertainty_decision": "accept",
            "confidence": 0.9,
            "grounded_item_id": "a",
            "generated_title": "Alpha",
            "prompt_hash": "p",
            "provider_metadata": {"cache_key": "c"},
        },
    }
    fallback = {**ours, "method": "ours_fallback_only", "predicted_items": ["t", "a"], "metadata": {"example_id": "u:1"}}
    _write_jsonl(
        runs / "r3_movielens_1m_real_llm_full_candidate500_ours_uncertainty_guided_real_seed13" / "predictions.jsonl",
        [ours],
    )
    _write_jsonl(
        runs / "r3_movielens_1m_real_llm_full_candidate500_ours_fallback_only_seed13" / "predictions.jsonl",
        [fallback],
    )
    export_case_studies(runs, output_dir=tmp_path / "tables", processed_dir=processed, seeds=(13,))
    for filename in CASE_FILES.values():
        path = tmp_path / "tables" / filename
        assert path.exists()
        with path.open("r", encoding="utf-8", newline="") as handle:
            assert "user_id" in next(csv.reader(handle))

