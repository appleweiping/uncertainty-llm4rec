from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_mock_llm_rerank_pipeline() -> None:
    result = run_all("configs/experiments/smoke_llm_rerank.yaml")
    run_dir = Path(result["run_dir"])
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert (run_dir / "metrics.json").exists()
    assert predictions
    assert predictions[0]["method"] == "llm_rerank_mock"
    assert predictions[0]["metadata"]["parse_success"] is True
    assert predictions[0]["metadata"]["target_excluded_from_prompt"] is True
    assert predictions[0]["metadata"]["excluded_item_ids"]
