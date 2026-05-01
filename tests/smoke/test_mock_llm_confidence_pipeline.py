from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_mock_llm_confidence_pipeline() -> None:
    result = run_all("configs/experiments/smoke_llm_confidence.yaml")
    run_dir = Path(result["run_dir"])
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert (run_dir / "metrics.csv").exists()
    assert predictions[0]["method"] == "llm_confidence_observation_mock"
    assert predictions[0]["metadata"]["verification_parse_success"] is True
    assert predictions[0]["metadata"]["candidate_normalized_parse_success"] is True
    assert predictions[0]["metadata"]["candidate_normalized_options"]
    assert predictions[0]["metadata"]["verification_prompt_template_id"]
    assert predictions[0]["metadata"]["candidate_normalized_prompt_template_id"]
    assert predictions[0]["raw_output"]
