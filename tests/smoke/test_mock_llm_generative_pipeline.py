from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_mock_llm_generative_pipeline() -> None:
    result = run_all("configs/experiments/smoke_llm_generative.yaml")
    run_dir = Path(result["run_dir"])
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert predictions
    assert predictions[0]["method"] == "llm_generative_mock"
    assert predictions[0]["raw_output"]
    assert predictions[0]["metadata"]["parse_success"] is True
    assert "confidence" in predictions[0]["metadata"]
    assert predictions[0]["metadata"]["target_excluded_from_prompt"] is True
    assert metrics["aggregate"]["confidence"]["confidence_count"] > 0
