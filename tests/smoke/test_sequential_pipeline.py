from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_sequential_pipeline_writes_unified_artifacts() -> None:
    result = run_all("configs/experiments/smoke_sequential.yaml")
    run_dir = Path(result["run_dir"])
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert predictions
    assert predictions[0]["method"] == "sequential_markov"
    assert predictions[0]["metadata"]["phase"] == "phase4_sequential_training_layer"
    assert predictions[0]["metadata"]["label_leakage"] is False
    assert (run_dir / "checkpoints" / "checkpoint_manifest.json").exists()
    assert metrics["aggregate"]["validity_rate"] >= 0.0
    assert metrics["aggregate"]["hallucination_rate"] >= 0.0
    assert "Phase 4 lightweight sequential baseline" in metrics["note"]
