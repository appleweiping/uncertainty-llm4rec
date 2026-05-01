from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_ours_method_pipeline() -> None:
    result = run_all("configs/experiments/smoke_ours_method.yaml")
    run_dir = Path(result["run_dir"])
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert run_dir.name == "smoke_ours_method_seed13"
    for filename in ["resolved_config.yaml", "environment.json", "logs.txt", "metrics.csv", "cost_latency.json"]:
        assert (run_dir / filename).exists()
    assert (run_dir / "artifacts").is_dir()
    assert predictions
    assert predictions[0]["method"] == "ours_uncertainty_guided"
    assert predictions[0]["metadata"]["ours_method"] is True
    assert predictions[0]["metadata"]["ablation_variant"] == "full"
    assert predictions[0]["metadata"]["prompt_hash"]
    assert metrics["is_experiment_result"] is False
    assert "Phase 6 OursMethod mock smoke metrics" in metrics["note"]
