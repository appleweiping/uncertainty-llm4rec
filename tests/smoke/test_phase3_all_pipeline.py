from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_phase3_all_pipeline() -> None:
    result = run_all("configs/experiments/smoke_phase3_all.yaml")
    assert result["baseline_methods"] == ["llm_generative", "llm_rerank", "llm_confidence_observation"]
    assert result["run_count"] == 3
    for run in result["runs"]:
        run_dir = Path(run["run_dir"])
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "environment.json").exists()
        assert (run_dir / "logs.txt").exists()
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "cost_latency.json").exists()
        assert (run_dir / "artifacts").is_dir()
        resolved_config = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        assert "provider: mock" in resolved_config
        assert "config_path: configs/llm/mock.yaml" in resolved_config
        predictions = read_jsonl(run_dir / "predictions.jsonl")
        assert predictions[0]["raw_output"]
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "confidence" in metrics["aggregate"]
        assert "calibration" in metrics["aggregate"]
        assert "efficiency" in metrics["aggregate"]
        assert "grounding_success_rate" in metrics["aggregate"]
