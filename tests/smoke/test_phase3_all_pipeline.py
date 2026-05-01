from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all


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
        assert (run_dir / "artifacts").is_dir()
