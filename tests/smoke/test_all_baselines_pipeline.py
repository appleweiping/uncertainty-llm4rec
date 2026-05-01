from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_all_baselines_pipeline() -> None:
    result = run_all("configs/experiments/smoke_all_baselines.yaml")
    assert result["baseline_methods"] == ["random", "popularity", "bm25", "mf"]
    assert result["run_count"] == 4
    for run in result["runs"]:
        run_dir = Path(run["run_dir"])
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "environment.json").exists()
        assert (run_dir / "logs.txt").exists()
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "artifacts").is_dir()
