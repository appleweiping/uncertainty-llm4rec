from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_phase5_all_pipeline() -> None:
    result = run_all("configs/experiments/smoke_phase5_all.yaml")
    assert result["baseline_methods"] == ["popularity", "bm25", "sequential_markov", "llm_generative"]
    assert result["run_count"] == 4
    assert "aggregation" in result["postprocess"]
    assert "tables" in result["postprocess"]
    for run in result["runs"]:
        run_dir = Path(run["run_dir"])
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "metrics.json").exists()
        assert "by_user_history_bucket" in run["metrics"]
