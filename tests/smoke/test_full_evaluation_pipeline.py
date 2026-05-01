from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_experiment


def test_full_evaluation_pipeline() -> None:
    result = run_experiment("configs/experiments/smoke_full_eval.yaml", preprocess=True)
    run_dir = Path(result["run_dir"])
    metrics = result["metrics"]
    assert (run_dir / "metrics.json").exists()
    assert "coverage_metrics" in metrics["aggregate"]
    assert "diversity" in metrics["aggregate"]
    assert "novelty" in metrics["aggregate"]
    assert "long_tail" in metrics["aggregate"]
    assert "by_domain" in metrics
    assert "by_user_history_bucket" in metrics
    assert "by_item_popularity_bucket" in metrics
