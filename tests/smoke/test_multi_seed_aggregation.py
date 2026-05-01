from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_multi_seed_aggregation() -> None:
    result = run_all("configs/experiments/smoke_multi_seed.yaml")
    assert result["run_count"] == 4
    assert result["seeds"] == [13, 17]
    assert Path("outputs/tables/aggregate_metrics.csv").exists()
    assert "aggregation" in result["postprocess"]
