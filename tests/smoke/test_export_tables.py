from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_export_tables_pipeline() -> None:
    result = run_all("configs/experiments/smoke_export_tables.yaml")
    assert result["run_count"] == 1
    assert "tables" in result["postprocess"]
    assert Path("outputs/tables/aggregate_metrics.csv").exists()
    assert Path("outputs/tables/aggregate_metrics.md").exists()
    assert Path("outputs/tables/aggregate_metrics.tex").exists()
    assert Path("outputs/tables/reliability_diagram.csv").exists()
    assert Path("outputs/tables/risk_coverage.csv").exists()
