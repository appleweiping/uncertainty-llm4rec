from __future__ import annotations

import json

from llm4rec.evaluation.table_export import export_phase5_tables


def test_export_phase5_tables_writes_aggregate_and_confidence_tables(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "smoke_llm_seed13"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "metadata": {"method": "llm_generative_mock", "methods": ["llm_generative_mock"], "dataset": "tiny", "seed": 13},
            "aggregate": {
                "recall@1": 0.0,
                "confidence": {"selective_risk_coverage_data": [{"coverage": 1.0, "risk": 1.0, "threshold": 0.8}]},
                "calibration": {"reliability_diagram_data": [{"bin": 8, "count": 1, "accuracy": 0.0, "mean_confidence": 0.8}]},
                "long_tail": {"confidence_by_popularity_bucket": {"head": {"count": 1, "mean_confidence": 0.8}}},
            },
            "by_domain": {"tiny": {"recall@1": 0.0}},
        }),
        encoding="utf-8",
    )
    manifest = export_phase5_tables(tmp_path / "runs", output_dir=tmp_path / "tables")
    assert (tmp_path / "tables" / "aggregate_metrics.csv").exists()
    assert (tmp_path / "tables" / "aggregate_metrics.md").exists()
    assert (tmp_path / "tables" / "aggregate_metrics.tex").exists()
    assert (tmp_path / "tables" / "reliability_diagram.csv").exists()
    assert (tmp_path / "tables" / "risk_coverage.csv").exists()
    assert (tmp_path / "tables" / "confidence_by_popularity_bucket.csv").exists()
    assert "experiment_summary_json" in manifest
