from __future__ import annotations

import json

from llm4rec.evaluation.aggregation import aggregate_run_metrics


def test_aggregate_run_metrics_writes_csv(tmp_path) -> None:
    for seed, value in [(1, 0.5), (2, 1.0)]:
        run_dir = tmp_path / "runs" / f"run_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({
                "metadata": {"method": "popularity", "methods": ["popularity"], "dataset": "tiny", "seed": seed},
                "aggregate": {"recall@1": value},
                "by_domain": {"tiny": {"recall@1": value}},
            }),
            encoding="utf-8",
        )
    manifest = aggregate_run_metrics(tmp_path / "runs", output_dir=tmp_path / "tables")
    assert manifest["run_count"] == 2
    assert (tmp_path / "tables" / "aggregate_metrics.csv").exists()
    recall_rows = [row for row in manifest["rows"] if row["domain"] == "aggregate" and row["metric"] == "recall@1"]
    assert recall_rows[0]["count"] == 2
    assert recall_rows[0]["mean"] == 0.75
