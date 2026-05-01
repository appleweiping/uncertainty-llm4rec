from __future__ import annotations

import json
from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_mf_pipeline() -> None:
    result = run_all("configs/experiments/smoke_mf.yaml")
    run_dir = Path(result["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert (run_dir / "logs.txt").exists()
    assert metrics["aggregate"]["validity_rate"] == 1.0
    assert metrics["aggregate"]["hallucination_rate"] == 0.0
