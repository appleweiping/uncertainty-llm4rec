from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_list_required_artifacts_for_config() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/list_required_artifacts.py",
            "--config",
            "configs/experiments/real_ours_method_template.yaml",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "predictions.jsonl" in payload["required_artifacts"]
    assert "metrics.json" in payload["required_artifacts"]
    assert payload["planned_output_dir"] == "outputs/real_runs/ours_method"
    assert payload["template"] is True


def test_list_required_artifacts_reports_missing_run_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("seed: 13\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/list_required_artifacts.py", "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "resolved_config.yaml" in payload["present_required_artifacts"]
    assert "predictions.jsonl" in payload["missing_required_artifacts"]
