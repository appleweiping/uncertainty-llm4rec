from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_validate_experiment_ready_accepts_safe_ours_template() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_experiment_ready.py",
            "--config",
            "configs/experiments/real_ours_method_template.yaml",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ready"] is True
    assert payload["checks"]["method_configured"] is True
    assert payload["checks"]["metrics_configured"] is True
    assert payload["checks"]["leakage_safeguards_present"] is True
    assert payload["dry_run"] is True
    assert payload["requires_confirm"] is True


def test_validate_experiment_ready_rejects_missing_method(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join(
            [
                "dry_run: true",
                "requires_confirm: true",
                "dataset:",
                "  name: tiny",
                "  processed_dir: TBD",
                "split: test",
                "candidate:",
                "  protocol: full",
                "metrics: [ranking]",
                "safety:",
                "  leakage_safeguards: true",
                "output:",
                f"  output_dir: {tmp_path}",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_experiment_ready.py", "--config", str(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ready"] is False
    assert "check failed: method_configured" in payload["errors"]
