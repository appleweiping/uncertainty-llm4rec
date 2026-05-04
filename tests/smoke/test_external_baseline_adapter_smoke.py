import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_recbole_smoke_skips_when_not_installed() -> None:
    if importlib.util.find_spec("recbole") is None:
        pytest.skip("RecBole optional baseline dependency is not installed")
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_external_baseline.py"), "--config", str(repo / "configs" / "experiments" / "baseline_sasrec_amazon_beauty.yaml")],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode in {0, 2}
