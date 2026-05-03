"""End-to-end CU-GR scripts on a tiny slice of real R3 artifacts (optional)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
RUNS = ROOT / "outputs" / "runs"
MARKER = RUNS / "r3_movielens_1m_real_llm_full_candidate500_bm25_seed13" / "predictions.jsonl"


@pytest.mark.skipif(not MARKER.is_file(), reason="R3 predictions not present")
def test_cu_gr_pipeline_smoke(tmp_path: Path) -> None:
    ds = tmp_path / "cu_gr_calibrator_dataset.csv"
    model = tmp_path / "model"
    tables = tmp_path / "tables"
    tables.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_calibrator_dataset.py"),
            "--runs",
            str(RUNS),
            "--output",
            str(ds),
            "--max-rows-per-seed",
            "120",
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        env=SUBENV,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_override_calibrator.py"),
            "--input",
            str(ds),
            "--output",
            str(model),
            "--tables-dir",
            str(tables),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        env=SUBENV,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "replay_cu_gr_policy.py"),
            "--input",
            str(ds),
            "--model",
            str(model),
            "--output",
            str(tables),
            "--runs",
            str(RUNS),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        env=SUBENV,
    )
    policy = tables / "cu_gr_policy_results.csv"
    assert policy.is_file()
    rows = list(csv.DictReader(policy.open(encoding="utf-8")))
    assert any(r.get("method") == "cu_gr" for r in rows)
    meta = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
    assert "feature_names" in meta
