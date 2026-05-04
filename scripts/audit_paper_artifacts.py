#!/usr/bin/env python3
"""Audit paper-facing CU-GR v2 artifacts for missing files and placeholders."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TABLES = [
    "paper_main_results.csv",
    "paper_main_results.md",
    "paper_main_results.tex",
    "paper_ablation.csv",
    "paper_ablation.md",
    "paper_ablation.tex",
    "paper_uncertainty_observation.csv",
    "paper_uncertainty_observation.md",
    "paper_uncertainty_observation.tex",
    "paper_panel_analysis.csv",
    "paper_panel_analysis.md",
    "paper_panel_analysis.tex",
    "paper_cost_latency.csv",
    "paper_cost_latency.md",
    "paper_cost_latency.tex",
    "figure_calibration_reliability.csv",
    "figure_risk_coverage.csv",
    "figure_delta_vs_fallback_by_dataset.csv",
    "figure_swap_outcomes.csv",
    "figure_panel_coverage.csv",
]

REQUIRED_DOCS = [
    "paper_results_summary.md",
    "cu_gr_v2_method_summary.md",
    "cu_gr_v2_experiment_summary.md",
    "cu_gr_v2_limitations.md",
    "cu_gr_v2_reviewer_checklist.md",
]

DATASETS = {"MovieLens 1M", "Amazon Beauty"}
PLACEHOLDERS = re.compile(r"\b(TBD|TODO|PLACEHOLDER|dummy|fabricated)\b", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()

    tables = _resolve(args.tables)
    docs = _resolve(args.docs)
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED_TABLES:
        path = tables / name
        if not path.exists():
            errors.append(f"missing required table: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required table: {path}")
        elif PLACEHOLDERS.search(path.read_text(encoding="utf-8", errors="ignore")):
            errors.append(f"placeholder token found in table: {path}")

    for name in REQUIRED_DOCS:
        path = docs / name
        if not path.exists():
            errors.append(f"missing required doc: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required doc: {path}")
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if PLACEHOLDERS.search(text):
                errors.append(f"placeholder token found in doc: {path}")

    for name in ["paper_main_results.csv", "paper_ablation.csv", "paper_panel_analysis.csv", "paper_cost_latency.csv"]:
        rows = _read_csv(tables / name)
        present = {r.get("Dataset") for r in rows}
        missing = DATASETS - present
        if missing:
            errors.append(f"{name} missing dataset rows: {sorted(missing)}")
        if not rows:
            errors.append(f"{name} has no data rows")

    _check_numeric_metrics(tables / "paper_main_results.csv", errors)
    _check_numeric_metrics(tables / "paper_ablation.csv", errors)

    commit = _git("rev-parse", "HEAD")
    env_commits = _environment_commits()
    if commit == "unavailable":
        warnings.append("git commit hash unavailable")
    if not env_commits:
        warnings.append("no environment/git metadata commit hashes discovered in run artifacts")

    negative_paths = [
        tables / "r3_movielens_1m_uncertainty.csv",
        tables / "r3_movielens_1m_main_results.csv",
        tables / "r3b_conservative_gate_main.csv",
        tables / "r3_case_studies_high_conf_wrong.csv",
    ]
    for path in negative_paths:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"negative v1/free-form evidence missing: {path}")

    for path in [
        tables / "cu_gr_v2_full_seed_cost_latency.csv",
        tables / "cu_gr_v2_amazon_beauty_cost_latency.csv",
    ]:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"cost/latency file missing: {path}")

    raw_artifacts = [
        ROOT / "outputs/runs/r3_v2_movielens_preference_signal_subgate_full_seeds_seed42/preference_signals.jsonl",
        ROOT / "outputs/runs/r3_v2_amazon_beauty_preference_full_seeds_seed42/preference_signals.jsonl",
    ]
    for path in raw_artifacts:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"raw LLM preference artifact missing: {path}")

    result = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "tables": str(tables),
        "docs": str(docs),
        "git_head": commit,
        "environment_commit_hashes_found": sorted(env_commits)[:10],
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def _check_numeric_metrics(path: Path, errors: list[str]) -> None:
    rows = _read_csv(path)
    numeric_cols = ["Recall@10", "NDCG@10", "MRR@10", "HitRate@10"]
    for idx, row in enumerate(rows, start=2):
        available = any(str(row.get(col, "")).strip() for col in numeric_cols)
        if not available:
            continue
        for col in numeric_cols:
            value = str(row.get(col, "")).strip()
            if value == "":
                continue
            try:
                float(value)
            except ValueError:
                errors.append(f"{path.name}:{idx} nonnumeric {col}: {value}")


def _environment_commits() -> set[str]:
    commits: set[str] = set()
    for path in (ROOT / "outputs/runs").glob("*/environment.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ["git_commit", "commit", "git_sha", "head"]:
            value = data.get(key)
            if isinstance(value, str) and value:
                commits.add(value)
    for path in (ROOT / "outputs/runs").glob("*/git_info.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ["commit", "git_commit", "sha", "head"]:
            value = data.get(key)
            if isinstance(value, str) and value:
                commits.add(value)
    return commits


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
