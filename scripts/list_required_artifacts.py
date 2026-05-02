"""List required artifacts for a config or inspect a run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_config  # noqa: E402
from llm4rec.experiments.runner import _config_for_baseline  # noqa: E402

REQUIRED_ARTIFACTS = [
    "resolved_config.yaml",
    "environment.json",
    "logs.txt",
    "predictions.jsonl",
    "metrics.json",
    "metrics.csv",
    "cost_latency.json",
    "artifacts/",
]

OPTIONAL_ARTIFACTS = [
    "git_info.json",
    "raw_llm_outputs.jsonl",
    "checkpoints/",
    "reliability_diagram.csv",
    "risk_coverage.csv",
    "confidence_by_popularity_bucket.csv",
    "ablation_table.csv",
    "failure_cases.jsonl",
]


def list_required_artifacts(
    *,
    config_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not config_path and not run_dir:
        raise ValueError("provide --config or --run-dir")
    payload: dict[str, Any] = {
        "required_artifacts": REQUIRED_ARTIFACTS,
        "optional_artifacts": OPTIONAL_ARTIFACTS,
    }
    if config_path:
        config = load_config(config_path)
        seeds = config.get("seeds") if isinstance(config.get("seeds"), list) and config.get("seeds") else [config.get("seed", "SEED_TBD")]
        planned_run_dirs = _planned_run_dirs(config, seeds)
        output = config.get("output") if isinstance(config.get("output"), dict) else {}
        output_dir = str(output.get("output_dir") or config.get("output_dir") or "outputs/runs")
        payload.update(
            {
                "config": str(config_path),
                "planned_output_dir": output_dir,
                "planned_run_dirs": planned_run_dirs,
                "template": bool(config.get("template", False)),
            }
        )
    if run_dir:
        inspected = _inspect_run_dir(Path(run_dir))
        payload.update(inspected)
    return payload


def _planned_run_dirs(config: dict[str, Any], seeds: list[Any]) -> list[str]:
    baselines = config.get("baselines")
    if isinstance(baselines, list) and baselines:
        run_dirs = []
        for seed in seeds:
            for baseline in baselines:
                child = _config_for_baseline(config, str(baseline), seed=int(seed))
                output = child.get("output") if isinstance(child.get("output"), dict) else {}
                output_dir = str(output.get("output_dir") or child.get("output_dir") or "outputs/runs")
                run_name = str(output.get("run_name") or child.get("run_name") or "RUN_NAME_TBD")
                run_dirs.append(str(Path(output_dir) / f"{run_name}_seed{seed}"))
        return run_dirs
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    output_dir = str(output.get("output_dir") or config.get("output_dir") or "outputs/runs")
    run_name = str(output.get("run_name") or config.get("run_name") or "RUN_NAME_TBD")
    return [str(Path(output_dir) / f"{run_name}_seed{seed}") for seed in seeds]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    args = parser.parse_args(argv)
    result = list_required_artifacts(config_path=args.config, run_dir=args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _inspect_run_dir(run_dir: Path) -> dict[str, Any]:
    missing = []
    present = []
    for artifact in REQUIRED_ARTIFACTS:
        path = run_dir / artifact.rstrip("/")
        if path.exists():
            present.append(artifact)
        else:
            missing.append(artifact)
    return {
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "present_required_artifacts": present,
        "missing_required_artifacts": missing,
    }


if __name__ == "__main__":
    raise SystemExit(main())
