"""Analyze grounded observation outputs and register ignored analysis artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.analysis import analyze_observation_run, append_registry_record  # noqa: E402


def _resolve(path: str | Path) -> Path:
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _default_output_dir(run_dir: Path | None, grounded_jsonl: Path) -> Path:
    if run_dir is not None:
        try:
            relative = run_dir.resolve().relative_to(ROOT / "outputs")
            return ROOT / "outputs" / "analysis" / relative
        except ValueError:
            return ROOT / "outputs" / "analysis" / run_dir.name
    return ROOT / "outputs" / "analysis" / grounded_jsonl.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Observation run directory containing grounded_predictions.jsonl.")
    parser.add_argument("--grounded-jsonl", help="Grounded predictions JSONL path.")
    parser.add_argument("--failed-jsonl", help="Failed cases JSONL path.")
    parser.add_argument("--manifest-json", help="Source observation manifest path.")
    parser.add_argument("--output-dir", help="Analysis output directory under outputs/ by default.")
    parser.add_argument("--registry-jsonl", default="outputs/run_registry/observation_runs.jsonl")
    parser.add_argument("--source-label")
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--low-confidence-tau", type=float, default=0.5)
    parser.add_argument("--high-confidence-tau", type=float, default=0.7)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=20)
    args = parser.parse_args(argv)

    run_dir = _resolve(args.run_dir) if args.run_dir else None
    if run_dir is None and not args.grounded_jsonl:
        raise SystemExit("Provide --run-dir or --grounded-jsonl")

    grounded_jsonl = (
        _resolve(args.grounded_jsonl)
        if args.grounded_jsonl
        else run_dir / "grounded_predictions.jsonl"  # type: ignore[operator]
    )
    failed_jsonl = (
        _resolve(args.failed_jsonl)
        if args.failed_jsonl
        else (run_dir / "failed_cases.jsonl" if run_dir else None)
    )
    manifest_json = (
        _resolve(args.manifest_json)
        if args.manifest_json
        else (run_dir / "manifest.json" if run_dir else None)
    )
    if not grounded_jsonl.exists():
        raise SystemExit(f"Grounded predictions not found: {grounded_jsonl}")
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else _default_output_dir(run_dir, grounded_jsonl)
    )
    analysis_manifest = analyze_observation_run(
        grounded_jsonl=grounded_jsonl,
        failed_jsonl=failed_jsonl if failed_jsonl and failed_jsonl.exists() else None,
        manifest_json=manifest_json if manifest_json and manifest_json.exists() else None,
        output_dir=output_dir,
        low_confidence_tau=args.low_confidence_tau,
        high_confidence_tau=args.high_confidence_tau,
        n_bins=args.n_bins,
        max_cases=args.max_cases,
    )
    registry_record = None
    if not args.no_registry:
        registry_record = append_registry_record(
            registry_jsonl=_resolve(args.registry_jsonl),
            analysis_manifest=analysis_manifest,
            source_label=args.source_label,
        )
    result = {
        **analysis_manifest,
        "registry_record": registry_record,
        "registry_jsonl": None if args.no_registry else str(_resolve(args.registry_jsonl)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
