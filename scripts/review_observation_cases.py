"""Generate pilot case-review and failure-taxonomy artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.analysis import review_observation_cases  # noqa: E402


def _resolve(path: str | Path) -> Path:
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _default_output_dir(run_dir: Path | None, grounded_jsonl: Path) -> Path:
    if run_dir is not None:
        try:
            relative = run_dir.resolve().relative_to(ROOT / "outputs")
            return ROOT / "outputs" / "case_reviews" / relative
        except ValueError:
            return ROOT / "outputs" / "case_reviews" / run_dir.name
    return ROOT / "outputs" / "case_reviews" / grounded_jsonl.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Observation run directory containing grounded_predictions.jsonl.")
    parser.add_argument("--grounded-jsonl", help="Grounded predictions JSONL path.")
    parser.add_argument("--input-jsonl", help="Observation input JSONL path. Defaults to manifest input_jsonl.")
    parser.add_argument("--failed-jsonl", help="Failed cases JSONL path.")
    parser.add_argument("--manifest-json", help="Observation run manifest path.")
    parser.add_argument("--output-dir", help="Case review output directory under outputs/ by default.")
    parser.add_argument("--low-confidence-tau", type=float, default=0.5)
    parser.add_argument("--high-confidence-tau", type=float, default=0.7)
    parser.add_argument("--ambiguity-tau", type=float, default=0.75)
    parser.add_argument("--popularity-ratio-tau", type=float, default=2.0)
    parser.add_argument("--max-history-titles", type=int, default=8)
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
    input_jsonl = _resolve(args.input_jsonl) if args.input_jsonl else None
    if not grounded_jsonl.exists():
        raise SystemExit(f"Grounded predictions not found: {grounded_jsonl}")
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else _default_output_dir(run_dir, grounded_jsonl)
    )

    review_manifest = review_observation_cases(
        grounded_jsonl=grounded_jsonl,
        failed_jsonl=failed_jsonl if failed_jsonl and failed_jsonl.exists() else None,
        manifest_json=manifest_json if manifest_json and manifest_json.exists() else None,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        low_confidence_tau=args.low_confidence_tau,
        high_confidence_tau=args.high_confidence_tau,
        ambiguity_tau=args.ambiguity_tau,
        popularity_ratio_tau=args.popularity_ratio_tau,
        max_history_titles=args.max_history_titles,
    )
    print(json.dumps(review_manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
