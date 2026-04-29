"""Compare completed observation analysis summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.analysis import write_observation_comparison  # noqa: E402
from storyflow.analysis.observation_comparison import load_json  # noqa: E402


def _resolve(path: str | Path) -> Path:
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _parse_run_spec(spec: str) -> tuple[str, Path, Path | None]:
    if "=" not in spec:
        raise SystemExit(
            "--run must use label=analysis_summary.json[,case_review_summary.json]"
        )
    label, paths = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit("--run label must not be empty")
    parts = [part.strip() for part in paths.split(",") if part.strip()]
    if not parts:
        raise SystemExit(f"--run {label}=... is missing analysis summary path")
    analysis_path = _resolve(parts[0])
    case_path = _resolve(parts[1]) if len(parts) > 1 else None
    if len(parts) > 2:
        raise SystemExit(f"--run {label}=... has too many comma-separated paths")
    if not analysis_path.exists():
        raise SystemExit(f"Analysis summary not found: {analysis_path}")
    if case_path is not None and not case_path.exists():
        raise SystemExit(f"Case review summary not found: {case_path}")
    return label, analysis_path, case_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeatable label=analysis_summary.json[,case_review_summary.json].",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis_comparisons/observation_runs",
        help="Ignored comparison artifact directory.",
    )
    parser.add_argument("--source-label")
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    for spec in args.run:
        label, analysis_path, case_path = _parse_run_spec(spec)
        runs.append(
            {
                "label": label,
                "analysis_summary": load_json(analysis_path),
                "case_review_summary": load_json(case_path) if case_path else {},
                "analysis_summary_path": str(analysis_path),
                "case_review_summary_path": str(case_path) if case_path else None,
            }
        )
    manifest = write_observation_comparison(
        runs=runs,
        output_dir=_resolve(args.output_dir),
        source_label=args.source_label,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
