"""Run synthetic confidence-guided exposure simulation from feature JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.confidence import (  # noqa: E402
    DEFAULT_RERANK_CONFIDENCE_SOURCE,
    SUPPORTED_RERANK_CONFIDENCE_SOURCES,
)
from storyflow.simulation import (  # noqa: E402
    SUPPORTED_EXPOSURE_POLICIES,
    ExposureSimulationConfig,
    simulate_exposure_feedback_jsonl,
)


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _default_output_dir(features_jsonl: Path) -> Path:
    resolved_parent = features_jsonl.resolve().parent
    for source_dir in (
        ROOT / "outputs" / "confidence_reranking",
        ROOT / "outputs" / "confidence_residuals",
        ROOT / "outputs" / "confidence_calibration",
        ROOT / "outputs" / "confidence_features",
    ):
        try:
            relative_parent = resolved_parent.relative_to(source_dir)
            return ROOT / "outputs" / "echo_simulation" / relative_parent
        except ValueError:
            continue
    return ROOT / "outputs" / "echo_simulation" / features_jsonl.stem


def _policies(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    unsupported = sorted(set(parsed) - set(SUPPORTED_EXPOSURE_POLICIES))
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported policies: {unsupported}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--summary-json")
    parser.add_argument("--manifest-json")
    parser.add_argument("--policies", type=_policies, default=SUPPORTED_EXPOSURE_POLICIES)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--group-key", default="input_id")
    parser.add_argument("--exposures-per-group", type=int, default=1)
    parser.add_argument(
        "--confidence-source",
        choices=sorted(SUPPORTED_RERANK_CONFIDENCE_SOURCES),
        default=DEFAULT_RERANK_CONFIDENCE_SOURCE,
    )
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--confidence-weight", type=float, default=0.5)
    parser.add_argument("--feedback-learning-rate", type=float, default=0.2)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args(argv)

    features_jsonl = _resolve(args.features_jsonl)
    if features_jsonl is None or not features_jsonl.exists():
        raise SystemExit(f"Feature JSONL not found: {features_jsonl}")

    output_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir(features_jsonl)
    assert output_dir is not None
    output_jsonl = (
        _resolve(args.output_jsonl)
        if args.output_jsonl
        else output_dir / "exposure_records.jsonl"
    )
    summary_json = (
        _resolve(args.summary_json)
        if args.summary_json
        else output_dir / "simulation_summary.json"
    )
    manifest_json = (
        _resolve(args.manifest_json)
        if args.manifest_json
        else output_dir / "manifest.json"
    )
    assert output_jsonl is not None
    assert summary_json is not None
    assert manifest_json is not None

    config = ExposureSimulationConfig(
        policies=args.policies,
        rounds=args.rounds,
        group_key=args.group_key,
        exposures_per_group=args.exposures_per_group,
        confidence_source=args.confidence_source,
        utility_weight=args.utility_weight,
        confidence_weight=args.confidence_weight,
        feedback_learning_rate=args.feedback_learning_rate,
    )
    manifest = simulate_exposure_feedback_jsonl(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        manifest_json=manifest_json,
        config=config,
        max_examples=args.max_examples,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
