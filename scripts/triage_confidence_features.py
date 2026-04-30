"""Assign uncertainty-aware diagnostic triage reason codes to feature JSONL."""

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
from storyflow.triage import TriageConfig, triage_features_jsonl  # noqa: E402


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
            return ROOT / "outputs" / "confidence_triage" / relative_parent
        except ValueError:
            continue
    return ROOT / "outputs" / "confidence_triage" / features_jsonl.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--manifest-json")
    parser.add_argument(
        "--confidence-source",
        choices=sorted(SUPPORTED_RERANK_CONFIDENCE_SOURCES),
        default=DEFAULT_RERANK_CONFIDENCE_SOURCE,
    )
    parser.add_argument("--high-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--low-grounding-threshold", type=float, default=0.45)
    parser.add_argument("--high-ambiguity-threshold", type=float, default=0.50)
    parser.add_argument("--high-echo-risk-threshold", type=float, default=0.35)
    parser.add_argument("--low-novelty-threshold", type=float, default=0.25)
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
        else output_dir / "triaged_features.jsonl"
    )
    manifest_json = (
        _resolve(args.manifest_json)
        if args.manifest_json
        else output_dir / "manifest.json"
    )
    assert output_jsonl is not None
    assert manifest_json is not None

    config = TriageConfig(
        confidence_source=args.confidence_source,
        high_confidence_threshold=args.high_confidence_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        low_grounding_threshold=args.low_grounding_threshold,
        high_ambiguity_threshold=args.high_ambiguity_threshold,
        high_echo_risk_threshold=args.high_echo_risk_threshold,
        low_novelty_threshold=args.low_novelty_threshold,
    )
    manifest = triage_features_jsonl(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        config=config,
        max_examples=args.max_examples,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
