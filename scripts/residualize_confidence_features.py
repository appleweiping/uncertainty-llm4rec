"""Fit/apply the split-audited CURE/TRUCE popularity residual scaffold."""

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
    DEFAULT_PROBABILITY_SOURCE,
    SUPPORTED_PROBABILITY_SOURCES,
    residualize_feature_rows,
)


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _default_output_dir(features_jsonl: Path) -> Path:
    try:
        relative_parent = features_jsonl.resolve().parent.relative_to(
            ROOT / "outputs" / "confidence_features"
        )
        return ROOT / "outputs" / "confidence_residuals" / relative_parent
    except ValueError:
        return ROOT / "outputs" / "confidence_residuals" / features_jsonl.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--manifest-json")
    parser.add_argument("--fit-splits", default="train")
    parser.add_argument("--eval-splits", default="validation,test")
    parser.add_argument(
        "--probability-source",
        choices=sorted(SUPPORTED_PROBABILITY_SOURCES),
        default=DEFAULT_PROBABILITY_SOURCE,
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--allow-same-split-eval",
        action="store_true",
        help="Allow fit/eval split overlap only for explicitly labeled diagnostics.",
    )
    args = parser.parse_args(argv)

    features_jsonl = _resolve(args.features_jsonl)
    if features_jsonl is None or not features_jsonl.exists():
        raise SystemExit(f"Feature JSONL not found: {features_jsonl}")

    output_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir(features_jsonl)
    assert output_dir is not None
    output_jsonl = (
        _resolve(args.output_jsonl)
        if args.output_jsonl
        else output_dir / "popularity_residualized_features.jsonl"
    )
    manifest_json = (
        _resolve(args.manifest_json)
        if args.manifest_json
        else output_dir / "manifest.json"
    )
    assert output_jsonl is not None
    assert manifest_json is not None

    manifest = residualize_feature_rows(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        fit_splits=args.fit_splits,
        eval_splits=args.eval_splits,
        probability_source=args.probability_source,
        max_examples=args.max_examples,
        allow_same_split_eval=args.allow_same_split_eval,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
