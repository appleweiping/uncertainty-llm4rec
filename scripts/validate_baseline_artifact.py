"""Validate external baseline ranking artifacts before observation adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.baselines import (  # noqa: E402
    default_baseline_artifact_manifest_path,
    validate_baseline_artifact,
)


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-jsonl", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--baseline-family",
        required=True,
        help="Baseline family that produced the ranking artifact, e.g. sasrec or lightgcn.",
    )
    parser.add_argument("--model-family")
    parser.add_argument("--run-label")
    parser.add_argument("--dataset")
    parser.add_argument("--processed-suffix")
    parser.add_argument("--split")
    parser.add_argument(
        "--trained-splits",
        default="train",
        help="Comma-separated split names used to fit the upstream ranker.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config-path")
    parser.add_argument("--source-manifest-json")
    parser.add_argument("--output-manifest-json")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing-inputs", action="store_true")
    parser.add_argument("--fail-on-extra-rankings", action="store_true")
    args = parser.parse_args(argv)

    input_jsonl = _resolve_repo_path(args.input_jsonl)
    ranking_jsonl = _resolve_repo_path(args.ranking_jsonl)
    output_manifest_json = (
        _resolve_repo_path(args.output_manifest_json)
        if args.output_manifest_json
        else default_baseline_artifact_manifest_path(
            ranking_jsonl=ranking_jsonl or args.ranking_jsonl,
            root=ROOT,
        )
    )
    config_path = _resolve_repo_path(args.config_path)
    source_manifest_json = _resolve_repo_path(args.source_manifest_json)
    trained_splits = [
        split.strip()
        for split in str(args.trained_splits).split(",")
        if split.strip()
    ]

    manifest = validate_baseline_artifact(
        ranking_jsonl=ranking_jsonl or args.ranking_jsonl,
        input_jsonl=input_jsonl or args.input_jsonl,
        baseline_family=args.baseline_family,
        output_manifest_json=output_manifest_json,
        model_family=args.model_family,
        run_label=args.run_label,
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        split=args.split,
        trained_splits=trained_splits,
        seed=args.seed,
        config_path=config_path,
        source_manifest_json=source_manifest_json,
        max_examples=args.max_examples,
        strict=args.strict,
        allow_missing_inputs=args.allow_missing_inputs,
        fail_on_extra_rankings=args.fail_on_extra_rankings,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if args.strict and manifest["validation_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
