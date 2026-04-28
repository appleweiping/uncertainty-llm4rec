"""Build prompt JSONL inputs from processed Storyflow observation examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.observation import (  # noqa: E402
    build_observation_input_records,
    default_observation_input_path,
    processed_dataset_dir,
    write_observation_inputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-suffix", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--prompt-template", default="forced_json")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument(
        "--allow-target-in-candidates",
        action="store_true",
        help=(
            "Only for controlled diagnostics. Default excludes target item "
            "from catalog-constrained candidates to avoid answer leakage."
        ),
    )
    parser.add_argument("--stratify-by-popularity", action="store_true")
    parser.add_argument("--output-jsonl")
    args = parser.parse_args(argv)

    processed_dir = processed_dataset_dir(
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        root=ROOT,
    )
    if not processed_dir.exists():
        raise SystemExit(f"Processed dataset not found: {processed_dir}")

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_observation_input_path(
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        split=args.split,
        prompt_template=args.prompt_template,
        candidate_count=args.candidate_count,
        root=ROOT,
    )
    records = build_observation_input_records(
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        split=args.split,
        processed_dir=processed_dir,
        max_examples=args.max_examples,
        stratify_by_popularity=args.stratify_by_popularity,
        prompt_template=args.prompt_template,
        candidate_count=args.candidate_count,
        allow_target_in_candidates=args.allow_target_in_candidates,
    )
    manifest = write_observation_inputs(
        records,
        output_jsonl=output_jsonl,
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        split=args.split,
        prompt_template=args.prompt_template,
        stratify_by_popularity=args.stratify_by_popularity,
        candidate_count=args.candidate_count,
        allow_target_in_candidates=args.allow_target_in_candidates,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
