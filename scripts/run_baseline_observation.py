"""Run lightweight baseline recommenders through the observation schema."""

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
    default_baseline_output_dir,
    run_baseline_observation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--baseline",
        default="popularity",
        choices=["popularity", "cooccurrence", "ranking_jsonl"],
    )
    parser.add_argument(
        "--ranking-jsonl",
        help=(
            "External ranking baseline JSONL with input_id plus ranked_item_ids "
            "or ranked_items. Required for baseline=ranking_jsonl."
        ),
    )
    parser.add_argument(
        "--strict-ranking",
        action="store_true",
        help="Fail instead of falling back to popularity when a ranking row is missing or unusable.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.is_absolute():
        input_jsonl = ROOT / input_jsonl
    if not input_jsonl.exists():
        raise SystemExit(f"Observation input JSONL not found: {input_jsonl}")

    ranking_jsonl = Path(args.ranking_jsonl) if args.ranking_jsonl else None
    if ranking_jsonl is not None and not ranking_jsonl.is_absolute():
        ranking_jsonl = ROOT / ranking_jsonl
    if args.baseline == "ranking_jsonl":
        if ranking_jsonl is None:
            raise SystemExit("--ranking-jsonl is required when --baseline ranking_jsonl")
        if not ranking_jsonl.exists():
            raise SystemExit(f"Ranking JSONL not found: {ranking_jsonl}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_baseline_output_dir(
            input_jsonl=input_jsonl,
            baseline=args.baseline,
            root=ROOT,
        )
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    manifest = run_baseline_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        baseline=args.baseline,
        ranking_jsonl=ranking_jsonl,
        strict_ranking=args.strict_ranking,
        max_examples=args.max_examples,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
