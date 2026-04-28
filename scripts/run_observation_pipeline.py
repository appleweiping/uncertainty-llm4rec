"""Run the Phase 2A no-API observation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.observation import default_mock_output_dir, run_mock_observation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["mock"])
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--mock-mode",
        default="popularity_biased",
        choices=["oracle-ish", "popularity_biased", "random"],
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--low-confidence-tau", type=float, default=0.5)
    parser.add_argument("--high-confidence-tau", type=float, default=0.7)
    args = parser.parse_args(argv)

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.is_absolute():
        input_jsonl = ROOT / input_jsonl
    if not input_jsonl.exists():
        raise SystemExit(f"Observation input JSONL not found: {input_jsonl}")
    output_dir = Path(args.output_dir) if args.output_dir else default_mock_output_dir(
        input_jsonl=input_jsonl,
        provider_mode=args.mock_mode,
        root=ROOT,
    )

    manifest = run_mock_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        provider_mode=args.mock_mode,
        max_examples=args.max_examples,
        resume=not args.no_resume,
        seed=args.seed,
        low_confidence_tau=args.low_confidence_tau,
        high_confidence_tau=args.high_confidence_tau,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
