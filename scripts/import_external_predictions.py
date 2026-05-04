#!/usr/bin/env python3
"""Import external per-candidate scores into TRUCE prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.external_baselines.prediction_import import import_scored_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-project", default="external")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = import_scored_candidates(
        scores_path=args.scores,
        examples_path=args.examples,
        output_path=args.output,
        method=args.method,
        source_project=args.source_project,
        model_name=args.model_name,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
