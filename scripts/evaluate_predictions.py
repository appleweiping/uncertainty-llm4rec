#!/usr/bin/env python3
"""Evaluate a prediction JSONL file with the TRUCE evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.evaluator import evaluate_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()
    metrics = evaluate_predictions(
        predictions_jsonl=args.predictions,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps({"metrics": metrics, "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
