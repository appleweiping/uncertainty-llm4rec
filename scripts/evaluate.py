"""Evaluate an existing Phase 1 prediction artifact."""

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
from llm4rec.experiments.config import load_config  # noqa: E402


def _run_dir(config: dict[str, object]) -> Path:
    seed = int(config.get("seed") or 0)
    run_name = str(config.get("run_name") or "smoke")
    return ROOT / str(config.get("output_dir") or "outputs/runs") / f"{run_name}_seed{seed}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(ROOT / args.config)
    run_dir = _run_dir(config)
    metrics = evaluate_predictions(
        predictions_jsonl=run_dir / "predictions.jsonl",
        output_dir=run_dir,
        top_k=[int(k) for k in config.get("top_k", [1, 5, 10])],
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
