"""Analyze R3 OursMethod decisions from saved artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.analysis.ours_error_decomposition import (  # noqa: E402
    decision_attribution,
    missing_required_artifacts,
    rerank_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runs_dir = ROOT / args.runs
    output_dir = ROOT / args.output
    missing = missing_required_artifacts(runs_dir)
    if missing:
        print(json.dumps({"status": "BLOCKER", "missing_artifacts": missing}, indent=2), file=sys.stderr)
        return 2
    attribution = decision_attribution(runs_dir, output_dir=output_dir)
    rerank = rerank_audit(runs_dir, output_dir=output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "decision_attribution_rows": len(attribution["decision_attribution"]),
                "delta_rows": len(attribution["deltas"]),
                "rerank_audit_rows": len(rerank),
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

