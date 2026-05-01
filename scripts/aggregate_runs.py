"""Aggregate metrics across run directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.aggregation import aggregate_run_metrics  # noqa: E402
from llm4rec.evaluation.table_export import export_phase5_tables  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output_dir = ROOT / args.output
    manifest = aggregate_run_metrics(ROOT / args.input, output_dir=output_dir)
    manifest["tables"] = export_phase5_tables(ROOT / args.input, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
