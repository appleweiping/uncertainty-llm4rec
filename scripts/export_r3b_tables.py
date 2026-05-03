"""Export R3b conservative-gate CSV tables from outputs/runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.analysis.r3b_table_export import export_r3b_tables  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--output", default="outputs/tables")
    args = parser.parse_args(argv)
    manifest = export_r3b_tables(runs_dir=ROOT / args.runs, output_dir=ROOT / args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
