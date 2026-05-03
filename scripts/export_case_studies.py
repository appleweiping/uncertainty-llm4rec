"""Export R3 OursMethod case studies from saved artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.analysis.case_studies import export_case_studies  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=50)
    args = parser.parse_args(argv)
    cases = export_case_studies(ROOT / args.runs, output_dir=ROOT / args.output, max_rows=args.max_rows)
    print(json.dumps({"status": "ok", "case_counts": {key: len(value) for key, value in cases.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

