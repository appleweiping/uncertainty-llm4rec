"""Write a project setup readiness report without running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.analysis.project_readiness import (  # noqa: E402
    build_project_readiness_manifest,
    write_project_readiness_report,
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/project_readiness/current")
    args = parser.parse_args(argv)

    outputs = write_project_readiness_report(
        output_dir=_resolve(args.output_dir),
        root=ROOT,
    )
    manifest = build_project_readiness_manifest(root=ROOT)
    manifest["outputs"] = outputs
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
