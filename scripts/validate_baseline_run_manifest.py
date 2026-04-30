"""Validate upstream baseline run manifests before artifact adaptation."""

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
    default_baseline_run_manifest_validation_path,
    validate_baseline_run_manifest,
)


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-validation-json")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-missing-artifact-paths",
        action="store_true",
        help=(
            "Allow declared input/ranking/log paths to be missing locally. Use "
            "only for template review or server-side preflight, not final claims."
        ),
    )
    args = parser.parse_args(argv)

    manifest_json = _resolve_repo_path(args.manifest_json)
    output_validation_json = (
        _resolve_repo_path(args.output_validation_json)
        if args.output_validation_json
        else default_baseline_run_manifest_validation_path(
            manifest_json=manifest_json or args.manifest_json,
            root=ROOT,
        )
    )
    validation = validate_baseline_run_manifest(
        manifest_json=manifest_json or args.manifest_json,
        output_validation_json=output_validation_json,
        strict=args.strict,
        require_artifact_paths=not args.allow_missing_artifact_paths,
        repo_root=ROOT,
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if args.strict and validation["validation_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
