"""Inspect Amazon Reviews 2023 readiness without downloading full data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.data import inspect_amazon_config, write_amazon_readiness_report  # noqa: E402
from storyflow.utils.config import load_simple_yaml  # noqa: E402


def _config_path(dataset: str) -> Path:
    return ROOT / "configs" / "datasets" / f"{dataset}.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--check-online", action="store_true")
    parser.add_argument("--sample-records", type=int, default=0)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    config_path = _config_path(args.dataset)
    if not config_path.exists():
        raise SystemExit(f"Unknown dataset config: {config_path}")
    config = load_simple_yaml(config_path)
    if str(config.get("type")) != "amazon_reviews_2023":
        raise SystemExit(f"Dataset is not Amazon Reviews 2023: {args.dataset}")
    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "outputs" / "amazon_reviews_2023" / args.dataset / "inspect"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = inspect_amazon_config(
        config,
        check_online=args.check_online,
        sample_records=args.sample_records,
    )
    manifest["config_path"] = str(config_path)
    manifest["dry_run"] = True
    manifest["note"] = "No full Amazon data was downloaded or processed."
    manifest_path = output_dir / "availability_manifest.json"
    report_path = output_dir / "readiness_report.md"
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_amazon_readiness_report(manifest, report_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] != "online_check_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
