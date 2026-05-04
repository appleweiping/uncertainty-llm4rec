#!/usr/bin/env python3
"""Export TRUCE processed data into RecBole atomic format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_config  # noqa: E402
from llm4rec.external_baselines.data_export import export_recbole_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/external_baselines/recbole"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    baseline = config.get("external_baseline") if isinstance(config.get("external_baseline"), dict) else {}
    training = baseline.get("training") if isinstance(baseline.get("training"), dict) else {}
    processed_dir = dataset.get("processed_dir")
    dataset_name = baseline.get("dataset_name") or dataset.get("name") or config.get("run_name")
    if not processed_dir:
        raise SystemExit("config.dataset.processed_dir is required")
    manifest = export_recbole_atomic(
        processed_dir=processed_dir,
        output_dir=args.output,
        dataset_name=str(dataset_name),
        seed=int(config.get("seed") or 0),
        sasrec_max_item_list_length=int(training.get("MAX_ITEM_LIST_LENGTH") or 50),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
