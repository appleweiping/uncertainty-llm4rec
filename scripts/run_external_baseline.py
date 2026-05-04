#!/usr/bin/env python3
"""Prepare and run an external baseline adapter.

The current implementation exports data and writes a RecBole config, then fails
clearly if RecBole is unavailable. It never reports external metrics as TRUCE
paper metrics.
"""

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
from llm4rec.external_baselines.recbole_adapter import build_recbole_config, run_recbole_training, write_recbole_config  # noqa: E402
from llm4rec.external_baselines.sasrec_adapter import sasrec_config  # noqa: E402
from llm4rec.external_baselines.lightgcn_adapter import lightgcn_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    baseline = config.get("external_baseline") if isinstance(config.get("external_baseline"), dict) else {}
    name = str(baseline.get("name") or "")
    out_dir = ROOT / str(config.get("output_dir") or "outputs/external_baselines")
    dataset_name = str(baseline.get("dataset_name") or dataset.get("name") or config.get("run_name"))
    processed_dir = str(dataset.get("processed_dir") or "")
    if not processed_dir:
        raise SystemExit("config.dataset.processed_dir is required")
    if name == "sasrec_recbole":
        ext_config = sasrec_config(dataset_name=dataset_name, processed_dir=processed_dir, output_dir=out_dir, seed=int(config.get("seed") or 0), training_config=baseline.get("training") or {})
    elif name == "lightgcn_recbole":
        ext_config = lightgcn_config(dataset_name=dataset_name, processed_dir=processed_dir, output_dir=out_dir, seed=int(config.get("seed") or 0), training_config=baseline.get("training") or {})
    else:
        raise SystemExit(f"unsupported external_baseline.name: {name}")
    manifest = export_recbole_atomic(processed_dir=processed_dir, output_dir=out_dir / "recbole_data", dataset_name=dataset_name, seed=ext_config.seed)
    recbole_cfg = build_recbole_config(ext_config, exported_dataset_dir=manifest["exported_dir"])
    cfg_path = write_recbole_config(out_dir / f"{dataset_name}_{name}_recbole_config.json", recbole_cfg)
    try:
        run_recbole_training(ext_config, exported_dataset_dir=manifest["exported_dir"])
    except Exception as exc:
        print(json.dumps({"status": "adapter_prepared_training_not_completed", "reason": str(exc), "export_manifest": manifest, "recbole_config": str(cfg_path)}, indent=2))
        return 2
    print(json.dumps({"status": "completed", "export_manifest": manifest, "recbole_config": str(cfg_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
