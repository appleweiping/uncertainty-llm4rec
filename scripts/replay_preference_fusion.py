#!/usr/bin/env python3
"""Replay CU-GR v2 offline fusion/table export from saved preference_signals (diagnostic subset)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm4rec.experiments.cu_gr_v2_preference import export_tables_after_signals  # noqa: E402


def _table_prefix_for_input(input_path: Path) -> str:
    name = input_path.name
    suffix = "_preference_dataset.csv"
    if name.startswith("cu_gr_v2_") and name.endswith(suffix):
        return name[: -len(suffix)]
    return "cu_gr_v2_full_seed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Optional dataset path for bookkeeping.")
    parser.add_argument("--signals", type=Path, default=None)
    parser.add_argument("--runs-root", nargs="*", type=Path, default=[Path("outputs/runs")])
    parser.add_argument("--run-name-prefix", default="r3_v2_movielens_preference_signal_subgate")
    parser.add_argument("--model", type=Path, default=Path("outputs/models/cu_gr_v2_preference_fusion"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--processed-dir", default="data/processed/movielens_1m/r2_full_single_dataset")
    args = parser.parse_args(argv)

    inp = Path(args.input)
    inp = inp if inp.is_absolute() else _ROOT / inp
    if not inp.exists():
        print(json.dumps({"warning": "dataset_csv_optional_missing", "input": str(inp)}, indent=2))

    mo = Path(args.model)
    model_dir = mo if mo.is_absolute() else _ROOT / mo
    table_prefix = _table_prefix_for_input(inp)
    full_seed_weights = _ROOT / "outputs" / "tables" / f"{table_prefix}_fusion_weights.csv"
    if full_seed_weights.exists() and (model_dir / "model.json").exists():
        out_dir = Path(args.output)
        out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "cu_gr_v2_replay_manifest.json"
        full_seed_files = [
            out_dir / name
            for name in (
                f"{table_prefix}_main.csv",
                f"{table_prefix}_by_seed.csv",
                f"{table_prefix}_fusion_weights.csv",
                f"{table_prefix}_swap_analysis.csv",
                f"{table_prefix}_parser_stats.csv",
                f"{table_prefix}_panel_coverage.csv",
                f"{table_prefix}_cost_latency.csv",
                f"{table_prefix}_failure_cases.csv",
            )
        ]
        marker.write_text(
            json.dumps(
                {
                    "mode": "full_seed_artifacts_already_materialized",
                    "dataset_input": str(inp),
                    "model": str(model_dir / "model.json"),
                    "files_refreshed": [str(p) for p in full_seed_files if p.exists()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"manifest": str(marker), "model": str(model_dir / "model.json")}, indent=2))
        return 0

    if args.signals:
        signals = args.signals if args.signals.is_absolute() else _ROOT / args.signals
    else:
        best = None
        best_mtime = -1.0
        for rr in args.runs_root:
            rt = rr if rr.is_absolute() else _ROOT / rr
            if not rt.is_dir():
                continue
            for candidate in rt.glob(f"{args.run_name_prefix}*_seed*/preference_signals.jsonl"):
                try:
                    mt = candidate.stat().st_mtime
                except OSError:
                    continue
                if mt > best_mtime:
                    best_mtime = mt
                    best = candidate
        signals = best
        if signals is None:
            raise SystemExit("no preference_signals.jsonl matched; pass --signals")

    export_tables_after_signals(
        ROOT=_ROOT,
        signals_path=signals,
        processed_dir=args.processed_dir,
        run_dir_hint=signals.parent,
    )
    tbl = _ROOT / "outputs/tables"
    touched = [
        tbl / name
        for name in (
            "cu_gr_v2_preference_dataset.csv",
            "cu_gr_v2_preference_parser_stats.csv",
            "cu_gr_v2_fusion_results.csv",
            "cu_gr_v2_vs_fallback.csv",
            "cu_gr_v2_swap_analysis.csv",
            "cu_gr_v2_feature_importance.csv",
            "cu_gr_v2_cost_latency.csv",
        )
    ]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "cu_gr_v2_replay_manifest.json"
    marker.write_text(
        json.dumps({"signals": str(signals), "dataset_input": str(inp), "files_refreshed": [str(p) for p in touched]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"signals": str(signals), "manifest": str(marker)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
