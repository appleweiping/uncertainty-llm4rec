#!/usr/bin/env python3
"""Recompute CU-GR v2 fusion tables + model.json offline from preference_signals.jsonl."""

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


def _pick_latest_signals(*, repo: Path, runs_roots: list[Path], run_name_prefix: str) -> Path | None:
    best: Path | None = None
    best_mtime = -1.0
    for rel in runs_roots:
        root = Path(rel)
        if not root.is_absolute():
            root = repo / root
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob(f"{run_name_prefix}*_seed*/preference_signals.jsonl")):
            try:
                m = candidate.stat().st_mtime
            except OSError:
                continue
            if m > best_mtime:
                best_mtime = m
                best = candidate
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", nargs="*", type=Path, default=[Path("outputs/runs")])
    parser.add_argument("--run-name-prefix", default="r3_v2_movielens_preference_signal_subgate")
    parser.add_argument("--signals", type=Path, default=None)
    parser.add_argument("--processed-dir", default="data/processed/movielens_1m/r2_full_single_dataset")
    parser.add_argument("--model-output", type=Path, default=Path("outputs/models/cu_gr_v2_preference_fusion"))
    args = parser.parse_args(argv)

    if args.signals:
        signals = args.signals if args.signals.is_absolute() else _ROOT / args.signals
        if not signals.is_file():
            raise SystemExit(f"signals missing: {signals}")
    else:
        sig = _pick_latest_signals(repo=_ROOT, runs_roots=list(args.runs_root), run_name_prefix=args.run_name_prefix)
        if sig is None:
            raise SystemExit("no preference_signals.jsonl found; pass --signals")
        signals = sig

    metrics = export_tables_after_signals(
        ROOT=_ROOT,
        signals_path=signals,
        processed_dir=args.processed_dir,
        run_dir_hint=signals.parent,
    )
    mo = Path(args.model_output)
    mo = mo if mo.is_absolute() else _ROOT / mo
    model_path = mo / "model.json"
    print(json.dumps({"signals": str(signals), "model_json": str(model_path), "ndcg_fallback": metrics.get("fallback_ndcg@10"), "fusion_delta": metrics.get("fusion_delta_ndcg10_vs_fallback")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
