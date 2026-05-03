#!/usr/bin/env python3
"""Rebuild cu_gr_v2 preference dataset CSV from preference_signals.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm4rec.experiments.cu_gr_v2_preference import build_dataset_csv_from_signal_paths, build_dataset_csv_from_signals  # noqa: E402


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


def _pick_latest_signal_group(*, repo: Path, runs_roots: list[Path], run_name_prefix: str) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    group_mtimes: dict[str, float] = {}
    for rel in runs_roots:
        root = Path(rel)
        if not root.is_absolute():
            root = repo / root
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob(f"{run_name_prefix}*_seed*/preference_signals.jsonl")):
            run_dir = candidate.parent.name
            if "_seed" not in run_dir:
                continue
            group = run_dir.rsplit("_seed", 1)[0]
            groups.setdefault(group, []).append(candidate)
            try:
                group_mtimes[group] = max(group_mtimes.get(group, -1.0), candidate.stat().st_mtime)
            except OSError:
                pass
    if not groups:
        return []
    latest = max(groups, key=lambda key: group_mtimes.get(key, -1.0))
    return sorted(groups[latest])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=[Path("outputs/runs")], help="directories to scan for *_seed/preference_signals.jsonl")
    parser.add_argument("--run-name-prefix", default="r3_v2_movielens_preference_signal_subgate")
    parser.add_argument("--preferences-jsonl", type=Path, default=None, help="If set, use this file directly")
    parser.add_argument("--processed-dir", default="data/processed/movielens_1m/r2_full_single_dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables"),
        help="Output directory or a full *.csv path",
    )
    args = parser.parse_args(argv)

    if args.preferences_jsonl:
        signals = args.preferences_jsonl
        signals = signals if signals.is_absolute() else _ROOT / signals
        if not signals.is_file():
            raise SystemExit(f"preferences-jsonl missing: {signals}")
        signal_group = [signals]
    else:
        signal_group = _pick_latest_signal_group(repo=_ROOT, runs_roots=list(args.runs), run_name_prefix=args.run_name_prefix)
        sig = signal_group[-1] if signal_group else _pick_latest_signals(repo=_ROOT, runs_roots=list(args.runs), run_name_prefix=args.run_name_prefix)
        if sig is None:
            raise SystemExit("no preference_signals.jsonl matched; specify --preferences-jsonl")
        signals = sig
        if not signal_group:
            signal_group = [signals]

    out = args.output
    if str(out).endswith(".csv"):
        dataset_csv = out if out.is_absolute() else _ROOT / out
        stats_parent = dataset_csv.parent
    else:
        stats_parent = out if out.is_absolute() else _ROOT / out
        dataset_csv = stats_parent / "cu_gr_v2_preference_dataset.csv"
    stats_csv = stats_parent / "cu_gr_v2_preference_parser_stats.csv"

    if len(signal_group) == 1:
        n = build_dataset_csv_from_signals(
            ROOT=_ROOT,
            signals_path=signal_group[0],
            processed_dir=args.processed_dir,
            dataset_csv=dataset_csv,
            parser_stats_csv=stats_csv,
        )
    else:
        n = build_dataset_csv_from_signal_paths(
            ROOT=_ROOT,
            signals_paths=signal_group,
            processed_dir=args.processed_dir,
            dataset_csv=dataset_csv,
            parser_stats_csv=stats_csv,
        )
    print(json.dumps({"signals": [str(p) for p in signal_group], "rows": n, "dataset_csv": str(dataset_csv), "parser_stats_csv": str(stats_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
