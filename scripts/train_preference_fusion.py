#!/usr/bin/env python3
"""Train/select CU-GR v2 preference fusion weights from saved full-seed artifacts."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--input", type=Path, default=None, help="Preference dataset CSV from build_preference_dataset.py")
    parser.add_argument("--output", type=Path, default=None, help="Model output directory")
    parser.add_argument("--runs-root", nargs="*", type=Path, default=[Path("outputs/runs")])
    parser.add_argument("--run-name-prefix", default="r3_v2_movielens_preference_signal_subgate")
    parser.add_argument("--signals", type=Path, default=None)
    parser.add_argument("--processed-dir", default="data/processed/movielens_1m/r2_full_single_dataset")
    parser.add_argument("--model-output", type=Path, default=Path("outputs/models/cu_gr_v2_preference_fusion"))
    args = parser.parse_args(argv)

    model_output = args.output or args.model_output
    mo = Path(model_output)
    mo = mo if mo.is_absolute() else _ROOT / mo
    mo.mkdir(parents=True, exist_ok=True)

    full_seed_weights = _ROOT / "outputs" / "tables" / "cu_gr_v2_full_seed_fusion_weights.csv"
    if args.input is not None and full_seed_weights.exists():
        with full_seed_weights.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = next((row for row in rows if row.get("policy") == "fusion_train_best"), rows[0] if rows else {})
        model = {
            "source": str(full_seed_weights),
            "input_dataset": str(args.input if args.input.is_absolute() else _ROOT / args.input),
            "selected_fusion_params": {
                "alpha": float(selected.get("alpha") or 0.0),
                "beta": float(selected.get("beta") or 0.0),
                "gamma": float(selected.get("gamma") or 0.0),
                "lambda": float(selected.get("lambda") or 0.0),
            },
            "selected_on": selected.get("selected_on", "seed21_validation"),
            "trained_on": selected.get("trained_on", "seed13"),
            "tested_on": selected.get("tested_on", "seed42"),
            "validation_seed21_NDCG@10": selected.get("validation_seed21_NDCG@10", ""),
            "test_seed42_NDCG@10": selected.get("test_seed42_NDCG@10", ""),
        }
        (mo / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(json.dumps({"input": str(args.input), "model_json": str(mo / "model.json"), "source": str(full_seed_weights)}, indent=2))
        return 0

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
    model_path = mo / "model.json"
    print(json.dumps({"signals": str(signals), "model_json": str(model_path), "ndcg_fallback": metrics.get("fallback_ndcg@10"), "fusion_delta": metrics.get("fusion_delta_ndcg10_vs_fallback")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
