#!/usr/bin/env python3
"""Build CU-GR offline dataset from R3 prediction artifacts (no API)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm4rec.analysis.calibrator_features import (
    build_dataset_context,
    build_override_topk,
    compute_offline_labels,
    example_id_key,
    extract_features,
    fallback_top_items_scores,
    train_seed_statistics,
)
from llm4rec.analysis.ours_error_decomposition import load_method_predictions, method_run_dir


def read_jsonl_indexed(path: Path, *, max_rows: int | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_rows is not None and i >= max_rows:
                break
            if line.strip():
                row = json.loads(line)
                out[example_id_key(row)] = row
    return out


def build_rows_for_seed(
    runs_dir: Path,
    seed: int,
    ours_method: str,
    *,
    context: dict[str, Any],
    train_stats: dict[str, float],
    max_rows: int | None,
) -> list[dict[str, Any]]:
    ours_path = method_run_dir(runs_dir, ours_method, seed) / "predictions.jsonl"
    bm_path = method_run_dir(runs_dir, "bm25", seed) / "predictions.jsonl"
    seq_path = method_run_dir(runs_dir, "sequential_markov", seed) / "predictions.jsonl"
    if not ours_path.exists() or not bm_path.exists():
        raise FileNotFoundError(f"missing {ours_path} or {bm_path}")
    ours_rows = read_jsonl_indexed(ours_path, max_rows=max_rows)
    bm_rows = read_jsonl_indexed(bm_path, max_rows=max_rows)
    seq_rows = read_jsonl_indexed(seq_path, max_rows=max_rows) if seq_path.exists() else {}
    rows_out: list[dict[str, Any]] = []
    for eid, ours in ours_rows.items():
        bm = bm_rows.get(eid)
        if bm is None:
            continue
        seq = seq_rows.get(eid) if seq_rows else None
        fb_items, _ = fallback_top_items_scores(bm, k=10)
        meta = dict(ours.get("metadata") or {})
        g_id = meta.get("grounded_item_id")
        g_str = str(g_id) if g_id not in (None, "") else ""
        cand = list(ours.get("candidate_items") or [])
        cand_set = set(cand)
        parse_ok = bool(meta.get("parse_success", False))
        ground_ok = bool(meta.get("grounding_success", False))
        adherent = bool(meta.get("candidate_adherent", False))
        valid_g = bool(g_str and g_str in cand_set and parse_ok and ground_ok and adherent)
        override_items = build_override_topk(fb_items, g_str if valid_g else None, cand_set, k=10)
        labels = compute_offline_labels(
            target_item=str(ours.get("target_item", "")),
            candidate_items=cand,
            fallback_items_topk=fb_items,
            override_items_topk=override_items,
            parse_success=parse_ok,
            grounding_success=ground_ok,
            candidate_adherent=adherent,
            hallucination_flag=bool(meta.get("is_hallucinated", False)),
        )
        feats = extract_features(ours, bm, seq, context=context, train_stats=train_stats)
        record: dict[str, Any] = {
            "example_id": eid,
            "run_seed": seed,
            "user_id": str(ours.get("user_id", "")),
            "target_item": str(ours.get("target_item", "")),
            "ours_method": ours_method,
            "candidate_items_json": json.dumps(cand, ensure_ascii=False),
            "fallback_top10_json": json.dumps(fb_items, ensure_ascii=False),
            "grounded_item_id": g_str,
        }
        record.update(feats)
        record.update(labels)
        rows_out.append(record)
    rows_out.sort(key=lambda r: r["example_id"])
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/cu_gr_calibrator_dataset.csv"))
    parser.add_argument("--ours-method", type=str, default="ours_uncertainty_guided_real")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--max-rows-per-seed", type=int, default=None)
    args = parser.parse_args()

    context = build_dataset_context(str(args.processed_dir) if args.processed_dir else None)
    runs_dir = args.runs
    train_ours = load_method_predictions(runs_dir, args.ours_method, 13)
    if args.max_rows_per_seed is not None:
        train_ours = train_ours[: args.max_rows_per_seed]
    train_stats = train_seed_statistics(train_ours)

    all_rows: list[dict[str, Any]] = []
    for seed in (13, 21, 42):
        all_rows.extend(
            build_rows_for_seed(
                runs_dir,
                seed,
                args.ours_method,
                context=context,
                train_stats=train_stats,
                max_rows=args.max_rows_per_seed,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    balance_path = args.output.parent / "cu_gr_class_balance.csv"
    by_seed_rows: list[dict[str, Any]] = []
    seeds_present = sorted({int(r["run_seed"]) for r in all_rows})
    for s in seeds_present:
        rows_s = [r for r in all_rows if int(r["run_seed"]) == s]
        n_tot = len(rows_s)
        ni = sum(int(r["override_improves"]) for r in rows_s)
        nh = sum(int(r["override_hurts"]) for r in rows_s)
        nn = sum(int(r["override_neutral"]) for r in rows_s)
        by_seed_rows.append({"run_seed": s, "n_improve": ni, "n_hurt": nh, "n_neutral": nn, "n_total": n_tot})
    with balance_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["run_seed", "n_improve", "n_hurt", "n_neutral", "n_total"])
        w.writeheader()
        for row in by_seed_rows:
            w.writerow(row)

    print(json.dumps({"dataset": str(args.output), "balance": str(balance_path), "n_rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
