"""Export canonical TRUCE data into external baseline formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_recbole_atomic(
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    seed: int,
) -> dict[str, Any]:
    """Export TRUCE processed artifacts into RecBole atomic files.

    The export preserves source user/item IDs as tokens. Candidate sets are kept
    in JSONL for later TRUCE-side scoring/import; RecBole is used only for model
    training/scoring, never for final paper metrics.
    """

    processed = Path(processed_dir)
    out = Path(output_dir)
    dataset_dir = out / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    examples = _read_jsonl(processed / "examples.jsonl")
    items = _read_csv(processed / "items.csv")
    interactions = _read_csv(processed / "interactions.csv")
    candidate_sets = _read_jsonl(processed / "candidate_sets.jsonl") if (processed / "candidate_sets.jsonl").exists() else []

    inter_path = dataset_dir / f"{dataset_name}.inter"
    item_path = dataset_dir / f"{dataset_name}.item"
    user_path = dataset_dir / f"{dataset_name}.user"
    candidates_path = dataset_dir / "truce_candidate_sets.jsonl"
    examples_path = dataset_dir / "truce_examples.jsonl"

    _write_recbole_inter(inter_path, examples, interactions)
    _write_recbole_item(item_path, items)
    _write_recbole_user(user_path, examples)
    _write_jsonl(candidates_path, candidate_sets)
    _write_jsonl(examples_path, examples)

    manifest = {
        "dataset_name": dataset_name,
        "processed_dir": str(processed),
        "exported_dir": str(dataset_dir),
        "seed": int(seed),
        "format": "recbole_atomic",
        "inter_file": str(inter_path),
        "item_file": str(item_path),
        "user_file": str(user_path),
        "candidate_sets": str(candidates_path),
        "examples": str(examples_path),
        "example_count": len(examples),
        "item_count": len(items),
        "interaction_count": len(interactions),
        "split_counts": _split_counts(examples),
        "metric_contract": "External model scores are imported into TRUCE prediction schema; TRUCE evaluator computes final metrics.",
    }
    (dataset_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_recbole_inter(path: Path, examples: list[dict[str, Any]], interactions: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in interactions:
        user = str(row.get("user_id") or "")
        item = str(row.get("item_id") or "")
        ts = str(row.get("timestamp") or "0")
        if not user or not item:
            continue
        key = (user, item, ts)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"user_id:token": user, "item_id:token": item, "timestamp:float": ts or "0"})
    if not rows:
        for ex in examples:
            user = str(ex.get("user_id") or "")
            for item in ex.get("history") or []:
                rows.append({"user_id:token": user, "item_id:token": str(item), "timestamp:float": "0"})
            rows.append({"user_id:token": user, "item_id:token": str(ex.get("target") or ""), "timestamp:float": "0"})
    _write_csv(path, rows, ["user_id:token", "item_id:token", "timestamp:float"], delimiter="\t")


def _write_recbole_item(path: Path, items: list[dict[str, str]]) -> None:
    rows = [
        {
            "item_id:token": str(row.get("item_id") or ""),
            "title:token_seq": str(row.get("title") or "").replace("\t", " "),
            "category:token_seq": str(row.get("category") or row.get("domain") or "").replace("\t", " "),
        }
        for row in items
    ]
    _write_csv(path, rows, ["item_id:token", "title:token_seq", "category:token_seq"], delimiter="\t")


def _write_recbole_user(path: Path, examples: list[dict[str, Any]]) -> None:
    users = sorted({str(row.get("user_id") or "") for row in examples if row.get("user_id")})
    rows = [{"user_id:token": user} for user in users]
    _write_csv(path, rows, ["user_id:token"], delimiter="\t")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _split_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in examples:
        split = str(row.get("split") or "")
        counts[split] = counts.get(split, 0) + 1
    return counts
