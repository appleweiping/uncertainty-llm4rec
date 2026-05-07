#!/usr/bin/env python3
"""Validate converted Week8 same-candidate TRUCE artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/week8_same_candidate"))
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--expected-users", type=int)
    parser.add_argument("--expected-candidates", type=int, default=101)
    parser.add_argument("--expected-negatives", type=int, default=100)
    args = parser.parse_args()
    rows = [
        validate_split(
            root=args.root,
            domain=domain,
            split=split,
            expected_users=args.expected_users,
            expected_candidates=args.expected_candidates,
            expected_negatives=args.expected_negatives,
        )
        for domain in args.domains
        for split in args.splits
    ]
    print(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def validate_split(
    *,
    root: Path,
    domain: str,
    split: str,
    expected_users: int | None,
    expected_candidates: int,
    expected_negatives: int,
) -> dict[str, Any]:
    base = root / f"{domain}_large10000_100neg" / split
    examples_path = base / "examples.jsonl"
    manifest_path = base / "preprocess_manifest.json"
    if not examples_path.exists():
        raise SystemExit(f"examples.jsonl not found: {examples_path}")
    examples = _read_jsonl(examples_path)
    users = {str(row.get("user_id") or "") for row in examples}
    bad_candidate_counts = []
    missing_target = []
    duplicate_candidates = []
    missing_event_ids = []
    target_insertions = []
    for row in examples:
        example_id = str(row.get("example_id") or "")
        target = str(row.get("target") or "")
        candidates = [str(item) for item in row.get("candidates") or []]
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if len(candidates) != expected_candidates:
            bad_candidate_counts.append({"example_id": example_id, "count": len(candidates)})
        if target and candidates.count(target) != 1:
            missing_target.append(example_id)
        if len(set(candidates)) != len(candidates):
            duplicate_candidates.append(example_id)
        if not meta.get("event_id") or not meta.get("source_event_id"):
            missing_event_ids.append(example_id)
        if meta.get("target_inserted_by_converter"):
            target_insertions.append(example_id)
    if bad_candidate_counts:
        raise SystemExit(f"{domain}/{split}: bad candidate counts: {bad_candidate_counts[:5]}")
    if missing_target:
        raise SystemExit(f"{domain}/{split}: target missing or duplicated: {missing_target[:5]}")
    if duplicate_candidates:
        raise SystemExit(f"{domain}/{split}: duplicate candidates: {duplicate_candidates[:5]}")
    if missing_event_ids:
        raise SystemExit(f"{domain}/{split}: missing event/source ids: {missing_event_ids[:5]}")
    if target_insertions:
        raise SystemExit(f"{domain}/{split}: converter inserted targets: {target_insertions[:5]}")
    if expected_users is not None and len(users) != expected_users:
        raise SystemExit(f"{domain}/{split}: expected {expected_users} users, found {len(users)}")
    manifest = _read_json(manifest_path)
    return {
        "domain": domain,
        "split": split,
        "status": "passed",
        "examples": len(examples),
        "users": len(users),
        "expected_candidates": expected_candidates,
        "expected_negatives": expected_negatives,
        "manifest": manifest,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
