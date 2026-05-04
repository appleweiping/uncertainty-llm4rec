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
    sasrec_max_item_list_length: int = 50,
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
    train_inter_path = dataset_dir / f"{dataset_name}.train.inter"
    valid_inter_path = dataset_dir / f"{dataset_name}.valid.inter"
    test_inter_path = dataset_dir / f"{dataset_name}.test.inter"
    sasrec_train_inter_path = dataset_dir / f"{dataset_name}.sasrec_train.inter"
    sasrec_valid_inter_path = dataset_dir / f"{dataset_name}.sasrec_valid.inter"
    sasrec_test_inter_path = dataset_dir / f"{dataset_name}.sasrec_test.inter"
    item_path = dataset_dir / f"{dataset_name}.item"
    user_path = dataset_dir / f"{dataset_name}.user"
    candidates_path = dataset_dir / "truce_candidate_sets.jsonl"
    examples_path = dataset_dir / "truce_examples.jsonl"

    benchmark_rows = _benchmark_interactions_from_examples(examples, interactions)
    _write_recbole_rows(inter_path, benchmark_rows["train"] + benchmark_rows["valid"] + benchmark_rows["test"])
    _write_recbole_rows(train_inter_path, benchmark_rows["train"])
    _write_recbole_rows(valid_inter_path, benchmark_rows["valid"])
    _write_recbole_rows(test_inter_path, benchmark_rows["test"])
    sasrec_rows = _sasrec_benchmark_rows(examples, interactions, max_item_list_length=sasrec_max_item_list_length)
    _write_sasrec_rows(sasrec_train_inter_path, sasrec_rows["train"])
    _write_sasrec_rows(sasrec_valid_inter_path, sasrec_rows["valid"])
    _write_sasrec_rows(sasrec_test_inter_path, sasrec_rows["test"])
    _write_recbole_item(item_path, items)
    _write_recbole_user(user_path, examples)
    _write_jsonl(candidates_path, candidate_sets)
    _write_jsonl(examples_path, examples)

    manifest = {
        "dataset_name": dataset_name,
        "processed_dir": str(processed),
        "exported_dir": str(dataset_dir),
        "seed": int(seed),
        "sasrec_max_item_list_length": int(sasrec_max_item_list_length),
        "format": "recbole_atomic",
        "inter_file": str(inter_path),
        "benchmark_files": {
            "train": str(train_inter_path),
            "valid": str(valid_inter_path),
            "test": str(test_inter_path),
        },
        "sasrec_benchmark_files": {
            "train": str(sasrec_train_inter_path),
            "valid": str(sasrec_valid_inter_path),
            "test": str(sasrec_test_inter_path),
        },
        "item_file": str(item_path),
        "user_file": str(user_path),
        "candidate_sets": str(candidates_path),
        "examples": str(examples_path),
        "example_count": len(examples),
        "item_count": len(items),
        "interaction_count": len(interactions),
        "split_counts": _split_counts(examples),
        "benchmark_interaction_counts": {key: len(value) for key, value in benchmark_rows.items()},
        "sasrec_benchmark_counts": {key: len(value) for key, value in sasrec_rows.items()},
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


def _benchmark_interactions_from_examples(
    examples: list[dict[str, Any]],
    interactions: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    by_user_train: dict[str, dict[str, Any]] = {}
    heldout_rows: list[dict[str, Any]] = []
    timelines = _interaction_timelines(interactions)
    for ex in examples:
        split = _normalize_split(ex.get("split"))
        user = str(ex.get("user_id") or "")
        if not user:
            continue
        if split == "train":
            current = by_user_train.get(user)
            if current is None or len(ex.get("history") or []) >= len(current.get("history") or []):
                by_user_train[user] = ex
        elif split in {"valid", "test"}:
            heldout_rows.append(ex)

    for user, ex in sorted(by_user_train.items()):
        rows = _training_rows_from_timeline(user, ex, timelines.get(user, []))
        if rows:
            by_split["train"].extend(rows)
            continue
        sequence = [str(item) for item in ex.get("history") or []]
        target = str(ex.get("target") or ex.get("target_item") or "")
        if target:
            sequence.append(target)
        by_split["train"].extend(_fallback_sequence_rows(user, sequence))

    for ex in sorted(heldout_rows, key=lambda row: (str(row.get("user_id") or ""), len(row.get("history") or []), str(row.get("example_id") or ""))):
        split = _normalize_split(ex.get("split"))
        user = str(ex.get("user_id") or "")
        target = str(ex.get("target") or ex.get("target_item") or "")
        if user and target and split in {"valid", "test"}:
            timestamp = _target_timestamp(ex, timelines.get(user, []))
            if timestamp is None:
                timestamp = float(len(ex.get("history") or []))
            by_split[split].append(_recbole_row(user, target, timestamp))

    if not by_split["valid"]:
        by_split["valid"] = list(by_split["test"])
    if not by_split["test"]:
        by_split["test"] = list(by_split["valid"])
    return by_split


def _recbole_row(user: str, item: str, timestamp: float) -> dict[str, Any]:
    return {"user_id:token": user, "item_id:token": item, "timestamp:float": timestamp}


def _interaction_timelines(interactions: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    timelines: dict[str, list[dict[str, Any]]] = {}
    for ordinal, row in enumerate(interactions):
        user = str(row.get("user_id") or "")
        item = str(row.get("item_id") or "")
        if not user or not item:
            continue
        timestamp = _safe_float(row.get("timestamp"), float(ordinal))
        timelines.setdefault(user, []).append({"item_id": item, "timestamp": timestamp, "ordinal": ordinal})
    for rows in timelines.values():
        rows.sort(key=lambda row: (float(row["timestamp"]), int(row["ordinal"])))
    return timelines


def _training_rows_from_timeline(
    user: str,
    ex: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_index = _target_index(ex)
    if target_index is None or target_index >= len(timeline):
        return []
    expected_target = str(ex.get("target") or ex.get("target_item") or "")
    if expected_target and str(timeline[target_index]["item_id"]) != expected_target:
        return []
    rows = []
    for row in timeline[: target_index + 1]:
        rows.append(_recbole_row(user, str(row["item_id"]), float(row["timestamp"])))
    return rows


def _target_timestamp(ex: dict[str, Any], timeline: list[dict[str, Any]]) -> float | None:
    target_index = _target_index(ex)
    if target_index is None or target_index >= len(timeline):
        return None
    expected_target = str(ex.get("target") or ex.get("target_item") or "")
    row = timeline[target_index]
    if expected_target and str(row["item_id"]) != expected_target:
        return None
    return float(row["timestamp"])


def _target_index(ex: dict[str, Any]) -> int | None:
    meta = ex.get("metadata") if isinstance(ex.get("metadata"), dict) else {}
    for value in (meta.get("target_index"), ex.get("target_index")):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    example_id = str(ex.get("example_id") or meta.get("example_id") or "")
    if ":" in example_id:
        suffix = example_id.rsplit(":", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return None
    history = ex.get("history") or ex.get("history_item_ids") or []
    return len(history) if history else None


def _fallback_sequence_rows(user: str, sequence: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(sequence):
        if item:
            rows.append(_recbole_row(user, item, float(index)))
    return rows


def _safe_float(value: Any, default: float) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_recbole_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, ["user_id:token", "item_id:token", "timestamp:float"], delimiter="\t")


def _sasrec_benchmark_rows(
    examples: list[dict[str, Any]],
    interactions: list[dict[str, str]],
    *,
    max_item_list_length: int,
) -> dict[str, list[dict[str, Any]]]:
    timelines = _interaction_timelines(interactions)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for ex in examples:
        split = _normalize_split(ex.get("split"))
        if split not in by_split:
            continue
        user = str(ex.get("user_id") or "")
        target = str(ex.get("target") or ex.get("target_item") or "")
        history = [str(item) for item in ex.get("history") or ex.get("history_item_ids") or [] if str(item)]
        history = history[-max_item_list_length:]
        if not user or not target or not history:
            continue
        timestamp = _target_timestamp(ex, timelines.get(user, []))
        if timestamp is None:
            timestamp = float(len(history))
        by_split[split].append(
            {
                "user_id:token": user,
                "item_id:token": target,
                "item_id_list:token_seq": " ".join(history),
                "timestamp:float": timestamp,
            }
        )
    return by_split


def _write_sasrec_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        ["user_id:token", "item_id:token", "item_id_list:token_seq", "timestamp:float"],
        delimiter="\t",
    )


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
        split = _normalize_split(row.get("split"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _normalize_split(value: Any) -> str:
    split = str(value or "").lower()
    if split in {"val", "validation"}:
        return "valid"
    return split
