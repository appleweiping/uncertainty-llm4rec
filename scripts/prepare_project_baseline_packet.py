#!/usr/bin/env python3
"""Prepare server input packets for official external LLM4Rec baselines.

The packet is an execution contract, not an experiment result. It exports
TRUCE examples, candidate maps, and project-facing data files while preserving
the rule that final paper metrics must be computed by the TRUCE evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    packet = prepare_packet(config=config, config_path=args.config)
    print(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def prepare_packet(*, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project = str(config.get("project") or "").lower()
    if project not in {"openp5", "tallrec"}:
        raise SystemExit(f"unsupported project: {project}")
    dataset = _as_dict(config.get("dataset"))
    processed_dir = ROOT / str(dataset.get("processed_dir") or "")
    if not processed_dir.exists():
        raise SystemExit(f"processed_dir not found: {processed_dir}")
    output_dir = ROOT / str(config.get("output_dir") or f"outputs/server_packets/{project}")
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _read_jsonl(processed_dir / "examples.jsonl")
    items = _read_items(processed_dir / "items.csv")
    shutil.copy2(processed_dir / "examples.jsonl", output_dir / "truce_examples.jsonl")
    shutil.copy2(processed_dir / "candidate_sets.jsonl", output_dir / "truce_candidate_sets.jsonl")
    _write_items_jsonl(output_dir / "item_catalog.jsonl", items)
    if project == "tallrec":
        project_files = _write_tallrec_files(config=config, output_dir=output_dir, examples=examples, items=items)
    else:
        project_files = _write_openp5_files(config=config, output_dir=output_dir, examples=examples, items=items)
    manifest = _manifest(config=config, config_path=config_path, output_dir=output_dir, examples=examples, items=items, project_files=project_files)
    (output_dir / "project_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _write_tallrec_files(
    *,
    config: dict[str, Any],
    output_dir: Path,
    examples: list[dict[str, Any]],
    items: dict[str, dict[str, str]],
) -> dict[str, str]:
    settings = _as_dict(config.get("tallrec"))
    max_history = int(settings.get("max_history_items") or 20)
    negatives_per_train = int(settings.get("negatives_per_train_example") or 63)
    seed = int(config.get("seed") or 0)
    rng = random.Random(seed)
    tallrec_dir = output_dir / "tallrec"
    tallrec_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    row_maps: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for split in ["train", "valid", "test"]:
        rows: list[dict[str, str]] = []
        for ex in examples:
            if _normalize_split(ex.get("split")) != split:
                continue
            if split == "train":
                candidates = _train_candidates(ex, negatives_per_train=negatives_per_train, rng=rng)
            else:
                candidates = [str(item) for item in ex.get("candidates") or ex.get("candidate_items") or []]
            for item_id in candidates:
                label = item_id == str(ex.get("target") or ex.get("target_item") or "")
                row_id = f"{_example_id(ex)}::{item_id}"
                rows.append({
                    "instruction": "Given the user's interaction history and one candidate item, answer whether the candidate is a good next recommendation. Answer exactly Yes. or No.",
                    "input": _pairwise_input(ex, item_id=item_id, items=items, max_history=max_history),
                    "output": "Yes." if label else "No.",
                    "truce_row_id": row_id,
                })
                row_maps[split].append({
                    "row_id": row_id,
                    "example_id": _example_id(ex),
                    "user_id": str(ex.get("user_id") or ""),
                    "item_id": item_id,
                    "label": int(label),
                    "split": split,
                })
        path = tallrec_dir / f"{split}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        map_path = tallrec_dir / f"{split}_row_map.jsonl"
        _write_jsonl(map_path, row_maps[split])
        files[f"tallrec_{split}_json"] = str(path)
        files[f"tallrec_{split}_row_map"] = str(map_path)
    scores_template = tallrec_dir / "candidate_scores_template.csv"
    _write_score_template(scores_template, row_maps["test"])
    files["candidate_scores_template"] = str(scores_template)
    return files


def _write_openp5_files(
    *,
    config: dict[str, Any],
    output_dir: Path,
    examples: list[dict[str, Any]],
    items: dict[str, dict[str, str]],
) -> dict[str, str]:
    settings = _as_dict(config.get("openp5"))
    max_history = int(settings.get("max_history_items") or 50)
    openp5_dir = output_dir / "openp5"
    openp5_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {"train": [], "valid": [], "test": []}
    for ex in examples:
        split = _normalize_split(ex.get("split"))
        if split not in rows_by_split:
            continue
        target = str(ex.get("target") or ex.get("target_item") or "")
        rows_by_split[split].append({
            "task": "sequential_recommendation",
            "example_id": _example_id(ex),
            "user_id": str(ex.get("user_id") or ""),
            "domain": str(ex.get("domain") or ""),
            "history_item_ids": [str(item) for item in (ex.get("history") or ex.get("history_item_ids") or [])][-max_history:],
            "history_text": _history_text(ex, items=items, max_history=max_history),
            "target_item_id": target,
            "target_text": _item_text(target, items),
            "candidate_item_ids": [str(item) for item in ex.get("candidates") or ex.get("candidate_items") or []],
            "split": split,
        })
    files: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        path = openp5_dir / f"{split}_sequential_tasks.jsonl"
        _write_jsonl(path, rows)
        files[f"openp5_{split}_sequential_tasks"] = str(path)
    id_map = openp5_dir / "item_id_mapping.csv"
    with id_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "openp5_token", "title"])
        writer.writeheader()
        for idx, item_id in enumerate(sorted(items), start=1):
            writer.writerow({"item_id": item_id, "openp5_token": f"<item_{idx}>", "title": items[item_id].get("title", "")})
    files["item_id_mapping"] = str(id_map)
    scores_template = openp5_dir / "candidate_scores_template.csv"
    test_map = [
        {"example_id": row["example_id"], "user_id": row["user_id"], "item_id": item_id, "split": "test"}
        for row in rows_by_split["test"]
        for item_id in row["candidate_item_ids"]
    ]
    _write_score_template(scores_template, test_map)
    files["candidate_scores_template"] = str(scores_template)
    return files


def _manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    examples: list[dict[str, Any]],
    items: dict[str, dict[str, str]],
    project_files: dict[str, str],
) -> dict[str, Any]:
    project = str(config.get("project") or "")
    split_counts: dict[str, int] = {}
    for ex in examples:
        split = _normalize_split(ex.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "schema_version": "truce_project_baseline_packet_v1",
        "project": project,
        "official_repo": str(config.get("official_repo") or ""),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "dataset": _as_dict(config.get("dataset")),
        "seed": int(config.get("seed") or 0),
        "split_counts": split_counts,
        "item_count": len(items),
        "candidate_protocol": _as_dict(config.get("candidate")),
        "metric_contract": "External project trains/generates/scores only; final Recall/NDCG/MRR must be computed by TRUCE evaluator.",
        "forbidden": [
            "Do not train on test rows.",
            "Do not copy official project evaluator metrics into paper tables.",
            "Do not change candidate sets or target inclusion.",
            "Do not tune Amazon Video Games gate with this packet.",
        ],
        "required_return_artifacts": [
            "candidate_scores.csv with columns example_id,user_id,item_id,score",
            "project_baseline_manifest.json with upstream commit and environment",
            "stdout/stderr logs",
            "checkpoint or generated-output reference",
        ],
        "truce_import_command_template": "py -3 scripts/import_external_predictions.py --scores <candidate_scores.csv> --examples <packet>/truce_examples.jsonl --output <run_dir>/predictions.jsonl --method <method> --source-project <project> --model-name <model> --seed <seed>",
        "project_files": project_files,
        "is_experiment_result": False,
        "is_paper_result": False,
    }


def _train_candidates(ex: dict[str, Any], *, negatives_per_train: int, rng: random.Random) -> list[str]:
    target = str(ex.get("target") or ex.get("target_item") or "")
    negatives = [str(item) for item in ex.get("candidates") or ex.get("candidate_items") or [] if str(item) != target]
    if negatives_per_train >= 0:
        negatives = rng.sample(negatives, k=min(negatives_per_train, len(negatives)))
    candidates = [target] if target else []
    candidates.extend(negatives)
    rng.shuffle(candidates)
    return candidates


def _pairwise_input(
    ex: dict[str, Any],
    *,
    item_id: str,
    items: dict[str, dict[str, str]],
    max_history: int,
) -> str:
    history = _history_text(ex, items=items, max_history=max_history)
    candidate = _item_text(item_id, items)
    return f"User history: {history}\nCandidate item: {candidate}\nCandidate item ID: {item_id}"


def _history_text(ex: dict[str, Any], *, items: dict[str, dict[str, str]], max_history: int) -> str:
    history = [str(item) for item in (ex.get("history") or ex.get("history_item_ids") or [])][-max_history:]
    return " ; ".join(_item_text(item, items) for item in history)


def _item_text(item_id: str, items: dict[str, dict[str, str]]) -> str:
    row = items.get(str(item_id), {})
    title = row.get("title") or str(item_id)
    category = row.get("category") or ""
    return f"{title} [{category}]" if category else title


def _write_items_jsonl(path: Path, items: dict[str, dict[str, str]]) -> None:
    _write_jsonl(path, [items[item_id] for item_id in sorted(items)])


def _write_score_template(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "user_id", "item_id", "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "example_id": row.get("example_id") or "",
                "user_id": row.get("user_id") or "",
                "item_id": row.get("item_id") or "",
                "score": "",
            })


def _read_items(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            item_id = str(row.get("item_id") or "")
            if item_id:
                rows[item_id] = {key: str(value or "") for key, value in row.items()}
        return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _example_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("example_id") or meta.get("example_id") or row.get("user_id") or "")


def _normalize_split(value: Any) -> str:
    split = str(value or "").lower()
    if split in {"val", "validation"}:
        return "valid"
    return split


if __name__ == "__main__":
    raise SystemExit(main())
