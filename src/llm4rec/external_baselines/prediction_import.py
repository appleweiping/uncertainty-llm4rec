"""Import external baseline scores into the TRUCE prediction schema."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def import_scored_candidates(
    *,
    scores_path: str | Path,
    examples_path: str | Path,
    output_path: str | Path,
    method: str,
    source_project: str,
    model_name: str,
    training_config: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    candidate_protocol: dict[str, Any] | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Convert per-candidate external scores into prediction JSONL."""

    scores = _read_scores(Path(scores_path))
    examples = _read_jsonl(Path(examples_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as handle:
        for ex in examples:
            if split is not None and _normalize_split(ex.get("split")) != _normalize_split(split):
                continue
            example_id = _example_id(ex)
            candidate_items = [str(x) for x in ex.get("candidates") or ex.get("candidate_items") or []]
            scored = {item: float(scores.get((example_id, item), scores.get((str(ex.get("user_id")), item), 0.0))) for item in candidate_items}
            ordered = sorted(candidate_items, key=lambda item: (-scored[item], item))
            rec = {
                "user_id": str(ex.get("user_id") or ""),
                "target_item": str(ex.get("target") or ex.get("target_item") or ""),
                "candidate_items": candidate_items,
                "predicted_items": ordered,
                "scores": [scored[item] for item in ordered],
                "method": method,
                "domain": str(ex.get("domain") or ""),
                "raw_output": None,
                "metadata": {
                    "example_id": example_id,
                    "event_id": _metadata_value(ex, "event_id", example_id),
                    "source_event_id": _metadata_value(ex, "source_event_id", example_id),
                    "split": ex.get("split"),
                    "external_baseline": True,
                    "library": source_project,
                    "source_project": source_project,
                    "source_project/library": source_project,
                    "recbole_version": _recbole_version() if source_project.lower() == "recbole" else "",
                    "model_name": model_name,
                    "training_config": training_config or {},
                    "checkpoint_path": str(checkpoint_path or ""),
                    "seed": int(seed),
                    "candidate_protocol": candidate_protocol or {},
                    "score_import_method": "per_candidate_score_csv",
                    "truce_evaluator_used": True,
                    "import_time": datetime.now(timezone.utc).isoformat(),
                },
            }
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return {"predictions": str(output), "count": n, "method": method, "model_name": model_name}


def _read_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            example_id = str(row.get("example_id") or row.get("user_id") or "")
            item_id = str(row.get("item_id") or "")
            if not example_id or not item_id:
                continue
            scores[(example_id, item_id)] = float(row.get("score") or 0.0)
    return scores


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _example_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("example_id") or meta.get("example_id") or row.get("user_id") or "")


def _metadata_value(row: dict[str, Any], key: str, default: Any = "") -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get(key) or meta.get(key) or default or "")


def _normalize_split(value: Any) -> str:
    split = str(value or "").lower()
    if split in {"val", "validation"}:
        return "valid"
    return split


def _recbole_version() -> str:
    try:
        import recbole

        return str(recbole.__version__)
    except Exception:
        return ""
