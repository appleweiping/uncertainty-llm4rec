from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BaselinePaths:
    baseline_name: str
    exp_name: str
    root: Path
    predictions_dir: Path
    metrics_dir: Path
    proxy_dir: Path
    logs_dir: Path


def ensure_baseline_dirs(
    baseline_name: str,
    exp_name: str,
    output_root: str | Path = "outputs/baselines",
) -> BaselinePaths:
    root = Path(output_root) / baseline_name / exp_name
    paths = BaselinePaths(
        baseline_name=baseline_name,
        exp_name=exp_name,
        root=root,
        predictions_dir=root / "predictions",
        metrics_dir=root / "metrics",
        proxy_dir=root / "proxy",
        logs_dir=root / "logs",
    )
    for path in [
        paths.root,
        paths.predictions_dir,
        paths.metrics_dir,
        paths.proxy_dir,
        paths.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return pd.read_json(path, lines=True).to_dict(orient="records")


def save_jsonl_records(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(path, orient="records", lines=True, force_ascii=False)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _normalize_history(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    item_id = str(
        candidate.get("item_id")
        or candidate.get("candidate_item_id")
        or candidate.get("id")
        or ""
    ).strip()
    title = str(candidate.get("title") or candidate.get("candidate_title") or item_id).strip()
    meta = str(
        candidate.get("meta")
        or candidate.get("candidate_meta")
        or candidate.get("candidate_text")
        or candidate.get("description")
        or ""
    ).strip()
    label = int(candidate.get("label", 0))
    return {
        "item_id": item_id,
        "title": title,
        "meta": meta,
        "label": label,
    }


def _normalize_grouped_row(row: dict[str, Any]) -> dict[str, Any]:
    candidates = [_normalize_candidate(candidate) for candidate in row.get("candidates", [])]
    candidates = [candidate for candidate in candidates if candidate["item_id"]]
    target_item_id = str(
        row.get("target_item_id")
        or next((candidate["item_id"] for candidate in candidates if int(candidate.get("label", 0)) == 1), "")
    ).strip()
    return {
        "user_id": str(row.get("user_id", "")).strip(),
        "history": _normalize_history(row.get("history")),
        "history_items": _normalize_history(row.get("history_items")),
        "target_item_id": target_item_id,
        "target_popularity_group": str(row.get("target_popularity_group", "unknown")).strip().lower() or "unknown",
        "candidates": candidates,
    }


def _group_pointwise_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = df.to_dict(orient="records")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        user_id = str(row.get("user_id", "")).strip()
        if not user_id:
            continue
        groups.setdefault(user_id, []).append(row)

    grouped: list[dict[str, Any]] = []
    for user_id, group_rows in groups.items():
        first = group_rows[0]
        candidates = []
        for row in group_rows:
            candidate = _normalize_candidate(
                {
                    "candidate_item_id": row.get("candidate_item_id"),
                    "candidate_title": row.get("candidate_title"),
                    "candidate_text": row.get("candidate_text"),
                    "label": row.get("label", 0),
                }
            )
            if candidate["item_id"]:
                candidates.append(candidate)

        grouped.append(
            {
                "user_id": user_id,
                "history": _normalize_history(first.get("history")),
                "history_items": _normalize_history(first.get("history_items")),
                "target_item_id": str(
                    first.get("target_item_id")
                    or next(
                        (
                            row.get("candidate_item_id")
                            for row in group_rows
                            if int(row.get("label", 0)) == 1 and row.get("candidate_item_id")
                        ),
                        "",
                    )
                ).strip(),
                "target_popularity_group": str(first.get("target_popularity_group", "unknown")).strip().lower() or "unknown",
                "candidates": candidates,
            }
        )
    return grouped


def load_grouped_candidate_samples(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grouped candidate input not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = pd.read_json(path, lines=True).to_dict(orient="records")
    elif path.suffix.lower() == ".csv":
        records = pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported grouped candidate input format: {path.suffix}")

    if not records:
        return []

    if isinstance(records[0].get("candidates"), list):
        grouped = [_normalize_grouped_row(row) for row in records]
    else:
        grouped = _group_pointwise_rows(pd.DataFrame(records))

    grouped = [row for row in grouped if row.get("user_id") and row.get("candidates")]
    if max_samples is not None and max_samples > 0:
        grouped = grouped[:max_samples]
    return grouped
