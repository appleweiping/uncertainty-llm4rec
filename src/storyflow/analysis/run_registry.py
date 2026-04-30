"""Ignored run registry helpers for observation analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_run_id(*parts: object, length: int = 16) -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def append_registry_record(
    *,
    registry_jsonl: str | Path,
    analysis_manifest: dict[str, Any],
    source_label: str | None = None,
) -> dict[str, Any]:
    """Append one analysis run record to an ignored JSONL registry."""

    registry_path = Path(registry_jsonl)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = stable_run_id(
        analysis_manifest.get("source_grounded_jsonl"),
        analysis_manifest.get("analysis_dir"),
        analysis_manifest.get("provider"),
        analysis_manifest.get("model"),
    )
    record = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "source_label": source_label,
        "analysis_dir": analysis_manifest.get("analysis_dir"),
        "summary": analysis_manifest.get("summary"),
        "report": analysis_manifest.get("report"),
        "source_grounded_jsonl": analysis_manifest.get("source_grounded_jsonl"),
        "source_failed_jsonl": analysis_manifest.get("source_failed_jsonl"),
        "source_manifest_json": analysis_manifest.get("source_manifest_json"),
        "provider": analysis_manifest.get("provider"),
        "model": analysis_manifest.get("model"),
        "baseline": analysis_manifest.get("baseline"),
        "source_kind": analysis_manifest.get("source_kind"),
        "claim_scope": analysis_manifest.get("claim_scope"),
        "confidence_semantics": analysis_manifest.get("confidence_semantics"),
        "confidence_is_calibrated": analysis_manifest.get("confidence_is_calibrated"),
        "claim_guardrails": analysis_manifest.get("claim_guardrails"),
        "dry_run": analysis_manifest.get("dry_run"),
        "api_called": analysis_manifest.get("api_called"),
        "count": analysis_manifest.get("count"),
        "failed_count": analysis_manifest.get("failed_count"),
        "is_experiment_result": False,
        "note": (
            "Local registry pointer only. Source outputs remain under ignored "
            "outputs/ paths and must not be treated as paper evidence unless "
            "a later approved pilot/full run supplies real manifests."
        ),
    }
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record
