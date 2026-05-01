"""Checkpoint manifest helpers for local smoke trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = "checkpoint_manifest.json"
MODEL_STATE_NAME = "model_state.json"
TRAINER_CONFIG_NAME = "trainer_config.yaml"


def save_checkpoint_artifacts(
    checkpoint_dir: str | Path,
    *,
    method: str,
    model_state: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_state_path = output / MODEL_STATE_NAME
    trainer_config_path = output / TRAINER_CONFIG_NAME
    manifest_path = output / MANIFEST_NAME
    model_state_path.write_text(
        json.dumps(model_state, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    trainer_config_path.write_text(dump_yaml(config), encoding="utf-8")
    manifest = {
        "schema_version": "llm4rec_checkpoint_v1",
        "method": method,
        "model_state": MODEL_STATE_NAME,
        "trainer_config": TRAINER_CONFIG_NAME,
        "metadata": metadata or {},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return {
        **manifest,
        "checkpoint_dir": str(output),
        "manifest_path": str(manifest_path),
        "model_state_path": str(model_state_path),
        "trainer_config_path": str(trainer_config_path),
    }


def load_checkpoint_manifest(checkpoint_dir: str | Path, *, expected_method: str | None = None) -> dict[str, Any]:
    path = Path(checkpoint_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"checkpoint manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "llm4rec_checkpoint_v1":
        raise ValueError(f"incompatible checkpoint schema: {manifest.get('schema_version')}")
    if expected_method is not None and manifest.get("method") != expected_method:
        raise ValueError(f"incompatible checkpoint method: {manifest.get('method')}")
    return manifest


def load_model_state(checkpoint_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or load_checkpoint_manifest(checkpoint_dir)
    state_path = Path(checkpoint_dir) / str(manifest.get("model_state") or MODEL_STATE_NAME)
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint model state not found: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def dump_yaml(value: Any, *, indent: int = 0) -> str:
    lines: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            item = value[key]
            prefix = " " * indent + f"{key}:"
            if isinstance(item, dict):
                lines.append(prefix)
                lines.append(dump_yaml(item, indent=indent + 2).rstrip("\n"))
            else:
                lines.append(prefix + " " + _format_scalar(item))
    else:
        lines.append(" " * indent + _format_scalar(value))
    return "\n".join(lines) + "\n"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    text = str(value)
    if text == "" or any(char in text for char in ":#[]{}") or text.strip() != text:
        return repr(text)
    return text
