"""Plan-only Qwen3 LoRA/SFT training contract.

The default path writes a reproducible server training plan without importing
heavy ML dependencies or starting a training job. Server execution is a future
approved step and must return manifests/logs before any result is claimed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storyflow.observation import read_jsonl, utc_now_iso, write_jsonl
from storyflow.utils.config import load_simple_yaml


REQUIRED_CONFIG_KEYS = {
    "backend",
    "model_name",
    "model_alias",
    "data",
    "training",
    "lora",
    "guards",
    "output_contract",
}


def _repo_relative(path: str | Path, *, root: str | Path = ".") -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _display_path(path: str | Path, *, root: str | Path = ".") -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(candidate)


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def load_qwen_lora_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the lightweight LoRA training config."""

    config = load_simple_yaml(config_path)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Qwen LoRA config missing keys: {missing}")
    if config["backend"] != "qwen_lora_sft":
        raise ValueError("Qwen LoRA config must set backend: qwen_lora_sft")
    if not str(config.get("model_name") or "").strip():
        raise ValueError("Qwen LoRA config must set model_name")
    if not str(config.get("model_alias") or "").strip():
        raise ValueError("Qwen LoRA config must set model_alias")
    data = _nested(config, "data")
    if not str(data.get("train_input_jsonl") or "").strip():
        raise ValueError("Qwen LoRA config must set data.train_input_jsonl")
    guards = _nested(config, "guards")
    if guards.get("server_execution_required") is not True:
        raise ValueError("guards.server_execution_required must be true")
    if guards.get("no_local_codex_training") is not True:
        raise ValueError("guards.no_local_codex_training must be true")
    return config


def default_qwen_lora_output_dir(
    *,
    config: dict[str, Any],
    root: str | Path = ".",
) -> Path:
    training = _nested(config, "training")
    configured = training.get("output_dir")
    if configured:
        return _repo_relative(str(configured), root=root)
    return Path(root) / "outputs" / "server_training" / str(config["model_alias"])


def _command_plan(
    *,
    config_path: Path,
    output_dir: Path,
    execute_server: bool,
) -> str:
    command = (
        "python scripts/server/run_qwen3_lora_sft.py "
        f"--config {config_path.as_posix()} "
        f"--output-dir {output_dir.as_posix()} "
        "--execute-server"
    )
    return "\n".join(
        [
            "# Qwen3-8B LoRA SFT Server Training Command Plan",
            "",
            "This is a command plan, not a completed run.",
            "Run only after the user explicitly starts the training experiment",
            "and the server environment, GPU, data paths, and artifact-return",
            "policy are recorded.",
            "",
            "```powershell",
            command if execute_server else command,
            "```",
            "",
            "The planned SFT target is title-level JSON generation from",
            "observation prompts. This is a baseline scaffold, not the final",
            "TRUCE/CURE objective.",
        ]
    ) + "\n"


def _expected_artifacts(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "adapter_dir": str(output_dir / "adapter"),
        "trainer_state": str(output_dir / "trainer_state.json"),
        "training_log": str(output_dir / "train.log"),
        "eval_metrics": str(output_dir / "eval_metrics.json"),
        "train_manifest": str(output_dir / "train_manifest.json"),
        "config_snapshot": str(output_dir / "config_snapshot.json"),
        "claim_scope": "server_training_artifacts_only_after_execute_server",
        "model_alias": str(config["model_alias"]),
        "is_experiment_result": False,
    }


def _preview_sft_rows(input_jsonl: Path, *, limit: int = 3) -> list[dict[str, Any]]:
    if not input_jsonl.exists():
        return []
    rows = []
    for row in read_jsonl(input_jsonl)[:limit]:
        rows.append(
            {
                "input_id": row.get("input_id"),
                "prompt_template": row.get("prompt_template"),
                "response_policy": "target_title_json_confidence_1",
                "target_title": row.get("target_title"),
                "split": row.get("split"),
            }
        )
    return rows


def build_qwen_lora_training_plan(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Write a plan-only LoRA training manifest and artifact contract."""

    resolved_config = _repo_relative(config_path, root=root)
    config = load_qwen_lora_training_config(resolved_config)
    output_path = (
        _repo_relative(output_dir, root=root)
        if output_dir
        else default_qwen_lora_output_dir(config=config, root=root)
    )
    output_path.mkdir(parents=True, exist_ok=True)
    data = _nested(config, "data")
    train_input = _repo_relative(str(data["train_input_jsonl"]), root=root)
    validation_value = data.get("validation_input_jsonl")
    validation_input = (
        _repo_relative(str(validation_value), root=root)
        if validation_value
        else None
    )
    expected = _expected_artifacts(config, output_path)
    expected_path = output_path / "expected_training_artifacts.json"
    expected_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    config_snapshot_path = output_path / "config_snapshot.json"
    config_snapshot_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    command_plan_path = output_path / "train_command_plan.md"
    command_plan_path.write_text(
        _command_plan(
            config_path=resolved_config,
            output_dir=output_path,
            execute_server=False,
        ),
        encoding="utf-8",
    )
    preview_path = output_path / "sft_preview_rows.jsonl"
    preview_rows = _preview_sft_rows(train_input)
    if preview_rows:
        write_jsonl(preview_path, preview_rows)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "backend": "qwen_lora_sft",
        "model": str(config["model_name"]),
        "model_alias": str(config["model_alias"]),
        "stage": str(config.get("stage") or "sft_baseline"),
        "seed": int(config.get("seed") or 0),
        "config_path": str(resolved_config),
        "output_dir": str(output_path),
        "train_input_jsonl": str(train_input),
        "train_input_exists": train_input.exists(),
        "validation_input_jsonl": str(validation_input) if validation_input else None,
        "validation_input_exists": validation_input.exists() if validation_input else None,
        "response_policy": str(data.get("response_policy") or "target_title_json_confidence_1"),
        "expected_artifacts": str(expected_path),
        "config_snapshot": str(config_snapshot_path),
        "command_plan": str(command_plan_path),
        "sft_preview_rows": str(preview_path) if preview_rows else None,
        "api_called": False,
        "server_executed": False,
        "model_inference_run": False,
        "model_training": False,
        "is_experiment_result": False,
        "claim_scope": "plan_only_not_training_not_paper_evidence",
        "note": (
            "Plan-only Qwen3 LoRA/SFT scaffold. It prepares the server training "
            "contract but does not train or load a model."
        ),
    }
    manifest_path = output_path / "train_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def run_qwen_lora_training(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    execute_server: bool = False,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Plan by default; refuse accidental local training."""

    if not execute_server:
        return build_qwen_lora_training_plan(
            config_path=config_path,
            output_dir=output_dir,
            root=root,
        )
    raise RuntimeError(
        "Qwen3 LoRA training execution is server-only and intentionally not "
        "started by local Codex. Use the generated command plan on approved "
        "server hardware after the user starts the training experiment."
    )
