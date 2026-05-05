#!/usr/bin/env python3
"""Prepare Qwen3-8B LoRA controlled-baseline inputs for project adapters.

This script creates an execution contract for server-side training. It does not
train a model locally and does not produce experiment results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_config  # noqa: E402


SUPPORTED_PROJECTS = {"tallrec", "openp5"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_controlled_baseline(config=config, config_path=args.config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def prepare_controlled_baseline(*, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project = str(config.get("project") or "").lower()
    if project not in SUPPORTED_PROJECTS:
        raise SystemExit(f"unsupported controlled baseline project: {project}")
    packet_dir = ROOT / str(config.get("packet_dir") or "")
    if not packet_dir.exists():
        raise SystemExit(f"packet_dir not found: {packet_dir}")
    output_dir = ROOT / str(config.get("output_dir") or f"outputs/server_training/{project}_qwen3_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    if project == "tallrec":
        files = _prepare_tallrec(config=config, packet_dir=packet_dir, output_dir=output_dir)
    else:
        files = _prepare_openp5(config=config, packet_dir=packet_dir, output_dir=output_dir)
    manifest = _manifest(config=config, config_path=config_path, output_dir=output_dir, files=files)
    (output_dir / "controlled_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_command_plan(output_dir / "server_command_plan.md", manifest=manifest)
    return manifest


def _prepare_tallrec(*, config: dict[str, Any], packet_dir: Path, output_dir: Path) -> dict[str, str]:
    project_dir = packet_dir / "tallrec"
    train_rows = json.loads((project_dir / "train.json").read_text(encoding="utf-8"))
    valid_rows = json.loads((project_dir / "valid.json").read_text(encoding="utf-8"))
    test_row_map = _read_jsonl(project_dir / "test_row_map.jsonl")
    max_train = _optional_int(config, "max_train_examples")
    max_valid = _optional_int(config, "max_valid_examples")
    train_sft = output_dir / "train_sft.jsonl"
    valid_sft = output_dir / "valid_sft.jsonl"
    _write_jsonl(train_sft, [_tallrec_sft_row(row) for row in train_rows[:max_train]])
    _write_jsonl(valid_sft, [_tallrec_sft_row(row) for row in valid_rows[:max_valid]])
    score_plan = output_dir / "test_score_plan.jsonl"
    _write_jsonl(score_plan, test_row_map)
    return {
        "train_sft": str(train_sft),
        "valid_sft": str(valid_sft),
        "test_score_plan": str(score_plan),
        "candidate_scores_template": str(project_dir / "candidate_scores_template.csv"),
        "test_row_map": str(project_dir / "test_row_map.jsonl"),
    }


def _prepare_openp5(*, config: dict[str, Any], packet_dir: Path, output_dir: Path) -> dict[str, str]:
    project_dir = packet_dir / "openp5"
    item_map = _read_item_map(project_dir / "item_id_mapping.csv")
    max_train = _optional_int(config, "max_train_examples")
    max_valid = _optional_int(config, "max_valid_examples")
    train_tasks = _read_jsonl(project_dir / "train_sequential_tasks.jsonl")
    valid_tasks = _read_jsonl(project_dir / "valid_sequential_tasks.jsonl")
    test_tasks = _read_jsonl(project_dir / "test_sequential_tasks.jsonl")
    train_sft = output_dir / "train_sft.jsonl"
    valid_sft = output_dir / "valid_sft.jsonl"
    test_score_plan = output_dir / "test_score_plan.jsonl"
    _write_jsonl(train_sft, [_openp5_sft_row(row, item_map=item_map) for row in train_tasks[:max_train]])
    _write_jsonl(valid_sft, [_openp5_sft_row(row, item_map=item_map) for row in valid_tasks[:max_valid]])
    _write_jsonl(test_score_plan, [_openp5_score_row(row, item_map=item_map) for row in test_tasks])
    return {
        "train_sft": str(train_sft),
        "valid_sft": str(valid_sft),
        "test_score_plan": str(test_score_plan),
        "candidate_scores_template": str(project_dir / "candidate_scores_template.csv"),
        "item_id_mapping": str(project_dir / "item_id_mapping.csv"),
    }


def _tallrec_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = f"{row.get('instruction', '')}\n\n{row.get('input', '')}".strip()
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(row.get("output") or "")},
        ],
        "metadata": {
            "source_project": "TALLRec",
            "truce_row_id": str(row.get("truce_row_id") or ""),
        },
    }


def _openp5_sft_row(row: dict[str, Any], *, item_map: dict[str, str]) -> dict[str, Any]:
    target = str(row.get("target_item_id") or "")
    prompt = _openp5_prompt(row, item_map=item_map)
    response = item_map.get(target, target)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "source_project": "OpenP5",
            "example_id": str(row.get("example_id") or ""),
            "target_item_id": target,
        },
    }


def _openp5_score_row(row: dict[str, Any], *, item_map: dict[str, str]) -> dict[str, Any]:
    candidates = [str(item) for item in row.get("candidate_item_ids") or []]
    return {
        "example_id": str(row.get("example_id") or ""),
        "user_id": str(row.get("user_id") or ""),
        "prompt": _openp5_prompt(row, item_map=item_map),
        "candidate_item_ids": candidates,
        "candidate_outputs": [item_map.get(item, item) for item in candidates],
        "scoring_contract": "Compute causal-LM log-likelihood for each candidate_output given prompt, then write candidate_scores.csv.",
    }


def _openp5_prompt(row: dict[str, Any], *, item_map: dict[str, str]) -> str:
    history_ids = [str(item) for item in row.get("history_item_ids") or []]
    candidate_ids = [str(item) for item in row.get("candidate_item_ids") or []]
    history_tokens = [item_map.get(item, item) for item in history_ids]
    candidate_tokens = [item_map.get(item, item) for item in candidate_ids]
    return (
        "Sequential recommendation task.\n"
        f"User history item tokens: {' '.join(history_tokens)}\n"
        f"Candidate item tokens: {' '.join(candidate_tokens)}\n"
        "Return exactly one candidate item token as the next recommendation."
    )


def _manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    files: dict[str, str],
) -> dict[str, Any]:
    project = str(config.get("project") or "").lower()
    return {
        "schema_version": "truce_qwen_lora_controlled_baseline_v1",
        "project": project,
        "controlled_baseline_name": str(config.get("controlled_baseline_name") or f"{project}_qwen3_lora"),
        "definition": "Controlled-backbone project adaptation; not an official upstream checkpoint result unless explicitly marked after training.",
        "base_model": str(config.get("base_model") or "/home/ajifang/models/Qwen/Qwen3-8B"),
        "packet_dir": str(config.get("packet_dir") or ""),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "seed": int(config.get("seed") or 13),
        "lora": _as_dict(config.get("lora")),
        "training": _as_dict(config.get("training")),
        "scoring": _as_dict(config.get("scoring")),
        "files": files,
        "is_experiment_result": False,
        "is_paper_result": False,
        "paper_table_policy": "Eligible for controlled main comparison only after real Qwen3-8B LoRA training, scoring, TRUCE import, and evaluator metrics are complete.",
        "required_return_artifacts": [
            "adapter checkpoint or checkpoint reference",
            "candidate_scores.csv",
            "predictions.jsonl imported by TRUCE",
            "metrics.json and metrics.csv from TRUCE evaluator",
            "environment.json, git_info.json, stdout/stderr logs",
        ],
    }


def _write_command_plan(path: Path, *, manifest: dict[str, Any]) -> None:
    name = manifest["controlled_baseline_name"]
    packet_dir = manifest["packet_dir"]
    output_dir = manifest["output_dir"]
    text = f"""# Server Command Plan: {name}

This is a controlled-backbone Qwen3-8B LoRA baseline plan. It is not complete
until real LoRA training and TRUCE evaluation have run.

## Inputs

- Packet: `{packet_dir}`
- SFT train: `{manifest['files']['train_sft']}`
- SFT valid: `{manifest['files']['valid_sft']}`
- Test score plan: `{manifest['files']['test_score_plan']}`
- Base model: `{manifest['base_model']}`

## Required Output

Write `candidate_scores.csv` with columns:

```text
example_id,user_id,item_id,score
```

Then import/evaluate with:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate
RUN=outputs/runs/{name}_seed{manifest['seed']}
mkdir -p "$RUN/artifacts"
cp {output_dir}/candidate_scores.csv "$RUN/artifacts/candidate_scores.csv"
python scripts/import_external_predictions.py \\
  --scores "$RUN/artifacts/candidate_scores.csv" \\
  --examples {packet_dir}/truce_examples.jsonl \\
  --output "$RUN/predictions.jsonl" \\
  --method {name} \\
  --source-project {manifest['project']} \\
  --model-name Qwen3-8B-LoRA \\
  --seed {manifest['seed']} \\
  --split test
python scripts/evaluate_predictions.py --predictions "$RUN/predictions.jsonl" --output-dir "$RUN"
```
"""
    path.write_text(text, encoding="utf-8")


def _read_item_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["item_id"]): str(row["openp5_token"]) for row in csv.DictReader(handle)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _optional_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value in (None, "", 0):
        return None
    return int(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
