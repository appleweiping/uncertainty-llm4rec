"""Project setup readiness reporting.

This module checks whether the repository has the engineering entry points
needed before starting real experiments. It does not run APIs, servers,
training, downloads, or baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storyflow.observation import utc_now_iso


READINESS_CHECKS: dict[str, tuple[str, ...]] = {
    "governance": (
        "Storyflow.md",
        "AGENTS.md",
        "README.md",
        "docs/experiment_protocol.md",
        "docs/implementation_plan.md",
    ),
    "data_pipeline": (
        "scripts/download_datasets.py",
        "scripts/prepare_dataset.py",
        "scripts/prepare_amazon_reviews_2023.py",
        "scripts/validate_processed_dataset.py",
        "scripts/audit_processed_dataset.py",
        "docs/dataset_matrix.md",
        "docs/data_validation.md",
    ),
    "api_observation": (
        "configs/providers/deepseek.yaml",
        "configs/providers/qwen_api.yaml",
        "configs/providers/kimi.yaml",
        "configs/providers/glm.yaml",
        "scripts/run_api_observation.py",
        "scripts/analyze_observation.py",
        "scripts/review_observation_cases.py",
        "src/storyflow/providers/openai_compatible.py",
        "src/storyflow/grounding/title.py",
    ),
    "qwen_server_observation": (
        "configs/server/qwen3_8b_observation.yaml",
        "scripts/server/run_qwen3_observation.py",
        "src/storyflow/server/qwen_observation.py",
        "docs/server_runbook.md",
    ),
    "qwen_lora_training": (
        "configs/server/qwen3_8b_lora_sft.yaml",
        "scripts/server/run_qwen3_lora_sft.py",
        "src/storyflow/training/qwen_lora.py",
    ),
    "baseline_contracts": (
        "scripts/run_baseline_observation.py",
        "scripts/validate_baseline_run_manifest.py",
        "scripts/validate_baseline_artifact.py",
        "configs/server/baseline_ranking_run_manifest.example.json",
        "src/storyflow/baselines/observation.py",
    ),
    "confidence_framework": (
        "scripts/build_confidence_features.py",
        "scripts/calibrate_confidence_features.py",
        "scripts/residualize_confidence_features.py",
        "scripts/rerank_confidence_features.py",
        "src/storyflow/confidence/features.py",
        "src/storyflow/confidence/calibration.py",
        "src/storyflow/confidence/residuals.py",
        "src/storyflow/confidence/reranking.py",
    ),
    "simulation_and_triage": (
        "scripts/simulate_echo_exposure.py",
        "scripts/triage_confidence_features.py",
        "src/storyflow/simulation/exposure.py",
        "src/storyflow/triage/reasons.py",
        "docs/echo_simulation_triage.md",
    ),
    "approval_and_run_packets": (
        "scripts/build_expansion_approval_checklist.py",
        "scripts/build_expansion_run_packet.py",
        "scripts/build_local_deepseek_experiment_plan.py",
        "docs/expansion_approval_gates.md",
        "docs/local_deepseek_experiments.md",
    ),
}


MODULE_READINESS_PROFILES: dict[str, dict[str, Any]] = {
    "governance": {
        "status_when_present": "ready",
        "ready_for": [
            "repository identity checks",
            "claim-safety review",
            "local report workflow",
        ],
        "blocked_until": [],
        "safe_command_refs": ["governance_checks"],
        "approval_operation_refs": [],
    },
    "data_pipeline": {
        "status_when_present": "ready_for_safe_preflight",
        "ready_for": [
            "processed-data validation",
            "processed-data audit",
            "observation input construction from existing processed data",
        ],
        "blocked_until": [
            "new raw-data downloads or full-data preparation require explicit approval",
        ],
        "safe_command_refs": [
            "validate_processed_dataset",
            "audit_processed_dataset",
            "build_observation_inputs",
        ],
        "approval_operation_refs": ["raw_data_download", "full_data_prepare"],
    },
    "api_observation": {
        "status_when_present": "ready_for_dry_run_and_mock",
        "ready_for": [
            "mock observation",
            "API dry-run request/parse/cache checks",
            "API readiness checks that do not call the network",
        ],
        "blocked_until": [
            "real paid API calls require explicit user approval and configured environment keys",
            "Qwen/Kimi/GLM provider configs still require confirmed endpoint/model values",
        ],
        "safe_command_refs": ["mock_observation", "api_readiness_check", "api_dry_run"],
        "approval_operation_refs": ["paid_api_observation"],
    },
    "qwen_server_observation": {
        "status_when_present": "plan_ready_execution_requires_approval",
        "ready_for": [
            "plan-only Qwen3 observation packet generation",
            "API-compatible server output contract review",
        ],
        "blocked_until": [
            "approved server hardware, model path, logs, and returned manifest are available",
        ],
        "safe_command_refs": ["qwen_observation_plan"],
        "approval_operation_refs": ["qwen_server_observation"],
    },
    "qwen_lora_training": {
        "status_when_present": "plan_ready_execution_requires_approval",
        "ready_for": [
            "plan-only LoRA/SFT training contract generation",
            "server artifact expectation review",
        ],
        "blocked_until": [
            "approved server training run with logs, config snapshot, and returned artifacts",
        ],
        "safe_command_refs": ["qwen_lora_plan"],
        "approval_operation_refs": ["qwen_lora_training"],
    },
    "baseline_contracts": {
        "status_when_present": "ready_for_safe_preflight",
        "ready_for": [
            "popularity/co-occurrence no-API sanity baselines",
            "external ranking artifact validation",
            "ranking-to-title adapter contract checks",
        ],
        "blocked_until": [
            "trained SASRec/BERT4Rec/GRU4Rec/LightGCN artifacts require approved training or user-provided outputs",
        ],
        "safe_command_refs": [
            "baseline_sanity",
            "baseline_run_manifest_validation",
            "baseline_artifact_validation",
        ],
        "approval_operation_refs": ["trained_baseline_execution"],
    },
    "confidence_framework": {
        "status_when_present": "ready_for_scaffold_diagnostics",
        "ready_for": [
            "feature building from completed grounded outputs",
            "split-audited calibration scaffold",
            "split-audited popularity residual scaffold",
            "deterministic reranking contract checks",
        ],
        "blocked_until": [
            "learned CURE/TRUCE method claims require approved training/evaluation artifacts",
            "exposure-counterfactual utility targets require approved exposure/relevance evidence",
        ],
        "safe_command_refs": [
            "build_confidence_features",
            "calibrate_confidence_features",
            "residualize_confidence_features",
            "rerank_confidence_features",
        ],
        "approval_operation_refs": ["learned_cure_truce_training"],
    },
    "simulation_and_triage": {
        "status_when_present": "ready_for_synthetic_diagnostics",
        "ready_for": [
            "synthetic exposure simulation",
            "diagnostic data-triage reason codes",
        ],
        "blocked_until": [
            "real feedback-loop or pruning claims require approved user-feedback/training evidence",
        ],
        "safe_command_refs": ["simulate_echo_exposure", "triage_confidence_features"],
        "approval_operation_refs": ["real_feedback_or_pruning_experiment"],
    },
    "approval_and_run_packets": {
        "status_when_present": "ready",
        "ready_for": [
            "non-executing approval checklists",
            "non-executing run packets",
            "local DeepSeek plan generation while experiments are paused",
        ],
        "blocked_until": [],
        "safe_command_refs": [
            "expansion_approval_checklist",
            "expansion_run_packet",
            "local_deepseek_plan",
            "project_readiness_report",
        ],
        "approval_operation_refs": [],
    },
}


SAFE_PREFLIGHT_COMMANDS: dict[str, dict[str, Any]] = {
    "governance_checks": {
        "command": "run_governance_checks_individually",
        "commands": [
            "Get-Location",
            "git branch --show-current",
            "git remote -v",
            "git status --short --branch",
        ],
        "purpose": "Confirm repository identity before editing or approving runs.",
    },
    "unit_tests": {
        "command": ".\\.venv\\bin\\python.exe -m pytest",
        "purpose": "Run the local pytest suite with committed fixtures only.",
    },
    "project_readiness_report": {
        "command": ".\\.venv\\bin\\python.exe scripts\\build_project_readiness_report.py",
        "purpose": "Write ignored setup-readiness artifacts without executing experiments.",
    },
    "validate_processed_dataset": {
        "command": "python scripts/validate_processed_dataset.py --dataset <dataset> --processed-suffix <suffix>",
        "purpose": "Validate existing processed artifacts before observation input construction.",
    },
    "audit_processed_dataset": {
        "command": "python scripts/audit_processed_dataset.py --dataset <dataset> --processed-suffix <suffix>",
        "purpose": "Audit repeat/no-repeat, split integrity, title quality, and bucket coverage.",
    },
    "build_observation_inputs": {
        "command": "python scripts/build_observation_inputs.py --dataset <dataset> --processed-suffix <suffix> --split test --max-examples <N>",
        "purpose": "Build ignored observation-input JSONL from existing processed data.",
    },
    "mock_observation": {
        "command": "python scripts/run_observation_pipeline.py --input-jsonl <input-jsonl> --provider-mode popularity_biased --max-examples <N>",
        "purpose": "Run the no-API mock observation sanity path.",
    },
    "api_readiness_check": {
        "command": "python scripts/check_api_pilot_readiness.py --provider-config <provider-yaml> --input-jsonl <input-jsonl> --execute-api-intended",
        "purpose": "Check API pilot readiness without calling the provider.",
    },
    "api_dry_run": {
        "command": "python scripts/run_api_observation.py --provider-config <provider-yaml> --input-jsonl <input-jsonl> --dry-run",
        "purpose": "Exercise request, parsing, cache, and manifest flow without network calls.",
    },
    "qwen_observation_plan": {
        "command": "python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl <input-jsonl> --output-dir <output-dir>",
        "purpose": "Write a plan-only Qwen3 observation contract.",
    },
    "qwen_lora_plan": {
        "command": "python scripts/server/run_qwen3_lora_sft.py --config configs/server/qwen3_8b_lora_sft.yaml --output-dir <output-dir>",
        "purpose": "Write a plan-only LoRA/SFT training contract.",
    },
    "baseline_sanity": {
        "command": "python scripts/run_baseline_observation.py --input-jsonl <input-jsonl> --baseline popularity --max-examples <N>",
        "purpose": "Run a local no-API lightweight baseline sanity path.",
    },
    "baseline_run_manifest_validation": {
        "command": "python scripts/validate_baseline_run_manifest.py --manifest-json <run-manifest> --strict",
        "purpose": "Validate user-provided trained-baseline provenance before artifact use.",
    },
    "baseline_artifact_validation": {
        "command": "python scripts/validate_baseline_artifact.py --ranking-jsonl <ranking-jsonl> --input-jsonl <input-jsonl> --baseline-family <family> --model-family <model> --dataset <dataset> --processed-suffix <suffix> --split <split> --trained-splits train --strict",
        "purpose": "Validate ranking JSONL compatibility before title-level adaptation.",
    },
    "build_confidence_features": {
        "command": "python scripts/build_confidence_features.py --grounded-jsonl <grounded-jsonl> --input-jsonl <input-jsonl> --catalog-csv <catalog-csv>",
        "purpose": "Build CURE/TRUCE feature rows from completed grounded observations.",
    },
    "calibrate_confidence_features": {
        "command": "python scripts/calibrate_confidence_features.py --features-jsonl <features-jsonl> --fit-splits train --eval-splits validation,test",
        "purpose": "Run split-audited calibration scaffold.",
    },
    "residualize_confidence_features": {
        "command": "python scripts/residualize_confidence_features.py --features-jsonl <features-jsonl> --fit-splits train --eval-splits validation,test",
        "purpose": "Run split-audited popularity residual scaffold.",
    },
    "rerank_confidence_features": {
        "command": "python scripts/rerank_confidence_features.py --features-jsonl <features-jsonl> --confidence-source calibrated_residualized --group-key input_id --top-k 1",
        "purpose": "Run deterministic CURE/TRUCE reranking contract.",
    },
    "simulate_echo_exposure": {
        "command": "python scripts/simulate_echo_exposure.py --features-jsonl <features-jsonl> --policies utility_only,confidence_only,utility_confidence,cure_truce --rounds 3",
        "purpose": "Run synthetic exposure diagnostics only.",
    },
    "triage_confidence_features": {
        "command": "python scripts/triage_confidence_features.py --features-jsonl <features-jsonl>",
        "purpose": "Write diagnostic triage reason codes without deleting training data.",
    },
    "expansion_approval_checklist": {
        "command": "python scripts/build_expansion_approval_checklist.py",
        "purpose": "Write a non-executing approval checklist.",
    },
    "expansion_run_packet": {
        "command": "python scripts/build_expansion_run_packet.py --track <track> --run-label <label> --input-jsonl <input-jsonl> --target-output-dir <target-output-dir>",
        "purpose": "Write a non-executing run packet for one selected track.",
    },
    "local_deepseek_plan": {
        "command": "python scripts/build_local_deepseek_experiment_plan.py --dataset <dataset> --gate-size <N>",
        "purpose": "Write a plan-only local DeepSeek experiment packet while execution remains paused.",
    },
}


APPROVAL_REQUIRED_OPERATIONS: dict[str, dict[str, Any]] = {
    "paid_api_observation": {
        "command_shape": "python scripts/run_api_observation.py ... --execute-api",
        "requires": [
            "explicit provider/model/budget/rate-limit approval",
            "environment variable API key",
            "manifested run label and output directory",
        ],
    },
    "qwen_server_observation": {
        "command_shape": "python scripts/server/run_qwen3_observation.py ... --execute-server",
        "requires": [
            "explicit server execution approval",
            "server GPU/model environment",
            "returned logs and manifest before any claim",
        ],
    },
    "qwen_lora_training": {
        "command_shape": "python scripts/server/run_qwen3_lora_sft.py ... --execute-server",
        "requires": [
            "explicit training approval",
            "server ML environment",
            "adapter path, logs, metrics, and train manifest",
        ],
    },
    "trained_baseline_execution": {
        "command_shape": "<baseline trainer command outside the local scaffold>",
        "requires": [
            "explicit baseline training approval",
            "train/eval split declaration",
            "ranking artifact plus source run manifest",
        ],
    },
    "raw_data_download": {
        "command_shape": "python scripts/download_datasets.py --dataset <dataset>",
        "requires": [
            "explicit download approval when network/raw data is involved",
            "license/access confirmation where applicable",
        ],
    },
    "full_data_prepare": {
        "command_shape": "python scripts/prepare_amazon_reviews_2023.py ... --allow-full",
        "requires": [
            "explicit full-preparation approval",
            "raw JSONL files available under data/raw/",
            "storage/runtime readiness",
        ],
    },
    "learned_cure_truce_training": {
        "command_shape": "<future learned CURE/TRUCE training command>",
        "requires": [
            "approved training/evaluation target",
            "completed observation artifacts",
            "server or local training plan with logs",
        ],
    },
    "real_feedback_or_pruning_experiment": {
        "command_shape": "<future feedback-loop or pruning experiment command>",
        "requires": [
            "approved exposure/relevance evidence",
            "noise/triage protocol",
            "paper-claim guardrails",
        ],
    },
}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _commands_for(refs: list[str]) -> list[dict[str, Any]]:
    commands = []
    for ref in refs:
        command = dict(SAFE_PREFLIGHT_COMMANDS[ref])
        command["name"] = ref
        command["requires_approval"] = False
        command["api_called"] = False
        command["server_executed"] = False
        command["model_training"] = False
        command["is_experiment_result"] = False
        commands.append(command)
    return commands


def _operations_for(refs: list[str]) -> list[dict[str, Any]]:
    operations = []
    for ref in refs:
        operation = dict(APPROVAL_REQUIRED_OPERATIONS[ref])
        operation["name"] = ref
        operation["requires_explicit_approval"] = True
        operation["forbidden_by_current_task"] = True
        operations.append(operation)
    return operations


def _build_module_readiness(checks: dict[str, Any]) -> dict[str, Any]:
    module_readiness: dict[str, Any] = {}
    for name, check in checks.items():
        profile = MODULE_READINESS_PROFILES.get(name, {})
        if check["missing"]:
            status = "blocked_missing_scaffold"
            blocked_until = [f"missing required paths: {check['missing']}"]
        else:
            status = str(profile.get("status_when_present") or "ready")
            blocked_until = list(profile.get("blocked_until") or [])
        module_readiness[name] = {
            "status": status,
            "paths_present": not check["missing"],
            "ready_for": list(profile.get("ready_for") or []),
            "blocked_until": blocked_until,
            "safe_preflight_commands": _commands_for(
                list(profile.get("safe_command_refs") or [])
            ),
            "approval_required_operations": _operations_for(
                list(profile.get("approval_operation_refs") or [])
            ),
        }
    module_readiness["paper_artifacts"] = {
        "status": "blocked_until_completed_full_runs",
        "paths_present": True,
        "ready_for": [
            "paper-safe summary templates from completed manifests only",
        ],
        "blocked_until": [
            "full approved observations, trained baselines, server/Qwen artifacts, and final analysis manifests exist",
            "paper figures/tables are generated from actual logs without manual numbers",
        ],
        "safe_preflight_commands": [],
        "approval_required_operations": [],
    }
    return module_readiness


def _all_safe_commands(module_readiness: dict[str, Any]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {
        "unit_tests": {
            **SAFE_PREFLIGHT_COMMANDS["unit_tests"],
            "name": "unit_tests",
            "requires_approval": False,
            "api_called": False,
            "server_executed": False,
            "model_training": False,
            "is_experiment_result": False,
        }
    }
    for module in module_readiness.values():
        for command in module.get("safe_preflight_commands", []):
            by_name[str(command["name"])] = command
    return [by_name[name] for name in sorted(by_name)]


def _all_approval_operations(module_readiness: dict[str, Any]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for module in module_readiness.values():
        for operation in module.get("approval_required_operations", []):
            by_name[str(operation["name"])] = operation
    return [by_name[name] for name in sorted(by_name)]


def build_project_readiness_manifest(*, root: str | Path = ".") -> dict[str, Any]:
    """Return a repository setup readiness manifest."""

    root_path = Path(root)
    checks: dict[str, Any] = {}
    missing_all: list[str] = []
    for name, required_paths in READINESS_CHECKS.items():
        missing = [
            path for path in required_paths if not (root_path / path).exists()
        ]
        present = [
            path for path in required_paths if (root_path / path).exists()
        ]
        checks[name] = {
            "passed": not missing,
            "present": present,
            "missing": missing,
        }
        missing_all.extend(missing)

    tests = sorted((root_path / "tests").glob("test_*.py"))
    tracked_fixture_count = len(list((root_path / "tests" / "fixtures").glob("*")))
    project_setup_complete = not missing_all and bool(tests)
    module_readiness = _build_module_readiness(checks)
    safe_preflight_commands = _all_safe_commands(module_readiness)
    approval_required_operations = _all_approval_operations(module_readiness)
    ready_modules = [
        name
        for name, module in module_readiness.items()
        if str(module["status"]).startswith("ready")
    ]
    blocked_modules = [
        name
        for name, module in module_readiness.items()
        if str(module["status"]).startswith("blocked")
    ]
    approval_required_modules = [
        name
        for name, module in module_readiness.items()
        if "requires_approval" in str(module["status"])
        or module.get("approval_required_operations")
    ]
    return {
        "created_at_utc": utc_now_iso(),
        "artifact_kind": "storyflow_project_setup_readiness",
        "project_setup_complete": project_setup_complete,
        "experiment_ready_scaffold": project_setup_complete,
        "ready_for_experiment_approval": project_setup_complete,
        "paper_results_ready": False,
        "real_experiments_started_by_this_report": False,
        "api_called": False,
        "server_executed": False,
        "model_training": False,
        "is_experiment_result": False,
        "checks": checks,
        "module_readiness": module_readiness,
        "ready_modules": sorted(ready_modules),
        "blocked_modules": sorted(blocked_modules),
        "approval_required_modules": sorted(approval_required_modules),
        "safe_preflight_commands": safe_preflight_commands,
        "approval_required_operations": approval_required_operations,
        "missing_required_paths": sorted(missing_all),
        "test_file_count": len(tests),
        "fixture_file_count": tracked_fixture_count,
        "experiment_start_policy": (
            "Do not start additional API, server, LoRA, or trained-baseline "
            "experiments until the user explicitly starts the experiment phase."
        ),
        "recommended_experiment_order": [
            "freeze a git commit and readiness report",
            "run local/API pilot or server Qwen observation on a small approved slice",
            "run trained ranking baseline artifact validation",
            "run Qwen3-8B LoRA SFT baseline on server",
            "run TRUCE/CURE framework diagnostics and reranking from completed outputs",
            "scale to title-rich Amazon categories with reviewer-facing caveats",
        ],
        "remaining_non_setup_blockers": [
            "real Qwen3/server inference artifacts are not present",
            "Qwen3-8B LoRA training has not been executed",
            "trained SASRec/BERT4Rec/GRU4Rec/LightGCN ranking artifacts are not present",
            "multi-provider API artifacts beyond DeepSeek are not present",
            "paper-level claims require completed logs, manifests, and analysis",
        ],
    }


def write_project_readiness_report(
    *,
    output_dir: str | Path,
    root: str | Path = ".",
) -> dict[str, str]:
    """Write JSON and Markdown readiness artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = build_project_readiness_manifest(root=root)
    json_path = output_path / "project_readiness_manifest.json"
    md_path = output_path / "project_readiness_report.md"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Storyflow / TRUCE-Rec Project Setup Readiness",
        "",
        f"- project_setup_complete: {manifest['project_setup_complete']}",
        f"- experiment_ready_scaffold: {manifest['experiment_ready_scaffold']}",
        f"- ready_for_experiment_approval: {manifest['ready_for_experiment_approval']}",
        f"- paper_results_ready: {manifest['paper_results_ready']}",
        "- api_called: false",
        "- server_executed: false",
        "- model_training: false",
        "",
        "## Checks",
        "",
    ]
    for name, check in manifest["checks"].items():
        lines.append(f"- {name}: {'pass' if check['passed'] else 'missing'}")
        if check["missing"]:
            lines.extend(f"  - missing `{path}`" for path in check["missing"])
    lines.extend(
        [
            "",
            "## Module Readiness",
            "",
            "| module | status | ready for | blocked until |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, module in manifest["module_readiness"].items():
        ready_for = "; ".join(module.get("ready_for") or [""])
        blocked_until = "; ".join(module.get("blocked_until") or [""])
        lines.append(
            f"| {name} | {module['status']} | {ready_for} | {blocked_until} |"
        )
    lines.extend(
        [
            "",
            "## Safe Preflight Commands",
            "",
        ]
    )
    for command in manifest["safe_preflight_commands"]:
        lines.append(f"- `{command['command']}`")
        for subcommand in command.get("commands") or []:
            lines.append(f"  - `{subcommand}`")
        lines.append(f"  - {command['purpose']}")
    lines.extend(
        [
            "",
            "## Approval-Required Operations",
            "",
        ]
    )
    for operation in manifest["approval_required_operations"]:
        requires = "; ".join(operation.get("requires") or [])
        lines.append(f"- `{operation['command_shape']}`")
        lines.append(f"  - requires: {requires}")
    lines.extend(
        [
            "",
            "## Experiment Start Policy",
            "",
            str(manifest["experiment_start_policy"]),
            "",
            "## Recommended Order",
            "",
        ]
    )
    lines.extend(f"{index + 1}. {item}" for index, item in enumerate(manifest["recommended_experiment_order"]))
    lines.extend(["", "## Remaining Non-Setup Blockers", ""])
    lines.extend(f"- {item}" for item in manifest["remaining_non_setup_blockers"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": _relative(json_path, Path(root)),
        "markdown": _relative(md_path, Path(root)),
    }
