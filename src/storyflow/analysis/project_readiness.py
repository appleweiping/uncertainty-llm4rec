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


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


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
    return {
        "created_at_utc": utc_now_iso(),
        "artifact_kind": "storyflow_project_setup_readiness",
        "project_setup_complete": project_setup_complete,
        "ready_for_experiment_approval": project_setup_complete,
        "real_experiments_started_by_this_report": False,
        "api_called": False,
        "server_executed": False,
        "model_training": False,
        "is_experiment_result": False,
        "checks": checks,
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
        f"- ready_for_experiment_approval: {manifest['ready_for_experiment_approval']}",
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
