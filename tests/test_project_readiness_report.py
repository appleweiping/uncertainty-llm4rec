from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts.build_project_readiness_report import main as readiness_main
from storyflow.analysis.project_readiness import build_project_readiness_manifest


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_project_readiness_manifest_checks_setup_without_execution() -> None:
    manifest = build_project_readiness_manifest(root=Path("."))

    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert manifest["model_training"] is False
    assert manifest["is_experiment_result"] is False
    assert manifest["experiment_ready_scaffold"] is True
    assert manifest["paper_results_ready"] is False
    assert "qwen_lora_training" in manifest["checks"]
    assert manifest["checks"]["qwen_lora_training"]["passed"] is True
    assert manifest["module_readiness"]["qwen_lora_training"]["status"] == (
        "plan_ready_execution_requires_approval"
    )
    assert "paper_artifacts" in manifest["blocked_modules"]
    assert "qwen_server_observation" in manifest["approval_required_modules"]
    safe_names = {row["name"] for row in manifest["safe_preflight_commands"]}
    approval_names = {row["name"] for row in manifest["approval_required_operations"]}
    assert "unit_tests" in safe_names
    assert "project_readiness_report" in safe_names
    assert "paid_api_observation" in approval_names
    assert "qwen_lora_training" in approval_names
    assert manifest["test_file_count"] > 0


def test_project_readiness_cli_writes_artifacts() -> None:
    workspace = _workspace("project_readiness")

    code = readiness_main(["--output-dir", str(workspace)])
    manifest = json.loads((workspace / "project_readiness_manifest.json").read_text(encoding="utf-8"))
    report = (workspace / "project_readiness_report.md").read_text(encoding="utf-8")

    assert code == 0
    assert manifest["real_experiments_started_by_this_report"] is False
    assert "Project Setup Readiness" in report
    assert "Module Readiness" in report
    assert "Safe Preflight Commands" in report
    assert "Approval-Required Operations" in report
    assert "Experiment Start Policy" in report
