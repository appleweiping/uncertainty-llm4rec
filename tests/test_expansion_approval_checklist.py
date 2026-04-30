from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts.build_expansion_approval_checklist import (
    TRACKS,
    build_expansion_approval_checklist,
    main as approval_main,
)


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_expansion_approval_checklist_has_all_guarded_tracks() -> None:
    checklist = build_expansion_approval_checklist()

    assert checklist["api_called"] is False
    assert checklist["server_executed"] is False
    assert checklist["model_training"] is False
    assert checklist["data_downloaded"] is False
    assert checklist["full_data_processed"] is False
    assert checklist["is_experiment_result"] is False
    assert {track["track"] for track in checklist["tracks"]} == set(TRACKS)

    for track in checklist["tracks"]:
        assert track["requires_user_approval"] is True
        assert track["required_confirmations"]
        assert track["preflight_commands"]
        assert track["execute_command_template"]
        assert track["forbidden_without_approval"]


def test_expansion_approval_checklist_marks_specific_blockers() -> None:
    checklist = build_expansion_approval_checklist(["api_provider", "qwen3_server"])
    tracks = {track["track"]: track for track in checklist["tracks"]}

    assert "explicit --execute-api command" in tracks["api_provider"]["required_confirmations"]
    assert any("real API calls" in item for item in tracks["api_provider"]["forbidden_without_approval"])
    assert "explicit --execute-server command" in tracks["qwen3_server"]["required_confirmations"]
    assert any("Qwen3 inference" in item for item in tracks["qwen3_server"]["forbidden_without_approval"])


def test_expansion_approval_cli_writes_json_and_markdown() -> None:
    workspace = _workspace("expansion_approval")

    code = approval_main(
        [
            "--track",
            "amazon_full_prepare",
            "--track",
            "baseline_artifact",
            "--output-dir",
            str(workspace),
        ]
    )

    manifest = json.loads((workspace / "expansion_approval_checklist.json").read_text(encoding="utf-8"))
    report = (workspace / "expansion_approval_checklist.md").read_text(encoding="utf-8")

    assert code == 0
    assert [track["track"] for track in manifest["tracks"]] == [
        "amazon_full_prepare",
        "baseline_artifact",
    ]
    amazon_track = manifest["tracks"][0]
    baseline_track = manifest["tracks"][1]
    assert "explicit --allow-full command" in amazon_track["required_confirmations"]
    assert any(category["dataset"] == "amazon_reviews_2023_books" for category in amazon_track["configured_amazon_categories"])
    assert "run manifest path" in baseline_track["required_confirmations"]
    assert "approval gate only" in report
    assert "--allow-full" in report
    assert "validate_baseline_run_manifest.py" in report
