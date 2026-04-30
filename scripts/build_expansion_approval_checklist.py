"""Write approval checklists for the next real Storyflow expansion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.utils.config import load_simple_yaml  # noqa: E402


TRACKS = ("api_provider", "qwen3_server", "amazon_full_prepare", "baseline_artifact")


def _provider_configs() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((ROOT / "configs" / "providers").glob("*.yaml")):
        config = load_simple_yaml(path)
        records.append(
            {
                "config_path": str(path),
                "provider_name": config.get("provider_name"),
                "model_name": config.get("model_name"),
                "api_key_env": config.get("api_key_env"),
                "base_url": config.get("base_url"),
                "endpoint": config.get("endpoint"),
                "requires_endpoint_confirmation": bool(config.get("requires_endpoint_confirmation")),
            }
        )
    return records


def _amazon_categories() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((ROOT / "configs" / "datasets").glob("amazon_reviews_2023_*.yaml")):
        config = load_simple_yaml(path)
        records.append(
            {
                "dataset": config.get("name"),
                "category_name": config.get("category_name"),
                "config_path": str(path),
                "raw_reviews_path": config.get("raw_reviews_path"),
                "raw_metadata_path": config.get("raw_metadata_path"),
                "sample_command": config.get("local_sample_command_template"),
                "full_command": config.get("full_mode_command_template"),
                "requires_large_download": bool(config.get("requires_large_download")),
                "server_scale": bool(config.get("server_scale")),
            }
        )
    return records


def _track_checklist(track: str) -> dict[str, Any]:
    if track == "api_provider":
        return {
            "track": track,
            "recommended_default": "DeepSeek only until a new provider/model/budget/rate gate is approved.",
            "requires_user_approval": True,
            "required_confirmations": [
                "provider",
                "model",
                "base_url_or_endpoint",
                "budget_label",
                "sample_size",
                "rate_limit",
                "max_concurrency",
                "environment variable exists",
                "explicit --execute-api command",
            ],
            "preflight_commands": [
                "python scripts/check_api_pilot_readiness.py --provider-config <provider-yaml> --input-jsonl <input-jsonl> --sample-size 5 --stage smoke --approved-provider <provider> --approved-model <model> --approved-rate-limit <rpm> --approved-max-concurrency <n> --approved-budget-label <label> --execute-api-intended",
                "python scripts/run_api_observation.py --provider-config <provider-yaml> --input-jsonl <input-jsonl> --max-examples 5 --dry-run",
            ],
            "execute_command_template": "python scripts/run_api_observation.py --provider-config <provider-yaml> --input-jsonl <input-jsonl> --max-examples <n> --execute-api --rate-limit <rpm> --max-concurrency <n> --run-label <label> --budget-label <label>",
            "forbidden_without_approval": ["real API calls", "provider sweep", "larger sample size"],
            "available_provider_configs": _provider_configs(),
        }
    if track == "qwen3_server":
        return {
            "track": track,
            "recommended_default": "Generate plan-only artifacts locally; execute only on approved server hardware.",
            "requires_user_approval": True,
            "required_confirmations": [
                "server path and environment",
                "GPU type and memory",
                "model source and revision",
                "input JSONL slice",
                "processed catalog path",
                "output artifact-return policy",
                "explicit --execute-server command",
            ],
            "preflight_commands": [
                "python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl <input-jsonl> --output-dir <plan-output-dir> --max-examples 20 --run-label qwen3_plan"
            ],
            "execute_command_template": "python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl <input-jsonl> --output-dir <server-output-dir> --execute-server --run-stage full --run-label <label>",
            "forbidden_without_approval": ["Qwen3 inference", "server command execution", "LoRA training"],
            "server_config": "configs/server/qwen3_8b_observation.yaml",
        }
    if track == "amazon_full_prepare":
        return {
            "track": track,
            "recommended_default": "Beauty remains first; Video_Games or Books are the next title-rich candidates after raw placement/server approval.",
            "requires_user_approval": True,
            "required_confirmations": [
                "dataset/category",
                "license/access accepted",
                "raw review JSONL path",
                "raw metadata JSONL path",
                "machine and disk budget",
                "whether full prepare is local or server",
                "explicit --allow-full command",
            ],
            "preflight_commands": [
                "python scripts/inspect_amazon_category_matrix.py --sample-records 3",
                "python scripts/prepare_amazon_reviews_2023.py --dataset <dataset> --dry-run",
            ],
            "execute_command_template": "python scripts/prepare_amazon_reviews_2023.py --dataset <dataset> --reviews-jsonl <reviews-jsonl> --metadata-jsonl <metadata-jsonl> --output-suffix full --allow-full",
            "forbidden_without_approval": ["full raw download", "full preprocessing", "paper/full-result claims"],
            "configured_amazon_categories": _amazon_categories(),
        }
    if track == "baseline_artifact":
        return {
            "track": track,
            "recommended_default": "Validate one trained SASRec-like ranking artifact before adding more baseline families.",
            "requires_user_approval": True,
            "required_confirmations": [
                "baseline family and model family",
                "training machine/server provenance",
                "train/evaluation split declaration",
                "ranking JSONL path",
                "run manifest path",
                "leakage guard flags",
                "artifact-return policy",
            ],
            "preflight_commands": [
                "python scripts/validate_baseline_run_manifest.py --manifest-json <run-manifest-json> --strict",
                "python scripts/validate_baseline_artifact.py --ranking-jsonl <ranking-jsonl> --input-jsonl <input-jsonl> --baseline-family <family> --model-family <model> --dataset <dataset> --processed-suffix <suffix> --split <split> --trained-splits train --strict",
            ],
            "execute_command_template": "python scripts/run_baseline_observation.py --input-jsonl <input-jsonl> --baseline ranking_jsonl --ranking-jsonl <ranking-jsonl> --strict-ranking",
            "forbidden_without_approval": ["large baseline training", "ranking artifact treated as result before validation", "bypassing title grounding"],
            "manifest_template": "configs/server/baseline_ranking_run_manifest.example.json",
        }
    raise ValueError(f"unknown track: {track}")


def build_expansion_approval_checklist(tracks: list[str] | None = None) -> dict[str, Any]:
    selected_tracks = tracks or list(TRACKS)
    unknown = sorted(set(selected_tracks) - set(TRACKS))
    if unknown:
        raise ValueError("unknown track(s): " + ", ".join(unknown))
    return {
        "artifact_kind": "storyflow_expansion_approval_checklist",
        "api_called": False,
        "server_executed": False,
        "model_training": False,
        "data_downloaded": False,
        "full_data_processed": False,
        "is_experiment_result": False,
        "claim_scope": "approval_gate_only_not_paper_evidence",
        "tracks": [_track_checklist(track) for track in selected_tracks],
    }


def _write_markdown(checklist: dict[str, Any], path: Path) -> None:
    lines = [
        "# Storyflow Expansion Approval Checklist",
        "",
        "This is an approval gate only. It does not call APIs, execute servers, train models, download data, or create paper evidence.",
        "",
        f"- claim scope: {checklist['claim_scope']}",
        f"- api called: {checklist['api_called']}",
        f"- server executed: {checklist['server_executed']}",
        f"- model training: {checklist['model_training']}",
        f"- data downloaded: {checklist['data_downloaded']}",
        "",
    ]
    for track in checklist["tracks"]:
        lines.extend(
            [
                f"## {track['track']}",
                "",
                f"- recommended default: {track['recommended_default']}",
                f"- requires user approval: {track['requires_user_approval']}",
                "",
                "Required confirmations:",
            ]
        )
        lines.extend(f"- {item}" for item in track["required_confirmations"])
        lines.extend(["", "Preflight commands:"])
        lines.extend(f"- `{command}`" for command in track["preflight_commands"])
        lines.extend(
            [
                "",
                "Execute command template:",
                "",
                "```powershell",
                track["execute_command_template"],
                "```",
                "",
                "Forbidden without approval:",
            ]
        )
        lines.extend(f"- {item}" for item in track["forbidden_without_approval"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_expansion_approval_checklist(checklist: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "expansion_approval_checklist.json"
    report_path = output_dir / "expansion_approval_checklist.md"
    json_path.write_text(
        json.dumps(checklist, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(checklist, report_path)
    return {"json": str(json_path), "report": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="append", choices=TRACKS)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    checklist = build_expansion_approval_checklist(args.track)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "approval_gates" / "next_expansion"
    )
    checklist["outputs"] = write_expansion_approval_checklist(checklist, output_dir)
    print(json.dumps(checklist, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
