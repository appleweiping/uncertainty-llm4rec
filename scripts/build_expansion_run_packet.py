"""Build a non-executing run packet for approved Storyflow expansions."""

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

try:  # noqa: E402
    from scripts.build_expansion_approval_checklist import TRACKS, build_expansion_approval_checklist
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from build_expansion_approval_checklist import TRACKS, build_expansion_approval_checklist


CONFIRMATION_FIELDS = {
    "api_provider": {
        "provider": "provider",
        "model": "model",
        "base_url_or_endpoint": "base_url_or_endpoint",
        "budget_label": "budget_label",
        "sample_size": "sample_size",
        "rate_limit": "rate_limit",
        "max_concurrency": "max_concurrency",
        "environment variable exists": "api_key_env_present",
    },
    "qwen3_server": {
        "server path and environment": "server_environment",
        "GPU type and memory": "gpu_spec",
        "model source and revision": "model_source",
        "input JSONL slice": "input_jsonl",
        "processed catalog path": "catalog_path",
        "output artifact-return policy": "artifact_return_policy",
    },
    "amazon_full_prepare": {
        "dataset/category": "dataset",
        "license/access accepted": "license_accepted",
        "raw review JSONL path": "reviews_jsonl",
        "raw metadata JSONL path": "metadata_jsonl",
        "machine and disk budget": "machine_disk_budget",
        "whether full prepare is local or server": "execution_location",
    },
    "baseline_artifact": {
        "baseline family and model family": "baseline_family",
        "training machine/server provenance": "training_provenance",
        "train/evaluation split declaration": "split_declaration",
        "ranking JSONL path": "ranking_jsonl",
        "run manifest path": "run_manifest_json",
        "leakage guard flags": "leakage_guards",
        "artifact-return policy": "artifact_return_policy",
    },
}


EXPECTED_ARTIFACTS = {
    "api_provider": [
        "request_records.jsonl",
        "raw_responses.jsonl",
        "parsed_predictions.jsonl",
        "failed_cases.jsonl",
        "grounded_predictions.jsonl",
        "metrics.json",
        "report.md",
        "manifest.json",
        "analysis_summary.json after scripts/analyze_observation.py",
    ],
    "qwen3_server": [
        "plan-only: request_records.jsonl",
        "plan-only: expected_output_contract.json",
        "plan-only: server_command_plan.md",
        "plan-only: manifest.json",
        "executed server run, if separately approved: raw/parsed/grounded/metrics/report/manifest",
    ],
    "amazon_full_prepare": [
        "item_catalog.csv",
        "interactions.csv",
        "item_popularity.csv",
        "user_sequences.jsonl",
        "observation_examples.jsonl",
        "preprocess_manifest.json",
        "validation manifest after scripts/validate_processed_dataset.py",
    ],
    "baseline_artifact": [
        "baseline run-manifest validation JSON",
        "ranking artifact validation JSON",
        "grounded_predictions.jsonl",
        "metrics.json",
        "report.md",
        "manifest.json",
        "analysis_summary.json after scripts/analyze_observation.py",
    ],
}


FORBIDDEN_CLAIMS = [
    "Do not describe this packet as an executed run.",
    "Do not claim API, server, training, full-data, or baseline results from this packet.",
    "Do not treat dry-run, plan-only, validation, or readiness artifacts as paper evidence.",
    "Do not evaluate correctness or confidence before generated/selected titles are grounded to the catalog.",
]


def _value(value: Any, placeholder: str) -> str:
    if value is None or value == "":
        return f"<{placeholder}>"
    return str(value)


def _provided_confirmations(track: str, options: dict[str, Any]) -> dict[str, Any]:
    fields = CONFIRMATION_FIELDS[track]
    return {label: options.get(option_name) for label, option_name in fields.items()}


def _missing_confirmations(track: str, options: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for label, option_name in CONFIRMATION_FIELDS[track].items():
        value = options.get(option_name)
        if value is None or value == "" or value is False:
            missing.append(label)
    return missing


def _commands(track: str, options: dict[str, Any]) -> dict[str, list[str] | str]:
    input_jsonl = _value(options.get("input_jsonl"), "input-jsonl")
    output_dir = _value(options.get("target_output_dir"), "run-output-dir")
    run_label = _value(options.get("run_label"), "run-label")
    dataset = _value(options.get("dataset"), "dataset")

    if track == "api_provider":
        provider_config = _value(options.get("provider_config"), "provider-yaml")
        provider = _value(options.get("provider"), "provider")
        model = _value(options.get("model"), "model")
        budget = _value(options.get("budget_label"), "budget-label")
        rate = _value(options.get("rate_limit"), "rpm")
        concurrency = _value(options.get("max_concurrency"), "max-concurrency")
        sample_size = _value(options.get("sample_size"), "sample-size")
        return {
            "safe_local_preflight": [
                f"python scripts/check_api_pilot_readiness.py --provider-config {provider_config} --input-jsonl {input_jsonl} --sample-size {sample_size} --stage smoke --approved-provider {provider} --approved-model {model} --approved-rate-limit {rate} --approved-max-concurrency {concurrency} --approved-budget-label {budget} --execute-api-intended",
                f"python scripts/run_api_observation.py --provider-config {provider_config} --input-jsonl {input_jsonl} --max-examples {sample_size} --dry-run",
            ],
            "approval_required_execute": f"python scripts/run_api_observation.py --provider-config {provider_config} --input-jsonl {input_jsonl} --output-dir {output_dir} --max-examples {sample_size} --execute-api --rate-limit {rate} --max-concurrency {concurrency} --run-label {run_label} --budget-label {budget}",
        }

    if track == "qwen3_server":
        server_config = _value(options.get("server_config"), "configs/server/qwen3_8b_observation.yaml")
        return {
            "safe_local_preflight": [
                f"python scripts/server/run_qwen3_observation.py --config {server_config} --input-jsonl {input_jsonl} --output-dir {output_dir}_plan --max-examples 20 --run-label {run_label}_plan"
            ],
            "approval_required_execute": f"python scripts/server/run_qwen3_observation.py --config {server_config} --input-jsonl {input_jsonl} --output-dir {output_dir} --execute-server --run-stage full --run-label {run_label}",
        }

    if track == "amazon_full_prepare":
        reviews_jsonl = _value(options.get("reviews_jsonl"), "reviews-jsonl")
        metadata_jsonl = _value(options.get("metadata_jsonl"), "metadata-jsonl")
        suffix = _value(options.get("processed_suffix"), "full")
        return {
            "safe_local_preflight": [
                "python scripts/inspect_amazon_category_matrix.py --sample-records 3",
                f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --dry-run",
            ],
            "approval_required_execute": f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --reviews-jsonl {reviews_jsonl} --metadata-jsonl {metadata_jsonl} --output-suffix {suffix} --allow-full",
        }

    if track == "baseline_artifact":
        ranking_jsonl = _value(options.get("ranking_jsonl"), "ranking-jsonl")
        run_manifest = _value(options.get("run_manifest_json"), "run-manifest-json")
        family = _value(options.get("baseline_family"), "baseline-family")
        model_family = _value(options.get("model_family"), "model-family")
        processed_suffix = _value(options.get("processed_suffix"), "processed-suffix")
        split = _value(options.get("split"), "split")
        return {
            "safe_local_preflight": [
                f"python scripts/validate_baseline_run_manifest.py --manifest-json {run_manifest} --strict",
                f"python scripts/validate_baseline_artifact.py --ranking-jsonl {ranking_jsonl} --input-jsonl {input_jsonl} --baseline-family {family} --model-family {model_family} --dataset {dataset} --processed-suffix {processed_suffix} --split {split} --trained-splits train --strict",
            ],
            "approval_required_execute": f"python scripts/run_baseline_observation.py --input-jsonl {input_jsonl} --baseline ranking_jsonl --ranking-jsonl {ranking_jsonl} --output-dir {output_dir} --strict-ranking",
        }

    raise ValueError(f"unknown track: {track}")


def build_expansion_run_packet(track: str, **options: Any) -> dict[str, Any]:
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track}")
    approval_track = build_expansion_approval_checklist([track])["tracks"][0]
    missing = _missing_confirmations(track, options)
    return {
        "artifact_kind": "storyflow_expansion_run_packet",
        "track": track,
        "run_label": options.get("run_label") or f"next_{track}",
        "approval_status": "not_authorized_by_packet",
        "requires_user_approval": True,
        "api_called": False,
        "server_executed": False,
        "model_inference_run": False,
        "model_training": False,
        "data_downloaded": False,
        "full_data_processed": False,
        "baseline_training_run": False,
        "is_experiment_result": False,
        "claim_scope": "run_packet_only_not_execution_not_paper_evidence",
        "grounding_required_before_correctness": True,
        "source_approval_track": approval_track,
        "provided_confirmations": _provided_confirmations(track, options),
        "missing_confirmations": missing,
        "ready_for_execution_after_user_approval": len(missing) == 0,
        "commands": _commands(track, options),
        "expected_artifacts": EXPECTED_ARTIFACTS[track],
        "forbidden_claims": FORBIDDEN_CLAIMS + approval_track["forbidden_without_approval"],
    }


def _write_markdown(packet: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Storyflow Expansion Run Packet: {packet['run_label']}",
        "",
        "This is a run packet only. It does not execute the command, call an API, run a server, train a model, download data, or process full data.",
        "",
        f"- track: {packet['track']}",
        f"- claim scope: {packet['claim_scope']}",
        f"- approval status: {packet['approval_status']}",
        f"- ready after explicit user approval: {packet['ready_for_execution_after_user_approval']}",
        f"- api called: {packet['api_called']}",
        f"- server executed: {packet['server_executed']}",
        f"- model training: {packet['model_training']}",
        f"- full data processed: {packet['full_data_processed']}",
        "",
        "## Provided Confirmations",
        "",
    ]
    for label, value in packet["provided_confirmations"].items():
        lines.append(f"- {label}: {value if value not in (None, '', False) else '<missing>'}")
    lines.extend(["", "## Missing Confirmations", ""])
    if packet["missing_confirmations"]:
        lines.extend(f"- {item}" for item in packet["missing_confirmations"])
    else:
        lines.append("- none in this packet; execution still requires explicit user approval in the current turn")
    lines.extend(["", "## Safe Local Preflight", ""])
    lines.extend(f"- `{command}`" for command in packet["commands"]["safe_local_preflight"])
    lines.extend(
        [
            "",
            "## Approval-Required Execute Command",
            "",
            "```powershell",
            packet["commands"]["approval_required_execute"],
            "```",
            "",
            "## Expected Artifacts",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in packet["expected_artifacts"])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend(f"- {item}" for item in packet["forbidden_claims"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_expansion_run_packet(packet: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "expansion_run_packet.json"
    report_path = output_dir / "expansion_run_packet.md"
    json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_markdown(packet, report_path)
    return {"json": str(json_path), "report": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--run-label")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-output-dir")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--provider-config")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--base-url-or-endpoint", dest="base_url_or_endpoint")
    parser.add_argument("--budget-label")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--rate-limit", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--api-key-env-present", action="store_true")
    parser.add_argument("--server-config", default="configs/server/qwen3_8b_observation.yaml")
    parser.add_argument("--server-environment")
    parser.add_argument("--gpu-spec")
    parser.add_argument("--model-source")
    parser.add_argument("--catalog-path")
    parser.add_argument("--artifact-return-policy")
    parser.add_argument("--dataset")
    parser.add_argument("--processed-suffix")
    parser.add_argument("--reviews-jsonl")
    parser.add_argument("--metadata-jsonl")
    parser.add_argument("--license-accepted", action="store_true")
    parser.add_argument("--machine-disk-budget")
    parser.add_argument("--execution-location")
    parser.add_argument("--baseline-family")
    parser.add_argument("--model-family")
    parser.add_argument("--training-provenance")
    parser.add_argument("--split-declaration")
    parser.add_argument("--ranking-jsonl")
    parser.add_argument("--run-manifest-json")
    parser.add_argument("--leakage-guards")
    parser.add_argument("--split")
    args = parser.parse_args(argv)

    track = args.track
    options = vars(args)
    options.pop("track", None)
    packet = build_expansion_run_packet(track, **options)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "run_packets" / track / packet["run_label"]
    )
    packet["outputs"] = write_expansion_run_packet(packet, output_dir)
    print(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
