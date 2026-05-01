"""Build a local-only DeepSeek experiment plan for Storyflow/TRUCE-Rec.

The plan is an ignored readiness artifact. It does not call APIs, execute
servers, train models, download data, or claim results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.generation import (  # noqa: E402
    CATALOG_CONSTRAINED_JSON_TEMPLATE,
    FORCED_JSON_TEMPLATE,
    RETRIEVAL_CONTEXT_JSON_TEMPLATE,
)
from storyflow.observation import default_observation_input_path  # noqa: E402
from storyflow.providers import load_provider_config  # noqa: E402


DEFAULT_DATASETS = (
    "amazon_reviews_2023_beauty",
    "amazon_reviews_2023_health",
    "amazon_reviews_2023_video_games",
)
PROMPT_TEMPLATES = (
    FORCED_JSON_TEMPLATE.name,
    RETRIEVAL_CONTEXT_JSON_TEMPLATE.name,
    CATALOG_CONSTRAINED_JSON_TEMPLATE.name,
)
SERVER_DEFERRED_TRACKS = (
    "qwen3_8b_server_observation",
    "qwen3_8b_lora_training",
    "server_trained_ranking_baselines",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _gate_output_path(
    *,
    dataset: str,
    processed_suffix: str,
    split: str,
    prompt_template: str,
    candidate_count: int | None,
    max_examples: int | None,
    repeat_target_policy: str,
) -> Path:
    base_path = default_observation_input_path(
        dataset=dataset,
        processed_suffix=processed_suffix,
        split=split,
        prompt_template=prompt_template,
        candidate_count=candidate_count,
        repeat_target_policy=repeat_target_policy,
        root=ROOT,
    )
    if max_examples is None:
        return base_path
    prefix = f"{split}_"
    stem = base_path.stem
    if stem.startswith(prefix):
        stem = f"{split}_gate{max_examples}_{stem[len(prefix):]}"
    else:
        stem = f"{stem}_gate{max_examples}"
    return base_path.with_name(f"{stem}{base_path.suffix}")


def _template_candidate_count(prompt_template: str, candidate_count: int) -> int | None:
    if prompt_template in {
        CATALOG_CONSTRAINED_JSON_TEMPLATE.name,
        RETRIEVAL_CONTEXT_JSON_TEMPLATE.name,
    }:
        return candidate_count
    return None


def _build_gate_command(
    *,
    dataset: str,
    processed_suffix: str,
    split: str,
    gate_size: int,
    candidate_count: int,
    repeat_target_policy: str,
    stratify_by_popularity: bool,
) -> str:
    parts = [
        "python scripts/build_observation_gate_inputs.py",
        f"--dataset {dataset}",
        f"--processed-suffix {processed_suffix}",
        f"--split {split}",
        f"--max-examples {gate_size}",
        f"--candidate-count {candidate_count}",
        f"--repeat-target-policy {repeat_target_policy}",
    ]
    if stratify_by_popularity:
        parts.append("--stratify-by-popularity")
    return " ".join(parts)


def _planned_run_dir(
    *,
    provider: str,
    dataset: str,
    processed_suffix: str,
    input_jsonl: Path,
    run_label: str,
) -> Path:
    return (
        ROOT
        / "outputs"
        / "api_observations"
        / provider
        / dataset
        / processed_suffix
        / f"{input_jsonl.stem}_api_{_sanitize(run_label)}"
    )


def _post_api_commands(
    *,
    run_dir: Path,
    input_jsonl: Path,
    catalog_csv: Path,
    source_label: str,
    same_split_diagnostic: bool,
) -> list[str]:
    grounded_jsonl = run_dir / "grounded_predictions.jsonl"
    features_dir = ROOT / "outputs" / "confidence_features" / run_dir.relative_to(ROOT / "outputs")
    features_jsonl = features_dir / "features.jsonl"
    commands = [
        (
            "python scripts/analyze_observation.py "
            f"--run-dir {_display_path(run_dir)} "
            f"--input-jsonl {_display_path(input_jsonl)} "
            f"--source-label {source_label}"
        ),
        (
            "python scripts/review_observation_cases.py "
            f"--run-dir {_display_path(run_dir)} "
            f"--input-jsonl {_display_path(input_jsonl)}"
        ),
        (
            "python scripts/build_confidence_features.py "
            f"--grounded-jsonl {_display_path(grounded_jsonl)} "
            f"--input-jsonl {_display_path(input_jsonl)} "
            f"--catalog-csv {_display_path(catalog_csv)}"
        ),
    ]
    if same_split_diagnostic:
        commands.extend(
            [
                (
                    "python scripts/calibrate_confidence_features.py "
                    f"--features-jsonl {_display_path(features_jsonl)} "
                    "--fit-splits test --eval-splits test "
                    "--allow-same-split-eval --n-bins 5"
                ),
                (
                    "python scripts/residualize_confidence_features.py "
                    f"--features-jsonl {_display_path(features_jsonl)} "
                    "--fit-splits test --eval-splits test "
                    "--allow-same-split-eval"
                ),
            ]
        )
    return commands


def _variant_plan(
    *,
    dataset: str,
    processed_suffix: str,
    split: str,
    prompt_template: str,
    gate_size: int,
    candidate_count: int,
    repeat_target_policy: str,
    provider_config: Path,
    provider: str,
    model: str,
    run_stage: str,
    rate_limit: int,
    max_concurrency: int,
    budget_label: str | None,
    run_label_prefix: str,
    allow_over_20: bool,
    catalog_csv: Path,
    same_split_diagnostic: bool,
) -> dict[str, Any]:
    template_candidate_count = _template_candidate_count(prompt_template, candidate_count)
    input_jsonl = _gate_output_path(
        dataset=dataset,
        processed_suffix=processed_suffix,
        split=split,
        prompt_template=prompt_template,
        candidate_count=template_candidate_count,
        max_examples=gate_size,
        repeat_target_policy=repeat_target_policy,
    )
    run_label = _sanitize(
        f"{run_label_prefix}_{dataset}_{processed_suffix}_{split}_{prompt_template}_n{gate_size}"
    )
    run_dir = _planned_run_dir(
        provider=provider,
        dataset=dataset,
        processed_suffix=processed_suffix,
        input_jsonl=input_jsonl,
        run_label=run_label,
    )
    budget = budget_label or "<budget-label>"
    readiness_parts = [
        "python scripts/check_api_pilot_readiness.py",
        f"--provider-config {_display_path(provider_config)}",
        f"--input-jsonl {_display_path(input_jsonl)}",
        f"--sample-size {gate_size}",
        f"--stage {run_stage}",
        f"--approved-provider {provider}",
        f"--approved-model {model}",
        f"--approved-rate-limit {rate_limit}",
        f"--approved-max-concurrency {max_concurrency}",
        f"--approved-budget-label {budget}",
        "--execute-api-intended",
    ]
    if allow_over_20 and run_stage == "pilot" and gate_size > 20:
        readiness_parts.append("--allow-over-20")
    dry_run_command = (
        "python scripts/run_api_observation.py "
        f"--provider-config {_display_path(provider_config)} "
        f"--input-jsonl {_display_path(input_jsonl)} "
        f"--output-dir {_display_path(run_dir)}_dry_run "
        f"--max-examples {gate_size} "
        f"--run-stage {run_stage} "
        "--dry-run"
    )
    execute_command = (
        "python scripts/run_api_observation.py "
        f"--provider-config {_display_path(provider_config)} "
        f"--input-jsonl {_display_path(input_jsonl)} "
        f"--output-dir {_display_path(run_dir)} "
        f"--max-examples {gate_size} "
        f"--execute-api "
        f"--rate-limit {rate_limit} "
        f"--max-concurrency {max_concurrency} "
        f"--run-label {run_label} "
        f"--budget-label {budget} "
        f"--run-stage {run_stage}"
    )
    return {
        "prompt_template": prompt_template,
        "candidate_count": template_candidate_count,
        "input_jsonl": _display_path(input_jsonl),
        "input_exists": input_jsonl.exists(),
        "input_record_count": _jsonl_count(input_jsonl),
        "planned_run_label": run_label,
        "planned_output_dir": _display_path(run_dir),
        "safe_preflight_commands": [
            " ".join(readiness_parts),
            dry_run_command,
        ],
        "approval_required_execute_command": execute_command,
        "post_api_commands": _post_api_commands(
            run_dir=run_dir,
            input_jsonl=input_jsonl,
            catalog_csv=catalog_csv,
            source_label=run_label,
            same_split_diagnostic=same_split_diagnostic,
        ),
        "api_called": False,
        "server_executed": False,
        "model_training": False,
        "is_experiment_result": False,
    }


def build_local_deepseek_experiment_plan(
    *,
    datasets: list[str],
    processed_suffix: str,
    split: str,
    gate_size: int,
    candidate_count: int,
    repeat_target_policy: str,
    provider_config_path: Path,
    run_stage: str,
    rate_limit: int,
    max_concurrency: int,
    budget_label: str | None,
    run_label_prefix: str,
    stratify_by_popularity: bool = True,
    allow_over_20: bool = True,
    same_split_diagnostic: bool = True,
) -> dict[str, Any]:
    if gate_size < 1:
        raise ValueError("gate_size must be >= 1")
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    if run_stage not in {"smoke", "pilot", "full"}:
        raise ValueError("run_stage must be smoke, pilot, or full")
    if repeat_target_policy not in {"all", "exclude", "only"}:
        raise ValueError("repeat_target_policy must be all, exclude, or only")

    provider_config = _resolve(provider_config_path)
    config = load_provider_config(provider_config)
    provider = config.provider_name
    model = config.model_name
    api_key_env_present = bool(os.environ.get(config.api_key_env))
    missing_confirmations = []
    if not budget_label:
        missing_confirmations.append("budget_label")
    if not api_key_env_present:
        missing_confirmations.append(f"environment variable {config.api_key_env}")
    missing_confirmations.append("current-turn explicit approval before running any --execute-api command")

    dataset_plans: list[dict[str, Any]] = []
    for dataset in datasets:
        processed_dir = ROOT / "data" / "processed" / dataset / processed_suffix
        catalog_csv = processed_dir / "item_catalog.csv"
        build_gate_command = _build_gate_command(
            dataset=dataset,
            processed_suffix=processed_suffix,
            split=split,
            gate_size=gate_size,
            candidate_count=candidate_count,
            repeat_target_policy=repeat_target_policy,
            stratify_by_popularity=stratify_by_popularity,
        )
        variants = [
            _variant_plan(
                dataset=dataset,
                processed_suffix=processed_suffix,
                split=split,
                prompt_template=template,
                gate_size=gate_size,
                candidate_count=candidate_count,
                repeat_target_policy=repeat_target_policy,
                provider_config=provider_config,
                provider=provider,
                model=model,
                run_stage=run_stage,
                rate_limit=rate_limit,
                max_concurrency=max_concurrency,
                budget_label=budget_label,
                run_label_prefix=run_label_prefix,
                allow_over_20=allow_over_20,
                catalog_csv=catalog_csv,
                same_split_diagnostic=same_split_diagnostic,
            )
            for template in PROMPT_TEMPLATES
        ]
        dataset_plans.append(
            {
                "dataset": dataset,
                "processed_suffix": processed_suffix,
                "split": split,
                "processed_dir": _display_path(processed_dir),
                "processed_dir_exists": processed_dir.exists(),
                "catalog_csv": _display_path(catalog_csv),
                "catalog_csv_exists": catalog_csv.exists(),
                "validation_commands": [
                    (
                        "python scripts/validate_processed_dataset.py "
                        f"--dataset {dataset} --processed-suffix {processed_suffix}"
                    ),
                    (
                        "python scripts/audit_processed_dataset.py "
                        f"--dataset {dataset} --processed-suffix {processed_suffix}"
                    ),
                ],
                "build_gate_inputs_command": build_gate_command,
                "prompt_variants": variants,
                "api_called": False,
                "server_executed": False,
                "model_training": False,
                "is_experiment_result": False,
            }
        )

    return {
        "artifact_kind": "storyflow_local_deepseek_experiment_plan",
        "created_at_utc": _utc_now_iso(),
        "claim_scope": "plan_only_not_execution_not_paper_evidence",
        "recommended_execution_order": [
            "validate_processed_dataset",
            "audit_processed_dataset",
            "build_observation_gate_inputs",
            "check_api_pilot_readiness",
            "run_api_observation --dry-run",
            "run_api_observation --execute-api only after explicit approval",
            "analyze_observation",
            "review_observation_cases",
            "build_confidence_features",
            "diagnostic calibration/residualization only when split caveats are recorded",
        ],
        "provider_config": _display_path(provider_config),
        "provider": provider,
        "model": model,
        "api_key_env": config.api_key_env,
        "api_key_env_present": api_key_env_present,
        "api_key_value_printed": False,
        "run_stage": run_stage,
        "gate_size": gate_size,
        "candidate_count": candidate_count,
        "repeat_target_policy": repeat_target_policy,
        "rate_limit_requests_per_minute": rate_limit,
        "max_concurrency": max_concurrency,
        "budget_label": budget_label,
        "missing_execution_confirmations": missing_confirmations,
        "server_deferred": True,
        "server_deferred_tracks": list(SERVER_DEFERRED_TRACKS),
        "small_model_training_deferred": True,
        "api_called": False,
        "server_executed": False,
        "model_training": False,
        "data_downloaded": False,
        "full_data_processed": False,
        "is_experiment_result": False,
        "claim_guardrails": [
            "This plan does not authorize a paid API call by itself.",
            "Do not report dry-run, readiness, or same-split diagnostic artifacts as paper results.",
            "Retrieval-context and catalog-constrained target-excluding gates diagnose prompt/grounding behavior, not recommendation accuracy.",
            "Server Qwen3 observation and LoRA training are explicitly deferred.",
        ],
        "datasets": dataset_plans,
    }


def _write_markdown(plan: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Local DeepSeek Experiment Plan: {plan['provider']} / {plan['model']}",
        "",
        "This is a plan-only artifact. It does not call APIs, execute server jobs, train models, download data, or create paper evidence.",
        "",
        "## Scope",
        "",
        f"- claim scope: {plan['claim_scope']}",
        f"- run stage: {plan['run_stage']}",
        f"- gate size: {plan['gate_size']}",
        f"- repeat target policy: {plan['repeat_target_policy']}",
        f"- rate limit: {plan['rate_limit_requests_per_minute']} requests/minute",
        f"- max concurrency: {plan['max_concurrency']}",
        f"- API key env present in current process: {plan['api_key_env_present']}",
        f"- server deferred: {plan['server_deferred']}",
        "",
        "## Missing Before Execution",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["missing_execution_confirmations"])
    lines.extend(["", "## Execution Order", ""])
    lines.extend(f"- {item}" for item in plan["recommended_execution_order"])
    for dataset in plan["datasets"]:
        lines.extend(
            [
                "",
                f"## {dataset['dataset']} / {dataset['processed_suffix']}",
                "",
                f"- processed dir exists: {dataset['processed_dir_exists']}",
                f"- catalog exists: {dataset['catalog_csv_exists']}",
                "",
                "Validation:",
            ]
        )
        lines.extend(f"- `{command}`" for command in dataset["validation_commands"])
        lines.extend(["", "Build inputs:", "", f"- `{dataset['build_gate_inputs_command']}`", ""])
        for variant in dataset["prompt_variants"]:
            lines.extend(
                [
                    f"### {variant['prompt_template']}",
                    "",
                    f"- input: `{variant['input_jsonl']}`",
                    f"- input exists: {variant['input_exists']}",
                    f"- input records: {variant['input_record_count']}",
                    f"- planned output: `{variant['planned_output_dir']}`",
                    "",
                    "Safe preflight:",
                ]
            )
            lines.extend(f"- `{command}`" for command in variant["safe_preflight_commands"])
            lines.extend(
                [
                    "",
                    "Approval-required execution:",
                    "",
                    "```powershell",
                    variant["approval_required_execute_command"],
                    "```",
                    "",
                    "Post-API local diagnostics:",
                ]
            )
            lines.extend(f"- `{command}`" for command in variant["post_api_commands"])
    lines.extend(["", "## Guardrails", ""])
    lines.extend(f"- {item}" for item in plan["claim_guardrails"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local_deepseek_experiment_plan(plan: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "local_deepseek_experiment_plan.json"
    report_path = output_dir / "local_deepseek_experiment_plan.md"
    outputs = {"json": _display_path(json_path), "report": _display_path(report_path)}
    plan["outputs"] = outputs
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_markdown(plan, report_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--processed-suffix", default="full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gate-size", type=int, default=60)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--repeat-target-policy", default="exclude", choices=["all", "exclude", "only"])
    parser.add_argument("--provider-config", default="configs/providers/deepseek.yaml")
    parser.add_argument("--run-stage", choices=["smoke", "pilot", "full"], default="full")
    parser.add_argument("--rate-limit", type=int, default=60)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--budget-label")
    parser.add_argument("--run-label-prefix", default="local_deepseek_fulldata_gate")
    parser.add_argument("--no-stratify-by-popularity", action="store_true")
    parser.add_argument("--no-allow-over-20", action="store_true")
    parser.add_argument("--no-same-split-diagnostic", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    plan = build_local_deepseek_experiment_plan(
        datasets=args.datasets or list(DEFAULT_DATASETS),
        processed_suffix=args.processed_suffix,
        split=args.split,
        gate_size=args.gate_size,
        candidate_count=args.candidate_count,
        repeat_target_policy=args.repeat_target_policy,
        provider_config_path=Path(args.provider_config),
        run_stage=args.run_stage,
        rate_limit=args.rate_limit,
        max_concurrency=args.max_concurrency,
        budget_label=args.budget_label,
        run_label_prefix=args.run_label_prefix,
        stratify_by_popularity=not args.no_stratify_by_popularity,
        allow_over_20=not args.no_allow_over_20,
        same_split_diagnostic=not args.no_same_split_diagnostic,
    )
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "experiment_plans" / _sanitize(args.run_label_prefix)
    )
    plan["outputs"] = write_local_deepseek_experiment_plan(plan, output_dir)
    print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
