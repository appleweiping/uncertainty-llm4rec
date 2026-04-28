"""Run API observation in dry-run mode by default.

Real API execution is blocked unless --execute-api is explicitly passed and the
provider config has confirmed endpoint/model settings plus an API key env var.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.grounding import TitleGrounder  # noqa: E402
from storyflow.observation import (  # noqa: E402
    catalog_records,
    compute_observation_metrics,
    load_catalog_rows,
    observation_metrics_markdown,
    read_jsonl,
    utc_now_iso,
    write_jsonl,
)
from storyflow.observation_parsing import parse_observation_response  # noqa: E402
from storyflow.providers import (  # noqa: E402
    APIRequestRecord,
    APIResponseRecord,
    DryRunProvider,
    OpenAICompatibleProvider,
    ProviderExecutionError,
    ResponseCache,
    build_cache_key,
    load_provider_config,
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _default_output_dir(
    *,
    provider: str,
    input_jsonl: Path,
    dry_run: bool,
) -> Path:
    parts = input_jsonl.parts
    dataset = parts[-3] if len(parts) >= 3 else "dataset"
    processed_suffix = parts[-2] if len(parts) >= 2 else "processed"
    mode = "dry_run" if dry_run else "api"
    return ROOT / "outputs" / "api_observations" / provider / dataset / processed_suffix / f"{input_jsonl.stem}_{mode}"


def _completed_ids(*paths: Path) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if "input_id" in row:
                completed.add(str(row["input_id"]))
    return completed


def _cache_dir(config_cache_dir: str) -> Path:
    path = Path(config_cache_dir)
    return path if path.is_absolute() else ROOT / path


def _api_request_from_input(
    input_record: dict[str, Any],
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    request_options: dict[str, Any] | None,
) -> APIRequestRecord:
    cache_key = build_cache_key(
        provider=provider,
        model=model,
        prompt_template=str(input_record["prompt_template"]),
        temperature=temperature,
        input_hash=str(input_record["prompt_hash"]),
        max_tokens=max_tokens,
        request_options=request_options,
    )
    return APIRequestRecord(
        request_id=f"{provider}:{input_record['input_id']}",
        input_id=str(input_record["input_id"]),
        provider=provider,
        model=model,
        prompt_template=str(input_record["prompt_template"]),
        prompt_hash=str(input_record["prompt_hash"]),
        prompt=str(input_record["prompt"]),
        cache_key=cache_key,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata={
            "example_id": input_record.get("example_id"),
            "dataset": input_record.get("dataset"),
            "split": input_record.get("split"),
        },
    )


def _response_from_cache(
    cached: dict[str, Any],
    *,
    request: APIRequestRecord,
    dry_run: bool,
) -> APIResponseRecord:
    return APIResponseRecord(
        request_id=request.request_id,
        input_id=request.input_id,
        provider=request.provider,
        model=request.model,
        raw_text=cached.get("raw_text"),
        status=str(cached.get("status") or "ok"),
        cache_key=request.cache_key,
        cache_hit=True,
        dry_run=bool(cached.get("dry_run", dry_run)),
        usage=dict(cached.get("usage") or {}),
        raw_payload=dict(cached.get("raw_payload") or {}),
        error=cached.get("error"),
        created_at_unix=float(cached.get("created_at_unix") or time.time()),
    )


def _generate_with_retry(
    provider: DryRunProvider | OpenAICompatibleProvider,
    *,
    request: APIRequestRecord,
    input_record: dict[str, Any],
    attempts: int,
    backoff_seconds: float,
) -> APIResponseRecord:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return provider.generate(request, input_record)
        except ProviderExecutionError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(backoff_seconds * attempt)
    raise ProviderExecutionError(str(last_error))


def run_api_observation(
    *,
    provider_config_path: Path,
    input_jsonl: Path,
    output_dir: Path | None,
    max_examples: int | None,
    dry_run: bool,
    resume: bool,
    max_concurrency: int | None,
    rate_limit: int | None,
    cache_enabled_override: bool | None = None,
) -> dict[str, Any]:
    config = load_provider_config(provider_config_path)
    output_dir = output_dir or _default_output_dir(
        provider=config.provider_name,
        input_jsonl=input_jsonl,
        dry_run=dry_run,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request_records.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    parsed_path = output_dir / "parsed_predictions.jsonl"
    grounded_path = output_dir / "grounded_predictions.jsonl"
    failed_path = output_dir / "failed_cases.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"

    inputs = read_jsonl(input_jsonl)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    if not inputs:
        raise ValueError("input_jsonl contains no records")

    if not dry_run:
        if not os.environ.get(config.api_key_env):
            raise SystemExit(f"Missing API key env var for real execution: {config.api_key_env}")
        if config.requires_endpoint_confirmation or config.endpoint_is_placeholder:
            raise SystemExit("Provider endpoint/model still contains TODO confirmation fields")

    if not resume:
        for path in (request_path, raw_path, parsed_path, grounded_path, failed_path):
            path.write_text("", encoding="utf-8")

    completed = _completed_ids(grounded_path, failed_path) if resume else set()
    catalog_csv = _resolve(inputs[0]["source"]["catalog_csv"])
    catalog_rows = load_catalog_rows(catalog_csv)
    grounder = TitleGrounder(catalog_records(catalog_rows))
    cache_enabled = config.cache.enabled if cache_enabled_override is None else cache_enabled_override
    cache = ResponseCache(_cache_dir(config.cache.cache_dir), enabled=cache_enabled)
    provider = DryRunProvider(config) if dry_run else OpenAICompatibleProvider(config)
    effective_concurrency = max_concurrency or config.max_concurrency
    effective_rate_limit = rate_limit or config.rate_limit.requests_per_minute
    if effective_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if effective_rate_limit < 1:
        raise ValueError("rate_limit must be >= 1")

    newly_processed = 0
    failed_count = 0
    request_ids = {str(input_record["input_id"]) for input_record in inputs}
    for index, input_record in enumerate(inputs):
        input_id = str(input_record["input_id"])
        if input_id in completed:
            continue
        request = _api_request_from_input(
            input_record,
            provider=config.provider_name,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_options=config.extra_body,
        )
        write_jsonl(request_path, [asdict(request)], append=True)
        cached = cache.get(request.cache_key)
        if cached is not None:
            response = _response_from_cache(cached, request=request, dry_run=dry_run)
        else:
            try:
                response = _generate_with_retry(
                    provider,
                    request=request,
                    input_record=input_record,
                    attempts=config.retry.max_attempts,
                    backoff_seconds=config.retry.backoff_seconds,
                )
            except ProviderExecutionError as exc:
                failed_count += 1
                write_jsonl(
                    failed_path,
                    [
                        {
                            "input_id": input_id,
                            "request_id": request.request_id,
                            "provider": config.provider_name,
                            "error": str(exc),
                            "failure_stage": "provider",
                            "dry_run": dry_run,
                            "created_at_utc": utc_now_iso(),
                        }
                    ],
                    append=True,
                )
                continue
            cache.put(request.cache_key, asdict(response))
        write_jsonl(raw_path, [asdict(response)], append=True)
        parsed = parse_observation_response(response.raw_text or "")
        parsed_row = {
            "input_id": input_id,
            "request_id": request.request_id,
            "provider": config.provider_name,
            "model": config.model_name,
            "cache_key": request.cache_key,
            "cache_hit": response.cache_hit,
            "dry_run": dry_run,
            "parse": asdict(parsed),
            "created_at_utc": utc_now_iso(),
        }
        write_jsonl(parsed_path, [parsed_row], append=True)
        if not parsed.success:
            failed_count += 1
            write_jsonl(
                failed_path,
                [
                    {
                        **parsed_row,
                        "failure_stage": "parse",
                        "error": parsed.error,
                    }
                ],
                append=True,
            )
            continue
        grounded = grounder.ground(
            parsed.generated_title or "",
            prediction_id=request.request_id,
        )
        correctness = int(
            grounded.is_grounded and grounded.item_id == input_record["target_item_id"]
        )
        write_jsonl(
            grounded_path,
            [
                {
                    "input_id": input_id,
                    "request_id": request.request_id,
                    "example_id": input_record["example_id"],
                    "user_id": input_record["user_id"],
                    "split": input_record["split"],
                    "provider": config.provider_name,
                    "model": config.model_name,
                    "cache_key": request.cache_key,
                    "cache_hit": response.cache_hit,
                    "dry_run": dry_run,
                    "generated_title": parsed.generated_title,
                    "confidence": parsed.confidence,
                    "is_likely_correct": parsed.is_likely_correct,
                    "parse_strategy": parsed.parse_strategy,
                    "target_item_id": input_record["target_item_id"],
                    "target_title": input_record["target_title"],
                    "target_popularity": input_record["target_popularity"],
                    "target_popularity_bucket": input_record["target_popularity_bucket"],
                    "grounded_item_id": grounded.item_id,
                    "grounding_status": grounded.status.value,
                    "grounding_score": grounded.score,
                    "grounding_ambiguity": grounded.ambiguity,
                    "correctness": correctness,
                    "usage": response.usage,
                    "is_experiment_result": False,
                }
            ],
            append=True,
        )
        newly_processed += 1
        if not dry_run and index < len(inputs) - 1:
            time.sleep(60.0 / effective_rate_limit)

    grounded_rows = [
        row for row in read_jsonl(grounded_path) if str(row["input_id"]) in request_ids
    ] if grounded_path.exists() else []
    metrics: dict[str, Any] | None = None
    if grounded_rows:
        metrics = compute_observation_metrics(grounded_rows)
        metrics.update(
            {
                "provider": config.provider_name,
                "model": config.model_name,
                "dry_run": dry_run,
                "is_experiment_result": False,
                "note": (
                    "API framework dry-run metrics only; not paper evidence."
                    if dry_run
                    else "Approved small API smoke/pilot metrics only; not a full run or paper result."
                ),
            }
        )
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        report_path.write_text(
            observation_metrics_markdown(
                metrics,
                title="API Observation Dry-Run Report" if dry_run else "API Observation Smoke/Pilot Report",
            ),
            encoding="utf-8",
        )

    manifest = {
        "created_at_utc": utc_now_iso(),
        "provider_config": str(provider_config_path),
        "provider": config.provider_name,
        "model": config.model_name,
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "request_records": str(request_path),
        "raw_responses": str(raw_path),
        "parsed_predictions": str(parsed_path),
        "grounded_predictions": str(grounded_path),
        "failed_cases": str(failed_path),
        "metrics": str(metrics_path) if metrics else None,
        "report": str(report_path) if metrics else None,
        "requested_input_count": len(inputs),
        "newly_processed_count": newly_processed,
        "failed_count": failed_count,
        "total_grounded_count": len(grounded_rows),
        "resume": resume,
        "dry_run": dry_run,
        "execute_api": not dry_run,
        "max_concurrency": effective_concurrency,
        "rate_limit_requests_per_minute": effective_rate_limit,
        "cache_enabled": cache_enabled,
        "request_options": config.extra_body,
        "cache_dir": str(_cache_dir(config.cache.cache_dir)),
        "api_called": not dry_run,
        "is_experiment_result": False,
        "note": (
            "Dry-run does not call network or paid APIs."
            if dry_run
            else "Approved small API smoke/pilot run; not a full run or paper result."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run. This is also the default.")
    parser.add_argument("--execute-api", action="store_true", help="Explicitly allow real API execution if config and key are valid.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--rate-limit", type=int)
    parser.add_argument("--no-cache", action="store_true", help="Bypass provider cache for one-off diagnostics.")
    args = parser.parse_args(argv)

    provider_config = _resolve(args.provider_config)
    input_jsonl = _resolve(args.input_jsonl)
    output_dir = _resolve(args.output_dir) if args.output_dir else None
    dry_run = True
    if args.execute_api and not args.dry_run:
        dry_run = False
    resume = not args.no_resume
    manifest = run_api_observation(
        provider_config_path=provider_config,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        max_examples=args.max_examples,
        dry_run=dry_run,
        resume=resume,
        max_concurrency=args.max_concurrency,
        rate_limit=args.rate_limit,
        cache_enabled_override=False if args.no_cache else None,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
