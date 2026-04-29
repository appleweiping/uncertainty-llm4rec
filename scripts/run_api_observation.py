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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock
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
    run_label: str | None,
    budget_label: str | None,
    execution_mode: str,
) -> APIRequestRecord:
    cache_key = build_cache_key(
        provider=provider,
        model=model,
        prompt_template=str(input_record["prompt_template"]),
        temperature=temperature,
        input_hash=str(input_record["prompt_hash"]),
        max_tokens=max_tokens,
        request_options=request_options,
        execution_mode=execution_mode,
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
            "run_label": run_label,
            "budget_label": budget_label,
            "execution_mode": execution_mode,
        },
    )


class _StartRateLimiter:
    """Thread-safe start-time limiter for external API requests."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be >= 1")
        self.interval_seconds = 60.0 / requests_per_minute
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed)
            self._next_allowed = scheduled + self.interval_seconds
            delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def _valid_cached_response(cached: dict[str, Any] | None, *, dry_run: bool) -> dict[str, Any] | None:
    if cached is None:
        return None
    if bool(cached.get("dry_run")) != dry_run:
        return None
    return cached


def _failure_row(
    *,
    input_id: str,
    request_id: str,
    provider: str,
    dry_run: bool,
    failure_stage: str,
    error: str | None,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **(row or {}),
        "input_id": input_id,
        "request_id": request_id,
        "provider": provider,
        "error": error,
        "failure_stage": failure_stage,
        "dry_run": dry_run,
        "created_at_utc": utc_now_iso(),
    }


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


def _run_note(*, dry_run: bool, run_stage: str) -> str:
    if dry_run:
        return "Dry-run does not call network or paid APIs."
    if run_stage == "full":
        return (
            "Approved full-slice API observation artifact with cache/resume; "
            "not a paper result until downstream analysis, scope labels, and "
            "reviewer-facing caveats are finalized."
        )
    if run_stage == "smoke":
        return "Approved tiny API smoke artifact; not a full run or paper result."
    return "Approved API pilot artifact; not a full run or paper result."


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
    run_label: str | None = None,
    budget_label: str | None = None,
    run_stage: str = "pilot",
) -> dict[str, Any]:
    if run_stage not in {"smoke", "pilot", "full"}:
        raise ValueError("run_stage must be smoke, pilot, or full")
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
    execution_mode = "dry_run" if dry_run else "execute_api"
    limiter = None if dry_run else _StartRateLimiter(effective_rate_limit)
    request_ids = {str(input_record["input_id"]) for input_record in inputs}
    pending_inputs = [
        input_record
        for input_record in inputs
        if str(input_record["input_id"]) not in completed
    ]

    def process_one(input_record: dict[str, Any]) -> dict[str, Any]:
        input_id = str(input_record["input_id"])
        request = _api_request_from_input(
            input_record,
            provider=config.provider_name,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_options=config.extra_body,
            run_label=run_label,
            budget_label=budget_label,
            execution_mode=execution_mode,
        )
        cached = _valid_cached_response(cache.get(request.cache_key), dry_run=dry_run)
        if cached is not None:
            response = _response_from_cache(cached, request=request, dry_run=dry_run)
        else:
            try:
                if limiter is not None:
                    limiter.wait()
                response = _generate_with_retry(
                    provider,
                    request=request,
                    input_record=input_record,
                    attempts=config.retry.max_attempts,
                    backoff_seconds=config.retry.backoff_seconds,
                )
            except ProviderExecutionError as exc:
                return {
                    "request": asdict(request),
                    "raw": None,
                    "parsed": None,
                    "grounded": None,
                    "failed": _failure_row(
                        input_id=input_id,
                        request_id=request.request_id,
                        provider=config.provider_name,
                        error=str(exc),
                        failure_stage="provider",
                        dry_run=dry_run,
                    ),
                }
            cache.put(request.cache_key, asdict(response))
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
        if not parsed.success:
            return {
                "request": asdict(request),
                "raw": asdict(response),
                "parsed": parsed_row,
                "grounded": None,
                "failed": _failure_row(
                    input_id=input_id,
                    request_id=request.request_id,
                    provider=config.provider_name,
                    error=parsed.error,
                    failure_stage="parse",
                    dry_run=dry_run,
                    row=parsed_row,
                ),
            }
        grounded = grounder.ground(
            parsed.generated_title or "",
            prediction_id=request.request_id,
        )
        correctness = int(
            grounded.is_grounded and grounded.item_id == input_record["target_item_id"]
        )
        grounded_row = {
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
            "target_in_history": bool(input_record.get("target_in_history", False)),
            "target_history_occurrence_count": int(
                input_record.get("target_history_occurrence_count") or 0
            ),
            "target_same_timestamp_as_history": bool(
                input_record.get("target_same_timestamp_as_history", False)
            ),
            "history_duplicate_item_count": int(
                input_record.get("history_duplicate_item_count") or 0
            ),
            "history_unique_item_count": int(
                input_record.get("history_unique_item_count") or 0
            ),
            "grounded_item_id": grounded.item_id,
            "grounding_status": grounded.status.value,
            "grounding_score": grounded.score,
            "grounding_ambiguity": grounded.ambiguity,
            "correctness": correctness,
            "usage": response.usage,
            "is_experiment_result": False,
        }
        return {
            "request": asdict(request),
            "raw": asdict(response),
            "parsed": parsed_row,
            "grounded": grounded_row,
            "failed": None,
        }

    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        future_to_index = {
            executor.submit(process_one, input_record): index
            for index, input_record in enumerate(pending_inputs)
        }
        for future in as_completed(future_to_index):
            result = future.result()
            write_jsonl(request_path, [result["request"]], append=True)
            if result["raw"] is not None:
                write_jsonl(raw_path, [result["raw"]], append=True)
            if result["parsed"] is not None:
                write_jsonl(parsed_path, [result["parsed"]], append=True)
            if result["failed"] is not None:
                failed_count += 1
                write_jsonl(failed_path, [result["failed"]], append=True)
            if result["grounded"] is not None:
                newly_processed += 1
                write_jsonl(grounded_path, [result["grounded"]], append=True)

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
                "run_stage": run_stage,
                "is_experiment_result": False,
                "note": _run_note(dry_run=dry_run, run_stage=run_stage),
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
        "run_label": run_label,
        "budget_label": budget_label,
        "run_stage": run_stage,
        "execution_mode": execution_mode,
        "cache_dir": str(_cache_dir(config.cache.cache_dir)),
        "api_called": not dry_run,
        "is_experiment_result": False,
        "note": _run_note(dry_run=dry_run, run_stage=run_stage),
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
    parser.add_argument("--run-label", help="Human-readable run label written to request records and manifest.")
    parser.add_argument("--budget-label", help="User-approved budget/provenance label written to request records and manifest.")
    parser.add_argument("--run-stage", choices=["smoke", "pilot", "full"], default="pilot")
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
        run_label=args.run_label,
        budget_label=args.budget_label,
        run_stage=args.run_stage,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
