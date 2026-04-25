from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.llm.base import normalize_generation_result
from src.llm.parser import parse_evidence_response, parse_relevance_evidence_response, parse_response


EVIDENCE_OUTPUT_SCHEMAS = {
    "evidence",
    "evidence_confidence",
    "evidence_posterior",
    "calibrated_evidence_posterior",
}

RELEVANCE_EVIDENCE_OUTPUT_SCHEMAS = {
    "relevance_evidence",
    "candidate_relevance_evidence",
    "candidate_relevance_evidence_posterior",
}

EVIDENCE_RESULT_KEYS = (
    "relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "evidence_margin",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "evidence_risk",
    "raw_confidence",
    "parse_success",
    "parse_error",
)


_CHECKPOINT_WRITE_LOCK = threading.Lock()


class _RateLimiter:
    def __init__(self, requests_per_minute: int | None = None) -> None:
        self.requests_per_minute = int(requests_per_minute or 0)
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def wait(self) -> None:
        if self.requests_per_minute <= 0:
            return

        interval = 60.0 / float(self.requests_per_minute)
        with self._lock:
            now = time.perf_counter()
            sleep_seconds = max(0.0, self._next_allowed_time - now)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
                now = time.perf_counter()
            self._next_allowed_time = now + interval


def _normalize_candidate_from_pointwise_sample(sample: dict[str, Any]) -> dict[str, Any]:
    candidate_item_id = (
        sample.get("candidate_item_id")
        or sample.get("item_id")
        or sample.get("target_item_id")
        or ""
    )

    candidate_title = (
        sample.get("candidate_title")
        or sample.get("title")
        or ""
    )

    candidate_meta = (
        sample.get("candidate_meta")
        or sample.get("candidate_description")
        or sample.get("candidate_text")
        or sample.get("text")
        or ""
    )

    return {
        "item_id": candidate_item_id,
        "title": candidate_title,
        "meta": candidate_meta,
    }


def _parse_pointwise_response(raw_text: str, output_schema: str | None) -> dict[str, Any]:
    schema = str(output_schema or "verbalized_confidence").strip().lower()
    if schema in RELEVANCE_EVIDENCE_OUTPUT_SCHEMAS:
        return parse_relevance_evidence_response(raw_text)
    if schema in EVIDENCE_OUTPUT_SCHEMAS:
        return parse_evidence_response(raw_text)
    return parse_response(raw_text)


def _add_optional_evidence_fields(result: dict[str, Any], parsed: dict[str, Any]) -> None:
    for key in EVIDENCE_RESULT_KEYS:
        if key in parsed:
            result[key] = parsed[key]


def _infer_parse_success(parsed: dict[str, Any]) -> bool:
    if "parse_success" in parsed:
        return bool(parsed["parse_success"])
    recommend_ok = parsed.get("recommend") in {"yes", "no"}
    confidence = parsed.get("confidence", -1.0)
    try:
        confidence_ok = 0.0 <= float(confidence) <= 1.0
    except (TypeError, ValueError):
        confidence_ok = False
    return bool(recommend_ok and confidence_ok)


def _build_result_record(
    *,
    sample: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
    raw_text: str,
    generation: dict[str, Any],
    parsed: dict[str, Any],
    label: int,
    method_variant: str | None = None,
    sample_id: int | str | None = None,
    input_row_index: int | None = None,
    split_name: str | None = None,
    retry_count: int = 0,
    error_type: str = "",
) -> dict[str, Any]:
    user_id = sample.get("user_id", "")
    target_item_id = (
        sample.get("target_item_id")
        or sample.get("candidate_item_id")
        or candidate.get("item_id", "")
    )
    popularity_group = sample.get("target_popularity_group", "unknown")

    result = {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "candidate_item_id": candidate.get("item_id", ""),
        "label": int(label),
        "target_popularity_group": popularity_group,
        "prompt": prompt,
        "raw_response": raw_text,
        "recommend": parsed.get("recommend", "unknown"),
        "confidence": parsed.get("confidence", parsed.get("raw_confidence", -1.0)),
        "reason": parsed.get("reason", ""),
        "response_latency": generation.get("latency", 0.0),
        "response_model_name": generation.get("model_name", ""),
        "response_provider": generation.get("provider", ""),
        "response_usage": generation.get("usage", {}),
    }
    if sample_id is not None:
        result["sample_id"] = sample_id
    if input_row_index is not None:
        result["input_row_index"] = int(input_row_index)
    if split_name:
        result["split_name"] = split_name
    result["parse_success"] = _infer_parse_success(parsed)
    result["error_type"] = error_type if error_type else ("" if result["parse_success"] else "parse_failed")
    result["retry_count"] = int(retry_count)
    result["latency_seconds"] = float(generation.get("latency", 0.0) or 0.0)
    if method_variant:
        result["method_variant"] = method_variant
    _add_optional_evidence_fields(result, parsed)
    return result


def _build_error_record(
    *,
    task: dict[str, Any],
    prompt: str,
    exc: Exception,
    retry_count: int,
    latency_seconds: float,
    method_variant: str | None,
    split_name: str | None,
) -> dict[str, Any]:
    sample = task["sample"]
    candidate = task["candidate"]
    label = int(task["label"])
    error_type = type(exc).__name__
    parsed = {
        "recommend": "unknown",
        "confidence": -1.0,
        "reason": "",
        "parse_success": False,
        "parse_error": error_type,
    }
    generation = {
        "raw_text": "",
        "latency": latency_seconds,
        "model_name": "",
        "provider": "",
        "usage": {},
    }
    result = _build_result_record(
        sample=sample,
        candidate=candidate,
        prompt=prompt,
        raw_text="",
        generation=generation,
        parsed=parsed,
        label=label,
        method_variant=method_variant,
        sample_id=task["sample_id"],
        input_row_index=task["input_row_index"],
        split_name=split_name,
        retry_count=retry_count,
        error_type=error_type,
    )
    result["error_message"] = str(exc)
    return result


def _flatten_pointwise_tasks(samples: list[dict]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    sample_id = 0
    for input_row_index, sample in enumerate(samples):
        if "candidates" in sample:
            target_item_id = sample["target_item"]["item_id"]
            for candidate_index, candidate in enumerate(sample["candidates"]):
                tasks.append(
                    {
                        "sample_id": sample_id,
                        "input_row_index": input_row_index,
                        "candidate_index": candidate_index,
                        "sample": sample,
                        "candidate": candidate,
                        "label": int(candidate["item_id"] == target_item_id),
                    }
                )
                sample_id += 1
            continue

        candidate = _normalize_candidate_from_pointwise_sample(sample)
        tasks.append(
            {
                "sample_id": sample_id,
                "input_row_index": input_row_index,
                "candidate_index": None,
                "sample": sample,
                "candidate": candidate,
                "label": int(sample.get("label", 0)),
            }
        )
        sample_id += 1
    return tasks


def _read_finished_sample_ids(output_path: str | Path) -> set[int]:
    path = Path(output_path)
    finished: set[int] = set()
    if not path.exists():
        return finished

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("sample_id")
            try:
                finished.add(int(sample_id))
            except (TypeError, ValueError):
                continue
    return finished


def _load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def _append_checkpoint_record(output_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CHECKPOINT_WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def _is_failed_record(record: dict[str, Any]) -> bool:
    error_type = str(record.get("error_type", "") or "").strip()
    parse_success = bool(record.get("parse_success", False))
    return bool(error_type) or not parse_success


def _write_sorted_records(output_path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            sample_id = int(record.get("sample_id"))
        except (TypeError, ValueError):
            continue
        deduped[sample_id] = record

    ordered = [deduped[key] for key in sorted(deduped)]
    with path.open("w", encoding="utf-8") as f:
        for record in ordered:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_single_task_with_retries(
    *,
    task: dict[str, Any],
    llm_backend,
    prompt_builder,
    output_schema: str | None,
    method_variant: str | None,
    split_name: str | None,
    max_retries: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None = None,
    rate_limiter: _RateLimiter | None = None,
) -> dict[str, Any]:
    prompt = prompt_builder.build_pointwise_prompt(task["sample"], task["candidate"])
    start = time.perf_counter()
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.wait()
            generation = normalize_generation_result(
                llm_backend.generate(
                    prompt,
                    timeout=timeout_seconds,
                    max_retries=0,
                ),
                default_provider=getattr(llm_backend, "provider", None),
                default_model_name=getattr(llm_backend, "model_name", "unknown"),
            )
            raw_text = generation["raw_text"]
            parsed = _parse_pointwise_response(raw_text, output_schema)
            result = _build_result_record(
                sample=task["sample"],
                candidate=task["candidate"],
                prompt=prompt,
                raw_text=raw_text,
                generation=generation,
                parsed=parsed,
                label=task["label"],
                method_variant=method_variant,
                sample_id=task["sample_id"],
                input_row_index=task["input_row_index"],
                split_name=split_name,
                retry_count=attempt,
            )
            if not result["parse_success"]:
                result["error_type"] = result.get("error_type") or "parse_failed"
                if attempt < max_retries:
                    sleep_seconds = retry_backoff_seconds * (2 ** attempt)
                    time.sleep(sleep_seconds)
                    continue
            return result
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_seconds = retry_backoff_seconds * (2 ** attempt)
            time.sleep(sleep_seconds)

    latency_seconds = time.perf_counter() - start
    assert last_exc is not None
    return _build_error_record(
        task=task,
        prompt=prompt,
        exc=last_exc,
        retry_count=max_retries,
        latency_seconds=latency_seconds,
        method_variant=method_variant,
        split_name=split_name,
    )


def run_pointwise_inference(
    samples: list[dict],
    llm_backend,
    prompt_builder,
    output_schema: str | None = None,
    method_variant: str | None = None,
) -> list[dict]:
    results: list[dict] = []

    for sample in tqdm(samples, desc="Running pointwise inference"):
        # Legacy toy schema: one record contains target_item and multiple candidates.
        if "candidates" in sample:
            user_id = sample["user_id"]
            target_item_id = sample["target_item"]["item_id"]
            popularity_group = sample.get("target_popularity_group", "tail")

            for candidate in sample["candidates"]:
                prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
                generation = normalize_generation_result(
                    llm_backend.generate(prompt),
                    default_provider=getattr(llm_backend, "provider", None),
                    default_model_name=getattr(llm_backend, "model_name", "unknown"),
                )
                raw_text = generation["raw_text"]
                parsed = _parse_pointwise_response(raw_text, output_schema)

                result = {
                    "user_id": user_id,
                    "target_item_id": target_item_id,
                    "candidate_item_id": candidate["item_id"],
                    "label": int(candidate["item_id"] == target_item_id),
                    "target_popularity_group": popularity_group,
                    "prompt": prompt,
                    "raw_response": raw_text,
                    "recommend": parsed.get("recommend", "unknown"),
                    "confidence": parsed.get("confidence", parsed.get("raw_confidence", -1.0)),
                    "reason": parsed.get("reason", ""),
                    "response_latency": generation.get("latency", 0.0),
                    "response_model_name": generation.get("model_name", ""),
                    "response_provider": generation.get("provider", ""),
                    "response_usage": generation.get("usage", {}),
                }
                if method_variant:
                    result["method_variant"] = method_variant
                _add_optional_evidence_fields(result, parsed)
                results.append(result)

            continue

        # Current pointwise schema: each record is already a user-candidate sample.
        candidate = _normalize_candidate_from_pointwise_sample(sample)
        prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
        generation = normalize_generation_result(
            llm_backend.generate(prompt),
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        raw_text = generation["raw_text"]
        parsed = _parse_pointwise_response(raw_text, output_schema)

        label = sample.get("label", 0)

        result = _build_result_record(
            sample=sample,
            candidate=candidate,
            prompt=prompt,
            raw_text=raw_text,
            generation=generation,
            parsed=parsed,
            label=label,
            method_variant=method_variant,
        )
        results.append(result)

    return results


def run_pointwise_inference_concurrent(
    samples: list[dict],
    llm_backend,
    prompt_builder,
    output_path: str | Path,
    output_schema: str | None = None,
    method_variant: str | None = None,
    split_name: str | None = None,
    max_workers: int = 4,
    requests_per_minute: int | None = None,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    timeout_seconds: float | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    failed_output_path: str | Path | None = None,
) -> list[dict]:
    output_path = Path(output_path)
    failed_output_path = Path(failed_output_path) if failed_output_path else output_path.parent / "failed_samples.jsonl"
    # In resume mode the existing JSONL is the checkpoint, so preserve it even
    # when an older experiment config still has overwrite=true.
    if overwrite and not resume and output_path.exists():
        output_path.unlink()
    if overwrite and not resume and failed_output_path.exists():
        failed_output_path.unlink()

    tasks = _flatten_pointwise_tasks(samples)
    finished_sample_ids = _read_finished_sample_ids(output_path) if resume else set()
    pending_tasks = [task for task in tasks if int(task["sample_id"]) not in finished_sample_ids]

    print(
        f"Concurrent inference: total={len(tasks)} finished={len(finished_sample_ids)} "
        f"pending={len(pending_tasks)} max_workers={max_workers} "
        f"requests_per_minute={requests_per_minute or 'unlimited'} checkpoint_every={checkpoint_every}"
    )

    if pending_tasks:
        rate_limiter = _RateLimiter(requests_per_minute=requests_per_minute)
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            future_to_task = {
                executor.submit(
                    _run_single_task_with_retries,
                    task=task,
                    llm_backend=llm_backend,
                    prompt_builder=prompt_builder,
                    output_schema=output_schema,
                    method_variant=method_variant,
                    split_name=split_name,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    timeout_seconds=timeout_seconds,
                    rate_limiter=rate_limiter,
                ): task
                for task in pending_tasks
            }

            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Running concurrent pointwise inference",
            ):
                record = future.result()
                _append_checkpoint_record(output_path, record)
                if _is_failed_record(record):
                    _append_checkpoint_record(failed_output_path, record)

    records = _load_jsonl_records(output_path)
    _write_sorted_records(output_path, records)
    sorted_records = _load_jsonl_records(output_path)
    failed_count = sum(1 for record in sorted_records if _is_failed_record(record))
    print(
        f"Concurrent inference complete: rows={len(sorted_records)} "
        f"failed_or_parse_failed={failed_count} failed_output={failed_output_path}"
    )
    return sorted_records
