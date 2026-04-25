from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.llm import build_backend_from_config
from src.llm.base import normalize_generation_result
from src.llm.parser import (
    parse_recommendation_list_evidence_response,
    parse_recommendation_list_plain_response,
)
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


_WRITE_LOCK = threading.Lock()


class _RateLimiter:
    def __init__(self, requests_per_minute: int | None) -> None:
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


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    df = pd.read_json(path, lines=True)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


def write_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def load_existing(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_sorted_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    deduped: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            deduped[int(record["sample_id"])] = record
        except (KeyError, TypeError, ValueError):
            continue
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(deduped):
            f.write(json.dumps(deduped[key], ensure_ascii=False) + "\n")


def build_history_block(sample: dict[str, Any]) -> str:
    history = sample.get("history") or []
    if not isinstance(history, list):
        return str(history).strip()
    lines = [f"{idx}. {str(item).strip() or '[EMPTY]'}" for idx, item in enumerate(history, start=1)]
    return "\n".join(lines)


def build_candidate_pool_block(sample: dict[str, Any]) -> str:
    ids = sample.get("candidate_item_ids") or []
    titles = sample.get("candidate_titles") or []
    texts = sample.get("candidate_texts") or []
    rows: list[str] = []
    for idx, item_id in enumerate(ids, start=1):
        title = str(titles[idx - 1] if idx - 1 < len(titles) else "").strip()
        text = str(texts[idx - 1] if idx - 1 < len(texts) else "").strip()
        desc = text or f"Title: {title}"
        rows.append(f"{idx}. candidate_item_id: {item_id}\n   {desc}")
    return "\n".join(rows)


def build_prompt(template: str, sample: dict[str, Any], top_k: int) -> str:
    return template.format(
        history_block=build_history_block(sample),
        candidate_pool_block=build_candidate_pool_block(sample),
        top_k=int(top_k),
    )


def parse_response(raw_text: str, setting: str, candidate_item_ids: list[str]) -> dict[str, Any]:
    if setting == "plain":
        return parse_recommendation_list_plain_response(raw_text, candidate_item_ids)
    if setting == "evidence":
        return parse_recommendation_list_evidence_response(raw_text, candidate_item_ids)
    raise ValueError("setting must be plain or evidence.")


def run_one(
    *,
    sample_id: int,
    sample: dict[str, Any],
    setting: str,
    template: str,
    top_k: int,
    backend,
    rate_limiter: _RateLimiter,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    prompt = build_prompt(template, sample, top_k=top_k)
    candidate_item_ids = [str(item_id) for item_id in (sample.get("candidate_item_ids") or [])]
    last_exc: Exception | None = None
    started = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            rate_limiter.wait()
            generation = normalize_generation_result(
                backend.generate(prompt, max_retries=0),
                default_provider=getattr(backend, "provider", None),
                default_model_name=getattr(backend, "model_name", "unknown"),
            )
            raw_text = generation["raw_text"]
            parsed = parse_response(raw_text, setting, candidate_item_ids)
            return {
                "sample_id": sample_id,
                "source_event_id": sample.get("source_event_id", ""),
                "user_id": sample.get("user_id", ""),
                "positive_item_id": sample.get("positive_item_id", ""),
                "candidate_item_ids": candidate_item_ids,
                "candidate_labels": sample.get("candidate_labels", []),
                "candidate_popularity_groups": sample.get("candidate_popularity_groups", []),
                "recommended_items": parsed.get("recommended_items", []),
                "global_uncertainty": parsed.get("global_uncertainty"),
                "selection_rationale": parsed.get("selection_rationale", ""),
                "parse_success": bool(parsed.get("parse_success", False)),
                "schema_valid": bool(parsed.get("schema_valid", False)),
                "invalid_item_count": int(parsed.get("invalid_item_count", 0) or 0),
                "duplicate_item_count": int(parsed.get("duplicate_item_count", 0) or 0),
                "parse_error": parsed.get("parse_error", ""),
                "raw_response": raw_text,
                "prompt": prompt,
                "setting": setting,
                "response_latency": generation.get("latency", 0.0),
                "response_model_name": generation.get("model_name", ""),
                "response_provider": generation.get("provider", ""),
                "response_usage": generation.get("usage", {}),
                "retry_count": attempt,
            }
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_seconds * (2 ** attempt))

    assert last_exc is not None
    return {
        "sample_id": sample_id,
        "source_event_id": sample.get("source_event_id", ""),
        "user_id": sample.get("user_id", ""),
        "positive_item_id": sample.get("positive_item_id", ""),
        "candidate_item_ids": candidate_item_ids,
        "candidate_labels": sample.get("candidate_labels", []),
        "candidate_popularity_groups": sample.get("candidate_popularity_groups", []),
        "recommended_items": [],
        "global_uncertainty": None,
        "selection_rationale": "",
        "parse_success": False,
        "schema_valid": False,
        "invalid_item_count": 0,
        "duplicate_item_count": 0,
        "parse_error": type(last_exc).__name__,
        "raw_response": "",
        "prompt": prompt,
        "setting": setting,
        "response_latency": time.perf_counter() - started,
        "response_model_name": "",
        "response_provider": "",
        "response_usage": {},
        "retry_count": max_retries,
        "error_message": str(last_exc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--setting", choices=["plain", "evidence"], required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(cfg.get("seed"))

    exp_name = str(cfg.get("exp_name", "recommendation_list"))
    output_root = str(cfg.get("output_root", "output-repaired"))
    paths = ensure_exp_dirs(exp_name, output_root)
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples")
    samples = read_jsonl(cfg["input_path"], max_samples=max_samples)
    top_k = int(cfg.get("top_k", 10))
    prompt_key = "plain_prompt_path" if args.setting == "plain" else "evidence_prompt_path"
    template = Path(cfg[prompt_key]).read_text(encoding="utf-8")
    output_path = paths.predictions_dir / f"{args.setting}_list_raw.jsonl"

    if output_path.exists() and args.overwrite:
        output_path.unlink()

    existing = load_existing(output_path)
    finished = {int(row["sample_id"]) for row in existing if "sample_id" in row and bool(row.get("parse_success", False))}
    pending = [(idx, sample) for idx, sample in enumerate(samples) if idx not in finished]
    print(
        f"[{exp_name}:{args.setting}] total={len(samples)} finished={len(finished)} "
        f"pending={len(pending)} output={output_path}"
    )

    backend = build_backend_from_config(cfg["model_config"])
    rate_limiter = _RateLimiter(cfg.get("requests_per_minute"))
    max_workers = int(cfg.get("max_workers", 4))
    max_retries = int(cfg.get("max_retries", 3))
    retry_backoff_seconds = float(cfg.get("retry_backoff_seconds", 2.0))

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(
                    run_one,
                    sample_id=sample_id,
                    sample=sample,
                    setting=args.setting,
                    template=template,
                    top_k=top_k,
                    backend=backend,
                    rate_limiter=rate_limiter,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                ): sample_id
                for sample_id, sample in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {args.setting} list inference"):
                write_jsonl_record(output_path, future.result())

    rows = load_existing(output_path)
    write_sorted_records(output_path, rows)
    rows = load_existing(output_path)
    ok = sum(1 for row in rows if row.get("parse_success"))
    print(f"[{exp_name}:{args.setting}] complete rows={len(rows)} parse_success={ok}")


if __name__ == "__main__":
    main()
