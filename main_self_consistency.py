from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.llm import build_backend_from_config
from src.llm.prompt_builder import PromptBuilder
from src.llm.self_consistency import run_self_consistency
from src.uncertainty.consistency_confidence import compute_consistency_summary
from src.utils.io import save_jsonl
from src.utils.paths import ensure_exp_dirs


class RateLimiter:
    def __init__(self, requests_per_minute: int | None = None):
        self.requests_per_minute = requests_per_minute
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.requests_per_minute or self.requests_per_minute <= 0:
            return
        with self._lock:
            min_interval = 60.0 / float(self.requests_per_minute)
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.monotonic()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl_records(path: str | Path, max_samples: int | None = None) -> list[dict]:
    df = pd.read_json(path, lines=True)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_json(path, orient="records", lines=True, force_ascii=False, mode="a")


def load_existing_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return pd.read_json(path, lines=True).to_dict(orient="records")


def read_finished_sample_ids(path: str | Path) -> set[int]:
    finished: set[int] = set()
    for row in load_existing_rows(path):
        try:
            finished.add(int(row.get("sample_id")))
        except Exception:
            continue
    return finished


def write_sorted_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["sample_id"], keep="last")
    df = df.sort_values("sample_id").reset_index(drop=True)
    save_jsonl(df.to_dict(orient="records"), path)


def _normalize_candidate_from_pointwise_sample(sample: dict[str, Any]) -> dict[str, Any]:
    candidate_item_id = (
        sample.get("candidate_item_id")
        or sample.get("item_id")
        or sample.get("target_item_id")
        or ""
    )

    candidate_title = sample.get("candidate_title") or sample.get("title") or ""
    candidate_meta = (
        sample.get("candidate_meta")
        or sample.get("candidate_description")
        or sample.get("candidate_text")
        or sample.get("text")
        or ""
    )

    return {
        "item_id": candidate_item_id,
        "title": str(candidate_title).strip(),
        "meta": str(candidate_meta).strip(),
    }


def resolve_split_input_path(cfg: dict[str, Any], split_name: str | None) -> str | None:
    if not split_name:
        return None
    split_input_paths = cfg.get("split_input_paths") or {}
    if not isinstance(split_input_paths, dict):
        return None
    return split_input_paths.get(str(split_name).strip().lower())


def build_self_consistency_row(
    *,
    sample_id: int,
    sample: dict[str, Any],
    prompt_builder: PromptBuilder,
    llm_backend,
    num_samples: int,
    sleep_time: float,
    output_schema: str | None,
    generation_kwargs: dict[str, Any],
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    candidate = _normalize_candidate_from_pointwise_sample(sample)
    prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
    runs = run_self_consistency(
        llm_backend=llm_backend,
        prompt=prompt,
        num_samples=num_samples,
        sleep_time=sleep_time,
        generation_kwargs=generation_kwargs,
        output_schema=output_schema,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        before_generate=rate_limiter.wait if rate_limiter is not None else None,
    )
    summary = compute_consistency_summary(runs)

    row = {
        "sample_id": int(sample_id),
        "user_id": sample.get("user_id", ""),
        "target_item_id": sample.get("target_item_id", candidate.get("item_id", "")),
        "candidate_item_id": candidate.get("item_id", ""),
        "label": int(sample.get("label", 0)),
        "target_popularity_group": sample.get("target_popularity_group", "unknown"),
        "prompt": prompt,
        "consistency_runs": runs,
        "parse_success_rate": float(sum(1 for run in runs if run.get("parse_success")) / len(runs)) if runs else 0.0,
    }
    row.update(summary)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    parser.add_argument("--input_path", type=str, default=None, help="Optional explicit input jsonl path.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional explicit output path.")
    parser.add_argument("--split_name", type=str, default="test", help="Split name, e.g. test.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of repeated generations per sample.")
    parser.add_argument("--sleep_time", type=float, default=0.0, help="Optional sleep between repeated generations.")
    parser.add_argument("--output_schema", type=str, default=None, help="Optional parser schema, e.g. evidence_confidence.")
    parser.add_argument("--concurrent", action="store_true", help="Run samples concurrently with checkpoint/resume.")
    parser.add_argument("--max_workers", type=int, default=None, help="Concurrent worker count.")
    parser.add_argument("--requests_per_minute", type=int, default=None, help="Optional request rate limit.")
    parser.add_argument("--max_retries", type=int, default=None, help="Retries per generation.")
    parser.add_argument("--retry_backoff_seconds", type=float, default=None, help="Initial retry backoff.")
    parser.add_argument("--timeout_seconds", type=float, default=None, help="Optional API request timeout.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing self-consistency JSONL.")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume even if config enables it.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_name = str(cfg.get("exp_name", "")).strip()
    if not exp_name:
        raise ValueError("exp_name is required in experiment config.")

    model_config = cfg.get("model_config")
    prompt_path = cfg.get("prompt_path", "prompts/pointwise_yesno.txt")
    output_root = cfg.get("output_root", "outputs")
    split_name = str(args.split_name).strip().lower()
    input_path = Path(args.input_path or resolve_split_input_path(cfg, split_name) or cfg.get("input_path"))
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples")
    overwrite = bool(args.overwrite or cfg.get("overwrite", False))
    output_schema = args.output_schema if args.output_schema is not None else cfg.get("output_schema")
    concurrent = bool(args.concurrent or cfg.get("concurrent", cfg.get("enable_concurrent", False)))
    max_workers = int(args.max_workers if args.max_workers is not None else cfg.get("max_workers", 4))
    requests_per_minute = args.requests_per_minute if args.requests_per_minute is not None else cfg.get("requests_per_minute")
    max_retries = int(args.max_retries if args.max_retries is not None else cfg.get("max_retries", 0))
    retry_backoff_seconds = float(
        args.retry_backoff_seconds if args.retry_backoff_seconds is not None else cfg.get("retry_backoff_seconds", 2.0)
    )
    timeout_seconds = args.timeout_seconds if args.timeout_seconds is not None else cfg.get("timeout_seconds")
    resume = bool((args.resume or cfg.get("resume", True)) and not args.no_resume)

    paths = ensure_exp_dirs(exp_name, output_root)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else paths.root / "self_consistency" / f"{split_name}_self_consistency.jsonl"
    )

    if output_path.exists() and overwrite and not resume:
        output_path.unlink()

    if output_path.exists() and not overwrite and not (concurrent and resume):
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config or --overwrite to rerun.")
        return

    samples = load_jsonl_records(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples for self-consistency from: {input_path}")

    llm_backend = build_backend_from_config(model_config)
    prompt_builder = PromptBuilder(template_path=str(prompt_path))
    generation_kwargs: dict[str, Any] = {}
    if timeout_seconds is not None:
        generation_kwargs["timeout"] = float(timeout_seconds)
        generation_kwargs["max_retries"] = 0

    rows: list[dict[str, Any]] = load_existing_rows(output_path) if concurrent and resume else []
    finished_ids = read_finished_sample_ids(output_path) if concurrent and resume else set()
    pending = [(idx, sample) for idx, sample in enumerate(samples) if idx not in finished_ids]
    print(
        f"[{exp_name}] Self-consistency mode: concurrent={concurrent} "
        f"finished={len(finished_ids)} pending={len(pending)} num_trials={args.num_samples}"
    )

    if concurrent:
        rate_limiter = RateLimiter(int(requests_per_minute)) if requests_per_minute else None
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [
                executor.submit(
                    build_self_consistency_row,
                    sample_id=idx,
                    sample=sample,
                    prompt_builder=prompt_builder,
                    llm_backend=llm_backend,
                    num_samples=args.num_samples,
                    sleep_time=args.sleep_time,
                    output_schema=output_schema,
                    generation_kwargs=generation_kwargs,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    rate_limiter=rate_limiter,
                )
                for idx, sample in pending
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running self-consistency"):
                row = future.result()
                append_jsonl_record(output_path, row)
                rows.append(row)
    else:
        rows = []
        for idx, sample in tqdm(list(enumerate(samples)), desc="Running self-consistency"):
            row = build_self_consistency_row(
                sample_id=idx,
                sample=sample,
                prompt_builder=prompt_builder,
                llm_backend=llm_backend,
                num_samples=args.num_samples,
                sleep_time=args.sleep_time,
                output_schema=output_schema,
                generation_kwargs=generation_kwargs,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            rows.append(row)

    write_sorted_rows(output_path, rows)
    print(f"[{exp_name}] Saved {len(rows)} self-consistency rows to: {output_path}")


if __name__ == "__main__":
    main()
