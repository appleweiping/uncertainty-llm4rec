from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm import build_backend_from_config, load_model_config
from src.llm.base import normalize_generation_result
from src.llm.parser import (
    parse_candidate_ranking_response,
    parse_pairwise_preference_response,
    parse_pointwise_response,
)
from src.utils.io import ensure_dir


TASK_PROMPTS = {
    "pointwise_yesno": (
        "User history: item A, item B.\n"
        "Candidate item id: C.\n"
        "Answer in JSON with fields recommend, confidence, reason. "
        "recommend must be yes or no. Do not output chain-of-thought or <think> blocks; "
        "return only the final JSON object."
    ),
    "candidate_ranking": (
        "User history: item A, item B.\n"
        "Candidates: I1, I2, I3.\n"
        "Rank all candidates and answer in JSON with ranked_item_ids, confidence, reason. "
        "Do not output chain-of-thought or <think> blocks; return only the final JSON object."
    ),
    "pairwise_preference": (
        "User history: item A, item B.\n"
        "Item A id: I1. Item B id: I2.\n"
        "Choose the preferred item and answer in JSON with preferred_item, confidence, reason. "
        "Do not output chain-of-thought or <think> blocks; return only the final JSON object."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["pointwise_yesno", "candidate_ranking", "pairwise_preference"],
    )
    parser.add_argument("--status_path", type=str, default="outputs/summary/week7_day1_backend_check.csv")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def _parse_task_response(task: str, text: str) -> dict[str, Any]:
    if task == "pointwise_yesno":
        parsed = parse_pointwise_response(text)
        return {"parse_success": parsed.get("recommend") in {"yes", "no"}, "parse_mode": parsed.get("parse_mode", "")}
    if task == "candidate_ranking":
        parsed = parse_candidate_ranking_response(text, allowed_item_ids=["I1", "I2", "I3"], topk=3)
        return {"parse_success": bool(parsed.get("parse_success", False)), "parse_mode": parsed.get("parse_mode", "")}
    if task == "pairwise_preference":
        parsed = parse_pairwise_preference_response(text, item_a_id="I1", item_b_id="I2")
        return {"parse_success": bool(parsed.get("parse_success", False)), "parse_mode": parsed.get("parse_mode", "")}
    raise ValueError(f"Unsupported backend check task: {task}")


def main() -> None:
    args = parse_args()
    model_cfg = load_model_config(args.model_config)
    backend_name = str(model_cfg.get("backend_name", "")).strip()
    model_name = str(model_cfg.get("model_name") or model_cfg.get("model_name_or_path") or Path(args.model_config).stem)

    rows: list[dict[str, Any]] = []
    backend = None
    backend_error = ""
    load_success = False
    if not args.dry_run:
        try:
            backend = build_backend_from_config(args.model_config)
            load_success = True
        except Exception as exc:  # pragma: no cover - server environment dependent
            backend_error = f"{type(exc).__name__}: {exc}"

    for task in args.tasks:
        task = task.strip()
        prompt = TASK_PROMPTS.get(task)
        if prompt is None:
            raise ValueError(f"Unsupported backend check task: {task}")

        raw_text = ""
        latency = 0.0
        parse_success: bool | str = "not_run"
        parse_mode = "dry_run" if args.dry_run else "backend_load_failed"
        row_error = backend_error

        if args.dry_run:
            row_error = ""
        elif backend is not None:
            start = time.perf_counter()
            try:
                generation = normalize_generation_result(
                    backend.generate(prompt, max_new_tokens=args.max_new_tokens),
                    default_provider=getattr(backend, "provider", backend_name),
                    default_model_name=getattr(backend, "model_name", model_name),
                )
                latency = float(generation.get("latency") or (time.perf_counter() - start))
                raw_text = str(generation.get("raw_text", ""))
                parsed = _parse_task_response(task, raw_text)
                parse_success = bool(parsed["parse_success"])
                parse_mode = str(parsed.get("parse_mode", ""))
            except Exception as exc:  # pragma: no cover - server environment dependent
                row_error = f"{type(exc).__name__}: {exc}"
                load_success = False

        rows.append(
            {
                "task": task,
                "backend": backend_name,
                "model": model_name,
                "model_config": args.model_config,
                "load_success": load_success if not args.dry_run else "not_loaded_dry_run",
                "avg_latency": latency,
                "parse_success": parse_success,
                "parse_mode": parse_mode,
                "dry_run": bool(args.dry_run),
                "error_message": row_error,
                "raw_response_preview": raw_text[:160].replace("\n", " "),
            }
        )

    status_path = Path(args.status_path)
    ensure_dir(status_path.parent)
    pd.DataFrame(rows).to_csv(status_path, index=False)
    print(f"Saved backend check to: {status_path}")


if __name__ == "__main__":
    main()
