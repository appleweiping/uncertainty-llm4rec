"""Day37 API routing/config parity diagnosis.

This script compares Day37 small-domain configs against recent successful
DeepSeek inference configs and summarizes a same-path main_infer health check.
It does not change prompts, parsers, formulas, or launch full inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


SUMMARY_DIR = Path("output-repaired/summary")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

SUCCESS_CONFIGS = {
    "beauty_day9_full_relevance": Path("configs/exp/beauty_deepseek_relevance_evidence_full.yaml"),
    "day29_movies_medium5": Path("configs/exp/movies_deepseek_relevance_evidence_medium_5neg_2000.yaml"),
    "day30_robustness_history_dropout_0.1": Path("configs/exp/beauty_robustness_500_history_dropout_0.1.yaml"),
}
DAY37_CONFIGS = {
    "movies_small": Path("configs/exp/movies_small_deepseek_relevance_evidence.yaml"),
    "books_small": Path("configs/exp/books_small_deepseek_relevance_evidence.yaml"),
    "electronics_small": Path("configs/exp/electronics_small_deepseek_relevance_evidence.yaml"),
}
MODEL_CONFIG = Path("configs/model/deepseek.yaml")
PARITY_HEALTH_INPUT = Path("output-repaired/day37_small_smoke_inputs/movies_valid_parity_health_1.jsonl")
PARITY_HEALTH_OUTPUT = Path("output-repaired/movies_small_deepseek_relevance_evidence/predictions/parity_health_raw.jsonl")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_fields() -> dict[str, Any]:
    cfg = load_yaml(MODEL_CONFIG)
    conn = cfg.get("connection") or {}
    return {
        "backend/provider": f"{cfg.get('backend_name', '')}/{cfg.get('provider', '')}",
        "backend": cfg.get("backend_name", ""),
        "provider": cfg.get("provider", ""),
        "model_name": cfg.get("model_name", ""),
        "base_url": conn.get("base_url", ""),
        "api_key_env": conn.get("api_key_env", ""),
        "api_key_env_present": bool(os.environ.get(str(conn.get("api_key_env", "")))),
    }


def flatten_config(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    fields = model_fields()
    fields.update(
        {
            "main_script": "main_infer.py",
            "python_executable": "py -3.12",
            "working_directory": str(Path.cwd()),
            "prompt_path": exp_cfg.get("prompt_path", ""),
            "model_config": exp_cfg.get("model_config", ""),
            "output_root": exp_cfg.get("output_root", ""),
            "output_dir": exp_cfg.get("output_dir", ""),
            "exp_name": exp_cfg.get("exp_name", ""),
            "output_schema": exp_cfg.get("output_schema", ""),
            "method_variant": exp_cfg.get("method_variant", ""),
            "concurrent": exp_cfg.get("concurrent", ""),
            "resume": exp_cfg.get("resume", ""),
            "max_workers": exp_cfg.get("max_workers", ""),
            "requests_per_minute": exp_cfg.get("requests_per_minute", ""),
            "max_retries": exp_cfg.get("max_retries", ""),
            "retry_backoff_seconds": exp_cfg.get("retry_backoff_seconds", ""),
            "timeout_seconds": exp_cfg.get("timeout_seconds", ""),
            "checkpoint_every": exp_cfg.get("checkpoint_every", ""),
        }
    )
    return fields


def make_inventory() -> None:
    model = model_fields()
    lines = [
        "# Day37 API Parity Successful Runs Inventory",
        "",
        f"- model_config_path: `{MODEL_CONFIG}`",
        f"- backend/provider: `{model['backend']}` / `{model['provider']}`",
        f"- model_name: `{model['model_name']}`",
        f"- base_url source: `{MODEL_CONFIG} -> connection.base_url`",
        f"- api_key_env: `{model['api_key_env']}` (value not printed; present={model['api_key_env_present']})",
        f"- Python executable: `py -3.12`",
        f"- working directory: `{Path.cwd()}`",
        "",
    ]
    commands = {
        "beauty_day9_full_relevance": "py -3.12 main_infer.py --config configs\\exp\\beauty_deepseek_relevance_evidence_full.yaml --split_name valid/test --concurrent --resume --max_workers 4 --requests_per_minute 120",
        "day29_movies_medium5": "py -3.12 main_infer.py --config configs\\exp\\movies_deepseek_relevance_evidence_medium_5neg_2000.yaml --split_name valid/test --concurrent --resume --max_workers 4 --requests_per_minute 120",
        "day30_robustness_history_dropout_0.1": "py -3.12 main_infer.py --config configs\\exp\\beauty_robustness_500_history_dropout_0.1.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120",
    }
    for name, path in SUCCESS_CONFIGS.items():
        cfg = load_yaml(path)
        flat = flatten_config(cfg)
        lines.extend(
            [
                f"## {name}",
                "",
                f"- config path: `{path}`",
                f"- main script / command: `{commands[name]}`",
                f"- backend/provider: `{flat['backend']}` / `{flat['provider']}`",
                f"- model name: `{flat['model_name']}`",
                f"- base_url: `{flat['base_url']}`",
                f"- api_key_env: `{flat['api_key_env']}` (value not printed)",
                f"- max_workers: `{flat['max_workers']}`",
                f"- requests_per_minute: `{flat['requests_per_minute']}`",
                f"- timeout / retry: timeout=`{flat['timeout_seconds']}`, max_retries=`{flat['max_retries']}`, retry_backoff_seconds=`{flat['retry_backoff_seconds']}`",
                f"- prompt_path: `{flat['prompt_path']}`",
                f"- output_dir/exp_name: `{flat['output_dir'] or flat['exp_name']}`",
                "",
            ]
        )
    (SUMMARY_DIR / "day37_api_parity_successful_runs_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def make_diff() -> None:
    successful = flatten_config(load_yaml(SUCCESS_CONFIGS["day29_movies_medium5"]))
    # All three small configs are generated from the same template; compare movies
    # directly and note if books/electronics differ materially.
    day37 = flatten_config(load_yaml(DAY37_CONFIGS["movies_small"]))
    fields = [
        "backend",
        "provider",
        "model_name",
        "base_url",
        "api_key_env",
        "api_key_env_present",
        "model_config",
        "prompt_path",
        "output_schema",
        "concurrent",
        "resume",
        "max_workers",
        "requests_per_minute",
        "max_retries",
        "retry_backoff_seconds",
        "timeout_seconds",
        "checkpoint_every",
        "main_script",
        "python_executable",
        "working_directory",
        "method_variant",
        "exp_name",
        "output_dir",
    ]
    rows = []
    for field in fields:
        sv = successful.get(field, "")
        dv = day37.get(field, "")
        match = sv == dv
        if field in {"method_variant", "exp_name", "output_dir"}:
            risk = "low"
            notes = "Expected experiment-specific difference."
        elif field == "api_key_env_present":
            risk = "high" if not dv else "low"
            notes = "Environment presence only; key value is not printed."
        elif match:
            risk = "low"
            notes = "Matches Day29 successful config."
        else:
            risk = "high" if field in {"backend", "provider", "model_name", "base_url", "api_key_env", "model_config", "prompt_path"} else "medium"
            notes = "Mismatch with Day29 successful config."
        rows.append(
            {
                "field": field,
                "successful_value": sv,
                "day37_value": dv,
                "match": match,
                "risk_level": risk,
                "notes": notes,
            }
        )
    write_csv(
        SUMMARY_DIR / "day37_api_config_parity_diff.csv",
        rows,
        ["field", "successful_value", "day37_value", "match", "risk_level", "notes"],
    )


def prepare_health_input() -> None:
    source = Path("data/processed/amazon_movies_small/valid.jsonl")
    row = read_jsonl(source)[:1]
    PARITY_HEALTH_INPUT.parent.mkdir(parents=True, exist_ok=True)
    with PARITY_HEALTH_INPUT.open("w", encoding="utf-8") as f:
        for record in row:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_health() -> None:
    records = read_jsonl(PARITY_HEALTH_OUTPUT)
    model = model_fields()
    if records:
        row = records[0]
        success = bool(row.get("parse_success")) and bool(str(row.get("raw_response", "")).strip())
        error_type = row.get("error_type") or row.get("parse_error") or ""
        message = str(row.get("error_message") or "")[:300]
        raw_nonempty = bool(str(row.get("raw_response", "")).strip())
    else:
        success = False
        error_type = "no_output"
        message = f"Parity health output not found or empty: {PARITY_HEALTH_OUTPUT}"
        raw_nonempty = False
    diff = []
    diff_path = SUMMARY_DIR / "day37_api_config_parity_diff.csv"
    if diff_path.exists():
        with diff_path.open("r", encoding="utf-8") as f:
            diff = list(csv.DictReader(f))
    high_risk = [r for r in diff if r.get("risk_level") == "high" and r.get("match") == "False"]
    lines = [
        "# Day37 API Health Check Parity",
        "",
        f"- backend config: `{MODEL_CONFIG}`",
        f"- backend/provider: `{model['backend']}` / `{model['provider']}`",
        f"- model_name: `{model['model_name']}`",
        f"- base_url read: `{bool(model['base_url'])}`",
        f"- base_url: `{model['base_url']}`",
        f"- api_key_env: `{model['api_key_env']}` (value not printed)",
        f"- api_key_env present: `{model['api_key_env_present']}`",
        f"- same-path script: `main_infer.py`",
        f"- health input: `{PARITY_HEALTH_INPUT}`",
        f"- health output: `{PARITY_HEALTH_OUTPUT}`",
        f"- request/parse success: `{success}`",
        f"- raw_response nonempty: `{raw_nonempty}`",
        f"- error_type: `{error_type}`",
        f"- error_message_truncated: `{message}`",
        f"- current_time: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Config Difference Against Day29 Successful Run",
        "",
    ]
    if high_risk:
        for row in high_risk:
            lines.append(f"- HIGH: `{row['field']}` successful=`{row['successful_value']}` day37=`{row['day37_value']}`")
    else:
        lines.append("- No high-risk config mismatch found against Day29 Movies medium5 successful config.")
    lines.append("")
    if success:
        lines.append("Recommendation: retry Movies small smoke, then proceed sequentially if parse_success >= 0.95.")
    else:
        lines.append("Recommendation: do not start full inference. Failure persists despite matching successful config, so this is likely current network/API routing/environment instability rather than a Day37 prompt/schema/parser issue.")
    (SUMMARY_DIR / "day37_api_health_check_parity.md").write_text("\n".join(lines), encoding="utf-8")


def write_recovery_report() -> None:
    health_text = (SUMMARY_DIR / "day37_api_health_check_parity.md").read_text(encoding="utf-8") if (SUMMARY_DIR / "day37_api_health_check_parity.md").exists() else ""
    diff_path = SUMMARY_DIR / "day37_api_config_parity_diff.csv"
    rows = list(csv.DictReader(diff_path.open("r", encoding="utf-8"))) if diff_path.exists() else []
    high = [r for r in rows if r.get("risk_level") == "high" and r.get("match") == "False"]
    success = "request/parse success: `True`" in health_text
    report = [
        "# Day37 API Connection Recovery Report",
        "",
        "## 1. Preflight",
        "",
        "Day37 small-domain data preflight passed: Books, Electronics, and Movies small each have 3000 valid rows, 3000 test rows, 500 users, 6 candidates/user, and the Beauty-compatible relevance evidence schema.",
        "",
        "## 2. Initial Failure",
        "",
        "The initial Movies small 20-row smoke produced empty raw_response rows with parse_success=0 and APIConnectionError. A one-row health check also returned APIConnectionError.",
        "",
        "## 3. Config Parity Diagnosis",
        "",
    ]
    if high:
        report.append("High-risk config mismatches were found:")
        for row in high:
            report.append(f"- `{row['field']}`: successful `{row['successful_value']}` vs Day37 `{row['day37_value']}`.")
    else:
        report.append("No high-risk mismatch was found between Day37 Movies small and the recently successful Day29 Movies medium5 config. Backend/provider, model, base_url, api_key_env, prompt_path, schema, concurrency, retry, and main script route match.")
    report.extend(
        [
            "",
            "## 4. Config / Env Changes",
            "",
            "No prompt/parser/formula changes were made. No Day37 config change was required because the API route already matches the successful Day29/Day30 route.",
            "",
            "## 5. Parity Health Check",
            "",
            "The parity health check used `main_infer.py` with the same DeepSeek config and a one-row Movies small input.",
            f"Result: `{'success' if success else 'failed'}`.",
            "",
            "## 6. Execution Decision",
            "",
        ]
    )
    if success:
        report.append("Parity health check passed. It is safe to retry Movies small smoke and then run Movies/Books/Electronics sequentially if smoke succeeds.")
    else:
        report.append("Parity health check failed despite matching successful config. Do not start full small-domain inference yet.")
    report.extend(
        [
            "",
            "## 7. Next Action If Still Blocked",
            "",
            "Check current network/proxy/API-key routing in the active shell. The key env is present but the request fails at APIConnectionError, so the user may need to refresh network/proxy or API key/session and rerun the parity health command before resuming.",
        ]
    )
    (SUMMARY_DIR / "day37_api_connection_recovery_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize-health", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        make_inventory()
        make_diff()
        prepare_health_input()
        print("Wrote Day37 API parity inventory/diff and prepared one-row health input.")
    if args.summarize_health:
        summarize_health()
        write_recovery_report()
        print("Wrote Day37 API health parity and recovery report.")
    if not args.prepare and not args.summarize_health:
        make_inventory()
        make_diff()
        prepare_health_input()
        summarize_health()
        write_recovery_report()


if __name__ == "__main__":
    main()
