"""Day30 API health check and smoke recovery.

This script performs a single minimal backend request using the same DeepSeek
model config as Day29/Day30, writes a health report without exposing the API
key, and optionally retries a tiny robustness smoke after health succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.llm import build_backend_from_config


MODEL_CONFIG = Path("configs/model/deepseek.yaml")
SUMMARY_DIR = Path("output-repaired/summary")
HEALTH_REPORT = SUMMARY_DIR / "day30_api_health_check.md"
SMOKE_STATUS = SUMMARY_DIR / "day30_smoke_retry_status.csv"
MONITOR = SUMMARY_DIR / "day30_cep_robustness_runtime_monitor.md"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def truncate(text: str, limit: int = 500) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def process_status() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Process | Where-Object { $_.ProcessName -match 'python|py' } | "
                "Select-Object Id,ProcessName,CPU,StartTime,Path | ConvertTo-Json -Depth 3",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception:
        return []


def run_health_check() -> dict[str, Any]:
    cfg = load_yaml(MODEL_CONFIG)
    connection = cfg.get("connection", {}) or {}
    backend_name = str(cfg.get("backend_name", ""))
    provider = str(cfg.get("provider", backend_name))
    base_url = connection.get("base_url") or cfg.get("base_url")
    api_key_env = str(connection.get("api_key_env") or cfg.get("api_key_env") or "")
    api_key_present = bool(os.getenv(api_key_env)) if api_key_env else False
    current_time = datetime.now().isoformat(timespec="seconds")
    row = {
        "model_config_path": str(MODEL_CONFIG),
        "backend": backend_name,
        "provider": provider,
        "base_url_read": bool(base_url),
        "base_url": str(base_url or ""),
        "api_key_env": api_key_env,
        "api_key_env_present": api_key_present,
        "request_success": False,
        "error_type": "",
        "error_message_truncated": "",
        "current_time": current_time,
        "recommend_retry": False,
        "latency_seconds": None,
        "day29_python_processes": process_status(),
    }
    if not api_key_present:
        row["error_type"] = "MissingAPIKey"
        row["error_message_truncated"] = f"Environment variable is not set: {api_key_env}"
        return row
    prompt = (
        "Return strictly this JSON and nothing else: "
        "{\"relevance_probability\":0.5,\"positive_evidence\":0.1,"
        "\"negative_evidence\":0.1,\"ambiguity\":0.5,\"missing_information\":0.5,"
        "\"recommend\":\"no\",\"reason\":\"health check\"}"
    )
    start = time.perf_counter()
    try:
        backend = build_backend_from_config(MODEL_CONFIG)
        result = backend.generate(prompt, max_tokens=80, temperature=0.0, timeout=60)
        row["latency_seconds"] = round(time.perf_counter() - start, 4)
        row["request_success"] = bool(str(result.get("raw_text", "")).strip())
        if not row["request_success"]:
            row["error_type"] = "EmptyResponse"
            row["error_message_truncated"] = "Health check returned an empty raw_text."
        row["recommend_retry"] = row["request_success"]
    except Exception as exc:
        row["latency_seconds"] = round(time.perf_counter() - start, 4)
        row["error_type"] = type(exc).__name__
        row["error_message_truncated"] = truncate(str(exc))
        row["recommend_retry"] = False
    return row


def write_health_report(row: dict[str, Any]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    processes = row.get("day29_python_processes") or []
    if processes:
        process_lines = "\n".join(
            f"- PID {p.get('Id')}: {p.get('ProcessName')} CPU={p.get('CPU')} Start={p.get('StartTime')}"
            for p in processes
        )
        process_note = "Python/py processes were found:\n" + process_lines
    else:
        process_note = "No running python/py process was found. Day29 is not occupying API from this workspace."

    report = f"""# Day30 API Health Check

- model_config_path: `{row['model_config_path']}`
- backend/provider: `{row['backend']}` / `{row['provider']}`
- base_url read: `{row['base_url_read']}`
- base_url: `{row['base_url']}`
- api_key_env: `{row['api_key_env']}`
- api_key_env present: `{row['api_key_env_present']}` (key value is intentionally not printed)
- one-row request success: `{row['request_success']}`
- error_type: `{row['error_type']}`
- error_message_truncated: `{row['error_message_truncated']}`
- latency_seconds: `{row['latency_seconds']}`
- current_time: `{row['current_time']}`
- recommend_retry: `{row['recommend_retry']}`

## Day29 / Background API Process Check

{process_note}

Note: this process snapshot is captured while the health-check script itself is running, so short-lived `py`/`python` entries can include the health check. Re-check after the command finishes before treating any PID as a persistent Day29 task.

## Interpretation

If `one-row request success` is false, do not start Day30 full noisy inference. Run the health check again later, then resume with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\\run_day30_cep_robustness_pipeline.ps1
```
"""
    HEALTH_REPORT.write_text(report, encoding="utf-8")


def update_monitor_blocked(row: dict[str, Any]) -> None:
    lines = [
        "# Day30 CEP Robustness Runtime Monitor",
        "",
        f"Last update: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Status: `blocked_api_connection`",
        "",
        "- Day30 local data/noise/config/orchestrator is complete.",
        "- First smoke failed because of APIConnectionError / API health failure.",
        "- No full noisy inference was launched.",
        "- Next action is health check + smoke retry.",
        "",
        "| noise_type | noise_level | expected_rows | current_rows | rolling_parse_success | status | resume_command |",
        "|---|---:|---:|---:|---:|---|---|",
        "| history_dropout | 0.1 | 3000 | 0 | 0.0000 | blocked_api_connection | `powershell -ExecutionPolicy Bypass -File scripts\\run_day30_cep_robustness_pipeline.ps1` |",
    ]
    MONITOR.write_text("\n".join(lines), encoding="utf-8")


def smoke_retry(attempt_id: str, rows: int = 10) -> dict[str, Any]:
    noise_type = "history_dropout"
    noise_level = "0.1"
    source = Path("data/processed/amazon_beauty_robustness_500/noisy_history_dropout_0.1/test.jsonl")
    retry_input = Path(f"data/processed/amazon_beauty_robustness_500/noisy_history_dropout_0.1/smoke_retry_{attempt_id}_{rows}.jsonl")
    retry_output = Path(f"output-repaired/beauty_robustness_500/history_dropout_0.1/predictions/smoke_retry_{attempt_id}_raw.jsonl")
    if not retry_input.exists():
        lines = source.read_text(encoding="utf-8").splitlines()[:rows]
        retry_input.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "main_infer.py",
        "--config",
        "configs/exp/beauty_robustness_500_history_dropout_0.1.yaml",
        "--input_path",
        str(retry_input),
        "--output_path",
        str(retry_output),
        "--split_name",
        "smoke",
        "--concurrent",
        "--resume",
        "--max_workers",
        "1",
        "--requests_per_minute",
        "30",
    ]
    subprocess.run(cmd, check=False)
    total = 0
    api_errors = 0
    parse_ok = 0
    raw_nonempty = 0
    field_ok = 0
    if retry_output.exists():
        with retry_output.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                rec = json.loads(line)
                if rec.get("error_type") == "APIConnectionError" or rec.get("parse_error") == "APIConnectionError":
                    api_errors += 1
                if rec.get("parse_success") is True:
                    parse_ok += 1
                if str(rec.get("raw_response") or "").strip():
                    raw_nonempty += 1
                if all(
                    rec.get(k) is not None
                    for k in [
                        "relevance_probability",
                        "positive_evidence",
                        "negative_evidence",
                        "ambiguity",
                        "missing_information",
                    ]
                ):
                    field_ok += 1
    status = "success" if total and parse_ok / total >= 0.95 and api_errors == 0 and raw_nonempty / total >= 0.95 else "failed"
    row = {
        "noise_type": noise_type,
        "noise_level": noise_level,
        "attempt_id": attempt_id,
        "rows": total,
        "max_workers": 1,
        "requests_per_minute": 30,
        "api_error_rate": api_errors / total if total else 1.0,
        "parse_success_rate": parse_ok / total if total else 0.0,
        "raw_response_nonempty_rate": raw_nonempty / total if total else 0.0,
        "field_complete_rate": field_ok / total if total else 0.0,
        "status": status,
        "notes": f"output={retry_output}",
    }
    existing = pd.read_csv(SMOKE_STATUS) if SMOKE_STATUS.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(SMOKE_STATUS, index=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-smoke-if-healthy", action="store_true")
    parser.add_argument("--attempt-id", default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()
    health = run_health_check()
    write_health_report(health)
    update_monitor_blocked(health)
    print(json.dumps({k: v for k, v in health.items() if k != "day29_python_processes"}, indent=2))
    if args.retry_smoke_if_healthy:
        if health["request_success"]:
            retry = smoke_retry(args.attempt_id, rows=args.rows)
            print(json.dumps(retry, indent=2))
        else:
            print("Health check failed; smoke retry skipped.")


if __name__ == "__main__":
    import argparse

    main()
