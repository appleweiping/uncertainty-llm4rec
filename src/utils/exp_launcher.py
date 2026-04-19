from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml
from src.utils.exp_registry import now_iso


TASK_SCRIPT_MAP = {
    "pointwise_yesno": "main_infer.py",
    "pointwise": "main_infer.py",
    "candidate_ranking": "main_rank.py",
    "ranking": "main_rank.py",
    "pairwise_preference": "main_pairwise.py",
    "pairwise": "main_pairwise.py",
    "candidate_ranking_rerank": "main_rank_rerank.py",
    "rank_rerank": "main_rank_rerank.py",
    "rerank": "main_rank_rerank.py",
}


def infer_domain(exp_name: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or {}
    if cfg.get("domain"):
        return str(cfg["domain"])
    data_config = str(cfg.get("data_config", "")).lower()
    for domain in ["movies", "beauty", "books", "electronics"]:
        if exp_name.startswith(domain) or domain in data_config:
            return domain
    return "unknown"


def infer_task(cfg: dict[str, Any], exp_name: str) -> str:
    task_type = str(cfg.get("task_type", "")).strip().lower()
    if task_type:
        return task_type
    if exp_name.endswith("_rank") or "_rank_" in exp_name:
        return "candidate_ranking"
    if "structured_risk" in exp_name or exp_name.endswith("_rerank") or "_rerank_" in exp_name:
        return "candidate_ranking_rerank"
    if exp_name.endswith("_pairwise") or "_pairwise_" in exp_name:
        return "pairwise_preference"
    if exp_name.endswith("_pointwise") or "_pointwise_" in exp_name:
        return "pointwise_yesno"
    return "unknown"


def script_for_task(task: str) -> str:
    normalized = str(task).strip().lower()
    if normalized not in TASK_SCRIPT_MAP:
        raise ValueError(f"Unsupported task for batch launcher: {task}")
    return TASK_SCRIPT_MAP[normalized]


def expected_eval_ready(output_dir: Path, task: str) -> bool:
    task = str(task).strip().lower()
    if task == "candidate_ranking":
        return (output_dir / "tables" / "ranking_metrics.csv").exists()
    if task in {"candidate_ranking_rerank", "rank_rerank", "rerank"}:
        return (output_dir / "tables" / "rerank_results.csv").exists()
    if task == "pairwise_preference":
        return (output_dir / "tables" / "pairwise_metrics.csv").exists()
    if task in {"pointwise_yesno", "pointwise"}:
        return (output_dir / "tables" / "diagnostic_metrics.csv").exists()
    return False


def expected_prediction_ready(output_dir: Path, task: str) -> bool:
    task = str(task).strip().lower()
    if task == "candidate_ranking":
        return (output_dir / "predictions" / "rank_predictions.jsonl").exists()
    if task in {"candidate_ranking_rerank", "rank_rerank", "rerank"}:
        return (output_dir / "reranked" / "rank_reranked.jsonl").exists()
    if task == "pairwise_preference":
        return (output_dir / "predictions" / "pairwise_predictions.jsonl").exists()
    if task in {"pointwise_yesno", "pointwise"}:
        prediction_dir = output_dir / "predictions"
        return any(prediction_dir.glob("*_raw.jsonl")) if prediction_dir.exists() else False
    return False


def registry_status_for_existing_artifacts(*, eval_ready: bool, prediction_ready: bool) -> tuple[str, str]:
    if eval_ready and prediction_ready:
        return (
            "artifact_ready",
            "artifacts already exist; registry refreshed without re-running the command",
        )
    if prediction_ready:
        return (
            "prediction_ready",
            "prediction artifacts already exist; evaluation artifacts are still incomplete",
        )
    return (
        "dry_run_ready",
        "command built and input exists; use --run on server to execute",
    )


def build_experiment_row(
    *,
    spec: dict[str, Any],
    batch_name: str,
    python_bin: str,
) -> dict[str, Any]:
    config_path = Path(str(spec["config_path"]))
    cfg = load_yaml(config_path)
    exp_name = str(spec.get("exp_name") or cfg.get("exp_name") or config_path.stem)
    task = str(spec.get("task") or infer_task(cfg, exp_name))
    output_dir = Path(str(spec.get("output_dir") or cfg.get("output_dir") or Path(cfg.get("output_root", "outputs")) / exp_name))
    input_path = Path(str(spec.get("input_path") or cfg.get("input_path", "")))
    model_config = str(spec.get("model_config") or cfg.get("model_config", ""))
    command = [python_bin, script_for_task(task), "--config", str(config_path)]
    return {
        "batch_name": batch_name,
        "exp_name": exp_name,
        "domain": str(spec.get("domain") or infer_domain(exp_name, cfg)),
        "task": task,
        "model": str(spec.get("model") or Path(model_config).stem or "unknown"),
        "method_family": str(spec.get("method_family") or cfg.get("method_family") or "local_hf_inference"),
        "method_variant": str(spec.get("method_variant") or cfg.get("method_variant") or "base_only"),
        "is_current_best_family": bool(spec.get("is_current_best_family") or cfg.get("is_current_best_family") or False),
        "config_path": str(config_path),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "eval_ready": expected_eval_ready(output_dir, task),
        "prediction_ready": expected_prediction_ready(output_dir, task),
        "command": " ".join(command),
        "_command_list": command,
        "_input_exists": input_path.exists() if str(input_path) else False,
    }


def launch_experiment(
    *,
    row: dict[str, Any],
    log_dir: str | Path,
    dry_run: bool,
    retry_count: int = 0,
) -> dict[str, Any]:
    result = dict(row)
    command = list(result.pop("_command_list"))
    input_exists = bool(result.pop("_input_exists"))
    exp_name = str(result["exp_name"])
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{exp_name}.stdout.log"
    stderr_path = log_dir / f"{exp_name}.stderr.log"

    result.update(
        {
            "retry_count": retry_count,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "dry_run": dry_run,
            "started_at": now_iso(),
            "finished_at": "",
            "latency_sec": 0.0,
            "return_code": "",
            "error_message": "",
        }
    )

    if not input_exists:
        result.update(
            {
                "status": "input_missing",
                "finished_at": now_iso(),
                "error_message": f"Input file not found: {result.get('input_path')}",
                "notes": "registered but not runnable until input exists",
            }
        )
        return result

    if dry_run:
        status, notes = registry_status_for_existing_artifacts(
            eval_ready=bool(result.get("eval_ready")),
            prediction_ready=bool(result.get("prediction_ready")),
        )
        result.update(
            {
                "status": status,
                "finished_at": now_iso(),
                "notes": notes,
            }
        )
        return result

    started = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            completed = subprocess.run(
                command,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                check=False,
            )
        latency = time.perf_counter() - started
        result.update(
            {
                "status": "success" if completed.returncode == 0 else "failed",
                "finished_at": now_iso(),
                "latency_sec": round(latency, 3),
                "return_code": completed.returncode,
                "eval_ready": expected_eval_ready(Path(str(result["output_dir"])), str(result["task"])),
                "notes": "completed by batch launcher" if completed.returncode == 0 else "command returned non-zero exit code",
            }
        )
        if completed.returncode != 0:
            result["error_message"] = f"Return code {completed.returncode}; see {stderr_path}"
        return result
    except Exception as exc:
        latency = time.perf_counter() - started
        result.update(
            {
                "status": "launch_failed",
                "finished_at": now_iso(),
                "latency_sec": round(latency, 3),
                "return_code": "",
                "error_message": repr(exc),
                "notes": "launcher exception before command completed",
            }
        )
        return result
