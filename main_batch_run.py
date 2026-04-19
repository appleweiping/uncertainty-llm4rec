from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml
from src.utils.exp_registry import write_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", required=True)
    parser.add_argument("--status_path", type=str, default="outputs/summary/week6_magic7_batch_status.csv")
    parser.add_argument("--status_only", action="store_true")
    return parser.parse_args()


def infer_domain(exp_name: str) -> str:
    for domain in ["movies", "beauty", "books", "electronics"]:
        if exp_name.startswith(domain):
            return domain
    return "unknown"


def infer_task(cfg: dict[str, Any], exp_name: str) -> str:
    task_type = str(cfg.get("task_type", "")).strip()
    if task_type:
        return task_type
    if exp_name.endswith("_rank"):
        return "candidate_ranking"
    if exp_name.endswith("_pointwise"):
        return "pointwise_yesno"
    return "unknown"


def expected_eval_ready(output_dir: Path, task: str) -> bool:
    if task == "candidate_ranking":
        return (output_dir / "tables" / "ranking_metrics.csv").exists()
    if task == "pointwise_yesno":
        return (output_dir / "tables" / "diagnostic_metrics.csv").exists()
    return False


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for config in args.configs:
        config_path = Path(config)
        cfg = load_yaml(config_path)
        exp_name = str(cfg.get("exp_name", config_path.stem))
        output_dir = Path(str(cfg.get("output_dir") or Path(cfg.get("output_root", "outputs")) / exp_name))
        task = infer_task(cfg, exp_name)
        input_path = Path(str(cfg.get("input_path", "")))
        eval_ready = expected_eval_ready(output_dir, task)
        if eval_ready:
            status = "eval_ready"
        elif input_path.exists():
            status = "registered_input_ready"
        else:
            status = "registered_input_missing"
        rows.append(
            {
                "exp_name": exp_name,
                "domain": infer_domain(exp_name),
                "task": task,
                "model": Path(str(cfg.get("model_config", ""))).stem or "unknown",
                "status": status,
                "config_path": str(config_path),
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "eval_ready": eval_ready,
                "notes": "status-only registry row" if args.status_only else "registered for later batch execution",
            }
        )

    status_path = write_registry(rows, args.status_path)
    print(f"Saved batch status to: {status_path}")


if __name__ == "__main__":
    main()
