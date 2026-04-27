from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_framework_day5_tiny_train_qwen_lora import run_tiny_train


SPECS = {
    "listwise_shuffled": {
        "config": "configs/framework/qwen3_8b_lora_baseline_beauty_listwise_strict_shuffled_small.yaml",
        "metrics": Path("data_done/framework_day10_listwise_strict_shuffled_train_metrics.json"),
        "report": Path("data_done/framework_day10_listwise_strict_shuffled_train_report.md"),
        "title": "Framework-Day10 Listwise Strict Shuffled Train Report",
        "scope": "Beauty-only listwise strict Qwen-LoRA baseline on shuffled candidate order. No CEP/confidence/evidence framework.",
    },
    "pointwise_shuffled": {
        "config": "configs/framework/qwen3_8b_lora_baseline_beauty_pointwise_shuffled_small.yaml",
        "metrics": Path("data_done/framework_day10_pointwise_shuffled_train_metrics.json"),
        "report": Path("data_done/framework_day10_pointwise_shuffled_train_report.md"),
        "title": "Framework-Day10 Pointwise Shuffled Train Report",
        "scope": "Beauty-only pointwise relevance baseline on shuffled candidate order for audited comparison. No calibrated probability or CEP target.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day10 train shuffled-order Qwen-LoRA baselines.")
    parser.add_argument("--name", choices=sorted(SPECS), default="listwise_shuffled")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    spec = SPECS[args.name]
    result = run_tiny_train(
        args.config or str(spec["config"]),
        metrics_path=spec["metrics"],
        report_path=spec["report"],
        report_title=str(spec["title"]),
        report_scope=str(spec["scope"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
