from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_framework_day5_tiny_train_qwen_lora import run_tiny_train


OUTPUTS = {
    "listwise_strict": {
        "config": "configs/framework/qwen3_8b_lora_baseline_beauty_listwise_strict_small.yaml",
        "metrics": Path("data_done/framework_day9_listwise_strict_train_metrics.json"),
        "report": Path("data_done/framework_day9_listwise_strict_report.md"),
        "title": "Framework-Day9 Listwise-v2 Strict Qwen-LoRA Train Report",
        "scope": (
            "This trains the Beauty listwise-v2 strict prompt baseline only. It validates whether "
            "matching strict train/inference formulation improves the local Qwen-LoRA recommender. "
            "It does not implement confidence, evidence, calibrated posterior, or CEP fusion."
        ),
    },
    "pointwise": {
        "config": "configs/framework/qwen3_8b_lora_baseline_beauty_pointwise_small.yaml",
        "metrics": Path("data_done/framework_day9_pointwise_train_metrics.json"),
        "report": Path("data_done/framework_day9_pointwise_train_report.md"),
        "title": "Framework-Day9 Pointwise-v1 Qwen-LoRA Train Report",
        "scope": (
            "This trains the Beauty pointwise relevance baseline only, later aggregated into a "
            "candidate ranking. It uses raw relevance labels, not calibrated probabilities and not CEP."
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day9 Qwen-LoRA baseline formulation train entrypoint.")
    parser.add_argument("--name", choices=sorted(OUTPUTS), default="listwise_strict")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    spec = OUTPUTS[args.name]
    config = args.config or str(spec["config"])
    metrics = run_tiny_train(
        config,
        metrics_path=spec["metrics"],
        report_path=spec["report"],
        report_title=str(spec["title"]),
        report_scope=str(spec["scope"]),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
