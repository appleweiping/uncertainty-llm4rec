from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_framework_day5_tiny_train_qwen_lora import run_tiny_train


DAY6_METRICS = Path("data_done/framework_day6_beauty_listwise_small_train_metrics.json")
DAY6_REPORT = Path("data_done/framework_day6_beauty_listwise_small_train_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day6 Beauty listwise Qwen-LoRA small train.")
    parser.add_argument("--config", default="configs/framework/qwen3_8b_lora_baseline_beauty_listwise_small.yaml")
    args = parser.parse_args()
    result = run_tiny_train(
        args.config,
        metrics_path=DAY6_METRICS,
        report_path=DAY6_REPORT,
        report_title="Framework-Day6 Beauty Listwise Qwen-LoRA Small Train Report",
        report_scope=(
            "This is a small Beauty listwise Qwen-LoRA baseline train. It validates train -> save adapter "
            "readiness for a local recommender baseline. It does not implement confidence, evidence, CEP fusion, "
            "API calls, pointwise training, four-domain training, or formal long training."
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
