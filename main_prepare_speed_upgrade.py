from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.build_speed_upgrade_plan import (
    build_speed_upgrade_rows,
    write_speed_upgrade_markdown,
    write_speed_upgrade_plan,
)
from src.training.framework_artifacts import append_stage_status, update_framework_manifest, utc_now_iso
from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Week7.5 speed-up plan without changing research scope.")
    parser.add_argument("--config", type=str, required=True, help="Week7.5 framework config path.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional speed-up plan CSV override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    summary_cfg = config.get("summary", {}) or {}

    output_path = Path(
        args.output_path or summary_cfg.get("speed_upgrade_plan_path", "outputs/summary/week7_5_speed_upgrade_plan.csv")
    )
    markdown_path = Path(
        summary_cfg.get("speed_upgrade_markdown_path", "outputs/summary/week7_5_speed_upgrade.md")
    )
    train_status_path = Path(summary_cfg.get("train_status_path", "outputs/summary/week7_5_train_status.csv"))
    framework_manifest_path = Path(
        summary_cfg.get("framework_manifest_path", "outputs/beauty_qwen3_rank_framework_v1/framework_run_manifest.json")
    )

    rows = build_speed_upgrade_rows(args.config)
    write_speed_upgrade_plan(rows, output_path)
    write_speed_upgrade_markdown(rows, markdown_path)

    append_stage_status(
        {
            "run_name": str(config.get("run_name", "week7_5_framework")),
            "domain": str(config.get("domain", "beauty")),
            "task": "candidate_ranking_speed_upgrade",
            "method_family": str(config.get("method_family", "trainable_lora_framework")),
            "method_variant": str(config.get("method_variant", config.get("run_name", "framework_v1"))),
            "model": str(config.get("model_name", "qwen3_8b_local")),
            "stage": "speed_upgrade_plan",
            "status": "speed_plan_ready",
            "dry_run": False,
            "startup_check_only": False,
            "speed_upgrade_plan_path": str(output_path),
            "speed_upgrade_markdown_path": str(markdown_path),
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "notes": "Week7.5 speed-up path fixed as an execution-layer upgrade without deleting pointwise, pairwise, calibration, or baseline evidence.",
        },
        train_status_path,
    )
    update_framework_manifest(
        path=framework_manifest_path,
        run_name=str(config.get("run_name", "week7_5_framework")),
        domain=str(config.get("domain", "beauty")),
        model=str(config.get("model_name", "qwen3_8b_local")),
        method_family=str(config.get("method_family", "trainable_lora_framework")),
        method_variant=str(config.get("method_variant", config.get("run_name", "framework_v1"))),
        adapter_output_dir=str(config.get("adapter_output_dir", "")),
        framework_output_dir=str(config.get("framework_output_dir", "")),
        compare_csv_path=str(summary_cfg.get("framework_compare_path", "outputs/summary/week7_5_framework_compare.csv")),
        compare_markdown_path=str(summary_cfg.get("framework_compare_markdown_path", "outputs/summary/week7_5_framework_compare.md")),
        training_summary_path=str(summary_cfg.get("training_summary_path", "artifacts/logs/qwen3_rank_beauty_framework_v1/training_summary.csv")),
        startup_check_path=str(summary_cfg.get("startup_check_path", "outputs/summary/week7_5_startup_check.json")),
        dataset_preview_path=str(summary_cfg.get("dataset_preview_path", "outputs/summary/week7_5_dataset_preview.csv")),
        latest_stage="speed_upgrade_plan",
        latest_status="speed_plan_ready",
        extra_fields={
            "speed_upgrade_plan_path": str(output_path),
            "speed_upgrade_markdown_path": str(markdown_path),
            "speed_plan_row_count": len(rows),
        },
    )
    print(f"Saved Week7.5 speed-up plan to: {output_path}")
    print(f"Saved Week7.5 speed-up markdown to: {markdown_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
