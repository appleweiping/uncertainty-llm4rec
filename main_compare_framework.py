from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.aggregate_framework_compare import (
    FRAMEWORK_COMPARE_COLUMNS,
    build_framework_compare_rows,
    summarize_framework_compare,
    write_framework_compare,
)
from src.analysis.aggregate_framework_baseline_bridge import (
    build_framework_baseline_bridge,
    write_framework_baseline_bridge,
    write_framework_baseline_bridge_markdown,
)
from src.training.framework_artifacts import append_stage_status, update_framework_manifest, utc_now_iso, write_compare_markdown
from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Week7.5 framework config path.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional compare CSV override.")
    parser.add_argument("--init_only", action="store_true", help="Only initialize the compare schema if the file is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    summary_cfg = config.get("summary", {}) or {}
    default_output_path = summary_cfg.get("framework_compare_path", "outputs/summary/week7_5_framework_compare.csv")
    output_path = Path(args.output_path or default_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_markdown_path = Path(
        str(summary_cfg.get("framework_compare_markdown_path", "outputs/summary/week7_5_framework_compare.md"))
    )
    train_status_path = Path(str(summary_cfg.get("train_status_path", "outputs/summary/week7_5_train_status.csv")))
    framework_manifest_path = Path(
        str(summary_cfg.get("framework_manifest_path", "outputs/beauty_qwen3_rank_framework_v1/framework_run_manifest.json"))
    )
    baseline_matrix_source_path = Path(
        str(summary_cfg.get("baseline_matrix_source_path", "outputs/summary/week7_day4_baseline_matrix.csv"))
    )
    baseline_bridge_path = Path(
        str(summary_cfg.get("framework_baseline_bridge_path", "outputs/summary/week7_5_baseline_matrix.csv"))
    )
    baseline_bridge_markdown_path = Path(
        str(summary_cfg.get("framework_baseline_bridge_markdown_path", "outputs/summary/week7_5_baseline_matrix.md"))
    )

    if args.init_only and output_path.exists():
        print(f"Week7.5 framework compare already exists: {output_path}")
        return

    rows = build_framework_compare_rows(args.config)
    if not rows:
        rows = []
    write_framework_compare(rows, output_path)
    write_compare_markdown(rows, compare_markdown_path)
    bridged_rows = build_framework_baseline_bridge(
        baseline_matrix_path=baseline_matrix_source_path,
        framework_compare_path=output_path,
    )
    write_framework_baseline_bridge(bridged_rows, baseline_bridge_path)
    write_framework_baseline_bridge_markdown(bridged_rows, baseline_bridge_markdown_path)
    compare_summary = summarize_framework_compare(rows)
    append_stage_status(
        {
            "run_name": str(config.get("run_name", "week7_5_framework")),
            "domain": str(config.get("domain", "beauty")),
            "task": "candidate_ranking_compare",
            "method_family": str(config.get("method_family", "trainable_lora_framework")),
            "method_variant": str(config.get("method_variant", config.get("run_name", "framework_v1"))),
            "model": str(config.get("model_name", "qwen3_8b_local")),
            "stage": "framework_compare",
            "status": "compare_ready",
            "dry_run": False,
            "startup_check_only": False,
            "framework_compare_path": str(output_path),
            "framework_compare_markdown_path": str(compare_markdown_path),
            "framework_baseline_bridge_path": str(baseline_bridge_path),
            "framework_baseline_bridge_markdown_path": str(baseline_bridge_markdown_path),
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "notes": "Framework compare refreshed from direct ranking, structured risk, literature baselines, and any available framework metrics.",
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
        compare_csv_path=str(output_path),
        compare_markdown_path=str(compare_markdown_path),
        training_summary_path=str(
            summary_cfg.get("training_summary_path", "artifacts/logs/qwen3_rank_beauty_framework_v1/training_summary.csv")
        ),
        startup_check_path=str(summary_cfg.get("startup_check_path", "outputs/summary/week7_5_startup_check.json")),
        dataset_preview_path=str(summary_cfg.get("dataset_preview_path", "outputs/summary/week7_5_dataset_preview.csv")),
        latest_stage="framework_compare",
        latest_status="compare_ready",
        extra_fields={
            **compare_summary,
            "baseline_matrix_source_path": str(baseline_matrix_source_path),
            "framework_baseline_bridge_path": str(baseline_bridge_path),
            "framework_baseline_bridge_markdown_path": str(baseline_bridge_markdown_path),
        },
    )
    print(f"Saved Week7.5 framework compare to: {output_path}")
    print(f"Saved Week7.5 framework baseline bridge to: {baseline_bridge_path}")
    print(f"Saved Week7.5 framework baseline bridge markdown to: {baseline_bridge_markdown_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
