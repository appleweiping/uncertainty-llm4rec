"""Build an Amazon Reviews 2023 cross-category readiness matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.data import inspect_amazon_config  # noqa: E402
from storyflow.utils.config import load_simple_yaml  # noqa: E402


def _amazon_config_paths(datasets: list[str] | None = None) -> list[Path]:
    config_dir = ROOT / "configs" / "datasets"
    if datasets:
        paths = [config_dir / f"{dataset}.yaml" for dataset in datasets]
    else:
        paths = sorted(config_dir.glob("amazon_reviews_2023_*.yaml"))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(
            "Unknown dataset config(s): " + ", ".join(str(path) for path in missing)
        )
    return paths


def _next_action(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    if manifest.get("raw_reviews_path_exists") and manifest.get("raw_metadata_path_exists"):
        sample_command = config.get("local_sample_command_template")
        if sample_command:
            return "local_raw_available: run sample prepare/validation gate before any observation"
        return "local_raw_available: add explicit sample command template before use"
    if config.get("requires_large_download"):
        return "server_or_manual_raw_required: confirm license/access, place raw JSONL, then run readiness again"
    return "raw_missing: place review and metadata JSONL files, then run readiness again"


def build_category_matrix(
    *,
    datasets: list[str] | None = None,
    sample_records: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for config_path in _amazon_config_paths(datasets):
        config = load_simple_yaml(config_path)
        if str(config.get("type")) != "amazon_reviews_2023":
            continue
        manifest = inspect_amazon_config(config, sample_records=sample_records)
        records.append(
            {
                "dataset": config.get("name"),
                "category_name": config.get("category_name"),
                "status": manifest.get("status"),
                "config_path": str(config_path),
                "hf_review_config": config.get("hf_review_config"),
                "hf_meta_config": config.get("hf_meta_config"),
                "raw_reviews_path": manifest.get("raw_reviews", {}).get("selected"),
                "raw_reviews_path_exists": manifest.get("raw_reviews_path_exists"),
                "raw_metadata_path": manifest.get("raw_metadata", {}).get("selected"),
                "raw_metadata_path_exists": manifest.get("raw_metadata_path_exists"),
                "server_scale": bool(config.get("server_scale")),
                "requires_large_download": bool(config.get("requires_large_download")),
                "local_sample_command_template": config.get("local_sample_command_template"),
                "full_mode_command_template": config.get("full_mode_command_template"),
                "next_action": _next_action(config, manifest),
                "warnings": manifest.get("warnings") or [],
            }
        )
    ready_count = sum(
        1
        for record in records
        if record["raw_reviews_path_exists"] and record["raw_metadata_path_exists"]
    )
    return {
        "artifact_kind": "amazon_reviews_2023_category_readiness_matrix",
        "api_called": False,
        "server_executed": False,
        "full_download_attempted": False,
        "full_processed": False,
        "model_training": False,
        "is_experiment_result": False,
        "claim_scope": "readiness_only_not_paper_evidence",
        "sample_records": sample_records,
        "dataset_count": len(records),
        "local_raw_available_count": ready_count,
        "records": records,
    }


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "category_name",
        "status",
        "raw_reviews_path_exists",
        "raw_metadata_path_exists",
        "server_scale",
        "requires_large_download",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def _write_markdown(matrix: dict[str, Any], path: Path) -> None:
    lines = [
        "# Amazon Reviews 2023 Category Readiness Matrix",
        "",
        "This is a readiness artifact only. It does not download full data, call APIs, train models, or create paper evidence.",
        "",
        f"- dataset count: {matrix['dataset_count']}",
        f"- local raw available count: {matrix['local_raw_available_count']}",
        f"- sample records inspected per existing raw file: {matrix['sample_records']}",
        f"- claim scope: {matrix['claim_scope']}",
        "",
        "| dataset | category | status | review raw | metadata raw | next action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in matrix["records"]:
        lines.append(
            "| {dataset} | {category} | {status} | {review} | {meta} | {action} |".format(
                dataset=record.get("dataset") or "",
                category=record.get("category_name") or "",
                status=record.get("status") or "",
                review=str(record.get("raw_reviews_path_exists")),
                meta=str(record.get("raw_metadata_path_exists")),
                action=record.get("next_action") or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_category_matrix(matrix: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "amazon_category_matrix.json"
    csv_path = output_dir / "amazon_category_matrix.csv"
    report_path = output_dir / "amazon_category_matrix.md"
    json_path.write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(matrix["records"], csv_path)
    _write_markdown(matrix, report_path)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "report": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--sample-records", type=int, default=0)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "amazon_reviews_2023" / "category_matrix"
    )
    matrix = build_category_matrix(
        datasets=args.datasets,
        sample_records=args.sample_records,
    )
    matrix["outputs"] = write_category_matrix(matrix, output_dir)
    print(json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
