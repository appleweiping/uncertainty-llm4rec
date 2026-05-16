from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "domain",
    "display_method",
    "method",
    "sample_count",
    "HR@5",
    "NDCG@5",
    "HR@10",
    "NDCG@10",
    "HR@20",
    "NDCG@20",
    "MRR",
    "coverage@5",
    "coverage@10",
    "coverage@20",
    "head_exposure_ratio@10",
    "longtail_coverage@10",
    "completion_label",
    "status_label",
    "artifact_class",
    "implementation_status",
    "score_coverage_rate",
    "evidence_present",
    "evidence_sha256",
    "truce_evidence_tar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRUCE summary tables from Pony official baseline manifest.")
    parser.add_argument("--manifest-json", default="outputs/pony_official_baselines/manifest.json")
    parser.add_argument("--output-root", default="outputs/pony_official_baselines/tables")
    parser.add_argument("--output-name", default="pony_official_baseline_comparison")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def eligible_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in manifest.get("entries", []) if entry.get("main_table_eligible") is True]


def status_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("entries", []))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    fields = [
        "domain",
        "display_method",
        "sample_count",
        "HR@10",
        "NDCG@10",
        "MRR",
        "coverage@10",
        "completion_label",
        "artifact_class",
        "evidence_present",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def build_tables(manifest_path: Path, output_root: Path, output_name: str) -> dict[str, str]:
    manifest = load_manifest(manifest_path)
    main_rows = eligible_rows(manifest)
    all_rows = status_rows(manifest)
    output_root.mkdir(parents=True, exist_ok=True)

    main_csv = output_root / f"{output_name}.csv"
    main_md = output_root / f"{output_name}.md"
    status_csv = output_root / f"{output_name}_status.csv"
    status_md = output_root / f"{output_name}_status.md"

    _write_csv(main_rows, main_csv)
    main_md.write_text(_to_markdown(main_rows), encoding="utf-8")
    _write_csv(all_rows, status_csv)
    status_md.write_text(_to_markdown(all_rows), encoding="utf-8")

    return {
        "main_csv": str(main_csv),
        "main_md": str(main_md),
        "status_csv": str(status_csv),
        "status_md": str(status_md),
        "main_rows": str(len(main_rows)),
        "status_rows": str(len(all_rows)),
    }


def main() -> None:
    args = parse_args()
    result = build_tables(Path(args.manifest_json), Path(args.output_root), args.output_name)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
