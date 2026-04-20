from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.analysis.aggregate_framework_compare import build_framework_compare_rows, write_framework_compare
from src.training.framework_artifacts import write_compare_markdown


DEFAULT_CONFIGS = [
    "configs/lora/qwen3_rank_beauty_srpd_v1.yaml",
    "configs/lora/qwen3_rank_beauty_srpd_v2.yaml",
    "configs/lora/qwen3_rank_beauty_srpd_v3.yaml",
]
DEFAULT_OUTPUT = "outputs/summary/week7_6_srpd_framework_compare.csv"
DEFAULT_MARKDOWN = "outputs/summary/week7_6_srpd_framework_compare.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine SRPD-v1/v2/v3 framework compare rows.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS, help="SRPD LoRA config paths.")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT, help="Combined SRPD compare CSV.")
    parser.add_argument("--markdown_path", default=DEFAULT_MARKDOWN, help="Combined SRPD compare markdown.")
    return parser.parse_args()


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    if str(row.get("is_trainable_framework", "")).lower() == "true":
        return (
            str(row.get("domain", "")),
            str(row.get("task", "")),
            str(row.get("method_family", "")),
            str(row.get("method_variant", "")),
            str(row.get("adapter_path", "")),
        )
    return (
        str(row.get("domain", "")),
        str(row.get("task", "")),
        str(row.get("method_family", "")),
        str(row.get("method_variant", "")),
        str(row.get("source_file", "")),
    )


def combine_rows(configs: list[str]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for config in configs:
        for row in build_framework_compare_rows(config):
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            combined.append(row)
    return combined


def main() -> None:
    args = parse_args()
    rows = combine_rows([str(Path(config)) for config in args.configs])
    write_framework_compare(rows, args.output_path)
    write_compare_markdown(rows, args.markdown_path)
    print(f"Saved combined SRPD framework compare to: {args.output_path}")
    print(f"Saved combined SRPD framework markdown to: {args.markdown_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
