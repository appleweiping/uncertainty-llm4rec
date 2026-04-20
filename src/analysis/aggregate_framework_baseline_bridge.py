from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASELINE_BRIDGE_COLUMNS = [
    "domain",
    "model",
    "task",
    "baseline_family",
    "baseline_name",
    "HR@10",
    "NDCG@10",
    "MRR",
    "pairwise_accuracy",
    "ECE",
    "Brier",
    "coverage",
    "head_exposure",
    "longtail_coverage",
    "baseline_layer",
    "model_source_group",
    "model_family",
    "adapter_path",
    "method_family",
    "method_variant",
    "uncertainty_source",
    "evaluation_scope",
    "samples",
    "parse_success_rate",
    "is_same_task_baseline",
    "is_current_best_family",
    "current_role",
    "evidence_question",
    "compare_status",
    "source_file",
    "notes",
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_base_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = {column: row.get(column, "") for column in BASELINE_BRIDGE_COLUMNS}
        if not item.get("compare_status"):
            item["compare_status"] = "reference_ready"
        normalized.append(item)
    return normalized


def _framework_row_to_baseline_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": row.get("domain", ""),
        "model": row.get("model", ""),
        "task": row.get("task", ""),
        "baseline_family": row.get("baseline_family", ""),
        "baseline_name": row.get("method_variant", ""),
        "HR@10": row.get("HR@10", ""),
        "NDCG@10": row.get("NDCG@10", ""),
        "MRR": row.get("MRR", ""),
        "pairwise_accuracy": row.get("pairwise_accuracy", ""),
        "ECE": row.get("ECE", ""),
        "Brier": row.get("Brier", ""),
        "coverage": row.get("coverage", ""),
        "head_exposure": row.get("head_exposure", ""),
        "longtail_coverage": row.get("longtail_coverage", ""),
        "baseline_layer": row.get("baseline_layer", ""),
        "model_source_group": row.get("model_source_group", ""),
        "model_family": row.get("model_family", ""),
        "adapter_path": row.get("adapter_path", ""),
        "method_family": row.get("method_family", ""),
        "method_variant": row.get("method_variant", ""),
        "uncertainty_source": row.get("uncertainty_source", ""),
        "evaluation_scope": row.get("evaluation_scope", ""),
        "samples": row.get("samples", ""),
        "parse_success_rate": row.get("parse_success_rate", ""),
        "is_same_task_baseline": row.get("is_same_task_baseline", ""),
        "is_current_best_family": row.get("is_current_best_family", ""),
        "current_role": row.get("current_role", ""),
        "evidence_question": row.get("evidence_question", ""),
        "compare_status": row.get("compare_status", ""),
        "source_file": row.get("source_file", ""),
        "notes": row.get("notes", ""),
    }


def build_framework_baseline_bridge(
    *,
    baseline_matrix_path: str | Path,
    framework_compare_path: str | Path,
) -> list[dict[str, Any]]:
    base_rows = _normalize_base_rows(_read_csv_rows(Path(baseline_matrix_path)))
    framework_rows = [_framework_row_to_baseline_row(row) for row in _read_csv_rows(Path(framework_compare_path))]

    if not framework_rows:
        return base_rows

    existing_keys = {
        (
            row.get("domain", ""),
            row.get("model", ""),
            row.get("task", ""),
            row.get("method_family", ""),
            row.get("method_variant", ""),
        )
        for row in base_rows
    }

    for row in framework_rows:
        key = (
            row.get("domain", ""),
            row.get("model", ""),
            row.get("task", ""),
            row.get("method_family", ""),
            row.get("method_variant", ""),
        )
        if key not in existing_keys:
            base_rows.append(row)

    return base_rows


def write_framework_baseline_bridge(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_BRIDGE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in BASELINE_BRIDGE_COLUMNS})
    return output_path


def write_framework_baseline_bridge_markdown(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Week7.5 Baseline Matrix Bridge")
    lines.append("")
    lines.append(
        "这份 bridge 文档的作用不是产出新的实验结论，而是把 week7.5 的 LoRA framework compare 正式嵌入既有三层 baseline 坐标系。这样一来，后续服务器真实训练结果一旦回来，就可以沿着同一套 baseline 字段、角色标签和 summary 叙事进入更大的 baseline matrix。"
    )
    lines.append("")
    lines.append("| task | baseline_layer | method_family | method_variant | current_role | compare_status |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row.get('task','')} | {row.get('baseline_layer','')} | {row.get('method_family','')} | {row.get('method_variant','')} | {row.get('current_role','')} | {row.get('compare_status','')} |"
        )
    lines.append("")
    lines.append(
        "在当前 bridge 里，structured risk current best family 被固定为 strongest hand-crafted baseline，direct ranking 保持 same-task reference，literature-aligned baselines 保持外部或 task-aligned reference，而 trainable framework 行则作为待服务器真实结果回填的主 compare 槽位保留下来。"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
