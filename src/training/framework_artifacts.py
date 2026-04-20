from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_framework_manifest(
    *,
    path: str | Path,
    run_name: str,
    domain: str,
    model: str,
    method_family: str,
    method_variant: str,
    adapter_output_dir: str,
    framework_output_dir: str,
    compare_csv_path: str,
    compare_markdown_path: str,
    training_summary_path: str,
    startup_check_path: str,
    dataset_preview_path: str,
    latest_stage: str,
    latest_status: str,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    manifest: dict[str, Any] = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))

    manifest.update(
        {
            "run_name": run_name,
            "domain": domain,
            "model": model,
            "method_family": method_family,
            "method_variant": method_variant,
            "adapter_output_dir": adapter_output_dir,
            "framework_output_dir": framework_output_dir,
            "compare_csv_path": compare_csv_path,
            "compare_markdown_path": compare_markdown_path,
            "training_summary_path": training_summary_path,
            "startup_check_path": startup_check_path,
            "dataset_preview_path": dataset_preview_path,
            "latest_stage": latest_stage,
            "latest_status": latest_status,
            "updated_at": utc_now_iso(),
        }
    )
    if extra_fields:
        manifest.update(extra_fields)
    write_json(manifest, path)


def append_stage_status(status_row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    fieldnames = list(status_row.keys())
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
    existing_rows.append(status_row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_compare_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Week7.5 Framework Compare")
    lines.append("")
    lines.append("这份 compare 不是新的论文结论，而是 week7.5 Beauty 单域最小训练闭环的工程视图。它把 direct ranking、structured risk strongest hand-crafted baseline、literature-aligned ranking baselines 和未来的 LoRA-adapted framework 放在同一条结果路径里，确保后续服务器闭环完成后不需要再回头补 compare 结构。")
    lines.append("")
    if not rows:
        lines.append("当前 compare 仍处于 schema 初始化阶段，尚未接收到真实 framework metrics。")
    else:
        lines.append("| task | method_family | method_variant | stage_role | NDCG@10 | MRR | notes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row.get('task','')} | {row.get('method_family','')} | {row.get('method_variant','')} | {row.get('training_stage_role','')} | {row.get('NDCG@10','')} | {row.get('MRR','')} | {row.get('notes','')} |"
            )
        framework_rows = [row for row in rows if str(row.get("is_trainable_framework", "")).lower() == "true"]
        lines.append("")
        if framework_rows:
            lines.append("当前 compare 已经包含 trainable framework 行；如果后续服务器训练完成并写入 framework metrics，这些行会成为 Beauty 单域最小训练闭环的中心结果。")
        else:
            lines.append("当前 compare 仍未包含真实 framework metrics，这说明本地工程准备已完成，但真正的 Beauty LoRA 闭环仍等待服务器训练与评估产物回填。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
