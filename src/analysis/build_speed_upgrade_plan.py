from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


SPEED_UPGRADE_COLUMNS = [
    "domain",
    "model",
    "focus_task",
    "plan_layer",
    "action_name",
    "action_group",
    "preserve_research_content",
    "target_component",
    "current_state",
    "week8_link",
    "notes",
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_medium_row(rows: list[dict[str, Any]], task: str, method_family: str, method_variant: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            str(row.get("task", "")) == task
            and str(row.get("method_family", "")) == method_family
            and str(row.get("method_variant", "")) == method_variant
        ):
            return row
    return None


def build_speed_upgrade_rows(config_path: str | Path) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    summary_cfg = config.get("summary", {}) or {}
    speed_cfg = config.get("speed_upgrade", {}) or {}
    evaluation_cfg = config.get("evaluation", {}) or {}

    domain = str(config.get("domain", "beauty"))
    model = str(config.get("model_name", "qwen3_8b_local"))
    focus_task = str(speed_cfg.get("bottleneck_task", "candidate_ranking"))
    medium_summary_path = Path("outputs/summary/week7_day5_medium_scale_summary.csv")
    medium_rows = _read_csv_rows(medium_summary_path)
    rank_row = _find_medium_row(
        medium_rows,
        task="candidate_ranking",
        method_family="local_hf_base_only",
        method_variant="direct_candidate_ranking_medium_scale",
    )
    pointwise_row = _find_medium_row(
        medium_rows,
        task="pointwise_yesno",
        method_family="local_hf_base_only",
        method_variant="pointwise_medium_scale",
    )
    pairwise_row = _find_medium_row(
        medium_rows,
        task="pairwise_preference",
        method_family="local_hf_base_only",
        method_variant="pairwise_medium_scale",
    )

    preserve_components = [str(item) for item in speed_cfg.get("preserve_components", [])]
    batching_cfg = speed_cfg.get("batching", {}) or {}
    output_contract_cfg = speed_cfg.get("output_contract", {}) or {}
    reuse_cfg = speed_cfg.get("reuse_and_resume", {}) or {}
    serving_cfg = speed_cfg.get("serving_readiness", {}) or {}
    week8_conditions = [str(item) for item in speed_cfg.get("week8_entry_conditions", [])]

    pointwise_state = "present" if pointwise_row else "not_materialized_locally"
    pairwise_state = "present" if pairwise_row else "not_materialized_locally"
    ranking_state = "present" if rank_row else "not_materialized_locally"

    rows: list[dict[str, Any]] = [
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "bottleneck_diagnosis",
            "action_name": "ranking_is_primary_throughput_bottleneck",
            "action_group": "diagnosis",
            "preserve_research_content": True,
            "target_component": "candidate_ranking",
            "current_state": ranking_state,
            "week8_link": "week8 should expand only after ranking throughput path is explicit",
            "notes": "Ranking is treated as the primary throughput bottleneck because it carries the longest listwise prompt and the largest output contract under the same local-HF execution path. This diagnosis does not remove pointwise, pairwise, calibration, or baseline layers.",
        },
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "content_preservation",
            "action_name": "do_not_delete_supporting_layers",
            "action_group": "scope_guardrail",
            "preserve_research_content": True,
            "target_component": ",".join(preserve_components),
            "current_state": "guardrail_fixed",
            "week8_link": "week8 keeps pointwise, pairwise, calibration, and baseline evidence active",
            "notes": "Speed-up is not allowed to come from deleting pointwise diagnosis, pairwise mechanism evidence, calibration, or the three-layer baseline compare. The optimization target is execution overhead on ranking-heavy paths.",
        },
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "output_contract",
            "action_name": "tighten_ranking_output_contract",
            "action_group": "prompt_and_generation",
            "preserve_research_content": True,
            "target_component": "candidate_ranking_generation",
            "current_state": "planned",
            "week8_link": "week8 larger runs should use a minimal JSON contract by default",
            "notes": f"Default ranking output should move toward JSON-only generation with include_reason_by_default={output_contract_cfg.get('include_reason_by_default', False)} and a target max_new_tokens of {output_contract_cfg.get('max_new_tokens_target', evaluation_cfg.get('max_new_tokens', ''))}. Long free-form reasoning should be reserved for case studies or debugging, not default throughput runs.",
        },
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "batching",
            "action_name": "verify_batch_generate_is_real",
            "action_group": "backend_execution",
            "preserve_research_content": True,
            "target_component": ",".join(str(item) for item in batching_cfg.get("verify_entrypoints", [])),
            "current_state": "planned",
            "week8_link": "week8 ranking expansion assumes local-HF batching is actually used",
            "notes": f"batch_generate must be verified at the execution level rather than assumed from config. The current target batch size is {batching_cfg.get('target_batch_size', '')}, and the backend path should be checked from main_rank into local_hf_backend.batch_generate.",
        },
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "reuse_and_resume",
            "action_name": "reduce_repeated_loads_and_enable_shard_resume",
            "action_group": "execution_reliability",
            "preserve_research_content": True,
            "target_component": "ranking_eval_and_compare",
            "current_state": "planned",
            "week8_link": "week8 domain expansion should use shardable and resumable ranking evaluation",
            "notes": f"Execution should reduce repeated model loads, keep registry-based resume active, and reserve shard evaluation with shard_size={reuse_cfg.get('shard_size', '')}. This is meant to scale ranking evaluation without deleting content-bearing tasks or compares.",
        },
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "serving_readiness",
            "action_name": "reserve_resident_serving_transition",
            "action_group": "future_execution_mode",
            "preserve_research_content": True,
            "target_component": "local_hf_serving",
            "current_state": "reserved",
            "week8_link": "week8 may require a resident serving path if ranking volume exceeds the Beauty single-domain medium stage",
            "notes": f"The current execution mode stays {serving_cfg.get('current_mode', 'transformers_local')}, while {serving_cfg.get('future_mode', 'vllm_reserved')} remains a reserved next step once ranking volume grows beyond the first single-domain medium stage.",
        },
        {
            "domain": domain,
            "model": model,
            "focus_task": focus_task,
            "plan_layer": "week8_entry_gate",
            "action_name": "gate_week8_on_framework_and_speed_readiness",
            "action_group": "stage_transition",
            "preserve_research_content": True,
            "target_component": "week8_entry_conditions",
            "current_state": "gate_defined",
            "week8_link": ",".join(week8_conditions),
            "notes": "Week8 should open only after the Beauty trainable framework has a minimal real result, the framework compare role is fixed, the strongest hand-crafted baseline identity is fixed, and the ranking speed path is documented together with shard/resume expectations.",
        },
    ]
    return rows


def write_speed_upgrade_plan(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPEED_UPGRADE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SPEED_UPGRADE_COLUMNS})
    return output_path


def write_speed_upgrade_markdown(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Week7.5 Speed Upgrade")
    lines.append("")
    lines.append(
        "这份 speed upgrade 说明不是新的方法设计，而是 week7.5 对执行层的正式收口。它的目标不是通过删掉 pointwise、pairwise、calibration 或 baseline 来换取更快速度，而是把 ranking 吞吐瓶颈识别出来，并把后续的优化路径固定成可以进入 week8 的工程约束。"
    )
    lines.append("")
    lines.append("| action_name | action_group | current_state | week8_link |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row.get('action_name','')} | {row.get('action_group','')} | {row.get('current_state','')} | {row.get('week8_link','')} |"
        )
    lines.append("")
    lines.append(
        "当前排序主瓶颈被固定在 candidate ranking 路径上，而不是整个项目的所有任务上。提速路线围绕四个方向组织：更紧的输出契约、确认 batch_generate 真正生效、减少重复模型加载并引入 shard/resume、以及为更大规模阶段预留常驻服务化接口。"
    )
    lines.append("")
    lines.append(
        "这些动作会和 framework compare、baseline bridge 以及 week8 入场条件一起工作：只有当 trainable framework 已经有最小可用结果、structured risk strongest hand-crafted baseline 的 compare 身份已经固定、ranking speed path 也已经明文化之后，week8 才应该完整打开。"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
