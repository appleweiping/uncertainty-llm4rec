from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


FRAMEWORK_COMPARE_COLUMNS = [
    "domain",
    "model",
    "task",
    "method_family",
    "method_variant",
    "baseline_family",
    "baseline_layer",
    "model_source_group",
    "model_family",
    "adapter_path",
    "uncertainty_source",
    "evaluation_scope",
    "is_same_task_baseline",
    "is_current_best_family",
    "is_trainable_framework",
    "samples",
    "HR@10",
    "NDCG@10",
    "MRR",
    "pairwise_accuracy",
    "ECE",
    "Brier",
    "coverage",
    "head_exposure",
    "longtail_coverage",
    "parse_success_rate",
    "training_stage_role",
    "current_role",
    "evidence_question",
    "compare_status",
    "source_file",
    "notes",
]


def _model_source_group(model: str, model_family: str = "", *, adapter_path: str = "") -> str:
    text = f"{model} {model_family} {adapter_path}".lower()
    if "lora" in text or "adapter" in text:
        return "lora_adapted_model_group"
    if "local" in text or "hf" in text or "qwen3" in text or "llama" in text:
        return "local_hf_main_model_group"
    return "unspecified_model_group"


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_single_row(path: Path) -> dict[str, Any] | None:
    rows = _read_csv_rows(path)
    if not rows:
        return None
    return rows[0]


def _normalize_compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append({column: row.get(column, "") for column in FRAMEWORK_COMPARE_COLUMNS})
    return normalized


def build_framework_compare_rows(config_path: str | Path) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    summary_cfg = config.get("summary", {}) or {}
    support_signals = config.get("support_signals", {}) or {}

    rows: list[dict[str, Any]] = []
    domain = str(config.get("domain", "beauty"))
    model = str(config.get("model_name", "qwen3_8b_local"))

    direct_metrics = _load_single_row(Path(str(summary_cfg.get("direct_ranking_metrics_path", ""))))
    if direct_metrics:
        rows.append(
            {
                "domain": domain,
                "model": model,
                "task": "candidate_ranking",
                "method_family": "local_hf_base_only",
                "method_variant": "direct_candidate_ranking_medium_scale",
                "baseline_family": "decision_formulation",
                "baseline_layer": "decision_formulation",
                "model_source_group": _model_source_group(model, "local_hf_base_only"),
                "model_family": "local_hf_base_only",
                "adapter_path": "",
                "uncertainty_source": "none",
                "evaluation_scope": "same_candidate_schema",
                "is_same_task_baseline": True,
                "is_current_best_family": False,
                "is_trainable_framework": False,
                "samples": direct_metrics.get("sample_count", ""),
                "HR@10": direct_metrics.get("HR@10", ""),
                "NDCG@10": direct_metrics.get("NDCG@10", ""),
                "MRR": direct_metrics.get("MRR", ""),
                "pairwise_accuracy": "",
                "ECE": "",
                "Brier": "",
                "coverage": direct_metrics.get("coverage@10", ""),
                "head_exposure": direct_metrics.get("head_exposure_ratio@10", ""),
                "longtail_coverage": direct_metrics.get("longtail_coverage@10", ""),
                "parse_success_rate": direct_metrics.get("parse_success_rate", ""),
                "training_stage_role": "direct_ranking_baseline",
                "current_role": "same_task_reference",
                "evidence_question": "whether the trainable framework improves over the non-uncertainty direct ranking baseline under the same candidate set",
                "compare_status": "reference_ready",
                "source_file": str(summary_cfg.get("direct_ranking_metrics_path", "")),
                "notes": "Direct candidate ranking baseline inherited from Week7 medium-scale server run.",
            }
        )

    structured_risk_metrics = _load_single_row(Path(str(summary_cfg.get("structured_risk_metrics_path", ""))))
    if structured_risk_metrics:
        rows.append(
            {
                "domain": domain,
                "model": model,
                "task": "candidate_ranking_rerank",
                "method_family": "structured_risk_family",
                "method_variant": str(support_signals.get("structured_risk_variant", "nonlinear_structured_risk_rerank")),
                "baseline_family": "decision_formulation",
                "baseline_layer": "decision_formulation",
                "model_source_group": _model_source_group(model, "local_hf_base_only"),
                "model_family": "local_hf_base_only",
                "adapter_path": "",
                "uncertainty_source": "pointwise_calibrated",
                "evaluation_scope": "same_candidate_schema",
                "is_same_task_baseline": False,
                "is_current_best_family": True,
                "is_trainable_framework": False,
                "samples": structured_risk_metrics.get("sample_count", structured_risk_metrics.get("samples", "")),
                "HR@10": structured_risk_metrics.get("HR@10", ""),
                "NDCG@10": structured_risk_metrics.get("NDCG@10", ""),
                "MRR": structured_risk_metrics.get("MRR", ""),
                "pairwise_accuracy": "",
                "ECE": "",
                "Brier": "",
                "coverage": structured_risk_metrics.get("coverage", ""),
                "head_exposure": structured_risk_metrics.get("head_exposure", ""),
                "longtail_coverage": structured_risk_metrics.get("longtail_coverage", ""),
                "parse_success_rate": structured_risk_metrics.get("parse_success_rate", ""),
                "training_stage_role": "strongest_handcrafted_baseline",
                "current_role": "strongest_handcrafted_baseline",
                "evidence_question": "whether a trainable framework is strong enough to justify added training complexity beyond the current best hand-crafted uncertainty-aware baseline",
                "compare_status": "strong_reference_ready",
                "source_file": str(summary_cfg.get("structured_risk_metrics_path", "")),
                "notes": "Current best structured-risk family carried into Week7.5 as the strongest hand-crafted baseline.",
            }
        )

    framework_metrics = _load_single_row(Path(str(summary_cfg.get("framework_metrics_path", ""))))
    if framework_metrics:
        rows.append(
            {
                "domain": domain,
                "model": model,
                "task": "candidate_ranking",
                "method_family": "trainable_lora_framework",
                "method_variant": str(config.get("method_variant", config.get("run_name", "framework_v1"))),
                "baseline_family": "trainable_framework",
                "baseline_layer": "trainable_framework",
                "model_source_group": _model_source_group(
                    model,
                    "local_hf_lora_adapted",
                    adapter_path=str(config.get("adapter_output_dir", "")),
                ),
                "model_family": "local_hf_lora_adapted",
                "adapter_path": str(config.get("adapter_output_dir", "")),
                "uncertainty_source": "side_signal_reserved",
                "evaluation_scope": "same_candidate_schema",
                "is_same_task_baseline": False,
                "is_current_best_family": False,
                "is_trainable_framework": True,
                "samples": framework_metrics.get("sample_count", ""),
                "HR@10": framework_metrics.get("HR@10", ""),
                "NDCG@10": framework_metrics.get("NDCG@10", ""),
                "MRR": framework_metrics.get("MRR", ""),
                "pairwise_accuracy": "",
                "ECE": "",
                "Brier": "",
                "coverage": framework_metrics.get("coverage@10", ""),
                "head_exposure": framework_metrics.get("head_exposure_ratio@10", ""),
                "longtail_coverage": framework_metrics.get("longtail_coverage@10", ""),
                "parse_success_rate": framework_metrics.get("parse_success_rate", ""),
                "training_stage_role": "trainable_framework_candidate",
                "current_role": "trainable_framework_main_candidate",
                "evidence_question": "whether a Qwen3-8B plus LoRA ranking framework can outperform the strongest hand-crafted baseline under the same candidate-ranking schema",
                "compare_status": "framework_result_ready",
                "source_file": str(summary_cfg.get("framework_metrics_path", "")),
                "notes": "LoRA-adapted ranking framework result. Day1 initializes the compare row even before the first adapter is trained.",
            }
        )
    else:
        rows.append(
            {
                "domain": domain,
                "model": model,
                "task": "candidate_ranking",
                "method_family": "trainable_lora_framework",
                "method_variant": str(config.get("method_variant", config.get("run_name", "framework_v1"))),
                "baseline_family": "trainable_framework",
                "baseline_layer": "trainable_framework",
                "model_source_group": _model_source_group(
                    model,
                    "local_hf_lora_adapted",
                    adapter_path=str(config.get("adapter_output_dir", "")),
                ),
                "model_family": "local_hf_lora_adapted",
                "adapter_path": str(config.get("adapter_output_dir", "")),
                "uncertainty_source": "side_signal_reserved",
                "evaluation_scope": "same_candidate_schema",
                "is_same_task_baseline": False,
                "is_current_best_family": False,
                "is_trainable_framework": True,
                "samples": "",
                "HR@10": "",
                "NDCG@10": "",
                "MRR": "",
                "pairwise_accuracy": "",
                "ECE": "",
                "Brier": "",
                "coverage": "",
                "head_exposure": "",
                "longtail_coverage": "",
                "parse_success_rate": "",
                "training_stage_role": "trainable_framework_candidate",
                "current_role": "trainable_framework_main_candidate",
                "evidence_question": "whether a Qwen3-8B plus LoRA ranking framework can outperform the strongest hand-crafted baseline under the same candidate-ranking schema",
                "compare_status": "framework_pending_server_run",
                "source_file": str(summary_cfg.get("framework_metrics_path", "")),
                "notes": "Framework compare slot is reserved and awaits the first real Beauty-domain server run.",
            }
        )

    literature_summary_path = Path(str(summary_cfg.get("literature_baseline_summary_path", "")))
    if literature_summary_path.exists():
        for row in _read_csv_rows(literature_summary_path):
            if str(row.get("task", "")) != "candidate_ranking":
                continue
            rows.append(
                {
                    "domain": row.get("domain", domain),
                    "model": row.get("model", model),
                    "task": row.get("task", "candidate_ranking"),
                    "method_family": row.get("baseline_family", "literature_aligned_rank"),
                    "method_variant": row.get("baseline_name", ""),
                    "baseline_family": "literature_aligned",
                    "baseline_layer": "literature_aligned",
                    "model_source_group": _model_source_group(
                        str(row.get("model", model)),
                        str(row.get("model_family", "")),
                        adapter_path=str(row.get("adapter_path", "")),
                    ),
                    "model_family": row.get("model_family", ""),
                    "adapter_path": row.get("adapter_path", ""),
                    "uncertainty_source": "none",
                    "evaluation_scope": "same_candidate_or_pairwise_schema",
                    "is_same_task_baseline": True,
                    "is_current_best_family": False,
                    "is_trainable_framework": False,
                    "samples": row.get("samples", ""),
                    "HR@10": row.get("HR@10", ""),
                    "NDCG@10": row.get("NDCG@10", ""),
                    "MRR": row.get("MRR", ""),
                    "pairwise_accuracy": row.get("pairwise_accuracy", ""),
                    "ECE": "",
                    "Brier": "",
                    "coverage": "",
                    "head_exposure": "",
                    "longtail_coverage": "",
                    "parse_success_rate": row.get("parse_success_rate", ""),
                    "training_stage_role": "literature_aligned_reference",
                    "current_role": "literature_or_task_aligned_reference",
                    "evidence_question": "whether the trainable framework can be defended against external or task-aligned formulations under the same schema",
                    "compare_status": "reference_ready",
                    "source_file": str(literature_summary_path),
                    "notes": row.get("notes", ""),
                }
            )

    return _normalize_compare_rows(rows)


def write_framework_compare(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAMEWORK_COMPARE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FRAMEWORK_COMPARE_COLUMNS})


def summarize_framework_compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    framework_rows = [row for row in rows if str(row.get("is_trainable_framework", "")).lower() == "true"]
    current_best_rows = [row for row in rows if str(row.get("is_current_best_family", "")).lower() == "true"]
    framework_result_rows = [row for row in framework_rows if str(row.get("compare_status", "")) == "framework_result_ready"]
    return {
        "row_count": len(rows),
        "framework_row_count": len(framework_rows),
        "current_best_row_count": len(current_best_rows),
        "framework_metrics_ready": bool(framework_result_rows),
        "compare_ready_for_server_demo": bool(current_best_rows),
    }
