from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ESTIMATOR_COMPARE_PATH = Path("outputs/summary/week6_day3_estimator_compare.csv")
DEFAULT_DECISION_BASELINE_PATH = Path("outputs/summary/week6_day4_decision_baseline_compare.csv")
DEFAULT_LITERATURE_BASELINE_PATH = Path("outputs/summary/week7_day3_literature_baseline_summary.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/summary/week7_day4_baseline_matrix.csv")

REQUIRED_COLUMNS = [
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
]

OUTPUT_COLUMNS = REQUIRED_COLUMNS + [
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
    "source_file",
    "notes",
]


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and pd.notna(record[name]):
            return record[name]
    return pd.NA


def _text(record: dict[str, Any], *names: str, default: str = "") -> str:
    value = _value(record, *names)
    if pd.isna(value):
        return default
    return str(value)


def _bool_value(record: dict[str, Any], name: str, default: bool = False) -> bool:
    value = record.get(name, default)
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _model_source_group(model: str, model_family: str = "") -> str:
    text = f"{model} {model_family}".lower()
    if "local" in text or "llama" in text or "hf" in text:
        return "local_hf_main_model_group"
    if any(name in text for name in ["qwen", "deepseek", "doubao", "kimi", "glm"]):
        return "official_api_observation_group"
    if "lora" in text or "adapter" in text:
        return "lora_adapted_reserved_group"
    return "unspecified_model_group"


def _common_metrics(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "HR@10": _value(record, "HR@10"),
        "NDCG@10": _value(record, "NDCG@10"),
        "MRR": _value(record, "MRR"),
        "pairwise_accuracy": _value(record, "pairwise_accuracy"),
        "ECE": _value(record, "ECE"),
        "Brier": _value(record, "Brier"),
        "coverage": _value(record, "coverage", "coverage@10", "uncertainty_coverage"),
        "head_exposure": _value(record, "head_exposure", "head_exposure_ratio", "head_exposure_ratio@10"),
        "longtail_coverage": _value(record, "longtail_coverage", "longtail_coverage@10"),
        "samples": _value(record, "samples", "sample_count"),
        "parse_success_rate": _value(record, "parse_success_rate"),
    }


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS]


def _build_uncertainty_source_rows(path: Path) -> list[dict[str, Any]]:
    df = _safe_read(path)
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        estimator = _text(record, "estimator", "uncertainty_source", default="unknown_estimator")
        if estimator in {"", "none", "nan"}:
            continue
        model = _text(record, "model", default="unknown_model")
        method_variant = _text(record, "method_variant", "rerank_variant", "method", default=estimator)
        row = {
            "domain": _text(record, "domain", default="beauty"),
            "model": model,
            "task": _text(record, "task", default="unknown_task"),
            "baseline_family": "uncertainty_source_baseline",
            "baseline_name": f"{estimator}:{method_variant}",
            "baseline_layer": "uncertainty_source",
            "model_source_group": _model_source_group(model),
            "model_family": "",
            "adapter_path": "",
            "method_family": _text(record, "method_family"),
            "method_variant": method_variant,
            "uncertainty_source": estimator,
            "evaluation_scope": _text(record, "evaluation_scope"),
            "is_same_task_baseline": False,
            "is_current_best_family": _bool_value(record, "is_current_best_family", False),
            "current_role": "compare_uncertainty_definition",
            "evidence_question": "which uncertainty definition remains useful across decision granularities",
            "source_file": str(path),
            "notes": _text(record, "notes", default="Estimator compare row imported from the multitask uncertainty-source table."),
        }
        row.update(_common_metrics(record))
        rows.append(row)
    return rows


def _build_decision_formulation_rows(path: Path) -> list[dict[str, Any]]:
    df = _safe_read(path)
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        model = _text(record, "model", default="unknown_model")
        method_variant = _text(record, "method_variant", "method", default="unknown_method")
        is_same_task = _bool_value(record, "is_same_task_baseline", False)
        is_current_best = _bool_value(record, "is_current_best_family", False)
        if is_same_task:
            baseline_family = "decision_formulation_same_task_baseline"
            current_role = "same_task_reference"
        elif is_current_best:
            baseline_family = "decision_formulation_current_best_reference"
            current_role = "current_best_family_reference"
        else:
            baseline_family = "decision_formulation_uncertainty_aware_reference"
            current_role = "decision_variant_reference"
        row = {
            "domain": _text(record, "domain", default="beauty"),
            "model": model,
            "task": _text(record, "task", default="unknown_task"),
            "baseline_family": baseline_family,
            "baseline_name": method_variant,
            "baseline_layer": "decision_formulation",
            "model_source_group": _model_source_group(model),
            "model_family": "",
            "adapter_path": "",
            "method_family": _text(record, "method_family"),
            "method_variant": method_variant,
            "uncertainty_source": _text(record, "uncertainty_source", "estimator"),
            "evaluation_scope": _text(record, "evaluation_scope"),
            "is_same_task_baseline": is_same_task,
            "is_current_best_family": is_current_best,
            "current_role": current_role,
            "evidence_question": "whether uncertainty-aware decision formulation improves over same-task non-uncertainty references",
            "source_file": str(path),
            "notes": _text(record, "notes", default="Decision-formulation baseline row imported from the Part5 same-task baseline table."),
        }
        row.update(_common_metrics(record))
        rows.append(row)
    return rows


def _build_literature_aligned_rows(path: Path) -> list[dict[str, Any]]:
    df = _safe_read(path)
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        model = _text(record, "model", default="unknown_model")
        model_family = _text(record, "model_family")
        baseline_name = _text(record, "baseline_name", default="unknown_literature_baseline")
        row = {
            "domain": _text(record, "domain", default="beauty"),
            "model": model,
            "task": _text(record, "task", default="unknown_task"),
            "baseline_family": _text(record, "baseline_family", default="literature_aligned_baseline"),
            "baseline_name": baseline_name,
            "baseline_layer": "literature_aligned",
            "model_source_group": _model_source_group(model, model_family),
            "model_family": model_family,
            "adapter_path": _text(record, "adapter_path"),
            "method_family": _text(record, "baseline_family"),
            "method_variant": baseline_name,
            "uncertainty_source": "none",
            "evaluation_scope": "same_candidate_or_pairwise_schema",
            "is_same_task_baseline": True,
            "is_current_best_family": False,
            "current_role": "literature_or_task_aligned_reference",
            "evidence_question": "whether the method can be compared against external or task-aligned formulations under the same schema",
            "source_file": str(path),
            "notes": _text(record, "notes", default="Literature-aligned baseline imported from the Week7 first-round baseline entry."),
        }
        row.update(_common_metrics(record))
        rows.append(row)
    return rows


def build_baseline_matrix(
    estimator_compare_path: str | Path = DEFAULT_ESTIMATOR_COMPARE_PATH,
    decision_baseline_path: str | Path = DEFAULT_DECISION_BASELINE_PATH,
    literature_baseline_path: str | Path = DEFAULT_LITERATURE_BASELINE_PATH,
) -> pd.DataFrame:
    estimator_path = Path(estimator_compare_path)
    decision_path = Path(decision_baseline_path)
    literature_path = Path(literature_baseline_path)

    rows = (
        _build_uncertainty_source_rows(estimator_path)
        + _build_decision_formulation_rows(decision_path)
        + _build_literature_aligned_rows(literature_path)
    )
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return _ensure_columns(matrix)

    matrix = _ensure_columns(matrix)
    matrix = matrix.sort_values(
        by=["baseline_layer", "domain", "task", "model", "baseline_family", "baseline_name"],
        kind="stable",
    ).reset_index(drop=True)
    return matrix


def write_baseline_matrix(matrix: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=False)
    return path
