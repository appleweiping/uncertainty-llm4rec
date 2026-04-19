from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BATCH_STATUS_PATH = Path("outputs/summary/week7_day5_batch_status.csv")
DEFAULT_BASELINE_MATRIX_PATH = Path("outputs/summary/week7_day4_baseline_matrix.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/summary/week7_day5_medium_scale_summary.csv")

OUTPUT_COLUMNS = [
    "source_type",
    "domain",
    "model",
    "task",
    "method_family",
    "method_variant",
    "baseline_family",
    "is_current_best_family",
    "model_source_group",
    "model_family",
    "adapter_path",
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
    "execution_status",
    "eval_ready",
    "prediction_ready",
    "output_dir",
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


def _read_first_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _metrics_for_batch_row(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
    output_dir = Path(str(record.get("output_dir", "")))
    task = str(record.get("task", "")).strip().lower()
    if task == "pointwise_yesno":
        metrics_path = output_dir / "tables" / "diagnostic_metrics.csv"
        metrics = _read_first_metrics(metrics_path)
        return (
            {
                "samples": _value(metrics, "num_samples", "sample_count", "samples"),
                "ECE": _value(metrics, "ece", "ECE"),
                "Brier": _value(metrics, "brier_score", "Brier"),
                "parse_success_rate": _value(metrics, "parse_success_rate"),
            },
            str(metrics_path),
        )
    if task == "candidate_ranking":
        metrics_path = output_dir / "tables" / "ranking_metrics.csv"
        metrics = _read_first_metrics(metrics_path)
        return (
            {
                "samples": _value(metrics, "sample_count", "samples"),
                "HR@10": _value(metrics, "HR@10"),
                "NDCG@10": _value(metrics, "NDCG@10"),
                "MRR": _value(metrics, "MRR"),
                "coverage": _value(metrics, "coverage@10", "coverage"),
                "head_exposure": _value(metrics, "head_exposure_ratio@10", "head_exposure"),
                "longtail_coverage": _value(metrics, "longtail_coverage@10", "longtail_coverage"),
                "parse_success_rate": _value(metrics, "parse_success_rate"),
            },
            str(metrics_path),
        )
    if task == "pairwise_preference":
        metrics_path = output_dir / "tables" / "pairwise_metrics.csv"
        metrics = _read_first_metrics(metrics_path)
        return (
            {
                "samples": _value(metrics, "sample_count", "samples"),
                "pairwise_accuracy": _value(metrics, "pairwise_accuracy"),
                "parse_success_rate": _value(metrics, "parse_success_rate"),
            },
            str(metrics_path),
        )
    if task in {"candidate_ranking_rerank", "rank_rerank", "rerank"}:
        metrics_path = output_dir / "tables" / "rerank_results.csv"
        df = _safe_read(metrics_path)
        if not df.empty:
            variant = str(record.get("method_variant", ""))
            match = df[df.get("rerank_variant", pd.Series(dtype=str)).astype(str) == variant]
            metrics = (match.iloc[0] if not match.empty else df.iloc[-1]).to_dict()
        else:
            metrics = {}
        return (
            {
                "samples": _value(metrics, "sample_count", "samples"),
                "HR@10": _value(metrics, "HR@10"),
                "NDCG@10": _value(metrics, "NDCG@10"),
                "MRR": _value(metrics, "MRR"),
                "coverage": _value(metrics, "coverage@10", "coverage"),
                "head_exposure": _value(metrics, "head_exposure_ratio", "head_exposure_ratio@10", "head_exposure"),
                "longtail_coverage": _value(metrics, "longtail_coverage", "longtail_coverage@10"),
            },
            str(metrics_path),
        )
    return {}, ""


def _model_source_group(model: str, method_family: str) -> str:
    text = f"{model} {method_family}".lower()
    if "local" in text or "llama" in text or "hf" in text:
        return "local_hf_main_model_group"
    return "official_api_observation_group"


def _batch_rows(batch_status_path: Path) -> list[dict[str, Any]]:
    status_df = _safe_read(batch_status_path)
    rows: list[dict[str, Any]] = []
    for record in status_df.to_dict(orient="records"):
        metrics, metrics_path = _metrics_for_batch_row(record)
        method_family = str(record.get("method_family", ""))
        model = str(record.get("model", ""))
        row = {
            "source_type": "server_local_medium_batch",
            "domain": record.get("domain"),
            "model": model,
            "task": record.get("task"),
            "method_family": method_family,
            "method_variant": record.get("method_variant"),
            "baseline_family": "",
            "is_current_best_family": record.get("is_current_best_family", False),
            "model_source_group": _model_source_group(model, method_family),
            "model_family": "local_hf_base_only",
            "adapter_path": "",
            "execution_status": record.get("status"),
            "eval_ready": record.get("eval_ready"),
            "prediction_ready": record.get("prediction_ready"),
            "output_dir": record.get("output_dir"),
            "source_file": metrics_path or str(batch_status_path),
            "notes": "Medium-scale batch row; metrics are filled only after the server run and evaluation artifacts exist.",
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _baseline_reference_rows(baseline_matrix_path: Path) -> list[dict[str, Any]]:
    baseline_df = _safe_read(baseline_matrix_path)
    if baseline_df.empty:
        return []

    keep_layers = {"decision_formulation", "literature_aligned"}
    baseline_df = baseline_df[baseline_df["baseline_layer"].astype(str).isin(keep_layers)].copy()
    rows: list[dict[str, Any]] = []
    for record in baseline_df.to_dict(orient="records"):
        rows.append(
            {
                "source_type": "baseline_reference_matrix",
                "domain": record.get("domain"),
                "model": record.get("model"),
                "task": record.get("task"),
                "method_family": record.get("method_family"),
                "method_variant": record.get("method_variant"),
                "baseline_family": record.get("baseline_family"),
                "is_current_best_family": record.get("is_current_best_family", False),
                "model_source_group": record.get("model_source_group"),
                "model_family": record.get("model_family"),
                "adapter_path": record.get("adapter_path"),
                "samples": record.get("samples"),
                "HR@10": record.get("HR@10"),
                "NDCG@10": record.get("NDCG@10"),
                "MRR": record.get("MRR"),
                "pairwise_accuracy": record.get("pairwise_accuracy"),
                "ECE": record.get("ECE"),
                "Brier": record.get("Brier"),
                "coverage": record.get("coverage"),
                "head_exposure": record.get("head_exposure"),
                "longtail_coverage": record.get("longtail_coverage"),
                "parse_success_rate": record.get("parse_success_rate"),
                "execution_status": "reference_available",
                "eval_ready": True,
                "prediction_ready": pd.NA,
                "output_dir": "",
                "source_file": str(baseline_matrix_path),
                "notes": record.get("notes"),
            }
        )
    return rows


def build_medium_scale_summary(
    batch_status_path: str | Path = DEFAULT_BATCH_STATUS_PATH,
    baseline_matrix_path: str | Path = DEFAULT_BASELINE_MATRIX_PATH,
) -> pd.DataFrame:
    rows = _batch_rows(Path(batch_status_path)) + _baseline_reference_rows(Path(baseline_matrix_path))
    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS]


def write_medium_scale_summary(
    df: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
