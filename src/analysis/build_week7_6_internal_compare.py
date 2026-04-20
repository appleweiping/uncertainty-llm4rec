from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path("outputs/summary")
DEFAULT_OUTPUT_CSV = SUMMARY_DIR / "week7_6_deepseek_beauty_internal_compare.csv"
DEFAULT_OUTPUT_MD = SUMMARY_DIR / "week7_6_deepseek_beauty_internal_compare.md"


FIELDNAMES = [
    "compare_layer",
    "category",
    "domain",
    "model",
    "sample_scope",
    "task",
    "method_family",
    "method_name",
    "method_role",
    "status",
    "samples",
    "HR@10",
    "NDCG@10",
    "MRR",
    "pairwise_accuracy",
    "ECE",
    "Brier",
    "AUROC",
    "coverage",
    "uncertainty_coverage",
    "changed_ranking_fraction",
    "avg_position_shift",
    "parse_success_rate",
    "source_file",
    "selection_note",
]


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _value(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _base_row(**kwargs: Any) -> dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "domain": "beauty",
            "model": "deepseek",
            "sample_scope": "100_sample_compact",
            "status": "available",
        }
    )
    row.update({key: str(value) for key, value in kwargs.items() if key in row and value is not None})
    return row


def _add_direct_and_structured(rows: list[dict[str, str]]) -> None:
    source = "outputs/summary/beauty_deepseek_rank_structured_risk_compare.csv"
    for record in _read_rows(source):
        method = _value(record, "method")
        if method == "direct_candidate_ranking":
            rows.append(
                _base_row(
                    compare_layer="A",
                    category="same_task_ranking_baseline",
                    task="candidate_ranking",
                    method_family="direct_candidate_ranking",
                    method_name="direct_candidate_ranking",
                    method_role="non_uncertainty_same_task_reference",
                    samples=_value(record, "sample_count", "samples"),
                    **{
                        "HR@10": _value(record, "HR@10"),
                        "NDCG@10": _value(record, "NDCG@10"),
                        "MRR": _value(record, "MRR"),
                    },
                    coverage=_value(record, "coverage", "coverage@10"),
                    parse_success_rate=_value(record, "parse_success_rate"),
                    source_file=source,
                    selection_note="Direct DeepSeek candidate ranking is the same-task non-uncertainty reference.",
                )
            )
        elif "structured_risk" in method:
            rows.append(
                _base_row(
                    compare_layer="C",
                    category="decision_formulation_baseline",
                    task="candidate_ranking",
                    method_family="structured_risk_family",
                    method_name=_value(record, "rerank_variant", default="nonlinear_structured_risk_rerank"),
                    method_role="selected_strongest_handcrafted_uncertainty_aware_family",
                    samples=_value(record, "sample_count", "samples"),
                    **{
                        "HR@10": _value(record, "HR@10"),
                        "NDCG@10": _value(record, "NDCG@10"),
                        "MRR": _value(record, "MRR"),
                    },
                    coverage=_value(record, "coverage", "coverage@10"),
                    uncertainty_coverage=_value(record, "uncertainty_coverage", "avg_uncertainty_coverage_rate"),
                    changed_ranking_fraction=_value(record, "changed_ranking_fraction"),
                    avg_position_shift=_value(record, "avg_position_shift"),
                    parse_success_rate=_value(record, "parse_success_rate"),
                    source_file=source,
                    selection_note="Selected as the current best hand-crafted candidate-ranking decision family under the DeepSeek Beauty compact setting.",
                )
            )


def _add_uncertainty_sources(rows: list[dict[str, str]]) -> None:
    source = "outputs/beauty_deepseek/tables/estimator_comparison.csv"
    for record in _read_rows(source):
        rows.append(
            _base_row(
                compare_layer="B",
                category="uncertainty_source_baseline",
                task="pointwise_diagnosis",
                method_family="uncertainty_source",
                method_name=_value(record, "estimator"),
                method_role="uncertainty_signal_definition_and_calibration_reference",
                samples=_value(record, "num_eval_samples", "calib_num_samples"),
                **{
                    "NDCG@10": _value(record, "rerank_NDCG@10", "rank_NDCG@10"),
                    "MRR": _value(record, "rerank_MRR@10", "rank_MRR@10"),
                    "ECE": _value(record, "calib_ece"),
                    "Brier": _value(record, "calib_brier_score"),
                    "AUROC": _value(record, "calib_auroc"),
                },
                source_file=source,
                selection_note="Used to justify calibrated uncertainty as the signal source rather than as a final ranking method.",
            )
        )


def _add_linear_penalty(rows: list[dict[str, str]]) -> None:
    source = "outputs/beauty_deepseek/tables/rerank_results.csv"
    for record in _read_rows(source):
        method = _value(record, "method")
        if method == "uncertainty_aware_rerank":
            rows.append(
                _base_row(
                    compare_layer="C",
                    category="decision_formulation_baseline",
                    task="legacy_uncertainty_rerank",
                    method_family="linear_penalty_family",
                    method_name="linear_penalty_lambda_uncertainty",
                    method_role="first_proof_of_uncertainty_transfer_baseline",
                    samples=_value(record, "num_samples"),
                    **{
                        "HR@10": _value(record, "HR@10"),
                        "NDCG@10": _value(record, "NDCG@10"),
                        "MRR": _value(record, "MRR@10", "MRR"),
                    },
                    coverage=_value(record, "long_tail_coverage@10"),
                    source_file=source,
                    selection_note="Kept as the earliest linear penalty baseline; not selected because it does not improve over its paired baseline in this legacy scope.",
                )
            )


def _add_pairwise(rows: list[dict[str, str]]) -> None:
    sources = [
        ("outputs/summary/beauty_deepseek_pairwise_coverage_plain_to_rank_compare.csv", "plain_win_count"),
        ("outputs/summary/beauty_deepseek_pairwise_coverage_to_rank_compare.csv", "weighted_win_count"),
    ]
    for source, variant in sources:
        for record in _read_rows(source):
            method = _value(record, "method")
            if not method.startswith("pairwise_to_rank"):
                continue
            rows.append(
                _base_row(
                    compare_layer="D",
                    category="pairwise_mechanism_baseline",
                    task="pairwise_to_rank",
                    method_family="pairwise_to_rank",
                    method_name=variant,
                    method_role="mechanism_line_supporting_signal_not_main_candidate_ranking_decision_family",
                    samples=_value(record, "sample_count", "samples"),
                    **{
                        "HR@10": _value(record, "HR@10"),
                        "NDCG@10": _value(record, "NDCG@10"),
                        "MRR": _value(record, "MRR"),
                    },
                    coverage=_value(record, "pairwise_supported_event_fraction"),
                    uncertainty_coverage=_value(record, "uncertainty_coverage", "avg_uncertainty_coverage_rate"),
                    changed_ranking_fraction=_value(record, "changed_ranking_fraction"),
                    avg_position_shift=_value(record, "avg_position_shift"),
                    parse_success_rate=_value(record, "parse_success_rate"),
                    source_file=source,
                    selection_note="High-score mechanism evidence, but treated as a pairwise-derived supporting path rather than the selected main candidate-ranking decision formulation.",
                )
            )


def _add_literature_aligned(rows: list[dict[str, str]]) -> None:
    source = "outputs/summary/week6_final_literature_baseline_compare.csv"
    for record in _read_rows(source):
        method = _value(record, "method_name")
        if method in {"direct_candidate_ranking", "structured_risk_current_best"}:
            continue
        rows.append(
            _base_row(
                compare_layer="F",
                category="literature_aligned_baseline",
                task=_value(record, "task", default="candidate_ranking"),
                method_family=_value(record, "baseline_group"),
                method_name=method,
                method_role="task_aligned_literature_inspired_reference",
                samples=_value(record, "samples"),
                **{
                    "HR@10": _value(record, "HR@10"),
                    "NDCG@10": _value(record, "NDCG@10"),
                    "MRR": _value(record, "MRR"),
                },
                source_file=source,
                selection_note="Compact defensive baseline aligned with prior LLM4Rec evaluation logic; not a full reproduction of external papers.",
            )
        )


def _add_trainable_status(rows: list[dict[str, str]]) -> None:
    srpd_v1_source = "outputs/summary/week7_6_srpd_v1_data_summary.csv"
    srpd_v1_records = _read_rows(srpd_v1_source)
    srpd_v1_status = "pending_next_stage"
    srpd_v1_detail = ""
    if srpd_v1_records:
        record = srpd_v1_records[0]
        srpd_v1_status = _value(record, "status", default="srpd_teacher_data_ready")
        srpd_v1_detail = (
            f" Teacher-data alignment is ready with matched={_value(record, 'matched_rows')}, "
            f"train={_value(record, 'train_rows')}, valid={_value(record, 'valid_rows')}."
        )

    rows.append(
        _base_row(
            compare_layer="E",
            category="trainable_framework_baseline",
            task="candidate_ranking",
            method_family="ordinary_lora_sft",
            method_name="ordinary_lora_sft",
            method_role="trainable_baseline",
            status="not_applicable_in_deepseek_api_compact_setting",
            source_file="outputs/summary/week7_5_framework_compare.csv",
            selection_note="Ordinary LoRA SFT has been demoed with Qwen3 local, but is not part of the DeepSeek API 100-sample internal compare.",
        )
    )
    rows.append(
        _base_row(
            compare_layer="E",
            category="trainable_framework_method_candidate",
            task="candidate_ranking",
            method_family="structured_risk_aware_lora",
            method_name="SR-LoRA / SRPD-v1",
            method_role="ours_structured_risk_teacher_sft",
            status=srpd_v1_status,
            source_file=srpd_v1_source,
            selection_note="Ours v1 should distill structured-risk teacher rankings into LoRA using SFT; metrics remain pending until server training/evaluation runs." + srpd_v1_detail,
        )
    )


def _add_g_option(rows: list[dict[str, str]]) -> None:
    srpd_summaries = {
        "SRPD-v1_structured_risk_teacher_sft": "outputs/summary/week7_6_srpd_v1_data_summary.csv",
        "SRPD-v2_uncertainty_weighted_sft": "outputs/summary/week7_6_srpd_v2_data_summary.csv",
        "SRPD-v3_pairwise_preference_enhanced": "outputs/summary/week7_6_srpd_v3_data_summary.csv",
    }

    def _summary_status(method_name: str, fallback: str) -> tuple[str, str, str]:
        source = srpd_summaries[method_name]
        records = _read_rows(source)
        if not records:
            return fallback, source, ""
        record = records[0]
        matched = _value(record, "matched_rows")
        train_rows = _value(record, "train_rows")
        valid_rows = _value(record, "valid_rows")
        detail = f"teacher-data ready; matched={matched}, train={train_rows}, valid={valid_rows}"
        return _value(record, "status", default="srpd_teacher_data_ready"), source, detail

    v1_status, v1_source, v1_detail = _summary_status("SRPD-v1_structured_risk_teacher_sft", "planned")
    v2_status, v2_source, v2_detail = _summary_status("SRPD-v2_uncertainty_weighted_sft", "planned")
    v3_status, v3_source, v3_detail = _summary_status("SRPD-v3_pairwise_preference_enhanced", "planned_final")

    rows.extend(
        [
            _base_row(
                compare_layer="G",
                category="ours_structured_risk_preference_distillation",
                task="candidate_ranking",
                method_family="SRPD",
                method_name="SRPD-v1_structured_risk_teacher_sft",
                method_role="ours_sr_teacher_distillation",
                status=v1_status,
                source_file=v1_source,
                selection_note="G-v1 uses structured risk as a teacher ranking and trains LoRA with standard SFT over teacher JSON targets. " + v1_detail,
            ),
            _base_row(
                compare_layer="G",
                category="ours_structured_risk_preference_distillation",
                task="candidate_ranking",
                method_family="SRPD",
                method_name="SRPD-v2_uncertainty_weighted_sft",
                method_role="ours_sr_teacher_distillation_with_uncertainty_weights",
                status=v2_status,
                source_file=v2_source,
                selection_note="G-v2 extends SRPD-v1 by weighting samples with calibrated uncertainty, SR/direct disagreement, and risk-sensitive event strength. " + v2_detail,
            ),
            _base_row(
                compare_layer="G",
                category="ours_structured_risk_preference_distillation",
                task="candidate_ranking",
                method_family="SRPD",
                method_name="SRPD-v3_pairwise_preference_enhanced",
                method_role="ours_final_sr_preference_distillation",
                status=v3_status,
                source_file=v3_source,
                selection_note="G-v3 adds pairwise preference supervision on top of structured-risk distillation and is the intended final method candidate. " + v3_detail,
            ),
        ]
    )


def build_week7_6_internal_compare() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    _add_direct_and_structured(rows)
    _add_uncertainty_sources(rows)
    _add_linear_penalty(rows)
    _add_pairwise(rows)
    _add_literature_aligned(rows)
    _add_trainable_status(rows)
    _add_g_option(rows)
    return rows


def save_csv(rows: list[dict[str, str]], path: Path = DEFAULT_OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: str) -> str:
    if value == "":
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value
    if abs(number) >= 10:
        return f"{number:.0f}" if number.is_integer() else f"{number:.3f}"
    return f"{number:.4f}"


def save_markdown(rows: list[dict[str, str]], path: Path = DEFAULT_OUTPUT_MD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_rows = [
        row
        for row in rows
        if row["task"] in {"candidate_ranking", "pairwise_to_rank", "legacy_uncertainty_rerank"}
    ]
    method_rows = sorted(
        method_rows,
        key=lambda row: (
            row["compare_layer"],
            -(float(row["NDCG@10"]) if row["NDCG@10"] else -1),
        ),
    )
    lines = [
        "# Week7.6 DeepSeek Beauty Internal Compare",
        "",
        "This table aligns the internal A-G evidence under the Beauty + DeepSeek + 100-sample compact setting. Completed rows use existing artifacts, while G rows define the next SRPD method family without fabricating metrics.",
        "",
        "| layer | method | role | samples | NDCG@10 | MRR | status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in method_rows:
        lines.append(
            "| {layer} | {method} | {role} | {samples} | {ndcg} | {mrr} | {status} |".format(
                layer=row["compare_layer"],
                method=row["method_name"],
                role=row["method_role"],
                samples=_fmt(row["samples"]),
                ndcg=_fmt(row["NDCG@10"]),
                mrr=_fmt(row["MRR"]),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "Selection note: structured risk is selected as the strongest hand-crafted candidate-ranking decision family. G/SRPD is the planned ultimate method family that will distill structured-risk and pairwise preference signals into LoRA rather than remaining a post-hoc reranker.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_week7_6_internal_compare()
    save_csv(rows)
    save_markdown(rows)
    print(f"Saved Week7.6 internal compare CSV to: {DEFAULT_OUTPUT_CSV}")
    print(f"Saved Week7.6 internal compare markdown to: {DEFAULT_OUTPUT_MD}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
