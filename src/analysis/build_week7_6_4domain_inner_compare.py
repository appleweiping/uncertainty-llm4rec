from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DOMAINS = ["beauty", "books", "movies", "electronics"]
SUMMARY_DIR = Path("outputs/summary")
DEFAULT_OUTPUT_CSV = SUMMARY_DIR / "week7_6_4domain_inner_compare.csv"
DEFAULT_OUTPUT_MD = SUMMARY_DIR / "week7_6_4domain_inner_compare.md"
DEFAULT_STATUS_CSV = SUMMARY_DIR / "week7_6_4domain_inner_compare_status.csv"

POINTWISE_EXPS = {
    "beauty": "beauty_deepseek",
    "books": "books_small_deepseek",
    "movies": "movies_small_deepseek",
    "electronics": "electronics_small_deepseek",
}

STRUCTURED_RISK_COMPARE = SUMMARY_DIR / "week6_final_4domain_structured_risk_compare.csv"
FOUR_DOMAIN_COMPARE = SUMMARY_DIR / "week6_magic7_4domain_deepseek_compare.csv"
LITERATURE_COMPARE = SUMMARY_DIR / "week6_final_literature_baseline_compare.csv"
PAIRWISE_SOURCE_TEMPLATES = {
    "plain_win_count": "{domain}_deepseek_pairwise_coverage_plain_to_rank_compare.csv",
    "weighted_win_count": "{domain}_deepseek_pairwise_coverage_to_rank_compare.csv",
}
FIELDNAMES = [
    "domain",
    "model",
    "sample_scope",
    "task",
    "compare_layer",
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
    "next_action",
    "selection_note",
]


def _srpd_summary_path(domain: str, method_name: str) -> Path:
    stage = {
        "SRPD-v1_structured_risk_teacher_sft": "v1",
        "SRPD-v2_uncertainty_weighted_sft": "v2",
        "SRPD-v3_pairwise_preference_enhanced": "v3",
    }[method_name]
    if domain == "beauty":
        legacy = SUMMARY_DIR / f"week7_6_srpd_{stage}_data_summary.csv"
        if legacy.exists():
            return legacy
    return SUMMARY_DIR / f"week7_6_{domain}_srpd_{stage}_data_summary.csv"


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _pairwise_source_path(domain: str, variant: str) -> Path:
    template = PAIRWISE_SOURCE_TEMPLATES[variant]
    return SUMMARY_DIR / template.format(domain=domain)


def _value(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _base_row(domain: str, **kwargs: Any) -> dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "domain": domain,
            "model": "deepseek",
            "sample_scope": "100_sample_compact",
            "status": "missing_artifact",
            "next_action": "locate_or_generate_required_artifact",
        }
    )
    row.update({key: str(value) for key, value in kwargs.items() if key in row and value is not None})
    return row


def _index_structured_rows() -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_rows(STRUCTURED_RISK_COMPARE):
        domain = _value(row, "domain")
        family = _value(row, "method_family")
        if domain and family:
            indexed[(domain, family)] = row
    return indexed


def _index_pointwise_rows() -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in _read_rows(FOUR_DOMAIN_COMPARE):
        if _value(row, "task") == "pointwise_yesno":
            indexed[_value(row, "domain")] = row
    return indexed


def _add_pointwise(rows: list[dict[str, str]], pointwise_rows: dict[str, dict[str, str]], domain: str) -> None:
    record = pointwise_rows.get(domain, {})
    rows.append(
        _base_row(
            domain,
            task="pointwise_diagnosis",
            compare_layer="B",
            method_family="uncertainty_source",
            method_name="verbalized_raw_or_domain_default",
            method_role="domain_uncertainty_diagnosis_reference",
            status="available" if record else "missing_artifact",
            samples=_value(record, "samples"),
            ECE=_value(record, "ECE"),
            Brier=_value(record, "Brier"),
            parse_success_rate=_value(record, "parse_success_rate"),
            source_file=str(FOUR_DOMAIN_COMPARE) if record else "",
            next_action="use_as_uncertainty_diagnosis_layer" if record else "run_pointwise_eval_for_domain",
            selection_note="Pointwise diagnosis supports calibrated uncertainty and risk weighting; it is not the final ranking method.",
        )
    )


def _add_direct_and_structured(
    rows: list[dict[str, str]],
    structured_rows: dict[tuple[str, str], dict[str, str]],
    domain: str,
) -> None:
    direct = structured_rows.get((domain, "direct_candidate_ranking"), {})
    rows.append(
        _base_row(
            domain,
            task="candidate_ranking",
            compare_layer="A",
            method_family="direct_candidate_ranking",
            method_name="direct_candidate_ranking",
            method_role="direct_baseline",
            status="available" if direct else "missing_artifact",
            samples=_value(direct, "samples"),
            **{
                "HR@10": _value(direct, "HR@10"),
                "NDCG@10": _value(direct, "NDCG@10"),
                "MRR": _value(direct, "MRR"),
            },
            changed_ranking_fraction=_value(direct, "changed_ranking_fraction"),
            avg_position_shift=_value(direct, "avg_position_shift"),
            source_file=str(STRUCTURED_RISK_COMPARE) if direct else "",
            next_action="keep_as_same_task_direct_reference" if direct else "run_direct_candidate_ranking",
            selection_note="Direct ranking is the non-uncertainty same-task baseline for each domain.",
        )
    )

    structured = structured_rows.get((domain, "structured_risk_family"), {})
    rows.append(
        _base_row(
            domain,
            task="candidate_ranking",
            compare_layer="C",
            method_family="structured_risk_family",
            method_name=_value(structured, "method_variant", default="nonlinear_structured_risk_rerank"),
            method_role="strongest_handcrafted_uncertainty_aware_baseline",
            status="available" if structured else "missing_artifact",
            samples=_value(structured, "samples"),
            **{
                "HR@10": _value(structured, "HR@10"),
                "NDCG@10": _value(structured, "NDCG@10"),
                "MRR": _value(structured, "MRR"),
            },
            uncertainty_coverage=_value(structured, "uncertainty_coverage"),
            changed_ranking_fraction=_value(structured, "changed_ranking_fraction"),
            avg_position_shift=_value(structured, "avg_position_shift"),
            source_file=str(STRUCTURED_RISK_COMPARE) if structured else "",
            next_action="candidate_teacher_available_for_srpd" if structured else "run_structured_risk_rerank",
            selection_note="Structured risk is the selected hand-crafted uncertainty-aware teacher candidate before SRPD training.",
        )
    )


def _add_linear_penalty(rows: list[dict[str, str]], domain: str) -> None:
    exp = POINTWISE_EXPS[domain]
    source = Path("outputs") / exp / "tables" / "rerank_results.csv"
    record = {}
    for row in _read_rows(source):
        if _value(row, "method") == "uncertainty_aware_rerank":
            record = row
            break
    rows.append(
        _base_row(
            domain,
            task="legacy_uncertainty_rerank",
            compare_layer="C",
            method_family="linear_penalty_family",
            method_name="linear_penalty_lambda_uncertainty",
            method_role="internal_uncertainty_baseline",
            status="available" if record else "missing_artifact",
            samples=_value(record, "num_samples"),
            **{
                "HR@10": _value(record, "HR@10"),
                "NDCG@10": _value(record, "NDCG@10"),
                "MRR": _value(record, "MRR@10", "MRR"),
            },
            coverage=_value(record, "long_tail_coverage@10"),
            source_file=str(source) if record else "",
            next_action="keep_as_legacy_transfer_baseline" if record else "run_legacy_linear_uncertainty_rerank",
            selection_note="Linear penalty is the earliest uncertainty-transfer baseline and should remain below the SR/teacher line.",
        )
    )


def _add_literature_rows(rows: list[dict[str, str]], domain: str) -> None:
    literature_methods = [
        ("candidate_order_rank", "same_task_order_baseline", "heuristic_candidate_order_reference"),
        ("popularity_prior_rank", "literature_aligned_popularity_prior", "heuristic_popularity_reference"),
        ("longtail_prior_rank", "exposure_oriented_longtail_prior", "heuristic_longtail_reference"),
    ]
    literature = [row for row in _read_rows(LITERATURE_COMPARE) if _value(row, "domain") == domain]
    by_method = {_value(row, "method_name"): row for row in literature}
    for method_name, family, role in literature_methods:
        record = by_method.get(method_name, {})
        rows.append(
            _base_row(
                domain,
                task="candidate_ranking",
                compare_layer="F",
                method_family=family,
                method_name=method_name,
                method_role=role,
                status="available" if record else "missing_artifact",
                samples=_value(record, "samples"),
                **{
                    "HR@10": _value(record, "HR@10"),
                    "NDCG@10": _value(record, "NDCG@10"),
                    "MRR": _value(record, "MRR"),
                },
                source_file=str(LITERATURE_COMPARE) if record else "",
                next_action="keep_as_outer_bridge_reference" if record else "run_task_aligned_baseline_for_domain",
                selection_note="Task/literature-aligned compact baseline used to prevent inner methods from comparing only against weak references.",
            )
        )


def _add_pairwise_rows(rows: list[dict[str, str]], domain: str) -> None:
    for variant in PAIRWISE_SOURCE_TEMPLATES:
        source = _pairwise_source_path(domain, variant)
        record = {}
        for row in _read_rows(source):
            if _value(row, "method").startswith("pairwise_to_rank"):
                record = row
                break
        rows.append(
            _base_row(
                domain,
                task="pairwise_to_rank",
                compare_layer="D",
                method_family="pairwise_to_rank",
                method_name=variant,
                method_role="mechanism_baseline",
                status="available" if record else "missing_artifact",
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
                source_file=str(source) if record else "",
                next_action="use_as_srpd_v3_mechanism_signal" if record else "run_deepseek_pairwise_coverage_and_pairwise_to_rank_for_domain",
                selection_note="Pairwise-to-rank is mechanism evidence and a future SRPD-v3 auxiliary signal, not the same family as SR rerank.",
            )
        )


def _add_trainable_rows(rows: list[dict[str, str]], domain: str) -> None:
    domain_pairwise_ready = any(
        row["domain"] == domain and row["method_family"] == "pairwise_to_rank" and row["status"] == "available"
        for row in rows
    )
    rows.append(
        _base_row(
            domain,
            task="candidate_ranking",
            compare_layer="E",
            method_family="ordinary_lora_sft",
            method_name="ordinary_lora_sft",
            method_role="trainable_baseline",
            status="available_cross_backbone_reference" if domain == "beauty" else "pending_training",
            source_file="outputs/summary/week7_5_framework_compare.csv" if domain == "beauty" else "",
            next_action="keep_as_qwen3_trainable_baseline_reference" if domain == "beauty" else "train_ordinary_lora_sft_after_srpd_scope_is_fixed",
            selection_note="Ordinary LoRA SFT is a trainable baseline, not the final SRPD method and not a DeepSeek API row.",
        )
    )

    for method_name in [
        "SRPD-v1_structured_risk_teacher_sft",
        "SRPD-v2_uncertainty_weighted_sft",
        "SRPD-v3_pairwise_preference_enhanced",
    ]:
        record = {}
        source = _srpd_summary_path(domain, method_name)
        srpd_rows = _read_rows(source)
        record = srpd_rows[0] if srpd_rows else {}
        default_status = "teacher_candidate_available"
        default_next_action = "build_domain_srpd_teacher_data_from_structured_risk_then_train"
        if method_name.endswith("enhanced") and domain != "beauty":
            if domain_pairwise_ready:
                default_status = "pairwise_signal_ready"
                default_next_action = "build_domain_srpd_v3_teacher_data_then_train"
            else:
                default_status = "blocked_by_missing_pairwise_source"
                default_next_action = "build_pairwise_coverage_then_srpd_v3_teacher_data"
        rows.append(
            _base_row(
                domain,
                task="candidate_ranking",
                compare_layer="G",
                method_family="SRPD",
                method_name=method_name,
                method_role="ours_final_candidate" if method_name.endswith("enhanced") else "ours_intermediate_candidate",
                status=_value(record, "status", default=default_status),
                samples=_value(record, "matched_rows"),
                source_file=str(source) if record else (str(STRUCTURED_RISK_COMPARE) if default_status == "teacher_candidate_available" else ""),
                next_action="train_and_evaluate_srpd_on_server" if record else default_next_action,
                selection_note="SRPD is the planned trainable method family: SR teacher, uncertainty weighting, and pairwise preference enhancement.",
            )
        )


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    structured_rows = _index_structured_rows()
    pointwise_rows = _index_pointwise_rows()
    for domain in DOMAINS:
        _add_pointwise(rows, pointwise_rows, domain)
        _add_direct_and_structured(rows, structured_rows, domain)
        _add_linear_penalty(rows, domain)
        _add_pairwise_rows(rows, domain)
        _add_literature_rows(rows, domain)
        _add_trainable_rows(rows, domain)
    return rows


def _write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def _status_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain]
        available = sum(1 for row in domain_rows if row["status"] in {"available", "srpd_teacher_data_ready", "available_cross_backbone_reference"})
        missing = sum(1 for row in domain_rows if row["status"] == "missing_artifact")
        pending = sum(1 for row in domain_rows if row["status"] in {"pending_training", "teacher_candidate_available"})
        structured = next((row for row in domain_rows if row["method_family"] == "structured_risk_family"), {})
        pairwise_ready = any(row["task"] == "pairwise_to_rank" and row["status"] == "available" for row in domain_rows)
        srpd_ready = any(row["method_family"] == "SRPD" and row["status"] == "srpd_teacher_data_ready" for row in domain_rows)
        summary.append(
            {
                "domain": domain,
                "total_rows": str(len(domain_rows)),
                "available_or_ready_rows": str(available),
                "missing_rows": str(missing),
                "pending_rows": str(pending),
                "structured_risk_status": structured.get("status", "missing_artifact"),
                "structured_risk_ndcg": structured.get("NDCG@10", ""),
                "pairwise_ready": str(pairwise_ready),
                "srpd_teacher_data_ready": str(srpd_ready),
                "next_priority": (
                    "train_srpd_v1_v2_v3"
                    if srpd_ready and pairwise_ready
                    else "build_pairwise_or_srpd_teacher_data_after_structured_risk"
                ),
            }
        )
    return summary


def save_markdown(rows: list[dict[str, str]], status_rows: list[dict[str, str]], path: Path = DEFAULT_OUTPUT_MD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Week7.6 Four-Domain Inner Compare",
        "",
        "This table extends the Beauty inner compare into Beauty, Books, Movies, and Electronics. It separates available evidence from missing artifacts and keeps SRPD as the trainable final candidate without fabricating metrics.",
        "",
        "## Domain Status",
        "",
        "| domain | available/ready | missing | pending | structured risk | pairwise ready | SRPD teacher-data ready | next priority |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in status_rows:
        lines.append(
            f"| {row['domain']} | {row['available_or_ready_rows']} | {row['missing_rows']} | {row['pending_rows']} | "
            f"{row['structured_risk_status']} ({_fmt(row['structured_risk_ndcg'])}) | {row['pairwise_ready']} | "
            f"{row['srpd_teacher_data_ready']} | {row['next_priority']} |"
        )

    lines.extend(
        [
            "",
            "## Core Ranking Rows",
            "",
            "| domain | method | role | status | NDCG@10 | MRR | next action |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    core_rows = [
        row
        for row in rows
        if row["method_family"]
        in {
            "direct_candidate_ranking",
            "structured_risk_family",
            "pairwise_to_rank",
            "ordinary_lora_sft",
            "SRPD",
            "literature_aligned_popularity_prior",
            "same_task_order_baseline",
        }
    ]
    for row in core_rows:
        lines.append(
            f"| {row['domain']} | {row['method_name']} | {row['method_role']} | {row['status']} | "
            f"{_fmt(row['NDCG@10'])} | {_fmt(row['MRR'])} | {row['next_action']} |"
        )

    lines.extend(
        [
            "",
            "Conclusion: four-domain direct ranking and structured-risk evidence are already aligned. Beauty has the richest mechanism evidence and SRPD teacher-data readiness. The next minimal gap is to build SRPD teacher data and pairwise mechanism evidence for the additional domains before any outer SOTA comparison.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Week7.6 four-domain inner compare.")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--status_csv", default=str(DEFAULT_STATUS_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    status = _status_rows(rows)
    _write_csv(rows, Path(args.output_csv), FIELDNAMES)
    _write_csv(status, Path(args.status_csv), list(status[0].keys()))
    save_markdown(rows, status, Path(args.output_md))
    print(f"Saved four-domain inner compare to: {args.output_csv}")
    print(f"Saved four-domain status to: {args.status_csv}")
    print(f"Saved four-domain markdown to: {args.output_md}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
