from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts.compare_observation_runs import main as compare_main
from storyflow.analysis import (
    compare_observation_summaries,
    observation_comparison_row,
    write_observation_comparison,
)
from storyflow.analysis.observation import read_jsonl


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _analysis_summary(
    *,
    ground_hit: float,
    mean_confidence: float,
    wbc_tau: float,
    grounding_failures: int,
    candidate_rate: float | None = None,
    target_excluded_rate: float | None = None,
) -> dict[str, object]:
    candidate_summary = {
        "candidate_context_available": candidate_rate is not None,
        "rows_with_candidate_context": 10 if candidate_rate is not None else 0,
        "target_in_candidates_count": 0 if target_excluded_rate == 1.0 else 1,
        "target_excluded_from_candidates_rate": target_excluded_rate,
        "generated_in_candidate_set_count": (
            int(candidate_rate * 10) if candidate_rate is not None else 0
        ),
        "generated_in_candidate_set_rate": candidate_rate,
        "grounded_not_in_candidate_set_count": 1 if candidate_rate is not None else 0,
        "ungrounded_with_candidate_context_count": grounding_failures,
        "selected_history_item_rate": 0.1 if candidate_rate is not None else None,
        "mean_selected_candidate_rank": 4.2 if candidate_rate is not None else None,
        "selected_candidate_bucket_counts": {"head": 2, "mid": 3, "tail": 3}
        if candidate_rate is not None
        else {},
        "target_correctness_interpretable_as_recommendation_accuracy": (
            False if target_excluded_rate == 1.0 else True
        ),
    }
    return {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "dry_run": False,
        "api_called": True,
        "count": 10,
        "failed_count": 0,
        "parse_failure_count": 0,
        "provider_failure_count": 0,
        "ground_hit": ground_hit,
        "correctness": 0.0,
        "mean_confidence": mean_confidence,
        "confidence_metrics": {
            "ece": mean_confidence,
            "brier": 0.5,
            "cbu_tau": None,
            "wbc_tau": wbc_tau,
            "tail_underconfidence_gap": None,
        },
        "grounding_summary": {
            "failure_count": grounding_failures,
            "status_counts": {"exact": 10 - grounding_failures, "ungrounded": grounding_failures},
        },
        "quadrant_counts": {
            "wrong_high_confidence": 8,
            "correct_low_confidence": 0,
            "wrong_low_confidence": 2,
            "correct_confident": 0,
        },
        "bucket_summary": {
            "head": {"count": 2},
            "mid": {"count": 5},
            "tail": {"count": 3},
        },
        "candidate_diagnostic_summary": candidate_summary,
    }


def _case_summary() -> dict[str, object]:
    return {
        "taxonomy_counts": {
            "wrong_high_confidence": 8,
            "ungrounded_high_confidence": 2,
        },
        "tag_counts": {
            "self_verified_wrong": 8,
            "generated_more_popular_than_target": 2,
            "wrong_high_confidence_generated_head": 1,
        },
    }


def test_observation_comparison_row_flattens_candidate_and_case_fields() -> None:
    row = observation_comparison_row(
        label="retrieval",
        analysis_summary=_analysis_summary(
            ground_hit=0.9,
            mean_confidence=0.8,
            wbc_tau=0.7,
            grounding_failures=1,
            candidate_rate=0.8,
            target_excluded_rate=1.0,
        ),
        case_review_summary=_case_summary(),
    )

    assert row["label"] == "retrieval"
    assert row["ground_hit"] == 0.9
    assert row["grounding_failure_rate"] == 0.1
    assert row["generated_in_candidate_set_rate"] == 0.8
    assert row["target_correctness_interpretable_as_recommendation_accuracy"] is False
    assert row["case_self_verified_wrong_count"] == 8


def test_compare_observation_summaries_adds_deltas_and_guardrails() -> None:
    comparison = compare_observation_summaries(
        [
            {
                "label": "free_form",
                "analysis_summary": _analysis_summary(
                    ground_hit=0.2,
                    mean_confidence=0.78,
                    wbc_tau=0.97,
                    grounding_failures=8,
                ),
            },
            {
                "label": "retrieval_context",
                "analysis_summary": _analysis_summary(
                    ground_hit=0.9,
                    mean_confidence=0.82,
                    wbc_tau=0.96,
                    grounding_failures=1,
                    candidate_rate=0.8,
                    target_excluded_rate=1.0,
                ),
            },
            {
                "label": "catalog_constrained",
                "analysis_summary": _analysis_summary(
                    ground_hit=0.7,
                    mean_confidence=0.65,
                    wbc_tau=0.76,
                    grounding_failures=3,
                    candidate_rate=0.7,
                    target_excluded_rate=1.0,
                ),
            },
        ],
        source_label="unit",
    )

    rows = comparison["rows"]
    assert rows[1]["delta_ground_hit_vs_first"] == 0.7
    assert comparison["diagnostic_takeaways"]["highest_ground_hit_label"] == "retrieval_context"
    assert comparison["diagnostic_takeaways"]["lowest_wbc_tau_label"] == "catalog_constrained"
    assert comparison["claim_guardrails"]["recommendation_accuracy_comparison_allowed"] is False
    assert comparison["claim_guardrails"]["is_paper_result"] is False


def test_write_observation_comparison_and_cli() -> None:
    workspace = _workspace("comparison")
    analysis = workspace / "analysis.json"
    case = workspace / "case.json"
    analysis.write_text(
        json.dumps(
            _analysis_summary(
                ground_hit=0.9,
                mean_confidence=0.8,
                wbc_tau=0.7,
                grounding_failures=1,
                candidate_rate=0.8,
                target_excluded_rate=1.0,
            )
        ),
        encoding="utf-8",
    )
    case.write_text(json.dumps(_case_summary()), encoding="utf-8")

    manifest = write_observation_comparison(
        runs=[
            {
                "label": "retrieval",
                "analysis_summary": json.loads(analysis.read_text(encoding="utf-8")),
                "case_review_summary": json.loads(case.read_text(encoding="utf-8")),
            }
        ],
        output_dir=workspace / "direct",
        source_label="unit-direct",
    )
    assert Path(manifest["summary"]).exists()
    assert read_jsonl(manifest["rows"])[0]["label"] == "retrieval"

    cli_output = workspace / "cli"
    code = compare_main(
        [
            "--run",
            f"retrieval={analysis},{case}",
            "--output-dir",
            str(cli_output),
            "--source-label",
            "unit-cli",
        ]
    )
    assert code == 0
    assert (cli_output / "comparison_summary.json").exists()
    assert (cli_output / "comparison_table.csv").exists()
    assert (cli_output / "report.md").exists()
