"""Analysis modules for observation outputs."""

from storyflow.analysis.case_review import (
    case_review_markdown,
    review_observation_cases,
    summarize_case_review,
)
from storyflow.analysis.grounding_diagnostics import (
    analyze_grounding_diagnostics,
    catalog_grounding_summary,
    classify_grounding_failure,
    duplicate_title_groups,
    grounding_diagnostics_markdown,
    grounding_failure_review,
    grounding_margin_summary,
)
from storyflow.analysis.observation import (
    analyze_observation_run,
    bucket_summary,
    observation_analysis_markdown,
    popularity_confidence_slope,
    reliability_bins,
    reliability_by_popularity_bucket,
    risk_case_slices,
    summarize_observation_records,
)
from storyflow.analysis.run_registry import append_registry_record, stable_run_id

__all__ = [
    "analyze_observation_run",
    "analyze_grounding_diagnostics",
    "append_registry_record",
    "bucket_summary",
    "catalog_grounding_summary",
    "case_review_markdown",
    "classify_grounding_failure",
    "duplicate_title_groups",
    "grounding_diagnostics_markdown",
    "grounding_failure_review",
    "grounding_margin_summary",
    "observation_analysis_markdown",
    "popularity_confidence_slope",
    "reliability_bins",
    "reliability_by_popularity_bucket",
    "review_observation_cases",
    "risk_case_slices",
    "stable_run_id",
    "summarize_case_review",
    "summarize_observation_records",
]
