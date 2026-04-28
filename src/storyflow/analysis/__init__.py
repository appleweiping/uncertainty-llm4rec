"""Analysis modules for observation outputs."""

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
    "append_registry_record",
    "bucket_summary",
    "observation_analysis_markdown",
    "popularity_confidence_slope",
    "reliability_bins",
    "reliability_by_popularity_bucket",
    "risk_case_slices",
    "stable_run_id",
    "summarize_observation_records",
]
