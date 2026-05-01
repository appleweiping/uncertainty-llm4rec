from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all
from llm4rec.io.artifacts import read_jsonl


def test_ours_ablation_pipeline() -> None:
    result = run_all("configs/experiments/smoke_ours_ablation.yaml")
    assert result["run_count"] == 6
    assert result["baseline_methods"] == [
        "ours_ablation_no_uncertainty",
        "ours_ablation_no_grounding",
        "ours_ablation_no_candidate_normalized_confidence",
        "ours_ablation_no_popularity_adjustment",
        "ours_ablation_no_echo_guard",
        "ours_fallback_only",
    ]
    expected = {
        "smoke_ours_ablation_ours_no_uncertainty_seed13": "uncertainty_policy",
        "smoke_ours_ablation_ours_no_grounding_seed13": "grounding_check",
        "smoke_ours_ablation_ours_no_candidate_normalized_confidence_seed13": "candidate_normalized_confidence",
        "smoke_ours_ablation_ours_no_popularity_adjustment_seed13": "popularity_adjustment",
        "smoke_ours_ablation_ours_no_echo_guard_seed13": "echo_risk_guard",
        "smoke_ours_ablation_ours_fallback_only_seed13": "generation_acceptance",
    }
    for run in result["runs"]:
        run_dir = Path(run["run_dir"])
        assert run_dir.name in expected
        predictions = read_jsonl(run_dir / "predictions.jsonl")
        assert predictions
        assert expected[run_dir.name] in predictions[0]["metadata"]["disabled_components"]
        assert (run_dir / "metrics.json").exists()
