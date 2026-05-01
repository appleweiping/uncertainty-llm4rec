from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.runner import run_all


def test_phase6_all_pipeline() -> None:
    result = run_all("configs/experiments/smoke_phase6_all.yaml")
    assert result["run_count"] == 13
    assert result["baseline_methods"] == [
        "ours_uncertainty_guided",
        "ours_ablation_no_uncertainty",
        "ours_ablation_no_grounding",
        "ours_ablation_no_candidate_normalized_confidence",
        "ours_ablation_no_popularity_adjustment",
        "ours_ablation_no_echo_guard",
        "ours_fallback_only",
        "llm_generative",
        "llm_rerank",
        "bm25",
        "popularity",
        "mf",
        "sequential_markov",
    ]
    run_names = {Path(run["run_dir"]).name for run in result["runs"]}
    assert "smoke_phase6_ours_full_seed13" in run_names
    assert "smoke_phase6_ours_fallback_only_seed13" in run_names
    assert "smoke_phase6_llm_generative_seed13" in run_names
    assert "smoke_phase6_sequential_markov_seed13" in run_names
