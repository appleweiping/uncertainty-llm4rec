from __future__ import annotations

from llm4rec.experiments.config import load_config
from llm4rec.methods.ablation import resolve_ablation_settings


def test_ours_ablation_configs_disable_expected_components() -> None:
    expected = {
        "configs/methods/ours_uncertainty_guided.yaml": ("full", []),
        "configs/methods/ours_ablation_no_uncertainty.yaml": ("no_uncertainty", ["uncertainty_policy"]),
        "configs/methods/ours_ablation_no_grounding.yaml": ("no_grounding", ["grounding_check"]),
        "configs/methods/ours_ablation_no_candidate_normalized_confidence.yaml": (
            "no_candidate_normalized_confidence",
            ["candidate_normalized_confidence"],
        ),
        "configs/methods/ours_ablation_no_popularity_adjustment.yaml": (
            "no_popularity_adjustment",
            ["popularity_adjustment"],
        ),
        "configs/methods/ours_ablation_no_echo_guard.yaml": ("no_echo_guard", ["echo_risk_guard"]),
        "configs/methods/ours_fallback_only.yaml": (
            "fallback_only",
            [
                "generation_acceptance",
                "uncertainty_policy",
                "grounding_check",
                "candidate_normalized_confidence",
                "popularity_adjustment",
                "echo_risk_guard",
            ],
        ),
    }
    for path, (variant, disabled) in expected.items():
        settings = resolve_ablation_settings(load_config(path))
        assert settings.variant == variant
        assert settings.disabled_components == disabled


def test_smoke_ablation_experiment_lists_phase6_variants() -> None:
    config = load_config("configs/experiments/smoke_ours_ablation.yaml")
    assert config["baselines"] == [
        "ours_ablation_no_uncertainty",
        "ours_ablation_no_grounding",
        "ours_ablation_no_candidate_normalized_confidence",
        "ours_ablation_no_popularity_adjustment",
        "ours_ablation_no_echo_guard",
        "ours_fallback_only",
    ]
