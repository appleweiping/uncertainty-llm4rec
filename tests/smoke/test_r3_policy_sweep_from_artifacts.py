from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.analysis.policy_sweep import PolicyVariant, run_policy_sweep


def test_r3_policy_sweep_from_existing_artifacts(tmp_path: Path) -> None:
    runs = Path("outputs/runs")
    required = runs / "r3_movielens_1m_real_llm_full_candidate500_ours_uncertainty_guided_real_seed13" / "predictions.jsonl"
    if not required.exists():
        pytest.skip("R3 artifacts are not present in this checkout.")
    manifest = run_policy_sweep(
        runs_dir=runs,
        output_dir=tmp_path,
        processed_dir=Path("data/processed/movielens_1m/r2_full_single_dataset"),
        seeds=(13,),
        validation_seed=13,
        confirmation_seeds=(),
        variants=[
            PolicyVariant(
                name="ours_conservative_uncertainty_gate",
                min_accept_confidence=0.95,
                min_grounding_score=1.0,
                enable_rerank=False,
                require_candidate_adherent=True,
                require_candidate_normalized=True,
            )
        ],
    )
    assert (tmp_path / "r3_ours_policy_sweep.csv").exists()
    assert manifest["best_conservative_policy"]["policy"] == "ours_conservative_uncertainty_gate"

