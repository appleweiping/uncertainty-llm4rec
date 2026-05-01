from __future__ import annotations

import pytest

from llm4rec.evaluation.significance import paired_bootstrap_test, paired_t_test_or_wilcoxon


def test_paired_bootstrap_test_is_deterministic() -> None:
    result = paired_bootstrap_test([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], n_resamples=50, seed=7)
    repeat = paired_bootstrap_test([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], n_resamples=50, seed=7)
    assert result == repeat
    assert result["observed_delta"] == pytest.approx(1 / 3)
    assert result["is_smoke_scaffold"] is True


def test_paired_t_test_or_wilcoxon_has_safe_fallback_or_result() -> None:
    result = paired_t_test_or_wilcoxon([0.0, 1.0], [1.0, 1.0])
    assert result["is_smoke_scaffold"] is True
    assert result["status"] in {"ok", "degenerate", "dependency_unavailable"}


def test_paired_significance_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_test([1.0], [1.0, 0.0])
