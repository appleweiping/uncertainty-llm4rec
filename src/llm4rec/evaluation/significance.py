"""Significance-test scaffolds for paired recommendation runs."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import Any


def paired_bootstrap_test(
    baseline_scores: list[float],
    challenger_scores: list[float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    _validate_paired_scores(baseline_scores, challenger_scores)
    if not baseline_scores:
        return {
            "test": "paired_bootstrap",
            "status": "no_data",
            "observed_delta": 0.0,
            "p_value": None,
            "ci95": [0.0, 0.0],
            "n": 0,
            "is_smoke_scaffold": True,
        }
    observed = mean(challenger_scores) - mean(baseline_scores)
    rng = random.Random(seed)
    deltas = []
    n = len(baseline_scores)
    for _ in range(max(1, int(n_resamples))):
        indices = [rng.randrange(n) for _item in range(n)]
        baseline_mean = mean(baseline_scores[index] for index in indices)
        challenger_mean = mean(challenger_scores[index] for index in indices)
        deltas.append(challenger_mean - baseline_mean)
    p_value = sum(1 for delta in deltas if abs(delta) >= abs(observed)) / len(deltas)
    ordered = sorted(deltas)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {
        "test": "paired_bootstrap",
        "status": "ok",
        "observed_delta": observed,
        "p_value": p_value,
        "ci95": [lower, upper],
        "n": n,
        "is_smoke_scaffold": True,
    }


def paired_t_test_or_wilcoxon(
    baseline_scores: list[float],
    challenger_scores: list[float],
) -> dict[str, Any]:
    _validate_paired_scores(baseline_scores, challenger_scores)
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return {
            "test": "paired_t_test_or_wilcoxon",
            "status": "dependency_unavailable",
            "warning": "scipy is unavailable; no parametric significance test was run",
            "n": len(baseline_scores),
            "is_smoke_scaffold": True,
        }
    if not baseline_scores:
        return {"test": "paired_t_test", "status": "no_data", "p_value": None, "n": 0, "is_smoke_scaffold": True}
    differences = [challenger - baseline for baseline, challenger in zip(baseline_scores, challenger_scores)]
    if len(differences) >= 2 and stdev(differences) > 0:
        result = stats.ttest_rel(challenger_scores, baseline_scores)
        return {
            "test": "paired_t_test",
            "status": "ok",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": len(differences),
            "is_smoke_scaffold": True,
        }
    return {
        "test": "paired_t_test",
        "status": "degenerate",
        "p_value": None,
        "warning": "paired differences are constant; no significance claim is made",
        "n": len(differences),
        "is_smoke_scaffold": True,
    }


def _validate_paired_scores(baseline_scores: list[float], challenger_scores: list[float]) -> None:
    if len(baseline_scores) != len(challenger_scores):
        raise ValueError("paired significance tests require equal-length score lists")
    for value in [*baseline_scores, *challenger_scores]:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("paired significance scores must be finite numeric values")
