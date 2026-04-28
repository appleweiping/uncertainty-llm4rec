from __future__ import annotations

import math

import pytest

from storyflow.grounding import ground_title
from storyflow.metrics import (
    brier_score,
    cbu_tau,
    expected_calibration_error,
    ground_hit_rate,
    tail_underconfidence_gap,
    wbc_tau,
)
from storyflow.schemas import PopularityBucket
from tests.fixtures.synthetic_observation import synthetic_catalog


def test_expected_calibration_error() -> None:
    ece = expected_calibration_error(
        [0.9, 0.8, 0.2, 0.1],
        [1, 1, 0, 0],
        n_bins=2,
    )

    assert ece == pytest.approx(0.15)


def test_brier_score() -> None:
    score = brier_score([0.9, 0.2], [1, 0])

    assert score == pytest.approx(((0.1**2) + (0.2**2)) / 2)


def test_cbu_and_wbc_tau() -> None:
    probabilities = [0.9, 0.4, 0.8, 0.2]
    labels = [1, 1, 0, 0]

    assert cbu_tau(probabilities, labels, tau=0.5) == pytest.approx(0.5)
    assert wbc_tau(probabilities, labels, tau=0.7) == pytest.approx(0.5)
    assert math.isnan(cbu_tau([0.1, 0.2], [0, 0], tau=0.5))
    assert math.isnan(wbc_tau([0.9, 0.8], [1, 1], tau=0.5))


def test_ground_hit_rate() -> None:
    grounded = ground_title("Primer", synthetic_catalog())
    missing = ground_title("Not In Catalog", synthetic_catalog())

    assert ground_hit_rate([grounded, missing]) == pytest.approx(0.5)


def test_tail_underconfidence_gap() -> None:
    gap = tail_underconfidence_gap(
        [0.9, 0.6, 0.2, 0.8],
        [1, 1, 1, 0],
        [
            PopularityBucket.HEAD,
            PopularityBucket.TAIL,
            PopularityBucket.TAIL,
            PopularityBucket.HEAD,
        ],
    )

    assert gap == pytest.approx(0.9 - 0.4)
