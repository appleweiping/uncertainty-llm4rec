# src/uncertainty/consistency_confidence.py

from __future__ import annotations

import numpy as np


def compute_consistency_confidence(predictions):
    """
    predictions: list of dicts with 'recommend'
    """
    votes = [1 if p["recommend"] == "yes" else 0 for p in predictions]

    if len(votes) == 0:
        return 0.5

    return float(np.mean(votes))


def compute_consistency_uncertainty(predictions):
    """
    variance-based uncertainty
    """
    votes = [1 if p["recommend"] == "yes" else 0 for p in predictions]

    if len(votes) == 0:
        return 0.5

    return float(np.var(votes))