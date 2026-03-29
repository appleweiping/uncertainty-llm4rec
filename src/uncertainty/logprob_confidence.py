# src/uncertainty/logprob_confidence.py

from __future__ import annotations

import numpy as np


def compute_logprob_confidence(logprobs):
    """
    logprobs: list of token log probabilities
    """
    if logprobs is None or len(logprobs) == 0:
        return 0.5

    probs = np.exp(logprobs)
    return float(np.mean(probs))