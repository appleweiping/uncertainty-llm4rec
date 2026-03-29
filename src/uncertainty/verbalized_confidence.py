# src/uncertainty/verbalized_confidence.py

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


def _extract_first_number(text: str) -> Optional[float]:
    """
    Extract the first numeric token from a string.
    Supports values like:
    '0.83', '83%', 'confidence: 0.83'
    """
    if text is None:
        return None

    matches = re.findall(r"[-+]?\d*\.?\d+", str(text))
    if not matches:
        return None

    try:
        return float(matches[0])
    except ValueError:
        return None


def normalize_confidence_value(value) -> float:
    """
    Normalize a single confidence value into [0, 1].
    Rules:
    - numeric values in [0,1] stay unchanged
    - percentage-like values in (1,100] become value / 100
    - invalid / missing values fallback to 0.5
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5

    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
    else:
        val = _extract_first_number(str(value))
        if val is None:
            return 0.5

        # if original text contains %, treat as percentage
        if "%" in str(value):
            val = val / 100.0

    # handle percentage-like numeric input such as 83
    if val > 1.0 and val <= 100.0:
        val = val / 100.0

    # clip to [0,1]
    val = max(0.0, min(1.0, float(val)))
    return val


def normalize_confidence_column(
    df: pd.DataFrame,
    input_col: str = "confidence",
    output_col: str = "confidence"
) -> pd.DataFrame:
    """
    Normalize a dataframe confidence column into [0,1].
    """
    out = df.copy()
    if input_col not in out.columns:
        raise ValueError(f"Column `{input_col}` not found in dataframe.")

    out[output_col] = out[input_col].apply(normalize_confidence_value).astype(float)
    return out


def add_uncertainty_from_confidence(
    df: pd.DataFrame,
    confidence_col: str = "calibrated_confidence",
    output_col: str = "uncertainty"
) -> pd.DataFrame:
    """
    uncertainty = 1 - calibrated_confidence
    """
    out = df.copy()
    if confidence_col not in out.columns:
        raise ValueError(f"Column `{confidence_col}` not found in dataframe.")

    out[output_col] = 1.0 - out[confidence_col].astype(float).clip(0.0, 1.0)
    return out