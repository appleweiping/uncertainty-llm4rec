# src/uncertainty/calibration.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class SplitResult:
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


def user_level_split(
    df: pd.DataFrame,
    user_col: str = "user_id",
    valid_ratio: float = 0.5,
    random_state: int = 42
) -> SplitResult:
    """
    Split dataframe by user to avoid leakage.
    All samples from one user go to the same split.
    """
    if user_col not in df.columns:
        raise ValueError(f"Column `{user_col}` not found in dataframe.")

    unique_users = sorted(df[user_col].dropna().unique().tolist())
    if len(unique_users) < 2:
        raise ValueError("Need at least 2 unique users for valid/test split.")

    rng = np.random.default_rng(random_state)
    shuffled_users = unique_users.copy()
    rng.shuffle(shuffled_users)

    n_valid_users = max(1, int(round(len(shuffled_users) * valid_ratio)))
    n_valid_users = min(n_valid_users, len(shuffled_users) - 1)

    valid_users = set(shuffled_users[:n_valid_users])
    test_users = set(shuffled_users[n_valid_users:])

    valid_df = df[df[user_col].isin(valid_users)].copy().reset_index(drop=True)
    test_df = df[df[user_col].isin(test_users)].copy().reset_index(drop=True)

    return SplitResult(valid_df=valid_df, test_df=test_df)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip"
        )
        self.is_fitted = False

    def fit(self, confidence: np.ndarray, correctness: np.ndarray) -> None:
        x = np.asarray(confidence).astype(float)
        y = np.asarray(correctness).astype(float)
        self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted yet.")
        x = np.asarray(confidence).astype(float)
        return np.clip(self.model.predict(x), 0.0, 1.0)


class ConstantCalibrator:
    """
    Fallback calibrator for degenerate valid splits with a single target value.
    """
    def __init__(self, constant: float):
        self.constant = float(np.clip(constant, 0.0, 1.0))
        self.is_fitted = True

    def fit(self, confidence: np.ndarray, correctness: np.ndarray) -> None:
        return None

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        x = np.asarray(confidence).astype(float)
        return np.full(shape=x.shape, fill_value=self.constant, dtype=float)


class PlattCalibrator:
    """
    Logistic regression over confidence -> correctness
    """
    def __init__(self):
        self.model = LogisticRegression()
        self.is_fitted = False

    def fit(self, confidence: np.ndarray, correctness: np.ndarray) -> None:
        x = np.asarray(confidence).astype(float).reshape(-1, 1)
        y = np.asarray(correctness).astype(int)
        self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator is not fitted yet.")
        x = np.asarray(confidence).astype(float).reshape(-1, 1)
        return np.clip(self.model.predict_proba(x)[:, 1], 0.0, 1.0)


def _build_calibrator(method: str):
    method = method.lower()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "platt":
        return PlattCalibrator()
    raise ValueError("method must be either 'isotonic' or 'platt'.")


def fit_calibrator(
    valid_df: pd.DataFrame,
    method: str = "isotonic",
    confidence_col: str = "confidence",
    target_col: str = "is_correct"
):
    if confidence_col not in valid_df.columns:
        raise ValueError(f"Column `{confidence_col}` not found in valid_df.")
    if target_col not in valid_df.columns:
        raise ValueError(f"Column `{target_col}` not found in valid_df.")
    if valid_df.empty:
        raise ValueError("valid_df is empty; cannot fit calibrator.")

    x = valid_df[confidence_col].to_numpy()
    y = valid_df[target_col].to_numpy()

    unique_targets = np.unique(y.astype(int))
    if len(unique_targets) < 2:
        return ConstantCalibrator(constant=float(unique_targets[0]))

    calibrator = _build_calibrator(method)
    calibrator.fit(x, y)
    return calibrator


def fit_isotonic_calibrator(
    valid_df: pd.DataFrame,
    confidence_col: str = "confidence",
    target_col: str = "is_correct"
):
    return fit_calibrator(
        valid_df=valid_df,
        method="isotonic",
        confidence_col=confidence_col,
        target_col=target_col,
    )


def fit_platt_calibrator(
    valid_df: pd.DataFrame,
    confidence_col: str = "confidence",
    target_col: str = "is_correct"
):
    return fit_calibrator(
        valid_df=valid_df,
        method="platt",
        confidence_col=confidence_col,
        target_col=target_col,
    )


def apply_calibrator(
    df: pd.DataFrame,
    calibrator,
    input_col: str = "confidence",
    output_col: str = "calibrated_confidence"
) -> pd.DataFrame:
    out = df.copy()
    if input_col not in out.columns:
        raise ValueError(f"Column `{input_col}` not found in dataframe.")

    out[output_col] = calibrator.predict(out[input_col].to_numpy())
    out[output_col] = out[output_col].astype(float).clip(0.0, 1.0)
    return out


def build_split_metadata(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    return {
        "num_valid_samples": int(len(valid_df)),
        "num_test_samples": int(len(test_df)),
        "num_valid_users": int(valid_df["user_id"].nunique()) if "user_id" in valid_df.columns else -1,
        "num_test_users": int(test_df["user_id"].nunique()) if "user_id" in test_df.columns else -1,
    }
