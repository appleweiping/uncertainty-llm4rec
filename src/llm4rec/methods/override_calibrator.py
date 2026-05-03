"""Learned override calibrators for CU-GR (improve vs harm heads)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SKLEARN_NOTE = "non_final_rule_fallback"


@dataclass
class CalibratorBundle:
    improve: Any
    harm: Any
    feature_names: list[str]
    improve_kind: str
    harm_kind: str
    class_balance: dict[str, Any]
    metrics: dict[str, Any]


def _try_sklearn():
    try:
        from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        return CalibratedClassifierCV, LogisticRegression
    except ImportError:
        return None, None


class _RuleFallback:
    """Non-final constant-probability fallback when sklearn is missing."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _RuleFallback:
        pos = float(np.sum(y > 0.5)) / max(len(y), 1)
        self.p = min(0.99, max(0.01, pos))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        p = self.p
        return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])


def _build_base_estimator():
    _, LogisticRegression = _try_sklearn()
    if LogisticRegression is None:
        return None, "missing_sklearn"
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    lr = LogisticRegression(max_iter=8000, class_weight="balanced", random_state=42, solver="saga", tol=1e-3)
    base = Pipeline([("scaler", StandardScaler()), ("lr", lr)])
    return base, "logreg_scaled"


def _wrap_calibrated(base: Any) -> Any:
    CalibratedClassifierCV, _ = _try_sklearn()
    if CalibratedClassifierCV is None:
        return base
    return CalibratedClassifierCV(base, cv="prefit", method="sigmoid")


def fit_head(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[Any, str, dict[str, float]]:
    """Train on train rows; calibrate on validation rows (prefit)."""
    metrics: dict[str, float] = {}
    if len(np.unique(y_train)) < 2:
        model = _RuleFallback(float(np.mean(y_train))).fit(X_train, y_train)
        return model, "rule_single_class_train", metrics
    base, kind = _build_base_estimator()
    if base is None:
        model = _RuleFallback().fit(X_train, y_train)
        return model, f"rule_{_SKLEARN_NOTE}", metrics
    from sklearn import metrics as skm  # noqa: PLC0415

    base.fit(X_train, y_train)
    calib = _wrap_calibrated(base)
    try:
        calib.fit(X_val, y_val)
    except Exception:
        calib = base
    probs = calib.predict_proba(X_val)[:, 1]
    try:
        metrics["auroc"] = float(skm.roc_auc_score(y_val, probs))
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(skm.average_precision_score(y_val, probs))
    except Exception:
        metrics["auprc"] = float("nan")
    return calib, f"{kind}_sigmoid_calibrated", metrics


def train_bundle(
    X_train: np.ndarray,
    y_improve_train: np.ndarray,
    y_harm_train: np.ndarray,
    X_val: np.ndarray,
    y_improve_val: np.ndarray,
    y_harm_val: np.ndarray,
    feature_names: list[str],
) -> CalibratorBundle:
    n_improve = int(np.sum(y_improve_train > 0.5))
    n_hurt = int(np.sum(y_harm_train > 0.5))
    n_neutral = int(np.sum((y_improve_train < 0.5) & (y_harm_train < 0.5)))
    balance = {
        "n_improve": n_improve,
        "n_hurt": n_hurt,
        "n_neutral": max(0, n_neutral),
        "positive_rate_improve": float(np.mean(y_improve_train)) if len(y_improve_train) else 0.0,
        "positive_rate_harm": float(np.mean(y_harm_train)) if len(y_harm_train) else 0.0,
        "insufficient_improve_positives": n_improve < 20,
    }
    imp_model, imp_kind, imp_m = fit_head(X_train, y_improve_train, X_val, y_improve_val)
    harm_model, harm_kind, harm_m = fit_head(X_train, y_harm_train, X_val, y_harm_val)
    metrics = {"improve": imp_m, "harm": harm_m}
    return CalibratorBundle(
        improve=imp_model,
        harm=harm_model,
        feature_names=list(feature_names),
        improve_kind=imp_kind,
        harm_kind=harm_kind,
        class_balance=balance,
        metrics=metrics,
    )


def predict_heads(bundle: CalibratorBundle, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_i = bundle.improve.predict_proba(X)[:, 1]
    p_h = bundle.harm.predict_proba(X)[:, 1]
    return p_i, p_h


def save_bundle(path: str | Path, bundle: CalibratorBundle) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model.pkl").open("wb") as handle:
        pickle.dump({"improve": bundle.improve, "harm": bundle.harm}, handle, protocol=4)
    meta = {
        "feature_names": bundle.feature_names,
        "improve_kind": bundle.improve_kind,
        "harm_kind": bundle.harm_kind,
        "class_balance": bundle.class_balance,
        "metrics": bundle.metrics,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def load_bundle(path: str | Path) -> CalibratorBundle:
    out = Path(path)
    with (out / "model.pkl").open("rb") as handle:
        blob = pickle.load(handle)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return CalibratorBundle(
        improve=blob["improve"],
        harm=blob["harm"],
        feature_names=list(meta["feature_names"]),
        improve_kind=str(meta.get("improve_kind", "")),
        harm_kind=str(meta.get("harm_kind", "")),
        class_balance=dict(meta.get("class_balance", {})),
        metrics=dict(meta.get("metrics", {})),
    )
