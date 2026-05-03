"""Override calibrator training smoke (requires scikit-learn)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from llm4rec.methods.override_calibrator import predict_heads, train_bundle


def test_train_bundle_runs_small_synthetic():
    rng = np.random.default_rng(0)
    n, d = 80, 12
    X_train = rng.normal(size=(n, d))
    y_imp = (rng.random(n) > 0.7).astype(float)
    y_harm = (rng.random(n) > 0.85).astype(float)
    X_val = rng.normal(size=(n, d))
    y_imp_v = (rng.random(n) > 0.7).astype(float)
    y_harm_v = (rng.random(n) > 0.85).astype(float)
    names = [f"f{j}" for j in range(d)]
    bundle = train_bundle(X_train, y_imp, y_harm, X_val, y_imp_v, y_harm_v, names)
    pi, ph = predict_heads(bundle, X_val)
    assert pi.shape == (n,) and ph.shape == (n,)
    assert bundle.class_balance["n_improve"] >= 0
