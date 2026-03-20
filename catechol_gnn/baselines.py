from __future__ import annotations

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_gbdt() -> MultiOutputRegressor:
    # Paper hyperparameters
    base = HistGradientBoostingRegressor(
        max_iter=1200,
        learning_rate=0.025,
        max_depth=10,
        min_samples_leaf=5,
    )
    return MultiOutputRegressor(base)


def inverse_variance_ensemble(pred_a: np.ndarray, pred_b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # pred shape: (n_samples, 3)
    var_a = np.var(pred_a, axis=0)
    var_b = np.var(pred_b, axis=0)
    w_a = 1.0 / (var_a + eps)
    w_b = 1.0 / (var_b + eps)
    return (pred_a * w_a + pred_b * w_b) / (w_a + w_b)
