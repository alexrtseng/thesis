from __future__ import annotations

from typing import Dict

import numpy as np
from darts import TimeSeries

# Basic metric functions operating on numpy arrays


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Avoid division by zero: mask out zeros
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)


def _series_to_np(series: TimeSeries) -> np.ndarray:
    # Flatten in case of multivariate (should be univariate for target)
    return series.values(copy=False).astype(np.float32).reshape(-1)


def per_step_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
    max_steps: int | None = None,
    prefix: str = "val",
) -> Dict[str, float]:
    """Compute per-timestep metrics for each horizon step up to max_steps.

    For step k (1-indexed), metrics are computed only on that single timestep,
    comparing predicted[k-1:k] vs actual[k-1:k]. This matches the requested
    [k:k+1] slicing behavior.

    Parameters
    ----------
    actual : TimeSeries
        Ground truth validation segment.
    predicted : TimeSeries
        Model predictions aligned from the first forecasted timestep.
    max_steps : int | None
        Limit number of steps (defaults to length of shortest series).
    prefix : str
        Prefix used when forming metric keys.

    Returns
    -------
    Dict[str, float]
        Mapping of metric names to values, e.g. 'val_rmse_step_3': value for only step 3.
    """
    y_true = _series_to_np(actual)
    y_pred = _series_to_np(predicted)
    n = min(len(y_true), len(y_pred))
    if max_steps is not None:
        n = min(n, max_steps)

    out: Dict[str, float] = {}
    for k in range(1, n + 1):
        yt = y_true[k - 1 : k]
        yp = y_pred[k - 1 : k]
        out[f"{prefix}_mse_step_{k}"] = mse(yt, yp)
        out[f"{prefix}_rmse_step_{k}"] = rmse(yt, yp)
        out[f"{prefix}_mae_step_{k}"] = mae(yt, yp)
        out[f"{prefix}_mape_step_{k}"] = mape(yt, yp)

    # Optionally include full-horizon aggregates for convenience
    out[f"{prefix}_mse_full"] = mse(y_true[:n], y_pred[:n])
    out[f"{prefix}_rmse_full"] = rmse(y_true[:n], y_pred[:n])
    out[f"{prefix}_mae_full"] = mae(y_true[:n], y_pred[:n])
    out[f"{prefix}_mape_full"] = mape(y_true[:n], y_pred[:n])
    return out
