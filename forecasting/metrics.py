from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Union

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


ArrayLike = Union[np.ndarray, Sequence[float], List[float]]


def _to_1d_np(x: Union[TimeSeries, ArrayLike]) -> np.ndarray:
    if isinstance(x, TimeSeries):
        # Flatten in case of multivariate (should be univariate for target)
        return x.values(copy=False).astype(np.float32).reshape(-1)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr


def per_step_metrics(
    actual: Union[TimeSeries, ArrayLike],
    predicted: Union[TimeSeries, ArrayLike],
    max_steps: int | None = None,
    prefix: str = "val",
) -> Dict[str, float]:
    """Compute per-timestep metrics for each horizon step up to max_steps.

    Accepts either Darts TimeSeries or 1D array-like for actual/predicted.
    For step k (1-indexed), metrics are computed only on that single timestep.
    """
    y_true = _to_1d_np(actual)
    y_pred = _to_1d_np(predicted)
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


def panel_metrics_from_forecasts(
    val_y: TimeSeries,
    forecasts: Iterable[TimeSeries],
    *,
    prefix: str = "val",
    horizon: int,
) -> Dict[str, float]:
    """Aggregate metrics across a panel of historical multi-step forecasts.

    For each forecast p (length L <= horizon), we align to val_y by timestamp,
    compute residuals p - y_true, and then compute per-step and overall metrics
    across all available (true, pred) pairs.
    """
    # Collect per-step residuals and overall residuals
    step_residuals: List[List[float]] = [[] for _ in range(horizon)]
    all_residuals: List[float] = []

    mean_err_all_series: List[float] = []
    mean_err_first12_series: List[float] = []
    mean_err_remaining_series: List[float] = []

    for p in forecasts:
        true_slice = val_y.slice(p.start_time(), p.end_time())
        y_true = _to_1d_np(true_slice)
        y_pred = _to_1d_np(p)
        L = min(len(y_true), len(y_pred), horizon)
        if L <= 0:
            mean_err_all_series.append(float("nan"))
            mean_err_first12_series.append(float("nan"))
            mean_err_remaining_series.append(float("nan"))
            continue

        errs = (y_pred[:L] - y_true[:L]).astype(np.float32)

        # per-forecast summaries
        mean_err_all_series.append(float(np.nanmean(errs)))
        first_n = min(12, L)
        mean_err_first12_series.append(float(np.nanmean(errs[:first_n])))
        mean_err_remaining_series.append(
            float(np.nanmean(errs[first_n:])) if L > first_n else float("nan")
        )

        all_residuals.extend(errs.tolist())
        for k in range(L):
            step_residuals[k].append(float(errs[k]))

    out: Dict[str, float] = {}

    # per-step metrics for steps with data
    for k in range(horizon):
        res = np.asarray(step_residuals[k], dtype=np.float32)
        if res.size == 0:
            continue
        # Treat residuals vs zero to get RMSE/MAE without reconstructing pairs
        out[f"{prefix}_mse_step_{k + 1}"] = float(np.mean(res**2))
        out[f"{prefix}_rmse_step_{k + 1}"] = float(
            np.sqrt(out[f"{prefix}_mse_step_{k + 1}"])
        )
        out[f"{prefix}_mae_step_{k + 1}"] = float(np.mean(np.abs(res)))

    # overall aggregates
    all_res = np.asarray(all_residuals, dtype=np.float32)
    if all_res.size:
        out[f"{prefix}_mse_full"] = float(np.mean(all_res**2))
        out[f"{prefix}_rmse_full"] = float(np.sqrt(out[f"{prefix}_mse_full"]))
        out[f"{prefix}_mae_full"] = float(np.mean(np.abs(all_res)))
    else:
        out[f"{prefix}_mse_full"] = float("nan")
        out[f"{prefix}_rmse_full"] = float("nan")
        out[f"{prefix}_mae_full"] = float("nan")

    # legacy-style summary means
    out["mean_error_all"] = (
        float(np.nanmean(mean_err_all_series)) if mean_err_all_series else float("nan")
    )
    out["mean_error_first12"] = (
        float(np.nanmean(mean_err_first12_series))
        if mean_err_first12_series
        else float("nan")
    )
    out["mean_error_remaining"] = (
        float(np.nanmean(mean_err_remaining_series))
        if mean_err_remaining_series
        else float("nan")
    )

    return out
