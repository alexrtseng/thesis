from typing import Dict

import gurobipy as gp
import numpy as np
import pandas as pd
from darts import TimeSeries

from deterministic.single_market_battery import (
    DEFAULT_BATTERY,
    deterministic_arbitrage_opt,
)
from deterministic.warm_start_arb_solver import (
    build_battery_model,
    set_objective,
    update_initial_charge,
)


# Basic metric functions operating on numpy arrays
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _short_horizon_pred_performance(
    preds: list[TimeSeries], prices_df: pd.DataFrame, hf_horizon: int
) -> pd.DataFrame:
    # build prices_df_structure: hf_horizon 5-min steps then remaining day hourly
    df = prices_df.copy()
    start_time = prices_df.index[0]
    assert len(prices_df) == len(preds)
    start: pd.Timestamp = pd.to_datetime(start_time)
    hf_index = pd.date_range(start=start, periods=hf_horizon + 1, freq="5min")
    day_end = start + pd.Timedelta(days=1)
    hourly_start = (
        hf_index[-1] + pd.Timedelta(minutes=5) if len(hf_index) > 0 else start
    )
    if hourly_start >= day_end:
        hourly_index = pd.DatetimeIndex([])
    else:
        hourly_index = pd.date_range(start=hourly_start, end=day_end, freq="h")

    prices_index = hf_index.append(hourly_index)
    df["lmp_lf_avg"] = df["lmp"].rolling(window=13, center=True, min_periods=1).mean()
    avg_lmps = df["lmp_lf_avg"]

    pred_arrays = []
    for i, pred in enumerate(preds[: -24 * 12]):  # to avoid running out of data
        arr = np.ndarray(len(prices_index))
        arr[0] = prices_df.loc[start_time + pd.Timedelta(minutes=5 * i), "lmp"]
        arr[1 : hf_horizon + 1] = pred[:hf_horizon].values().reshape(-1)
        shifted_hourly_index = hourly_index + pd.Timedelta(minutes=5 * i)
        if len(hourly_index) > 0:
            # align avg_lmps to the shifted hourly index; missing entries become NaN
            hourly_values = avg_lmps.reindex(shifted_hourly_index).to_numpy(dtype=float)
            arr[hf_horizon + 1 :] = hourly_values
        else:
            # nothing to fill for hourly part
            arr[hf_horizon + 1 :] = np.array([], dtype=float)

        pred_arrays.append(arr)

    model, soe, charge, discharge, times, dt_vec, init_soe_constr = build_battery_model(
        prices_index=prices_index, battery=DEFAULT_BATTERY, requires_equivalent_soe=True
    )
    current_soe = DEFAULT_BATTERY.initial_charge_mwh
    charge_decisions = []
    discharge_decisions = []
    for i, arr in enumerate(pred_arrays):
        set_objective(model, charge, discharge, times, arr, dt_vec)
        update_initial_charge(model, init_soe_constr, soe, current_soe)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("MPC step did not reach optimal solution")
        current_soe = float(soe.X[1])
        charge_decisions.append(float(charge.X[0]))
        discharge_decisions.append(float(discharge.X[0]))

    df["charge_mw"] = pd.Series(charge_decisions, index=prices_df.index[: -24 * 12])
    df["discharge_mw"] = pd.Series(
        discharge_decisions, index=prices_df.index[: -24 * 12]
    )
    return df


def short_horizon_pred_performance(
    preds: list[TimeSeries], preds_df: pd.DataFrame
) -> tuple[pd.DataFrame, Dict[str, float]]:
    start = preds[0].time_index[0] - pd.Timedelta(minutes=5)
    end = preds[-1].time_index[0] - pd.Timedelta(minutes=5)
    df = preds_df.loc[start:end].copy()
    df = df[["actual"]].rename(columns={"actual": "lmp"}, inplace=False)
    df.dropna(subset=["lmp"], inplace=True)
    perf_decisions, _ = deterministic_arbitrage_opt(
        prices_df=df,
        require_equivalent_soe=True,
    )
    for_24_decisions = _short_horizon_pred_performance(preds, df, 24)[
        ["charge_mw", "discharge_mw", "lmp"]
    ]
    for_12_decisions = _short_horizon_pred_performance(preds, df, 12)[
        ["charge_mw", "discharge_mw"]
    ]
    for_9_decisions = _short_horizon_pred_performance(preds, df, 9)[
        ["charge_mw", "discharge_mw"]
    ]
    for_6_decisions = _short_horizon_pred_performance(preds, df, 6)[
        ["charge_mw", "discharge_mw"]
    ]
    for_3_decisions = _short_horizon_pred_performance(preds, df, 3)[
        ["charge_mw", "discharge_mw"]
    ]
    for_1_decisions = _short_horizon_pred_performance(preds, df, 1)[
        ["charge_mw", "discharge_mw"]
    ]
    # align and concatenate on index, keeping both sets of columns with clear suffixes
    combined_decisions = pd.concat(
        [
            perf_decisions.add_suffix("_perf"),
            for_24_decisions.add_suffix("_24"),
            for_12_decisions.add_suffix("_12"),
            for_9_decisions.add_suffix("_9"),
            for_6_decisions.add_suffix("_6"),
            for_3_decisions.add_suffix("_3"),
            for_1_decisions.add_suffix("_1"),
        ],
        axis=1,
        join="outer",
    )
    combined_decisions = combined_decisions.sort_index()
    combined_decisions = combined_decisions.dropna(how="any")
    perf_val = np.sum(
        (combined_decisions["charge_mw_perf"] - combined_decisions["discharge_mw_perf"])
        * 5.0
        / 60.0
        * combined_decisions["lmp_24"]
    )
    vals = {}
    for i, horizon in enumerate([1, 3, 6, 9, 12, 24]):
        vals[f"pct_perf_hor_{horizon}"] = (
            np.sum(
                (
                    combined_decisions[f"charge_mw_{horizon}"]
                    - combined_decisions[f"discharge_mw_{horizon}"]
                )
                * 5.0
                / 60.0
                * combined_decisions["lmp_24"]
            )
            / perf_val
        )

    return combined_decisions, vals


def calculate_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    if "actual" not in pred_df.columns:
        raise ValueError("pred_df must contain an 'actual' column")

    results: Dict[str, Dict[str, float]] = {}
    actual = pred_df["actual"]
    mae_total = 0
    rmse_total = 0
    for h in range(1, 25):
        col = f"h_{h}"
        if col not in pred_df.columns:
            # Skip missing columns gracefully
            continue
        pair = pd.concat([actual, pred_df[col]], axis=1, keys=["actual", col]).dropna()
        if pair.empty:
            # No overlapping data; return NaNs
            results[col] = {"mae": float("nan"), "rmse": float("nan")}
            continue
        y_true = pair["actual"].to_numpy(dtype=float)
        y_pred = pair[col].to_numpy(dtype=float)
        results[f"{col}_mae"] = mae(y_true, y_pred)
        results[f"{col}_rmse"] = rmse(y_true, y_pred)
        mae_total += results[f"{col}_mae"]
        rmse_total += results[f"{col}_rmse"]

    results["val_mae_full"] = mae_total
    results["val_rmse_full"] = rmse_total

    return results
