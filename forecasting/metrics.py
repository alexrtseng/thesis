from pathlib import Path
from typing import Dict

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

from deterministic.single_market_battery import (
    DEFAULT_BATTERY,
    deterministic_arbitrage_opt,
    txbx,
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
    print(f"Running short horizon performance with hf_horizon={hf_horizon}")
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
    preds: list[TimeSeries], preds_df: pd.DataFrame, granular_metrics: bool = False
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
    if granular_metrics:
        for_24_decisions = _short_horizon_pred_performance(preds, df, 24)[
            ["charge_mw", "discharge_mw"]
        ]
    for_12_decisions = _short_horizon_pred_performance(preds, df, 12)[
        ["charge_mw", "discharge_mw"]
    ]
    if granular_metrics:
        for_9_decisions = _short_horizon_pred_performance(preds, df, 9)[
            ["charge_mw", "discharge_mw"]
        ]
    for_6_decisions = _short_horizon_pred_performance(preds, df, 6)[
        ["charge_mw", "discharge_mw", "lmp"]
    ]
    for_3_decisions = _short_horizon_pred_performance(preds, df, 3)[
        ["charge_mw", "discharge_mw"]
    ]
    if granular_metrics:
        for_1_decisions = _short_horizon_pred_performance(preds, df, 1)[
            ["charge_mw", "discharge_mw"]
        ]
    # align and concatenate on index, keeping both sets of columns with clear suffixes
    frames = [perf_decisions.add_suffix("_perf")]

    # always include these horizons
    frames.append(for_12_decisions.add_suffix("_12"))
    frames.append(for_6_decisions.add_suffix("_6"))
    frames.append(for_3_decisions.add_suffix("_3"))

    # include granular horizons only when requested
    if granular_metrics:
        frames.append(for_24_decisions.add_suffix("_24"))
        frames.append(for_9_decisions.add_suffix("_9"))
        frames.append(for_1_decisions.add_suffix("_1"))

    combined_decisions = pd.concat(frames, axis=1, join="outer")
    combined_decisions.rename(columns={"lmp_6": "lmp"}, inplace=True)
    combined_decisions = combined_decisions.sort_index()
    combined_decisions = combined_decisions.dropna(how="any")
    perf_val = np.sum(
        (combined_decisions["discharge_mw_perf"] - combined_decisions["charge_mw_perf"])
        * (5.0 / 60.0)
        * combined_decisions["lmp"]
    )
    vals = {}
    horizon_list = [1, 3, 6, 9, 12, 24] if granular_metrics else [3, 6, 12]
    for i, horizon in enumerate(horizon_list):
        vals[f"pct_perf_hor_{horizon}"] = (
            np.sum(
                (
                    combined_decisions[f"discharge_mw_{horizon}"]
                    - combined_decisions[f"charge_mw_{horizon}"]
                )
                * (5.0 / 60.0)
                * combined_decisions["lmp"]
            )
            / perf_val
        )

    return combined_decisions, vals


def _long_horizon_pred_performance(
    preds_hourly: list[TimeSeries], _prices_series: pd.Series, hf_horizon: int
) -> pd.DataFrame:
    print(f"Running long horizon performance with hf_horizon={hf_horizon}")
    prices_series = _prices_series.copy()
    start_time = prices_series.index[0]
    assert len(prices_series) == len(preds_hourly)
    start: pd.Timestamp = pd.to_datetime(start_time)

    # Build mixed index: K 5-min steps then hourly to end-of-day
    hf_index = pd.date_range(start=start, periods=hf_horizon + 1, freq="5min")
    day_end = start + pd.Timedelta(days=1)
    hourly_start = hf_index[-1].ceil("h") if len(hf_index) > 0 else start
    hourly_index = (
        pd.date_range(start=hourly_start, end=day_end, freq="h")
        if hourly_start < day_end
        else pd.DatetimeIndex([])
    )
    prices_index = hf_index.append(hourly_index)

    pred_arrays = []
    # Avoid running past data; mirror the short-horizon guard
    limit = max(0, len(preds_hourly) - 24 * 12)
    for i, pred in enumerate(preds_hourly[:limit]):
        arr = np.empty(len(prices_index), dtype=float)
        # initial point aligned with origin i
        arr[0] = prices_series.loc[start_time + pd.Timedelta(minutes=5 * i)]
        # first K steps: actual 5-min prices
        actual_segment = prices_series.reindex(
            pd.date_range(
                start=start + pd.Timedelta(minutes=5 * i),
                periods=hf_horizon,
                freq="5min",
            )
        ).to_numpy(dtype=float)
        arr[1 : hf_horizon + 1] = actual_segment
        # hourly part: use hourly forecast values directly
        if len(hourly_index) > 0:
            hourly_values = pred.values().reshape(-1)
            h_needed = len(hourly_index)
            arr[hf_horizon + 1 :] = hourly_values[:h_needed]
        pred_arrays.append(arr)

    model, soe, charge, discharge, times, dt_vec, init_soe_constr = build_battery_model(
        prices_index=prices_index, battery=DEFAULT_BATTERY, requires_equivalent_soe=True
    )
    current_soe = DEFAULT_BATTERY.initial_charge_mwh
    charge_decisions = []
    discharge_decisions = []
    for arr in pred_arrays:
        set_objective(model, charge, discharge, times, arr, dt_vec)
        update_initial_charge(model, init_soe_constr, soe, current_soe)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("MPC step did not reach optimal solution")
        current_soe = float(soe.X[1])
        charge_decisions.append(float(charge.X[0]))
        discharge_decisions.append(float(discharge.X[0]))

    out_df = pd.DataFrame(
        {
            "charge_mw": charge_decisions,
            "discharge_mw": discharge_decisions,
        },
        index=prices_series.index[:limit],
    )
    return out_df


def long_horizon_pred_performance(
    preds: list[TimeSeries], lmp_series: pd.Series
) -> tuple[pd.DataFrame, Dict[str, float]]:
    start = preds[0].time_index[0] - pd.Timedelta(minutes=5)
    end = preds[-1].time_index[0] - pd.Timedelta(minutes=5)
    _lmp_series = lmp_series.loc[start:end].copy()
    _lmp_series = _lmp_series.rename("lmp")
    _lmp_series.dropna(inplace=True)

    # Perfect foresight baseline
    perf_decisions, _ = deterministic_arbitrage_opt(
        prices_df=_lmp_series.to_frame("lmp"),
        require_equivalent_soe=True,
    )

    # Evaluate 3, 6, 9 high-fidelity steps
    for_3 = _long_horizon_pred_performance(preds, _lmp_series, 3)[
        ["charge_mw", "discharge_mw"]
    ]
    for_6 = _long_horizon_pred_performance(preds, _lmp_series, 6)[
        ["charge_mw", "discharge_mw"]
    ]
    for_9 = _long_horizon_pred_performance(preds, _lmp_series, 9)[
        ["charge_mw", "discharge_mw"]
    ]

    frames = [perf_decisions.add_suffix("_perf")]
    frames.append(for_3.add_suffix("_3"))
    frames.append(for_6.add_suffix("_6"))
    frames.append(for_9.add_suffix("_9"))
    frames.append(_lmp_series.to_frame("lmp"))

    combined = pd.concat(frames, axis=1, join="outer").sort_index().dropna(how="any")

    perf_val = np.sum(
        (combined["charge_mw_perf"] - combined["discharge_mw_perf"])
        * (5.0 / 60.0)
        * combined["lmp"]
    )
    vals: Dict[str, float] = {}
    for horizon in [3, 6, 9]:
        vals[f"pct_perf_hor_{horizon}"] = (
            np.sum(
                (combined[f"charge_mw_{horizon}"] - combined[f"discharge_mw_{horizon}"])
                * (5.0 / 60.0)
                * combined["lmp"]
            )
            / perf_val
            if perf_val != 0
            else float("nan")
        )

    return combined, vals


def _short_and_long_pred_performance(
    hf_preds: list[TimeSeries],
    lf_preds: list[TimeSeries],
    _prices_series: pd.Series,
    hf_horizon: int,
):
    print(f"Running joint pred performance with hf_horizon={hf_horizon}")
    prices_series = _prices_series.copy()
    start_time = prices_series.index[0]
    # Ensure we have matching number of origins for LF forecasts
    assert len(prices_series) == len(lf_preds)
    start: pd.Timestamp = pd.to_datetime(start_time)

    # Build mixed index: K 5-min steps then hourly to end-of-day
    hf_index = pd.date_range(start=start, periods=hf_horizon + 1, freq="5min")
    day_end = start + pd.Timedelta(days=1)
    hourly_start = hf_index[-1].ceil("h") if len(hf_index) > 0 else start
    hourly_index = (
        pd.date_range(start=hourly_start, end=day_end, freq="h")
        if hourly_start < day_end
        else pd.DatetimeIndex([])
    )
    prices_index = hf_index.append(hourly_index)

    pred_arrays = []
    # Guard to avoid overruns similar to other helpers
    limit = max(0, min(len(hf_preds), len(lf_preds)) - 24 * 12)
    for i in range(limit):
        hf_pred = hf_preds[i]
        lf_pred = lf_preds[i]
        arr = np.empty(len(prices_index), dtype=float)
        # initial point aligned with origin i (use actual price at origin t)
        arr[0] = prices_series.loc[start_time + pd.Timedelta(minutes=5 * i)]
        # first K steps: use HF prediction values at 5-min resolution
        hf_vals = hf_pred[:hf_horizon].values().reshape(-1)
        arr[1 : hf_horizon + 1] = hf_vals
        # hourly part: use LF hourly forecast values directly
        if len(hourly_index) > 0:
            hourly_values = lf_pred.values().reshape(-1)
            h_needed = len(hourly_index)
            arr[hf_horizon + 1 :] = hourly_values[:h_needed]
        pred_arrays.append(arr)

    model, soe, charge, discharge, times, dt_vec, init_soe_constr = build_battery_model(
        prices_index=prices_index, battery=DEFAULT_BATTERY, requires_equivalent_soe=True
    )
    current_soe = DEFAULT_BATTERY.initial_charge_mwh
    charge_decisions = []
    discharge_decisions = []
    for arr in pred_arrays:
        set_objective(model, charge, discharge, times, arr, dt_vec)
        update_initial_charge(model, init_soe_constr, soe, current_soe)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("MPC step did not reach optimal solution")
        current_soe = float(soe.X[1])
        charge_decisions.append(float(charge.X[0]))
        discharge_decisions.append(float(discharge.X[0]))

    out_df = pd.DataFrame(
        {
            "charge_mw": charge_decisions,
            "discharge_mw": discharge_decisions,
        },
        index=prices_series.index[:limit],
    )
    return out_df


def short_and_long_pred_performance(
    hf_preds: list[TimeSeries],
    lf_preds: list[TimeSeries],
    feature_df: pd.DataFrame,
):
    print(f"Prior to leveling, hf and lf pred lengths are: {len(hf_preds)}, {len(lf_preds)}")
    start = max(
        hf_preds[0].time_index[0] - pd.Timedelta(minutes=5),
        lf_preds[0].time_index[0] - pd.Timedelta(minutes=5),
    )
    end = min(
        hf_preds[-1].time_index[0] - pd.Timedelta(minutes=5),
        lf_preds[-1].time_index[0] - pd.Timedelta(minutes=5),
    )
    _lmp_series = feature_df["lmp_rt"].loc[start:end].copy()
    _lmp_series = _lmp_series.rename("lmp")
    # align preds
    _lmp_series.dropna(inplace=True)

    def steps_between(t0: pd.Timestamp, t1: pd.Timestamp, step_minutes: int) -> int:
        # round to nearest step to avoid float/int truncation issues
        delta = (t1 - t0).total_seconds() / (60 * step_minutes)
        return int(round(delta))

    # compute how many origins to drop from the left/right to match [start, end]
    drop_left_hf = max(0, steps_between(hf_preds[0].time_index[0], start + pd.Timedelta(minutes=5), 5))
    drop_right_hf = max(0, steps_between(end + pd.Timedelta(minutes=5), hf_preds[-1].time_index[0], 5))
    left_hf = drop_left_hf
    right_hf_excl = len(hf_preds) - drop_right_hf
    hf_preds = hf_preds[left_hf:right_hf_excl].copy()

    drop_left_lf = max(0, steps_between(lf_preds[0].time_index[0], start + pd.Timedelta(minutes=5), 5))
    drop_right_lf = max(0, steps_between(end + pd.Timedelta(minutes=5), lf_preds[-1].time_index[0], 5))
    left_lf = drop_left_lf
    right_lf_excl = len(lf_preds) - drop_right_lf
    lf_preds = lf_preds[left_lf:right_lf_excl].copy()

    print(f"After leveling, hf and lf pred lengths are: {len(hf_preds)}, {len(lf_preds)}")

    # Perfect foresight and txbx baseline
    perf_decisions, _ = deterministic_arbitrage_opt(
        prices_df=_lmp_series.to_frame("lmp"),
        require_equivalent_soe=True,
    )
    tb4_decisions, _ = txbx(prices_df=feature_df.loc[start:end])

    # Evaluate combined HF+LF with 3, 6, 9 high-fidelity steps
    for_3 = _short_and_long_pred_performance(hf_preds, lf_preds, _lmp_series, 3)[
        ["charge_mw", "discharge_mw"]
    ]
    for_6 = _short_and_long_pred_performance(hf_preds, lf_preds, _lmp_series, 6)[
        ["charge_mw", "discharge_mw"]
    ]
    for_9 = _short_and_long_pred_performance(hf_preds, lf_preds, _lmp_series, 9)[
        ["charge_mw", "discharge_mw"]
    ]

    frames = [perf_decisions.add_suffix("_perf"), tb4_decisions.add_suffix("_tb4")]
    frames.append(for_3.add_suffix("_3"))
    frames.append(for_6.add_suffix("_6"))
    frames.append(for_9.add_suffix("_9"))
    frames.append(_lmp_series.to_frame("lmp"))

    combined = pd.concat(frames, axis=1, join="outer").sort_index().dropna(how="any")

    perf_val = np.sum(
        (combined["discharge_mw_perf"] - combined["charge_mw_perf"])
        * (5.0 / 60.0)
        * combined["lmp"]
    )
    tb4_val = np.sum(
        (combined["discharge_mw_tb4"] - combined["charge_mw_tb4"])
        * (5.0 / 60.0)
        * combined["lmp"]
    )
    vals: Dict[str, float] = {}
    vals["perf_val"] = perf_val
    vals["tb4_val"] = tb4_val
    vals["pct_tb4_vs_perf"] = (tb4_val / perf_val) if perf_val != 0 else float("nan")
    for horizon in [3, 6, 9]:
        val_generated = np.sum(
            (combined[f"discharge_mw_{horizon}"] - combined[f"charge_mw_{horizon}"])
            * (5.0 / 60.0)
            * combined["lmp"]
        )

        vals[f"pct_perf_hor_{horizon}"] = (
            val_generated / perf_val if perf_val != 0 else float("nan")
        )
        vals[f"pct_tb4_hor_{horizon}"] = (
            val_generated / tb4_val if tb4_val != 0 else float("nan")
        )

    return combined, vals


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


def calculate_residual_matrix(pred_df: pd.DataFrame):
    if "actual" not in pred_df.columns:
        raise ValueError("pred_df must contain an 'actual' column")
    # Residual bins: width 10 from -1000 to 950 -> 195 bins
    bin_start = -200
    bin_end = 1000
    bin_width = 4
    # Build bin edges inclusive of end
    bins = np.arange(bin_start, bin_end + bin_width, bin_width, dtype=float)
    # Prepare output matrix: 24 horizons x 300 bins
    num_horizons = 24
    num_bins = int((bin_end - bin_start) / bin_width)
    if len(bins) - 1 != num_bins:
        # Safety check to ensure 100 bins
        num_bins = len(bins) - 1

    matrix = np.zeros((num_horizons, num_bins), dtype=int)

    actual = pred_df["actual"].to_numpy(dtype=float)

    for h in range(1, num_horizons + 1):
        col = f"h_{h}"
        if col not in pred_df.columns:
            # Leave row as zeros if horizon column missing
            continue
        preds_h = pred_df[col].to_numpy(dtype=float)
        # Build residuals where both actual & prediction are present
        mask = ~np.isnan(actual) & ~np.isnan(preds_h)
        if not np.any(mask):
            continue
        residuals = preds_h[mask] - actual[mask]

        # Histogram count per bin using pandas cut for consistent binning
        # Values outside range are dropped (do not contribute to any bin)
        cats = pd.cut(residuals, bins=bins, right=False, include_lowest=True)
        counts = pd.Series(cats).value_counts(sort=False)
        # Align to expected number of bins
        counts = counts.reindex(
            pd.IntervalIndex.from_breaks(bins, closed="left"), fill_value=0
        )
        matrix[h - 1, :] = counts.to_numpy(dtype=int)

    # Return as DataFrame for readability: rows=horizons, cols=bin labels
    bin_labels = [f"[{int(bins[i])},{int(bins[i + 1])})" for i in range(len(bins) - 1)]
    horizon_labels = [f"h_{h}" for h in range(1, num_horizons + 1)]
    return pd.DataFrame(matrix, index=horizon_labels, columns=bin_labels)


def plot_residuals_and_save(pred_df: pd.DataFrame, output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    residuals_dir = output_dir / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)

    # Compute residual matrix
    matrix_df = calculate_residual_matrix(pred_df)
    matrix_csv = residuals_dir / "residual_matrix.csv"
    matrix_df.to_csv(matrix_csv)

    # Determine bin centers for x-axis in 2D plots
    # Columns are labels like "[start,end)"; parse start/end and use center
    bin_centers = []
    for label in matrix_df.columns:
        # label format: [a,b)
        a, b = label.strip()[1:-1].split(",")
        a = float(a)
        b = float(b[:-1]) if b.endswith(")") else float(b)
        bin_centers.append((a + b) / 2.0)
    bin_centers = np.array(bin_centers)

    # Determine central 95% range across all horizons (aggregate distribution) for consistent cropping
    total_counts = matrix_df.sum(axis=0).to_numpy(dtype=int)
    grand_total = total_counts.sum()
    if grand_total > 0:
        cum_counts = np.cumsum(total_counts)
        lower_cut = 0.025 * grand_total
        upper_cut = 0.975 * grand_total
        # Find first bin index where cumulative >= lower_cut
        lower_idx = int(np.searchsorted(cum_counts, lower_cut, side="left"))
        upper_idx = int(np.searchsorted(cum_counts, upper_cut, side="right")) - 1
        # Safety bounds
        lower_idx = max(lower_idx, 0)
        upper_idx = min(upper_idx, len(total_counts) - 1)
    else:
        lower_idx, upper_idx = 0, len(total_counts) - 1

    cropped_bin_centers = bin_centers[lower_idx : upper_idx + 1]
    cropped_columns = matrix_df.columns[lower_idx : upper_idx + 1]

    # Plot residual histograms for selected horizons (cropped to central 95%)
    horizons_to_plot = [1, 3, 6, 9, 12, 24]
    fig, axes = plt.subplots(
        len(horizons_to_plot), 1, figsize=(10, 2.4 * len(horizons_to_plot)), sharex=True
    )
    for ax, h in zip(axes, horizons_to_plot):
        row = matrix_df.loc[f"h_{h}"][cropped_columns]
        ax.bar(
            cropped_bin_centers,
            row.to_numpy(dtype=int),
            width=(cropped_bin_centers[1] - cropped_bin_centers[0]) * 0.9
            if len(cropped_bin_centers) > 1
            else 1.0,
            align="center",
            color="#4e79a7",
        )
        ax.set_title(f"Residuals Histogram — Horizon h_{h} (Central 95%)")
        ax.set_ylabel("freq")
    axes[-1].set_xlabel("residual (bin center)")
    # Annotate range in the top subplot
    axes[0].text(
        0.01,
        0.95,
        f"Central 95% range: {cropped_columns[0]} .. {cropped_columns[-1]}",
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        ha="left",
        color="#333",
    )
    fig.tight_layout()
    hist_png = residuals_dir / "residual_histograms_selected_central95.png"
    fig.savefig(hist_png, dpi=150)
    plt.close(fig)

    # 3D plot: horizon vs cropped bin index vs frequency (central 95%)
    H = matrix_df.shape[0]
    Bc = len(cropped_columns)
    X_bins = np.arange(Bc)  # cropped bin index
    Y_horizons = np.arange(1, H + 1)  # 1..24
    X, Y = np.meshgrid(X_bins, Y_horizons)
    Z = matrix_df[cropped_columns].to_numpy(dtype=int)

    fig3d = plt.figure(figsize=(12, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")
    # Use bar3d for clarity; each bar at (bin,horizon) with height frequency
    dx = 0.8
    dy = 0.8
    # Flatten grids for bar3d
    xs = X.ravel()
    ys = Y.ravel()
    zs = np.zeros_like(xs, dtype=float)
    hs = Z.ravel().astype(float)
    ax3d.bar3d(xs, ys, zs, dx, dy, hs, shade=True, color="#f28e2b")
    ax3d.set_xlabel("Bin")
    ax3d.set_ylabel("Horizon")
    ax3d.set_zlabel("Freq")
    ax3d.set_title("Residuals Over Horizons")
    fig3d.tight_layout()
    plot3d_png = residuals_dir / "residuals_3d.png"
    fig3d.savefig(plot3d_png, dpi=150)
    plt.close(fig3d)

    return residuals_dir
