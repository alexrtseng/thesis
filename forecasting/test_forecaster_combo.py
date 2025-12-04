from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from deterministic.single_market_battery import (
    DEFAULT_BATTERY,
    deterministic_arbitrage_opt,
)
from deterministic.warm_start_arb_solver import (
    build_battery_model,
    set_objective,
    update_initial_charge,
)
from forecasting.metrics import (
    calculate_metrics,
    long_horizon_pred_performance,
    short_horizon_pred_performance,
)


def _load_model_from_outputs(
    pnode_id: int, model_class_name: str, run_name: str
) -> ForecastingModel:
    base = Path("forecasting/outputs") / str(pnode_id) / model_class_name
    # Find directory that contains the run_name
    candidates = [d for d in base.glob(f"*{run_name}*") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No saved model dir found for {model_class_name} run '{run_name}' under {base}"
        )
    model_dir = sorted(candidates)[-1]
    model_file = model_dir / "model.pkl"
    ckpt_file = model_dir / "model.pkl.ckpt"
    if ckpt_file.exists():
        model_file = ckpt_file
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
    # Darts models support classmethod `load`
    return ForecastingModel.load(str(model_file))


def _build_series_from_feature_df(
    feature_df: pd.DataFrame,
) -> Tuple[TimeSeries, TimeSeries]:
    # Assumes feature_df has 5-min indexed columns 'lmp_rt' and covariates used by HF/LF models
    target_series = TimeSeries.from_dataframe(
        feature_df,
        time_col=None,
        value_cols="lmp_rt",
        freq="5min",
    )
    return target_series, TimeSeries.from_dataframe(
        feature_df,
        time_col=None,
        value_cols=[c for c in feature_df.columns if c != "lmp_rt"],
        freq="5min",
    )


def _joint_optimization_with_hf_lf(
    hf_preds: list[TimeSeries],
    lf_preds: list[TimeSeries],
    prices_df: pd.DataFrame,
    hf_horizon: int,
) -> pd.DataFrame:
    """Run MPC using HF forecasts for first K 5-min steps and LF hourly thereafter.

    Returns decisions indexed by 5-min origin timestamps over the evaluated window.
    """
    start_time = prices_df.index[0]
    start = pd.to_datetime(start_time)
    hf_index = pd.date_range(start=start, periods=hf_horizon + 1, freq="5min")
    day_end = start + pd.Timedelta(days=1)
    hourly_start = (
        hf_index[-1] + pd.Timedelta(minutes=5) if len(hf_index) > 0 else start
    )
    hourly_index = (
        pd.date_range(start=hourly_start, end=day_end, freq="h")
        if hourly_start < day_end
        else pd.DatetimeIndex([])
    )
    prices_index = hf_index.append(hourly_index)

    # Evaluate up to the guard used elsewhere to avoid overruns
    limit = max(0, min(len(hf_preds), len(lf_preds)) - 24 * 12)
    pred_arrays = []
    for i in range(limit):
        hf = hf_preds[i]
        lf = lf_preds[i] if i < len(lf_preds) else lf_preds[-1]
        arr = np.empty(len(prices_index), dtype=float)
        arr[0] = prices_df.loc[start_time + pd.Timedelta(minutes=5 * i), "lmp"]
        # First K steps: use HF forecast values
        hf_vals = hf[:hf_horizon].values().reshape(-1)
        arr[1 : hf_horizon + 1] = hf_vals
        # Hourly part: LF hourly forecast values
        if len(hourly_index) > 0:
            hourly_values = lf.values().reshape(-1)
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
        if model.Status != 2:  # gp.GRB.OPTIMAL
            raise RuntimeError("MPC step did not reach optimal solution")
        current_soe = float(soe.X[1])
        charge_decisions.append(float(charge.X[0]))
        discharge_decisions.append(float(discharge.X[0]))

    out_df = pd.DataFrame(
        {
            "charge_mw": charge_decisions,
            "discharge_mw": discharge_decisions,
        },
        index=prices_df.index[:limit],
    )
    return out_df


def evaluate_hf_lf_pair(
    feature_df: pd.DataFrame,
    pnode_id: int,
    hf_model_class: str,
    hf_run_id: str,
    lf_model_class: str,
    lf_run_id: str,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a HF forecaster and LF forecaster together.

    - Loads models from `forecasting/outputs/<pnode>/<ModelClass>/<sweep__run>/model.pkl(.ckpt)` by run id.
    - Generates predictions for HF (5-min horizon 24, rolling) and LF (hourly 24) over the validation window.
    - Calculates validation metrics for both, runs short- and long-horizon performance.
    - Runs joint optimization using HF for first K steps (K=3,6,9) and LF thereafter.
    - Also computes deterministic arbitrage baseline.

    Returns a dict of metric blocks.
    """
    # Build target and covariates TimeSeries (HF uses 5-min data; LF consumes hourly preds later)
    target_ts, fut_ts = _build_series_from_feature_df(feature_df)

    # Load models
    hf_model = _load_model_from_outputs(pnode_id, hf_model_class, hf_run_id)
    lf_model = _load_model_from_outputs(pnode_id, lf_model_class, lf_run_id)

    # Generate HF predictions: rolling origins, horizon 24
    hf_preds: list[TimeSeries] = []
    for i in range(50, len(val_y) - 24):
        hf_preds.append(hf_model.predict(n=24, series=val_y[:i]))

    # Generate LF predictions: hourly horizon 24, reusing same origins
    lf_preds: list[TimeSeries] = []
    for i in range(50, len(val_y) - 24):
        lf_preds.append(lf_model.predict(n=24, series=val_y[:i]))

    # Build preds_df aligned to validation timestamps from HF preds
    # Use the same helper logic as in train: actual as validation target values
    all_index = pd.Index(val_y.time_index)
    for pred in hf_preds:
        all_index = all_index.union(pd.Index(pred.time_index))
    all_index = all_index.sort_values()
    preds_df = pd.DataFrame(index=all_index)
    preds_df["actual"] = np.nan
    preds_df.loc[val_y.time_index, "actual"] = val_y.values().reshape(-1)
    for h in range(1, 25):
        preds_df[f"h_{h}"] = np.nan
    for pred in hf_preds:
        for h, ts in enumerate(pred.time_index, start=1):
            col = f"h_{h}"
            if pd.isna(preds_df.at[ts, col]):
                preds_df.at[ts, col] = pred.values()[h - 1]

    # Metrics for HF
    hf_metrics = calculate_metrics(preds_df)
    # Short-horizon performance (HF-only)
    sh_combined, sh_vals = short_horizon_pred_performance(
        hf_preds, preds_df, granular_metrics=False
    )

    # Long-horizon performance (LF-only) uses actual series for first K steps internally
    # Build lmp_series over same validation window
    lmp_series = pd.Series(
        val_y.values().reshape(-1), index=val_y.time_index, name="lmp"
    )
    lh_combined, lh_vals = long_horizon_pred_performance(lf_preds, lmp_series)

    # Joint optimization using both
    # Use actual prices frame aligned to HF origins span
    start = hf_preds[0].time_index[0] - pd.Timedelta(minutes=5)
    end = hf_preds[-1].time_index[0] - pd.Timedelta(minutes=5)
    prices_df = pd.DataFrame({"lmp": lmp_series.loc[start:end]})
    joint_vals: Dict[str, float] = {}
    for K in [3, 6, 9]:
        joint_df = _joint_optimization_with_hf_lf(
            hf_preds, lf_preds, prices_df, hf_horizon=K
        )
        # Compare to deterministic arbitrage over same window
        perf_df, _ = deterministic_arbitrage_opt(
            prices_df=prices_df, require_equivalent_soe=True
        )
        perf_val = np.sum(
            (perf_df["charge_mw"] - perf_df["discharge_mw"])
            * 5.0
            / 60.0
            * prices_df["lmp"]
        )
        joint_val = np.sum(
            (joint_df["charge_mw"] - joint_df["discharge_mw"])
            * 5.0
            / 60.0
            * prices_df["lmp"]
        )
        joint_vals[f"joint_pct_perf_hor_{K}"] = (
            (joint_val / perf_val) if perf_val != 0 else float("nan")
        )

    return {
        "hf_metrics": hf_metrics,
        "short_horizon": sh_vals,
        "long_horizon": lh_vals,
        "joint": joint_vals,
    }
