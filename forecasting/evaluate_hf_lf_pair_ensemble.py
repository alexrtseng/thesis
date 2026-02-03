import json
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from darts import TimeSeries

from deterministic.single_market_battery import (
    DEFAULT_BATTERY,
    deterministic_arbitrage_opt,
)
from forecasting.store_test_forecasts import get_cached_forecasts
from forecasting.test_forecaster_combo import (
    evaluate_hf_lf_pair,
    produce_forecasts_for_eval,
)
from forecasting.train import build_series_for_node
from stochastic.warm_start_arb import (
    build_two_stage_battery_model,
    set_two_stage_objective,
    update_two_stage_initial_charge,
)


def evaluate_hf_lf_pair_ensemble(
    pnode_id: int,
    hf_run_paths: list[str] | None,
    lf_run_paths: list[str] | None,
    test_size: int | None = None,
    hf_horizon: int = 6,
    cache_file=None,
):
    t_start = time.perf_counter()
    if cache_file is None:
        run_path_forecast_dict = produce_forecasts_for_eval(
            pnode_id, hf_run_paths, lf_run_paths, test_size, hf_horizon
        )
    else:
        run_path_forecast_dict = get_cached_forecasts(
            cache_file, run_paths=(hf_run_paths or []) + (lf_run_paths or [])
        )
    hf_pred_sets: list[list[TimeSeries]] = []
    lf_pred_sets: list[list[TimeSeries]] = []
    for hf_run_path in hf_run_paths:
        hf_pred_sets.append(run_path_forecast_dict[hf_run_path])

    for lf_run_path in lf_run_paths:
        lf_pred_sets.append(run_path_forecast_dict[lf_run_path])

    feature_df = build_series_for_node(pnode_id)

    # --- Align all HF/LF prediction origin ranges to a common [start, end] window ---
    def _steps_between(t0: pd.Timestamp, t1: pd.Timestamp, step_minutes: int) -> int:
        delta = (t1 - t0).total_seconds() / (60 * step_minutes)
        return int(round(delta))

    all_sets = hf_pred_sets + lf_pred_sets
    common_start = max(s[0].time_index[0] - pd.Timedelta(minutes=5) for s in all_sets)
    common_end = min(s[-1].time_index[0] - pd.Timedelta(minutes=5) for s in all_sets)
    lmp_series = feature_df["lmp_rt"].loc[common_start:common_end].copy().rename("lmp")
    lmp_series.dropna(inplace=True)
    if len(lmp_series) < 10:
        raise RuntimeError("Not enough overlapping data after leveling predictions")

    def _slice_preds(preds: list[TimeSeries]) -> list[TimeSeries]:
        left = max(
            0,
            _steps_between(
                preds[0].time_index[0], common_start + pd.Timedelta(minutes=5), 5
            ),
        )
        right_drop = max(
            0,
            _steps_between(
                common_end + pd.Timedelta(minutes=5), preds[-1].time_index[0], 5
            ),
        )
        right_excl = len(preds) - right_drop
        return preds[left:right_excl].copy()

    hf_pred_sets = [_slice_preds(p) for p in hf_pred_sets]
    lf_pred_sets = [_slice_preds(p) for p in lf_pred_sets]

    # --- Build the mixed index for the MPC problem (K 5-min then hourly to end-of-day) ---
    start_time = pd.to_datetime(lmp_series.index[0])
    hf_index = pd.date_range(start=start_time, periods=hf_horizon + 1, freq="5min")
    day_end = start_time + pd.Timedelta(days=1)
    hourly_start = hf_index[-1].ceil("h") if len(hf_index) > 0 else start_time
    hourly_index = (
        pd.date_range(start=hourly_start, end=day_end, freq="h")
        if hourly_start < day_end
        else pd.DatetimeIndex([])
    )
    prices_index = hf_index.append(hourly_index)

    H = len(hf_pred_sets)
    L = len(lf_pred_sets)
    branches = H * L
    print(f"Running two-stage ensemble with H={H}, L={L}, branches={branches}")

    (
        model,
        init_soe,
        init_charge,
        init_discharge,
        charge_branches,
        discharge_branches,
        _times,
        dt_vec,
        init_soe_constr,
    ) = build_two_stage_battery_model(
        prices_index=prices_index,
        battery=DEFAULT_BATTERY,
        branches=branches,
        initial_charge_mwh=DEFAULT_BATTERY.initial_charge_mwh,
        requires_equivalent_soe=True,
        verbose=False,
    )

    # Determine how many origins we can safely evaluate
    min_len = min(min(len(p) for p in hf_pred_sets), min(len(p) for p in lf_pred_sets))
    limit = max(0, min_len - 24 * 12)
    if limit == 0:
        raise RuntimeError("Not enough prediction origins (after guard) to evaluate")

    current_soe = float(DEFAULT_BATTERY.initial_charge_mwh)
    charge_decisions: list[float] = []
    discharge_decisions: list[float] = []
    realized_revenue: list[float] = []
    dt0 = float(dt_vec[0])

    # Pre-allocate hourly part length
    hourly_needed = len(hourly_index)
    future_len = len(prices_index) - 1

    for i in range(limit):
        init_price = float(lmp_series.iloc[i])
        price_series: list[np.ndarray] = []

        # Build one branch per (hf_model, lf_model) combination
        for h in range(H):
            hf_pred = hf_pred_sets[h][i]
            hf_vals = hf_pred[:hf_horizon].values().reshape(-1)
            if len(hf_vals) < hf_horizon:
                hf_vals = np.pad(hf_vals, (0, hf_horizon - len(hf_vals)), mode="edge")

            for lf_idx in range(L):
                lf_pred = lf_pred_sets[lf_idx][i]
                future = np.empty(future_len, dtype=float)
                future[:hf_horizon] = hf_vals[:hf_horizon]
                if hourly_needed > 0:
                    hourly_vals = lf_pred.values().reshape(-1)
                    if len(hourly_vals) < hourly_needed:
                        hourly_vals = np.pad(
                            hourly_vals,
                            (0, hourly_needed - len(hourly_vals)),
                            mode="edge",
                        )
                    future[hf_horizon:] = hourly_vals[:hourly_needed]
                price_series.append(future)

        if len(price_series) != branches:
            raise RuntimeError("Internal error: branch price series length mismatch")

        set_two_stage_objective(
            model=model,
            init_price=init_price,
            init_charge=init_charge,
            init_discharge=init_discharge,
            charge_branches=charge_branches,
            discharge_branches=discharge_branches,
            price_series=price_series,
            dt_vec=dt_vec,
        )
        update_two_stage_initial_charge(model, init_soe_constr, init_soe, current_soe)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("Two-stage MPC step did not reach optimal solution")

        c0 = float(init_charge.X)
        d0 = float(init_discharge.X)
        charge_decisions.append(c0)
        discharge_decisions.append(d0)

        realized_revenue.append((d0 - c0) * dt0 * init_price)

        # Apply decision to update SoE (use current_soe for self-discharge term)
        current_soe = float(
            np.clip(
                current_soe
                + (
                    c0 * DEFAULT_BATTERY.in_efficiency
                    - d0 / DEFAULT_BATTERY.out_efficiency
                    - current_soe * DEFAULT_BATTERY.self_discharge_percent_per_hour
                )
                * dt0,
                0.0,
                DEFAULT_BATTERY.capacity_mwh,
            )
        )

    decisions_index = lmp_series.index[:limit]
    two_stage_df = pd.DataFrame(
        {"charge_mw": charge_decisions, "discharge_mw": discharge_decisions},
        index=decisions_index,
    )

    # Baseline: perfect-foresight deterministic arbitrage over the same window
    perf_df, _ = deterministic_arbitrage_opt(
        prices_df=lmp_series.iloc[:limit].to_frame("lmp"),
        battery=DEFAULT_BATTERY,
        require_equivalent_soe=True,
        verbose=False,
        use_barrier=True,
    )
    dt_hours = (
        perf_df.index.to_series().shift(-1) - perf_df.index.to_series()
    ).dt.total_seconds().fillna(0.0) / 3600.0
    perf_val = float(
        (
            (perf_df["discharge_mw"] - perf_df["charge_mw"])
            * dt_hours
            * lmp_series.iloc[: len(perf_df)].to_numpy(dtype=float)
        ).sum()
    )
    two_stage_val = float(np.sum(realized_revenue))

    metrics = {
        "two_stage_val": two_stage_val,
        "perf_val": perf_val,
        "pct_perf": (two_stage_val / perf_val) if perf_val != 0 else float("nan"),
        "branches": branches,
        "hf_horizon": hf_horizon,
        "eval_points": int(limit),
        "hf_run_paths": hf_run_paths,
        "lf_run_paths": lf_run_paths,
        "total_time_taken_s": float(time.perf_counter() - t_start),
        "common_start": common_start.strftime("%Y-%m-%d %H:%M:%S"),
        "common_end": common_end.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_dir = (
        Path("forecasting/outputs") / "tests" / str(pnode_id) / "two_stage_ensemble"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    two_stage_df.to_csv(out_dir / "two_stage_decisions.csv")
    with open(out_dir / "two_stage_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, two_stage_df


if __name__ == "__main__":
    pnode_id = 2156113094
    hf_run_path = "watt-our/thesis-hf-forecasters/tks540ei"
    hf_run_path_2 = "watt-our/thesis-hf-forecasters/obvw0nqv"
    lf_run_path = "watt-our/thesis-lf-forecasters/tbk5xmvg"
    lf_run_path_2 = "watt-our/thesis-lf-forecasters/a3ilhshd"
    # evaluate_hf_lf_pair(pnode_id, hf_run_path, None, test_size=500, pjm_da_preds=True)
    joint_opt_metrics, _, _ = evaluate_hf_lf_pair(
        pnode_id,
        hf_run_path,
        lf_run_path,
        test_size=500,
        pjm_da_preds=False,
    )
    metrics, two_stage_df = evaluate_hf_lf_pair_ensemble(
        pnode_id,
        hf_run_paths=[hf_run_path, hf_run_path_2],
        lf_run_paths=[lf_run_path],
        test_size=500,
        pjm_da_preds=False,
        hf_horizon=6,
    )
    print("Joint opt metrics from single HF/LF pair:")
    print(json.dumps(joint_opt_metrics, indent=2))
    print("Two-stage ensemble metrics:")
    print(json.dumps(metrics, indent=2))
    print("Two-stage decisions head:")
    print(two_stage_df.head())
