import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

from data.data_output_functions import read_lmp_folder
from deterministic.single_market_battery import (
    DEFAULT_BATTERY,
    BatteryParams,
)


def build_battery_model(
    prices_index,
    battery: BatteryParams,
    initial_charge_mwh: float | None = None,
    requires_equivalent_soe: bool = False,
    verbose: bool = False,
):
    """Vectorized build of battery arbitrage LP using Gurobi MVar.

    Returns model plus MVars (soe, charge, discharge) and list of timestamps.
    Model uses dual simplex (Method=1) to enable basis warm starts across price updates.
    """
    model = gp.Model("battery_arbitrage")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.Method = 1  # dual simplex for warm starts

    times = list(prices_index)
    if len(times) >= 2:
        last_delta = times[-1] - times[-2]
        times.append(times[-1] + last_delta)
    T = len(times)  # artificially extend time index by one step
    if T < 2:
        raise ValueError("Need at least two timestamps for intervals")

    # Interval lengths (hours) for first T-1 intervals
    idx_series = pd.Series(times)
    dt_hours = (idx_series.shift(-1) - idx_series).dt.total_seconds().fillna(
        0.0
    ) / 3600.0
    dt_vec = np.asarray(dt_hours.iloc[:-1], dtype=float)  # length T-1

    # Decision MVars
    soe = model.addMVar(T, lb=0.0, ub=battery.capacity_mwh, name="soe")
    charge = model.addMVar(T - 1, lb=0.0, ub=battery.max_charge_mw, name="charge")
    discharge = model.addMVar(
        T - 1, lb=0.0, ub=battery.max_discharge_mw, name="discharge"
    )

    # Initial SoE equality constraint (store for later RHS updates in MPC)
    init_val = (
        battery.initial_charge_mwh
        if initial_charge_mwh is None
        else float(initial_charge_mwh)
    )
    init_soe_constr = model.addConstr(soe[0] == init_val, name="init_soe")
    if requires_equivalent_soe:
        model.addConstr(soe[-1] == init_val, name="last_soe_equivalence")

    # Vector dynamics: soe[1:] = soe[:-1] + (charge*in_eff - discharge/out_eff - soe[:-1]*self_discharge)*dt
    model.addConstr(
        soe[1:]
        == soe[:-1]
        + (
            charge * battery.in_efficiency
            - discharge / battery.out_efficiency
            - soe[:-1] * battery.self_discharge_percent_per_hour
        )
        * dt_vec,
        name="soe_dynamics_vec",
    )

    model.update()
    return model, soe, charge, discharge, times, dt_vec, init_soe_constr


def set_objective(model, charge, discharge, times, prices_series, dt_vec):
    """Vectorized objective: maximize sum price_t * (discharge_t - charge_t) * dt_t."""
    price_vec = np.asarray(prices_series, dtype=float)
    coeff = price_vec * dt_vec
    # MVar linear expression via dot product
    expr = coeff @ (discharge - charge)
    model.setObjective(expr, gp.GRB.MAXIMIZE)


def update_initial_charge(
    model: gp.Model,
    init_soe_constr: gp.Constr,
    soe_mvar: gp.MVar,
    new_initial_charge: float,
):
    init_soe_constr.RHS = float(new_initial_charge)
    # Tighten bounds (optional but can help dual simplex)
    soe_mvar[0].LB = float(new_initial_charge)
    soe_mvar[0].UB = float(new_initial_charge)
    model.update()


def battery_arb(
    hf_horizon: int,
    lf_horizon: int,
    prices_df: pd.DataFrame,
    battery_params: BatteryParams = DEFAULT_BATTERY,
    require_equivalent_soe: bool = False,
    use_lf_avg: bool = False,
    verbose: bool = False,
):
    required_horizon = hf_horizon + lf_horizon * 12
    df = prices_df.copy()
    charge_decisions = []
    discharge_decisions = []
    timestamps = []
    prices = []
    current_soe = battery_params.initial_charge_mwh
    hf = df.iloc[:hf_horizon].copy()
    lf_indices = [hf_horizon + j * 12 for j in range(lf_horizon)]
    lf = df.iloc[lf_indices].copy()
    if use_lf_avg:
        lf["lmp"] = lf["lmp_lf_avg"]
    window = pd.concat([hf, lf])
    # Build vectorized model once (warm start across iterations if prices change)
    model, soe, charge, discharge, times, dt_vec, init_soe_constr = build_battery_model(
        window.index,
        battery_params,
        initial_charge_mwh=current_soe,
        requires_equivalent_soe=require_equivalent_soe,
        verbose=verbose,
    )
    lmp_series = df["lmp"]
    for i in range(0, len(df) - required_horizon):
        # Update price window for current MPC step
        hf = lmp_series.iloc[i : i + hf_horizon].copy()
        lf_indices = [i + hf_horizon + j * 12 for j in range(lf_horizon)]
        lf = lmp_series.iloc[lf_indices].copy()
        if use_lf_avg:
            lf["lmp"] = lf["lmp_lf_avg"]
        price_window = pd.concat([hf, lf])
        set_objective(model, charge, discharge, times, price_window, dt_vec)
        update_initial_charge(model, init_soe_constr, soe, current_soe)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("MPC step did not reach optimal solution")
        # Extract first-interval decisions
        charge_decisions.append(float(charge.X[0]))
        discharge_decisions.append(float(discharge.X[0]))
        timestamps.append(df.index[i])
        prices.append(float(df.iloc[i]["lmp"]))
        # Advance SoE to next period
        current_soe = float(soe.X[1])
    decisions_df = pd.DataFrame(
        {
            "datetime_beginning_utc": timestamps,
            "price": prices,
            "charge_mw": charge_decisions,
            "discharge_mw": discharge_decisions,
        }
    )
    value_generated = (
        (decisions_df["discharge_mw"] - decisions_df["charge_mw"])
        * decisions_df["price"]
        * (5 / 60)
    ).sum()
    return decisions_df, value_generated


def cold_battery_arb(
    hf_horizon: int,
    lf_horizon: int,
    prices_df: pd.DataFrame,
    battery_params: BatteryParams = DEFAULT_BATTERY,
    require_equivalent_soe: bool = False,
    use_lf_avg: bool = False,
    verbose: bool = False,
):
    """MPC loop rebuilding the model every iteration (no warm start).

    Returns decisions_df, value_generated, and total_seconds runtime.
    """
    required_horizon = hf_horizon + lf_horizon * 12
    df = prices_df.copy()
    charge_decisions = []
    discharge_decisions = []
    timestamps = []
    prices = []
    current_soe = battery_params.initial_charge_mwh
    lmp_series = df["lmp"]
    start_ts = time.perf_counter()
    for i in range(0, len(df) - required_horizon):
        hf_slice = lmp_series.iloc[i : i + hf_horizon].copy()
        lf_indices = [i + hf_horizon + j * 12 for j in range(lf_horizon)]
        lf_slice = lmp_series.iloc[lf_indices].copy()
        if use_lf_avg:
            # If averaging column exists in original df, apply it; else skip silently
            if "lmp_lf_avg" in df.columns:
                lf_slice = df["lmp_lf_avg"].iloc[lf_indices].copy()
        price_window = pd.concat([hf_slice, lf_slice])
        model, soe, charge, discharge, times, dt_vec, init_constr = build_battery_model(
            price_window.index,
            battery_params,
            initial_charge_mwh=current_soe,
            requires_equivalent_soe=require_equivalent_soe,
            verbose=verbose,
        )
        set_objective(model, charge, discharge, times, price_window, dt_vec)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("Cold MPC step not optimal")
        charge_decisions.append(float(charge.X[0]))
        discharge_decisions.append(float(discharge.X[0]))
        timestamps.append(df.index[i])
        prices.append(float(df.iloc[i]["lmp"]))
        current_soe = float(soe.X[1])
    elapsed = time.perf_counter() - start_ts
    decisions_df = pd.DataFrame(
        {
            "datetime_beginning_utc": timestamps,
            "price": prices,
            "charge_mw": charge_decisions,
            "discharge_mw": discharge_decisions,
        }
    )
    value_generated = (
        (decisions_df["discharge_mw"] - decisions_df["charge_mw"])
        * decisions_df["price"]
        * (5 / 60)
    ).sum()
    return decisions_df, value_generated, elapsed


def warm_battery_arb_benchmark(
    hf_horizon_hours: int = 4,
    lf_horizon_hours: int = 0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    pnode_id: int = 2156113094,
    use_lf_avg: bool = False,
    require_equivalent_soe: bool = False,
    verbose: bool = False,
):
    """Benchmark warm-start vs cold-start MPC over a slice of data.

    Prints timing and speedup factor.
    """
    if start is None:
        start = pd.Timestamp(year=2024, month=1, day=1, tz="UTC")
    if end is None:
        end = pd.Timestamp(year=2024, month=2, day=1, tz="UTC")

    lmp_dir = Path("data/pjm_lmps")
    df = read_lmp_folder(lmp_dir)
    df = df[df["pnode_id"] == int(pnode_id)]
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = df.set_index("datetime_beginning_utc").sort_index()
    df = df.loc[start:end]
    df.rename(columns={"lmp_rt": "lmp"}, inplace=True)
    if use_lf_avg:
        df["lmp_lf_avg"] = (
            df["lmp"].rolling(window=13, center=True, min_periods=1).mean()
        )

    hf_intervals = hf_horizon_hours * 12
    lf_intervals = lf_horizon_hours

    # Warm start run
    t0 = time.perf_counter()
    warm_decisions, warm_value = battery_arb(
        hf_horizon=hf_intervals,
        lf_horizon=lf_intervals,
        prices_df=df,
        battery_params=DEFAULT_BATTERY,
        require_equivalent_soe=require_equivalent_soe,
        use_lf_avg=use_lf_avg,
        verbose=verbose,
    )
    warm_elapsed = time.perf_counter() - t0

    # Cold run (rebuild each iteration)
    cold_decisions, cold_value, cold_elapsed = cold_battery_arb(
        hf_horizon=hf_intervals,
        lf_horizon=lf_intervals,
        prices_df=df,
        battery_params=DEFAULT_BATTERY,
        require_equivalent_soe=require_equivalent_soe,
        use_lf_avg=use_lf_avg,
        verbose=verbose,
    )
    # cold_elapsed already measures total time for cold loop

    print("\n=== MPC Warm-Start Benchmark ===")
    print(
        f"Data window: {start} to {end} | HF horizon: {hf_horizon_hours}h | LF horizon: {lf_horizon_hours}h"
    )
    print(f"Iterations: {len(warm_decisions)}")
    print(f"Warm: {warm_elapsed:.4f}s  (value=${warm_value:,.2f})")
    print(f"Cold (loop rebuild): {cold_elapsed:.4f}s  (value=${cold_value:,.2f})")
    speedup = cold_elapsed / warm_elapsed if warm_elapsed > 0 else float("inf")
    print(f"Speedup (cold/warm): {speedup:.2f}x")

    return {
        "iterations": len(warm_decisions),
        "warm_seconds": warm_elapsed,
        "cold_seconds": cold_elapsed,
        "speedup": speedup,
        "warm_value": warm_value,
        "cold_value": cold_value,
    }


if __name__ == "__main__":
    # Simple benchmark invocation
    benchmark_stats = warm_battery_arb_benchmark(
        hf_horizon_hours=12,
        lf_horizon_hours=12,
        start=pd.Timestamp(year=2024, month=1, day=1, tz="UTC"),
        end=pd.Timestamp(year=2024, month=1, day=15, tz="UTC"),
        pnode_id=2156113094,
        use_lf_avg=False,
        require_equivalent_soe=False,
        verbose=False,
    )
    # Optional: print structured dict
    print("Benchmark summary:", benchmark_stats)
