import gurobipy as gp
import numpy as np
import pandas as pd

from deterministic.single_market_battery import DEFAULT_BATTERY, BatteryParams


def build_two_stage_battery_model(
    prices_index,
    battery: BatteryParams,
    branches: int,
    initial_charge_mwh: float | None = None,
    requires_equivalent_soe: bool = False,
    verbose: bool = False,
):
    """Vectorized build of stochastic battery arbitrage LP using Gurobi MVar.

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

    # First step variables
    init_soe = model.addVar(lb=0.0, ub=battery.capacity_mwh, name="soe")
    init_charge = model.addVar(lb=0.0, ub=battery.max_charge_mw, name="charge")
    init_discharge = model.addVar(lb=0.0, ub=battery.max_discharge_mw, name="discharge")

    # branching
    soe_branches = []
    charge_branches = []
    discharge_branches = []
    for b in range(branches):
        soe = model.addMVar(T - 1, lb=0.0, ub=battery.capacity_mwh, name="soe")
        charge = model.addMVar(T - 2, lb=0.0, ub=battery.max_charge_mw, name="charge")
        discharge = model.addMVar(
            T - 2, lb=0.0, ub=battery.max_discharge_mw, name="discharge"
        )
        soe_branches.append(soe)
        charge_branches.append(charge)
        discharge_branches.append(discharge)

    # Initial SoE equality constraint (store for later RHS updates in MPC)
    init_val = (
        battery.initial_charge_mwh
        if initial_charge_mwh is None
        else float(initial_charge_mwh)
    )
    init_soe_constr = model.addConstr(init_soe == init_val, name="init_soe")
    if requires_equivalent_soe:
        for b in range(branches):
            model.addConstr(soe_branches[b][-1] == init_soe, name=f"eq_soe_{b}")

    # Vector dynamics: soe[1:] = soe[:-1] + (charge*in_eff - discharge/out_eff - soe[:-1]*self_discharge)*dt
    for b in range(branches):
        soe = soe_branches[b]
        charge = charge_branches[b]
        discharge = discharge_branches[b]
        model.addConstr(
            soe[0]
            == init_soe
            + (
                init_charge * battery.in_efficiency
                - init_discharge / battery.out_efficiency
                - init_soe * battery.self_discharge_percent_per_hour
            )
            * float(dt_vec[0]),
            name=f"first_soe_dynamics_branch_{b}",
        )
        model.addConstr(
            soe[1:]
            == soe[:-1]
            + (
                charge * battery.in_efficiency
                - discharge / battery.out_efficiency
                - soe[:-1] * battery.self_discharge_percent_per_hour
            )
            * dt_vec[1:],
            name=f"soe_dynamics_vec_branch_{b}",
        )

    model.update()
    return (
        model,
        init_soe,
        init_charge,
        init_discharge,
        charge_branches,
        discharge_branches,
        times,
        dt_vec,
        init_soe_constr,
    )


def set_two_stage_objective(
    model,
    init_price,
    init_charge,
    init_discharge,
    charge_branches,
    discharge_branches,
    price_series: list,
    dt_vec,
):
    """Vectorized objective: maximize sum price_t * (discharge_t - charge_t) * dt_t."""
    assert len(charge_branches) == len(discharge_branches)
    assert len(charge_branches) == len(price_series)
    expr = gp.LinExpr()
    expr += init_price * dt_vec[0] * (init_discharge - init_charge)
    branch_fraction = 1.0 / len(charge_branches)
    for i, series in enumerate(price_series):
        price_vec = np.asarray(series, dtype=float)
        coeff = price_vec * dt_vec[1:]
        charge = charge_branches[i]
        discharge = discharge_branches[i]
        expr += branch_fraction * (coeff @ (discharge - charge))

    model.setObjective(expr, gp.GRB.MAXIMIZE)


def update_two_stage_initial_charge(
    model: gp.Model,
    init_soe_constr: gp.Constr,
    init_soe: gp.MVar,
    new_initial_charge: float,
):
    init_soe_constr.RHS = float(new_initial_charge)
    # Tighten bounds (optional but can help dual simplex)
    init_soe.LB = float(new_initial_charge)
    init_soe.UB = float(new_initial_charge)
    model.update()


if __name__ == "__main__":
    # Smoke test: two-stage stochastic MPC-style loop with two forecast branches.
    # This intentionally uses tiny dummy data to validate model build, objective set,
    # and RHS update for warm-started re-optimizations.

    def _mvar_to_scalar(x: gp.MVar) -> float:
        arr = np.asarray(x.X, dtype=float).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"Expected scalar MVar, got shape {arr.shape}")
        return float(arr[0])

    battery = DEFAULT_BATTERY
    branches = 3

    # 4 timestamps; builder appends one more to define 4 intervals.
    prices_index = pd.date_range("2025-01-01 00:00", periods=8, freq="h", tz="UTC")

    # Two simple future price forecasts (for the *future* intervals after the first action).
    # Must have length = len(dt_vec[1:]) which is 3 for this setup.
    forecast_high_late = [30.0, 80.0, 120.0, 150.0, 140.0, 150.0, 160.0]
    forecast_low_late = [70.0, 20.0, 10.0, 5.0, 10.0, 15.0, 20.0]
    # Third branch: descending from high to low to encourage earlier discharge.
    forecast_descending = [120.0, 60.0, 20.0, 10.0, 5.0, 2.0, 1.0]

    (
        model,
        init_soe,
        init_charge,
        init_discharge,
        charge_branches,
        discharge_branches,
        times,
        dt_vec,
        init_soe_constr,
    ) = build_two_stage_battery_model(
        prices_index=prices_index,
        battery=battery,
        branches=branches,
        initial_charge_mwh=battery.capacity_mwh / 2.0,
        requires_equivalent_soe=True,
        verbose=False,
    )

    # Minimal MPC-like loop: repeatedly re-solve after updating initial SoE.
    realized_prices = [100.0, 40.0, 80.0, 20.0, 150.0]  # for first interval of each iteration
    hist_iter: list[int] = []
    hist_price: list[float] = []
    hist_charge0: list[float] = []
    hist_discharge0: list[float] = []
    hist_soe: list[float] = []
    hist_next_soe: list[float] = []

    for k, init_price in enumerate(realized_prices, start=1):
        set_two_stage_objective(
            model=model,
            init_price=float(init_price),
            init_charge=init_charge,
            init_discharge=init_discharge,
            charge_branches=charge_branches,
            discharge_branches=discharge_branches,
            price_series=[forecast_high_late, forecast_low_late, forecast_descending],
            dt_vec=dt_vec,
        )
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Optimization failed with status={model.Status}")

        soe0 = _mvar_to_scalar(init_soe)
        c0 = _mvar_to_scalar(init_charge)
        d0 = _mvar_to_scalar(init_discharge)
        dt0 = float(dt_vec[0])

        print(f"\n=== MPC iteration {k} ===")
        print(f"init_price: {float(init_price):.2f} $/MWh")
        print(f"init_soe:   {soe0:.4f} MWh")
        print(f"charge0:    {c0:.4f} MW")
        print(f"discharge0: {d0:.4f} MW")
        print(f"obj:        {float(model.ObjVal):.4f}")

        # Apply the first action to get the next initial SoE (simple simulation step).
        next_soe = (
            soe0
            + (
                c0 * battery.in_efficiency
                - d0 / battery.out_efficiency
                - soe0 * battery.self_discharge_percent_per_hour
            )
            * dt0
        )
        next_soe = float(np.clip(next_soe, 0.0, battery.capacity_mwh))

        hist_iter.append(k)
        hist_price.append(float(init_price))
        hist_charge0.append(c0)
        hist_discharge0.append(d0)
        hist_soe.append(soe0)
        hist_next_soe.append(next_soe)

        update_two_stage_initial_charge(
            model=model,
            init_soe_constr=init_soe_constr,
            init_soe=init_soe,
            new_initial_charge=next_soe,
        )

    print("\nSmoke test complete.")

    # ---- Plotting (optional) ----
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable; skipping plots ({e}).")
        raise SystemExit(0)

    history_df = pd.DataFrame(
        {
            "price": hist_price,
            "charge0_mw": hist_charge0,
            "discharge0_mw": hist_discharge0,
            "soe_mwh": hist_soe,
            "next_soe_mwh": hist_next_soe,
        },
        index=pd.Index(hist_iter, name="mpc_iter"),
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_p, ax_u, ax_s = axes

    ax_p.plot(
        history_df.index, history_df["price"], marker="o", label="realized init_price"
    )
    ax_p.set_ylabel("$/MWh")
    ax_p.set_title("Two-stage MPC smoke test")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="best")

    ax_u.step(
        history_df.index, history_df["charge0_mw"], where="post", label="charge0 (MW)"
    )
    ax_u.step(
        history_df.index,
        history_df["discharge0_mw"],
        where="post",
        label="discharge0 (MW)",
    )
    ax_u.set_ylabel("MW")
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(loc="best")

    # SoE at start of each iteration, plus the next SoE after applying the action.
    ax_s.plot(history_df.index, history_df["soe_mwh"], marker="o", label="SoE start")
    ax_s.plot(
        history_df.index, history_df["next_soe_mwh"], marker="o", label="SoE next"
    )
    ax_s.set_xlabel("MPC iteration")
    ax_s.set_ylabel("MWh")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="best")

    fig.tight_layout()

    # Forecast branch sanity-check plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    horizon_steps = np.arange(1, 1 + len(forecast_high_late))
    ax.plot(
        horizon_steps,
        forecast_high_late,
        marker="o",
        label="forecast branch A (ascending)",
    )
    ax.plot(
        horizon_steps,
        forecast_low_late,
        marker="o",
        label="forecast branch B (descending low)",
    )
    ax.plot(
        horizon_steps,
        forecast_descending,
        marker="o",
        label="forecast branch C (descending high)",
    )
    ax.set_xlabel("forecast step (after first action)")
    ax.set_ylabel("$/MWh")
    ax.set_title("Dummy forecast branches")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig2.tight_layout()

    plt.show()
