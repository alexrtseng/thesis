from dataclasses import dataclass

import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class BatteryParams:
    capacity_mwh: float = 20.0
    max_charge_mw: float = 5.0
    max_discharge_mw: float = 5.0
    initial_charge_mwh: float = 5.0
    in_efficiency: float = 0.98
    out_efficiency: float = 0.98
    self_discharge_percent_per_hour: float = 0.0


DEFAULT_BATTERY = BatteryParams()


def txbx(
    prices_df: pd.DataFrame,
    x: int,
    battery: BatteryParams = DEFAULT_BATTERY,
    verbose: bool = False,
):
    """Heuristic schedule: charge at lowest x hours, discharge at highest x hours.

    Requirements:
    - max_charge_mw must equal max_discharge_mw
    - max_discharge_mw * x must be less than capacity_mwh

    Behavior:
    - Uses per-interval durations implied by the DatetimeIndex to find a set of
      disjoint intervals totaling x hours for charging (lowest prices) and x hours
      for discharging (highest prices). The final interval on each side may be
      partially allocated.
    - Produces a DataFrame with columns charge_mw and discharge_mw (MW), aligned to
      the input index; the last timestamp has zero duration and thus contributes
      nothing to revenue.
    - Returns (decisions_df, revenue_$).
    """
    # Input validation
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        raise ValueError("prices_df must be indexed by a DatetimeIndex")
    if "lmp" not in prices_df.columns:
        raise ValueError("prices_df must contain an 'lmp' column")
    if not isinstance(x, int) or x <= 0:
        raise ValueError("x must be a positive integer (hours)")

    prices_df = prices_df.sort_index()

    # Battery checks
    if float(battery.max_charge_mw) != float(battery.max_discharge_mw):
        raise ValueError("max_charge_mw must equal max_discharge_mw for txbx()")
    if float(battery.max_discharge_mw) * float(x) > float(battery.capacity_mwh):
        raise ValueError("max_discharge_mw * x must be less than capacity_mwh")

    # Build interval table (ignore the last row which has no forward interval)
    times = prices_df.index
    if len(times) < 2:
        raise ValueError("Need at least two timestamps to define time intervals")
    dt_hours = (times[1:] - times[:-1]).to_numpy(dtype="timedelta64[s]").astype(
        float
    ) / 3600.0
    # Construct per-interval DataFrame aligned to the starting timestamps
    intervals = pd.DataFrame(
        {
            "price": prices_df["lmp"].iloc[:-1].astype(float).values,
            "dt_hours": dt_hours,
        },
        index=times[:-1],
    )

    total_hours = float(intervals["dt_hours"].sum())
    if total_hours < 2.0 * float(x):
        raise ValueError(
            "Insufficient data length: need at least 2*x hours of intervals to allocate charge and discharge disjointly."
        )

    max_mw = float(battery.max_charge_mw)

    # Helper: greedy selection to accumulate up to x hours with possible partial last interval
    def select_intervals(
        df: pd.DataFrame, ascending: bool, hours: float
    ) -> pd.DataFrame:
        cand = df.sort_values("price", ascending=ascending).copy()
        cand["alloc"] = 0.0
        remaining = float(hours)
        for idx, row in cand.iterrows():
            if remaining <= 0:
                break
            h = float(row["dt_hours"]) if float(row["dt_hours"]) > 0 else 0.0
            if h <= 0:
                continue
            take = min(h, remaining)
            cand.at[idx, "alloc"] = take / h  # fraction in [0,1]
            remaining -= take
        # Keep only rows with nonzero allocation
        return cand[cand["alloc"] > 0].copy()

    # 1) Choose charge intervals (lowest prices)
    charge_sel = select_intervals(intervals, ascending=True, hours=float(x))
    used_index = set(charge_sel.index)

    # 2) Choose discharge intervals (highest prices) from remaining rows
    discharge_pool = intervals.loc[~intervals.index.isin(used_index)]
    discharge_sel = select_intervals(discharge_pool, ascending=False, hours=float(x))

    # Construct decisions aligned to full index
    decisions = pd.DataFrame(
        {"charge_mw": 0.0, "discharge_mw": 0.0}, index=prices_df.index
    )
    # Apply allocations (scale MW by fraction to handle partial intervals)
    for idx, row in charge_sel.iterrows():
        frac = float(row["alloc"])  # fraction of that interval
        decisions.at[idx, "charge_mw"] = max_mw * frac
        decisions.at[idx, "discharge_mw"] = 0.0
    for idx, row in discharge_sel.iterrows():
        frac = float(row["alloc"])  # fraction of that interval
        decisions.at[idx, "discharge_mw"] = max_mw * frac
        # Ensure disjointness already enforced; no need to zero charge here

    # Compute revenue using forward interval dt
    # Align dt_hours back to the full index (last timestamp gets 0)
    dt_full = pd.Series(0.0, index=prices_df.index, dtype=float)
    dt_full.iloc[:-1] = intervals["dt_hours"].values
    net_mw = (decisions["discharge_mw"] - decisions["charge_mw"]).astype(float)
    revenue = (prices_df["lmp"].astype(float) * net_mw * dt_full).sum()

    return decisions, float(revenue)


def deterministic_arbitrage_opt(
    prices_df: pd.DataFrame,
    battery: BatteryParams = DEFAULT_BATTERY,
    verbose: bool = False,
    require_equivalent_soe: bool = False,
    initial_charge_mwh: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Solve a simple deterministic arbitrage for a single battery.

    prices_df: DataFrame indexed by DatetimeIndex with column 'lmp' (price in $/MWh).
        battery: BatteryParams with fields capacity_mwh, max_charge_mw, max_discharge_mw,
            initial_charge_mwh, in_efficiency, out_efficiency, self_discharge_percent_per_hour.
    Returns a DataFrame indexed by timestamp with columns:
      state_of_energy_mwh, charge_mw, discharge_mw
    """

    # Basic validation and normalization
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        raise ValueError("prices_df must be indexed by a DatetimeIndex")
    if "lmp" not in prices_df.columns:
        raise ValueError("prices_df must contain an 'lmp' column")
    prices_df = prices_df.sort_index()

    capacity_mwh = float(battery.capacity_mwh)
    max_charge_mw = float(battery.max_charge_mw)
    max_discharge_mw = float(battery.max_discharge_mw)
    initial_charge_mwh = (
        float(battery.initial_charge_mwh)
        if initial_charge_mwh is None
        else float(initial_charge_mwh)
    )
    in_efficiency = float(battery.in_efficiency)
    out_efficiency = float(battery.out_efficiency)
    self_discharge_percent_per_hour = float(battery.self_discharge_percent_per_hour)

    # Build model
    model = gp.Model("battery_arbitrage")
    if not verbose:
        model.Params.OutputFlag = 0

    times = list(prices_df.index)
    T = len(times)
    if T < 2:
        raise ValueError("Need at least two timestamps to define time intervals")

    # Decision variables per time step
    soe: list[gp.Var] = [model.addVar(lb=0.0, ub=capacity_mwh, name="soe_0")]
    # Fix initial SoE
    model.addConstr(soe[0] == initial_charge_mwh, name="init_soe")
    charge: list[gp.Var] = []
    discharge: list[gp.Var] = []

    # Dynamics and objective over intervals [t, t+1)
    obj_expr = gp.LinExpr()
    for t in range(T - 1):
        dt_hours = (times[t + 1] - times[t]).total_seconds() / 3600.0
        c = model.addVar(lb=0.0, ub=max_charge_mw, name=f"charge_{t}")
        d = model.addVar(lb=0.0, ub=max_discharge_mw, name=f"discharge_{t}")
        charge.append(c)
        discharge.append(d)

        next_soe = model.addVar(lb=0.0, ub=capacity_mwh, name=f"soe_{t + 1}")
        soe.append(next_soe)

        # SoE dynamics with efficiency and self-discharge
        model.addConstr(
            next_soe
            == soe[t]
            + (
                c * in_efficiency
                - d / out_efficiency
                - soe[t] * self_discharge_percent_per_hour
            )
            * dt_hours,
            name=f"soe_dyn_{t}",
        )

        # Revenue over interval [t, t+1): price at t
        price = float(prices_df.iloc[t]["lmp"])  # $/MWh
        obj_expr += price * (d - c) * dt_hours

    if require_equivalent_soe:
        model.addConstr(soe[-1] == initial_charge_mwh, name="final_soe_equal_init")
    model.setObjective(obj_expr, gp.GRB.MAXIMIZE)
    model.optimize()

    # Extract numeric results
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError("Optimization did not find optimal solution")
    else:
        result_df = pd.DataFrame(
            {
                "state_of_energy_mwh": [v.X for v in soe],
                "charge_mw": [v.X for v in charge] + [0.0],
                "discharge_mw": [v.X for v in discharge] + [0.0],
            },
            index=times,
        )
    return result_df, model.ObjVal


# Reserved for potential future use
def calc_reg_profit(reg_prices: pd.DataFrame, battery: BatteryParams) -> float:
    if not isinstance(reg_prices.index, pd.DatetimeIndex):
        raise ValueError("reg_prices must be indexed by a DatetimeIndex")
    if "mcp" not in reg_prices.columns:
        raise ValueError("prices_df must contain an 'mcp' column")

    T = len(reg_prices)
    if T < 2:
        raise ValueError("Need at least two timestamps to define time intervals")

    profit = 0.0
    for i in range(T - 2):
        dt_hours = (
            reg_prices.index[i + 1] - reg_prices.index[i]
        ).total_seconds() / 3600.0
        profit += (
            reg_prices["mcp"].iloc[i]
            * dt_hours
            * min(battery.max_charge_mw, battery.max_discharge_mw)
        )

    return profit


if __name__ == "__main__":
    # Build a tiny artificial price series that clearly incentivizes charge then discharge
    # Pattern: low (10), high (100), low (10), high (100), ... hourly steps
    idx = pd.date_range("2025-01-01 00:00", periods=8, freq="h", tz="UTC")
    prices = [10, 100, 10, 100, 10, 100, 10, 100]
    toy_df = pd.DataFrame({"lmp": prices}, index=idx)

    print("Toy LMP input (first 8 hours):")
    print(toy_df)

    res, profit = deterministic_arbitrage_opt(toy_df, DEFAULT_BATTERY)
    print("\nOptimization results:")
    print(res)

    # Compute realized profit for readability
    dt_hours = (
        res.index.to_series().shift(-1) - res.index.to_series()
    ).dt.total_seconds().fillna(0) / 3600.0
    interval_price = toy_df["lmp"].shift(0)  # price at start of interval
    net_mw = (res["discharge_mw"] - res["charge_mw"]).shift(0)
    revenue = (interval_price * net_mw * dt_hours).fillna(0.0)
    print(f"\nTotal profit ($): {revenue.sum():.2f}")

    # Plot

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Price
    axes[0].plot(toy_df.index, toy_df["lmp"], label="LMP ($/MWh)", color="tab:blue")
    axes[0].set_ylabel("$/MWh")
    axes[0].legend(loc="upper left")

    # Charge/Discharge
    axes[1].step(
        res.index,
        res["charge_mw"],
        where="post",
        label="Charge (MW)",
        color="tab:green",
    )
    axes[1].step(
        res.index,
        -res["discharge_mw"],
        where="post",
        label="-Discharge (MW)",
        color="tab:red",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("MW")
    axes[1].legend(loc="upper left")

    # SoE
    axes[2].step(
        res.index,
        res["state_of_energy_mwh"],
        where="post",
        label="SoE (MWh)",
        color="tab:purple",
    )
    axes[2].set_ylabel("MWh")
    axes[2].set_xlabel("Time (UTC)")
    axes[2].legend(loc="upper left")

    fig.suptitle("Deterministic Arbitrage Toy Example")
    fig.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # More realistic two-day synthetic price scenario
    # ---------------------------------------------------------------
    print(
        "\n\nTwo-day synthetic price scenario (diurnal pattern with midday lows and evening peaks):"
    )
    idx2 = pd.date_range("2025-01-03 00:00", periods=48, freq="h", tz="UTC")
    prices2: list[float] = []
    for ts in idx2:
        h = ts.hour
        if 0 <= h <= 5:
            p = 18.0  # overnight low
        elif 6 <= h <= 8:
            p = 45.0  # morning ramp
        elif 9 <= h <= 15:
            p = 0.0  # midday renewable surplus, occasionally negative
        elif 16 <= h <= 20:
            p = 110.0  # evening peak
        else:  # 21-23
            p = 35.0  # late evening
        prices2.append(p)
    two_day_df = pd.DataFrame({"lmp": prices2}, index=idx2)

    print(two_day_df.head(12))
    print("...")
    print(two_day_df.tail(12))

    res2, profit2 = deterministic_arbitrage_opt(two_day_df, DEFAULT_BATTERY)
    # Summarize rather than print the entire 48 rows
    print("\nOptimization results (head):")
    print(res2.head(12))
    print("...\nOptimization results (tail):")
    print(res2.tail(12))

    dt_hours2 = (
        res2.index.to_series().shift(-1) - res2.index.to_series()
    ).dt.total_seconds().fillna(0) / 3600.0
    interval_price2 = two_day_df["lmp"].reindex(res2.index).shift(0)
    net_mw2 = (res2["discharge_mw"] - res2["charge_mw"]).shift(0)
    revenue2 = (interval_price2 * net_mw2 * dt_hours2).fillna(0.0)
    print(f"\nTotal profit over 2 days ($): {revenue2.sum():.2f}")

    # Plot two-day scenario
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes2[0].plot(
        two_day_df.index, two_day_df["lmp"], label="LMP ($/MWh)", color="tab:blue"
    )
    axes2[0].set_ylabel("$/MWh")
    axes2[0].legend(loc="upper left")

    axes2[1].step(
        res2.index,
        res2["charge_mw"],
        where="post",
        label="Charge (MW)",
        color="tab:green",
    )
    axes2[1].step(
        res2.index,
        -res2["discharge_mw"],
        where="post",
        label="-Discharge (MW)",
        color="tab:red",
    )
    axes2[1].axhline(0, color="black", linewidth=0.8)
    axes2[1].set_ylabel("MW")
    axes2[1].legend(loc="upper left")

    axes2[2].step(
        res2.index,
        res2["state_of_energy_mwh"],
        where="post",
        label="SoE (MWh)",
        color="tab:purple",
    )
    axes2[2].set_ylabel("MWh")
    axes2[2].set_xlabel("Time (UTC)")
    axes2[2].legend(loc="upper left")

    fig2.suptitle("Deterministic Arbitrage — Two-day Synthetic Scenario")
    fig2.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # Overlay deterministic vs txbx (x=2) on a multi-day scenario
    # ---------------------------------------------------------------
    print("\n\nOverlay: Deterministic vs txbx (x=2) over 48 hours (2 days)")
    idx3 = pd.date_range("2025-01-07 00:00", periods=48, freq="h", tz="UTC")
    lmp_values3: list[float] = []
    for ts in idx3:
        h = ts.hour
        # Use the same diurnal structure as above
        if 0 <= h <= 5:
            p = 18.0
        elif 6 <= h <= 8:
            p = 45.0
        elif 9 <= h <= 15:
            p = 0.0
        elif 16 <= h <= 20:
            p = 110.0
        else:  # 21-23
            p = 35.0
        lmp_values3.append(p)
    three_day_df = pd.DataFrame({"lmp": lmp_values3}, index=idx3)

    det_res3, det_profit3 = deterministic_arbitrage_opt(three_day_df, DEFAULT_BATTERY)
    tx_res3, tx_profit3 = txbx(three_day_df, x=2, battery=DEFAULT_BATTERY)

    print(f"Deterministic objective ($): {det_profit3:.2f}")
    print(f"txbx revenue ($): {tx_profit3:.2f}")

    # Build net MW series for both
    det_net3 = (det_res3["discharge_mw"] - det_res3["charge_mw"]).astype(float)
    tx_net3 = (tx_res3["discharge_mw"] - tx_res3["charge_mw"]).astype(float)

    fig3, axes3 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Prices
    axes3[0].plot(
        three_day_df.index, three_day_df["lmp"], label="LMP ($/MWh)", color="tab:blue"
    )
    axes3[0].set_ylabel("$/MWh")
    axes3[0].legend(loc="upper left")

    # Charge overlay
    axes3[1].step(
        det_res3.index,
        det_res3["charge_mw"],
        where="post",
        label="Deterministic Charge",
        color="tab:green",
    )
    axes3[1].step(
        tx_res3.index,
        tx_res3["charge_mw"],
        where="post",
        label="txbx Charge",
        color="tab:olive",
        linestyle="--",
    )
    axes3[1].set_ylabel("MW")
    axes3[1].legend(loc="upper left")

    # Discharge overlay
    axes3[2].step(
        det_res3.index,
        det_res3["discharge_mw"],
        where="post",
        label="Deterministic Discharge",
        color="tab:red",
    )
    axes3[2].step(
        tx_res3.index,
        tx_res3["discharge_mw"],
        where="post",
        label="txbx Discharge",
        color="tab:pink",
        linestyle="--",
    )
    axes3[2].set_ylabel("MW")
    axes3[2].legend(loc="upper left")

    # Net MW overlay
    axes3[3].step(
        det_net3.index,
        det_net3,
        where="post",
        label="Deterministic Net MW",
        color="tab:purple",
    )
    axes3[3].step(
        tx_net3.index,
        tx_net3,
        where="post",
        label="txbx Net MW",
        color="tab:brown",
        linestyle="--",
    )
    axes3[3].axhline(0, color="black", linewidth=0.8)
    axes3[3].set_ylabel("MW")
    axes3[3].set_xlabel("Time (UTC)")
    axes3[3].legend(loc="upper left")

    fig3.suptitle("Deterministic vs txbx (x=2) — 2-day Overlay")
    fig3.tight_layout()
    plt.show()
