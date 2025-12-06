import math
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
    battery: BatteryParams = DEFAULT_BATTERY,
):
    assert "lmp_rt" in prices_df.columns, "prices_df must contain 'lmp_rt' column"
    assert "lmp_da" in prices_df.columns, "prices_df must contain 'lmp_da' column"
    X = math.floor(
        battery.capacity_mwh / max(battery.max_discharge_mw, battery.max_charge_mw)
    )
    print(f"txbx heuristic with X={X} based on battery parameters")
    if X <= 0:
        raise ValueError("X must be a positive integer")
    if X > 12:  # Arbitrary upper limit for practicality
        raise ValueError("X is too large; must be <= 12")

    # Prepare decisions
    dec = pd.DataFrame(index=prices_df.index)
    dec["lmp"] = prices_df["lmp_rt"]
    dec["lmp_da"] = prices_df["lmp_da"]
    dec["charge_mw"] = 0.0
    dec["discharge_mw"] = 0.0

    # Group by day and pick top/bottom X hours
    for _, grp in dec.groupby(dec.index.normalize()):
        five_minute_intervals = grp.index
        if len(five_minute_intervals) == 0:
            continue
        k = min(X * 12, len(five_minute_intervals) // 2)
        # Bottom X for charging
        bottom_idx = grp.nsmallest(k, columns="lmp_da").index
        top_idx = grp.nlargest(k, columns="lmp_da").index
        dec.loc[bottom_idx, "charge_mw"] = battery.max_charge_mw / battery.in_efficiency
        dec.loc[top_idx, "discharge_mw"] = (
            battery.max_discharge_mw * battery.out_efficiency
        )

    # Revenue (hourly): (discharge - charge) * price * 1h
    revenue = float(
        ((dec["discharge_mw"] - dec["charge_mw"]) * dec["lmp"] * (5.0 / 60.0)).sum()
    )
    return dec, revenue


def deterministic_arbitrage_opt(
    prices_df: pd.DataFrame,
    battery: BatteryParams = DEFAULT_BATTERY,
    verbose: bool = False,
    require_equivalent_soe: bool = False,
    initial_charge_mwh: float | None = None,
    use_barrier: bool = True,
) -> tuple[pd.DataFrame, float]:
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
    if use_barrier:
        # Force barrier algorithm and disable crossover for speed (no basis needed).
        model.Params.Method = 2  # Barrier
        # model.Params.Crossover = 0  # Skip crossover
        model.Params.Presolve = 2  # Aggressive presolve
        model.Params.BarHomogeneous = 1  # Homogeneous form (often more robust)

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
