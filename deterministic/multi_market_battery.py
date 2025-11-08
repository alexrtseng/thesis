import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd
from single_market_battery import DEFAULT_BATTERY, BatteryParams


def deterministic_reg_arbitrage_battery_opt(
    reg_prices: pd.DataFrame,
    arbitrage_prices: pd.DataFrame,
    battery: BatteryParams = DEFAULT_BATTERY,
    verbose: bool = False,
) -> tuple[pd.DataFrame, float]:
    # Basic validation and normalization
    if not isinstance(reg_prices.index, pd.DatetimeIndex):
        raise ValueError("reg_prices must be indexed by a DatetimeIndex")
    if "mcp" not in reg_prices.columns:
        raise ValueError("reg_prices must contain an 'mcp' column")

    if not isinstance(arbitrage_prices.index, pd.DatetimeIndex):
        raise ValueError("arbitrage_prices must be indexed by a DatetimeIndex")
    if "lmp" not in arbitrage_prices.columns:
        raise ValueError("arbitrage_prices must contain an 'lmp' column")
    reg_prices = reg_prices.sort_index()
    arbitrage_prices = arbitrage_prices.sort_index()
    # keep in mind that this is inner join
    prices_df = pd.concat([reg_prices, arbitrage_prices], axis=1, join="inner")

    capacity_mwh = float(battery.capacity_mwh)
    max_charge_mw = float(battery.max_charge_mw)
    max_discharge_mw = float(battery.max_discharge_mw)
    initial_charge_mwh = float(battery.initial_charge_mwh)
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
    reg_allocation: list[gp.Var] = []
    charge: list[gp.Var] = []
    discharge: list[gp.Var] = []

    # Dynamics and objective over intervals [t, t+1)
    obj_expr = gp.LinExpr()
    for t in range(T - 1):
        dt_hours = (times[t + 1] - times[t]).total_seconds() / 3600.0
        c = model.addVar(lb=0.0, ub=max_charge_mw, name=f"charge_{t}")
        d = model.addVar(lb=0.0, ub=max_discharge_mw, name=f"discharge_{t}")
        r_allocation = model.addVar(lb=0.0, name=f"reg_allocation_{t}")
        charge.append(c)
        discharge.append(d)
        reg_allocation.append(r_allocation)

        next_soe = model.addVar(lb=0.0, ub=capacity_mwh, name=f"soe_{t + 1}")
        model.addConstr(c + r_allocation <= max_charge_mw, name=f"charge_cap_{t}")
        model.addConstr(d + r_allocation <= max_discharge_mw, name=f"discharge_cap_{t}")
        model.addConstr(
            soe[t] + r_allocation * dt_hours <= capacity_mwh, name=f"reg_soc_cap_{t}"
        )
        model.addConstr(
            soe[t] - r_allocation * dt_hours >= 0.0, name=f"reg_soc_floor_{t}"
        )
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
        arb_price = float(prices_df.iloc[t]["lmp"])  # $/MWh
        reg_price = float(prices_df.iloc[t]["mcp"])  # $/MW h
        obj_expr += arb_price * (d - c) * dt_hours
        obj_expr += reg_price * r_allocation * dt_hours

    model.setObjective(obj_expr, gp.GRB.MAXIMIZE)
    model.optimize()

    # Extract numeric results
    result_df = pd.DataFrame(
        {
            "state_of_energy_mwh": [v.X for v in soe],
            "charge_mw": [v.X for v in charge] + [0.0],
            "discharge_mw": [v.X for v in discharge] + [0.0],
            "reg_allocation_mw": [v.X for v in reg_allocation] + [0.0],
        },
        index=times,
    )
    return result_df, model.ObjVal


if __name__ == "__main__":
    # Super simple, easy-to-visualize 12-hour scenario
    # - LMP: low -> moderate -> high peak -> moderate
    # - MCP: reg revenue available mostly in moderate periods
    idx = pd.date_range("2025-01-05 00:00", periods=12, freq="H", tz="UTC")

    lmp_values: list[float] = []
    mcp_values: list[float] = []
    for ts in idx:
        h = ts.hour
        # Arbitrage prices ($/MWh)
        if 0 <= h <= 2:
            lmp = 12.0  # cheap
        elif 3 <= h <= 5:
            lmp = 40.0  # moderate
        elif 6 <= h <= 8:
            lmp = 110.0  # peak
        else:  # 9-11
            lmp = 35.0  # evening moderate
        lmp_values.append(lmp)

        # Regulation MCP ($/MW·h)
        if 0 <= h <= 2:
            mcp = 0.0  # encourage pure charging
        elif 3 <= h <= 5:
            mcp = 30.0  # allocate to reg while maybe mild charging/discharging
        elif 6 <= h <= 8:
            mcp = 0.0  # focus on arbitrage discharge at peak
        else:  # 9-11
            mcp = 20.0  # some reg revenue in the evening
        mcp_values.append(mcp)

    reg_df = pd.DataFrame({"mcp": mcp_values}, index=idx)
    arb_df = pd.DataFrame({"lmp": lmp_values}, index=idx)

    print("Regulation MCP (first 12 hours):")
    print(reg_df)
    print("\nArbitrage LMP (first 12 hours):")
    print(arb_df)

    res, total_profit = deterministic_reg_arbitrage_battery_opt(
        reg_df, arb_df, DEFAULT_BATTERY
    )
    print("\nOptimization results:")
    print(res)

    # Revenue breakdown
    dt_hours = (
        res.index.to_series().shift(-1) - res.index.to_series()
    ).dt.total_seconds().fillna(0) / 3600.0
    arb_revenue = (
        arb_df["lmp"].shift(0)
        * (res["discharge_mw"] - res["charge_mw"]).shift(0)
        * dt_hours
    ).fillna(0.0)
    reg_revenue = (
        reg_df["mcp"].shift(0) * res["reg_allocation_mw"].shift(0) * dt_hours
    ).fillna(0.0)
    total_profit_check = float((arb_revenue + reg_revenue).sum())
    print(f"\nArbitrage revenue ($): {arb_revenue.sum():.2f}")
    print(f"Regulation revenue ($): {reg_revenue.sum():.2f}")
    print(
        f"Total profit ($): {total_profit:.2f} (recomputed: {total_profit_check:.2f})"
    )

    # Plots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Prices
    axes[0].plot(arb_df.index, arb_df["lmp"], label="LMP ($/MWh)", color="tab:blue")
    axes[0].set_ylabel("$/MWh")
    axes[0].legend(loc="upper left")

    axes[1].plot(
        reg_df.index, reg_df["mcp"], label="Reg MCP ($/MW·h)", color="tab:orange"
    )
    axes[1].set_ylabel("$/MW·h")
    axes[1].legend(loc="upper left")

    # Charge/Discharge
    axes[2].step(
        res.index,
        res["charge_mw"],
        where="post",
        label="Charge (MW)",
        color="tab:green",
    )
    axes[2].step(
        res.index,
        -res["discharge_mw"],
        where="post",
        label="-Discharge (MW)",
        color="tab:red",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("MW")
    axes[2].legend(loc="upper left")

    # SoE and Regulation allocation
    axes[3].step(
        res.index,
        res["state_of_energy_mwh"],
        where="post",
        label="SoE (MWh)",
        color="tab:purple",
    )
    ax3b = axes[3].twinx()
    ax3b.step(
        res.index,
        res["reg_allocation_mw"],
        where="post",
        label="Reg (MW)",
        color="tab:brown",
    )
    axes[3].set_ylabel("MWh")
    ax3b.set_ylabel("MW")
    axes[3].set_xlabel("Time (UTC)")
    # Manage legends for twin axes
    lines, labels = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    axes[3].legend(lines + lines2, labels + labels2, loc="upper left")

    fig.suptitle("Deterministic Regulation + Arbitrage — 12-hour Demo")
    fig.tight_layout()
    plt.show()
