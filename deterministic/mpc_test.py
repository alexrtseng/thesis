import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data.data_output_functions import read_lmp_folder
from deterministic.single_market_battery import (
    DEFAULT_BATTERY,
    BatteryParams,
    deterministic_arbitrage_opt,
)


def _horizon_worker(
    hf_horizon: int,
    lf_horizon: int,
    prices_df: pd.DataFrame,
    battery_params: BatteryParams,
    require_equivalent_soe: bool,
    label: str,
):
    """Worker for horizon sweeps. Returns a dict with label, hours, value, and steps."""
    decisions, value = battery_arb_mpc_tester(
        hf_horizon=hf_horizon,
        lf_horizon=lf_horizon,
        prices_df=prices_df,
        battery_params=battery_params,
        require_equivalent_soe=require_equivalent_soe,
    )
    hours = hf_horizon / 12
    return {
        "label": label,
        "hf_hours": hours,
        "value_$": value,
        "steps": len(decisions),
    }


def _fidelity_worker(
    fidelity_hours: float,
    total_horizon_hours: int,
    df: pd.DataFrame,
    battery: BatteryParams,
    use_lf_avg: bool = False,
):
    """Worker for fidelity sweeps. Mirrors the sequential math and returns result row."""
    # work on a local copy and optionally add +/-30min centered average (5-min freq -> window=13)
    df = df.copy()
    if use_lf_avg:
        df["lmp_lf_avg"] = (
            df["lmp"].rolling(window=13, center=True, min_periods=1).mean()
        )

    hf_horizon = int(fidelity_hours * 12)  # in 5-min intervals
    lf_horizon = math.floor(total_horizon_hours - fidelity_hours)
    correction_factor = int((1 - fidelity_hours + math.floor(fidelity_hours)) * 12)

    decisions, value = battery_arb_mpc_tester(
        hf_horizon=hf_horizon,
        lf_horizon=lf_horizon,
        prices_df=df[
            : (len(df) - correction_factor) if correction_factor != 12 else len(df)
        ],
        battery_params=battery,
        use_lf_avg=use_lf_avg,
    )
    return {
        "fidelity_hours": fidelity_hours,
        "lf_horizon_hours": lf_horizon,
        "value_$": value,
        "steps": len(decisions),
        "lf_avg": bool(use_lf_avg),  # tag scenario
    }


def battery_arb_mpc_tester(
    hf_horizon: int,
    lf_horizon: int,
    prices_df: pd.DataFrame,
    battery_params: BatteryParams = DEFAULT_BATTERY,
    require_equivalent_soe: bool = False,
    use_lf_avg: bool = False,
):
    required_horizon = hf_horizon + lf_horizon * 12
    df = prices_df.copy()
    charge_decisions = []
    discharge_decisions = []
    timestamps = []
    prices = []
    current_soe = battery_params.initial_charge_mwh
    for i in range(0, len(df) - required_horizon):
        hf = df.iloc[i : i + hf_horizon].copy()
        lf_indices = [i + hf_horizon + j * 12 for j in range(lf_horizon)]
        lf = df.iloc[lf_indices].copy()
        if use_lf_avg:
            lf["lmp"] = lf["lmp_lf_avg"]
        window = pd.concat([hf, lf])
        result_df, _ = deterministic_arbitrage_opt(
            prices_df=window,
            battery=battery_params,
            require_equivalent_soe=require_equivalent_soe,
            initial_charge_mwh=current_soe,
        )
        charge_decisions.append(result_df.iloc[0]["charge_mw"])
        discharge_decisions.append(result_df.iloc[0]["discharge_mw"])
        timestamps.append(df.index[i])
        prices.append(df.iloc[i]["lmp"])
        current_soe = result_df.iloc[1]["state_of_energy_mwh"]
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


def test_mpc_horizon(start, end, parallel: bool = True, max_workers: int | None = None):
    # Horizons expressed in HOURS
    potential_horizons_hours = [1, 4, 8, 16, 24, 36, 48, 72]
    # Define several battery sizes (other params constant) for multi-line comparison
    capacities = [4, 8, 10]
    battery_configs = []
    for cap in capacities:
        battery_configs.append(
            BatteryParams(
                capacity_mwh=cap,
                max_charge_mw=1,
                max_discharge_mw=1,
                initial_charge_mwh=0,
                in_efficiency=0.98,
                out_efficiency=0.98,
            )
        )
    pnode_id = 2156113094  # Example node
    lmp_dir = Path("data/pjm_lmps")
    df = read_lmp_folder(lmp_dir)
    df = df[df["pnode_id"] == int(pnode_id)]
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = df.set_index("datetime_beginning_utc").sort_index()
    df = df.loc[start:end]
    df.rename(columns={"lmp_rt": "lmp"}, inplace=True)
    start_time = time.perf_counter()
    # Collect results per battery (possibly in parallel)
    all_results: dict[str, pd.DataFrame] = {}
    tasks = []

    # Build tasks for normal battery configs
    for b in battery_configs:
        label = f"Cap {b.capacity_mwh} MWh"
        for hours in potential_horizons_hours:
            hf_intervals = hours * 12
            prices_slice = df[
                : len(df) + (hf_intervals - (potential_horizons_hours[-1] * 12))
            ]
            tasks.append((hf_intervals, 0, prices_slice, b, False, label))

    # Build tasks for EqSoE variant (uses first battery config)
    label_eq = "Cap 4 MWh (EqSoE)"
    b_eq = battery_configs[0]
    for hours in potential_horizons_hours:
        hf_intervals = hours * 12
        prices_slice = df[
            : len(df) + (hf_intervals - (potential_horizons_hours[-1] * 12))
        ]
        tasks.append((hf_intervals, 0, prices_slice, b_eq, True, label_eq))

    results_rows: dict[str, list[dict]] = {}

    if parallel:
        worker_count = max_workers or os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            future_to_task = {ex.submit(_horizon_worker, *t): t for t in tasks}
            for fut in as_completed(future_to_task):
                res = fut.result()
                results_rows.setdefault(res["label"], []).append(
                    {k: res[k] for k in ("hf_hours", "value_$", "steps")}
                )
    else:
        for t in tasks:
            res = _horizon_worker(*t)
            results_rows.setdefault(res["label"], []).append(
                {k: res[k] for k in ("hf_hours", "value_$", "steps")}
            )

    # Assemble dataframes, sort, and print like before
    for label, rows in results_rows.items():
        df_line = pd.DataFrame(rows).sort_values("hf_hours").reset_index(drop=True)
        all_results[label] = df_line
        for _, r in df_line.iterrows():
            print(
                f"[{label}] Horizon {int(r['hf_hours'])}h -> value ${r['value_$']:,.2f} (steps={int(r['steps'])})"
            )

    elapsed = time.perf_counter() - start_time
    print(f"MPC horizon sweep completed in {elapsed:.2f}s (parallel={parallel})")

    # Plot multi-line
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, df_line in all_results.items():
        ax.plot(
            df_line["hf_hours"],
            df_line["value_$"],
            marker="o",
            linestyle="-",
            label=label,
        )
        # annotate last point for quick comparison
        last_row = df_line.iloc[-1]
        ax.annotate(
            f"{last_row['value_$']:.0f}",
            (last_row["hf_hours"], last_row["value_$"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    ax.set_xlabel("MPC High-Frequency Horizon (hours)")
    ax.set_ylabel("Accumulated First-Step Value ($)")
    ax.set_title(f"Battery MPC Value vs Horizon (2024 Year, Pnode {pnode_id})")
    ax.grid(True, alpha=0.35)
    ax.legend(title="Battery")

    output_dir = Path("deterministic/output/mpc")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "mpc_value_vs_horizon_multibattery.png"
    fig.tight_layout()
    fig.savefig(fig_path)
    print(f"Saved multi-line plot to {fig_path}")

    # Save each battery's data and a combined file
    for label, df_line in all_results.items():
        safe_label = label.replace(" ", "_").replace(".", "_")
        csv_path = output_dir / f"mpc_value_vs_horizon_{safe_label}.csv"
        df_line.to_csv(csv_path, index=False)
        print(f"Saved data for {label} to {csv_path}")

    combined = pd.concat(
        [df.assign(battery=lbl) for lbl, df in all_results.items()], ignore_index=True
    )
    combined_path = output_dir / "mpc_value_vs_horizon_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined data to {combined_path}")


def test_fidelity_horizon(
    start,
    end,
    total_horizon_hours: int = 24,
    parallel: bool = True,
    max_workers: int | None = None,
):
    # Test varying fidelity levels within a fixed total horizon
    fidelity_levels = [
        0,
        (1.0 / 12.0),
        (1.0 / 6.0),
        0.25,
        0.5,
        1,
        2,
        4,
        8,
        12,
        24,
    ]  # in hours
    battery = BatteryParams(
        capacity_mwh=8,
        max_charge_mw=1,
        max_discharge_mw=1,
        initial_charge_mwh=0,
        in_efficiency=0.98,
        out_efficiency=0.98,
    )
    pnode_id = 2156113094  # Example node
    lmp_dir = Path("data/pjm_lmps")
    df = read_lmp_folder(lmp_dir)
    df = df[df["pnode_id"] == int(pnode_id)]
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = df.set_index("datetime_beginning_utc").sort_index()
    df = df.loc[start:end]
    df.rename(columns={"lmp_rt": "lmp"}, inplace=True)

    start_time = time.perf_counter()
    results_no_avg: list[dict] = []
    results_lf_avg: list[dict] = []

    if parallel:
        worker_count = max_workers or os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futures = {}
            for f in fidelity_levels:
                # submit both scenarios for each fidelity level
                fut_a = ex.submit(
                    _fidelity_worker, f, total_horizon_hours, df, battery, False
                )
                fut_b = ex.submit(
                    _fidelity_worker, f, total_horizon_hours, df, battery, True
                )
                futures[fut_a] = False
                futures[fut_b] = True

            for fut in as_completed(futures):
                res = fut.result()
                if futures[fut]:
                    results_lf_avg.append(res)
                else:
                    results_no_avg.append(res)
    else:
        for f in fidelity_levels:
            results_no_avg.append(
                _fidelity_worker(f, total_horizon_hours, df, battery, use_lf_avg=False)
            )
            results_lf_avg.append(
                _fidelity_worker(f, total_horizon_hours, df, battery, use_lf_avg=True)
            )

    # keep original order by fidelity for both scenarios
    results_no_avg.sort(key=lambda r: r["fidelity_hours"])
    results_lf_avg.sort(key=lambda r: r["fidelity_hours"])

    # Print both for quick scan
    for r in results_no_avg:
        print(
            f"[Fidelity {r['fidelity_hours']}h | no_avg]  LF Horizon {r['lf_horizon_hours']}h -> value ${r['value_$']:,.2f} (steps={r['steps']})"
        )
    for r in results_lf_avg:
        print(
            f"[Fidelity {r['fidelity_hours']}h | lf_avg]  LF Horizon {r['lf_horizon_hours']}h -> value ${r['value_$']:,.2f} (steps={r['steps']})"
        )

    # Save to CSVs
    output_dir = Path("deterministic/output/mpc")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_no = pd.DataFrame(results_no_avg)
    df_lf = pd.DataFrame(results_lf_avg)

    csv_no = output_dir / "mpc_value_vs_fidelity_noavg.csv"
    csv_lf = output_dir / "mpc_value_vs_fidelity_lfavg.csv"
    df_no.to_csv(csv_no, index=False)
    df_lf.to_csv(csv_lf, index=False)
    print(f"Saved fidelity results (no_avg) to {csv_no}")
    print(f"Saved fidelity results (lf_avg) to {csv_lf}")

    # Combined CSV with scenario column
    df_no2 = df_no.copy()
    df_no2["scenario"] = "no_avg"
    df_lf2 = df_lf.copy()
    df_lf2["scenario"] = "lf_avg"
    combined = pd.concat([df_no2, df_lf2], ignore_index=True)
    combined_path = output_dir / "mpc_value_vs_fidelity_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined fidelity results to {combined_path}")

    # Plot both lines
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_no["fidelity_hours"], df_no["value_$"], marker="o", label="No Avg")
    ax.plot(
        df_lf["fidelity_hours"], df_lf["value_$"], marker="s", label="LF Avg (rolling)"
    )

    ax.set_xlabel("High-fidelity horizon (hours)")
    ax.set_ylabel("Accumulated First-Step Value ($)")
    ax.set_title("MPC Value vs Fidelity Hours")
    ax.grid(True, alpha=0.35)

    # annotate last points for both curves if present
    if not df_no.empty:
        last_no = df_no.iloc[-1]
        ax.annotate(
            f"{last_no['value_$']:.0f}",
            (last_no["fidelity_hours"], last_no["value_$"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )
    if not df_lf.empty:
        last_lf = df_lf.iloc[-1]
        ax.annotate(
            f"{last_lf['value_$']:.0f}",
            (last_lf["fidelity_hours"], last_lf["value_$"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    ax.legend(title="Scenario")

    png_path = output_dir / "mpc_value_vs_fidelity.png"
    fig.tight_layout()
    fig.savefig(png_path)
    print(f"Saved fidelity plot (both scenarios) to {png_path}")

    elapsed = time.perf_counter() - start_time
    print(f"Fidelity sweep completed in {elapsed:.2f}s (parallel={parallel})")


if __name__ == "__main__":
    start = pd.Timestamp(year=2024, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=2025, month=1, day=1, tz="UTC")
    # test_mpc_horizon(start, end)
    test_fidelity_horizon(start, end)
