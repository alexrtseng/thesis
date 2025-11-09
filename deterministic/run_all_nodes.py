from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Flexible imports so the module works whether executed as a script or imported as a package
try:  # When imported as part of the deterministic package
    from .multi_market_battery import deterministic_reg_arbitrage_battery_opt
    from .pjm_reg_arbitrage_battery import (
        pjm_deterministic_reg_arbitrage_battery_opt,
    )
    from .single_market_battery import (
        DEFAULT_BATTERY,
        BatteryParams,
        calc_reg_profit,
        deterministic_arbitrage_opt,
        txbx,
    )
except Exception:  # When executed directly from this folder
    from multi_market_battery import (  # type: ignore
        deterministic_reg_arbitrage_battery_opt,
    )
    from pjm_reg_arbitrage_battery import (  # type: ignore
        pjm_deterministic_reg_arbitrage_battery_opt,
    )
    from single_market_battery import (  # type: ignore
        DEFAULT_BATTERY,
        BatteryParams,
        calc_reg_profit,
        deterministic_arbitrage_opt,
        txbx,
    )

# Use shared data I/O and 5-min filling utilities
try:
    # Prefer normal import if repo_root is already on sys.path and data is importable
    from data.data_output_functions import (
        fill_missing_5min_slots,
    )
    from data.data_output_functions import (
        read_lmp_folder as _read_lmp_folder_shared,
    )
    from data.data_output_functions import (
        read_reg_folder as _read_reg_folder_shared,
    )
except Exception:
    # Fallback: import module directly from file path (works without data/__init__.py)
    import importlib.util

    _repo_root = Path(__file__).resolve().parents[1]
    _mod_path = _repo_root / "data" / "data_output_functions.py"
    _spec = importlib.util.spec_from_file_location("data_output_functions", _mod_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Unable to load data_output_functions from {_mod_path}")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    _read_lmp_folder_shared = _mod.read_lmp_folder
    _read_reg_folder_shared = _mod.read_reg_folder
    fill_missing_5min_slots = _mod.fill_missing_5min_slots  # type: ignore


def _process_node_worker(
    node_id: int,
    g: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    reg_series: pd.DataFrame,
    battery: "BatteryParams",
    txbx_x: int,
    reg_missing_5min_steps: int,
) -> Optional[dict]:
    """Process a single node's objectives; designed to run in a separate process.

    Returns a dict of computed metrics or None if insufficient data.
    """
    try:
        print(f"Processing node {node_id}...")
        node_prices = (
            g[["datetime_beginning_utc", "lmp"]]
            .dropna()
            .drop_duplicates(subset=["datetime_beginning_utc"])
            .set_index("datetime_beginning_utc")
            .sort_index()
        )
        if len(node_prices.index) < 2:
            return None

        expected_5min = pd.date_range(
            start=start_ts.floor("5min"), end=end_ts.ceil("5min"), freq="5min", tz="UTC"
        )

        # Local helper to avoid importing parent's version
        def _count_missing_5min(
            index: pd.DatetimeIndex, expected: pd.DatetimeIndex
        ) -> int:
            idx = pd.DatetimeIndex(index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            return int(len(expected) - len(idx.intersection(expected)))

        lmp_missing_steps = _count_missing_5min(node_prices.index, expected_5min)
        print(f"  - LMP missing 5-min steps: {lmp_missing_steps}")

        # Fill to 5-min grid using prior-day same-slot only
        node_prices_filled = fill_missing_5min_slots(
            node_prices[["lmp"]],
            expected_index=expected_5min,
            columns=["lmp"],
            prior_day_backfill=True,
            ffill=False,
            bfill=False,
        )

        # 1) Arbitrage-only optimization
        _, arb_profit = deterministic_arbitrage_opt(
            node_prices_filled[["lmp"]], battery
        )

        # 2) Regulation-only simple profit (node-agnostic)
        reg_profit = float(calc_reg_profit(reg_series, battery))

        # 3) Multi-market optimization
        _, mm_profit = deterministic_reg_arbitrage_battery_opt(
            reg_series, node_prices_filled
        )

        # 4) PJM-specific reg+arbitrage optimization
        _, pjm_profit = pjm_deterministic_reg_arbitrage_battery_opt(
            reg_series, node_prices_filled
        )

        # 5) Heuristic txbx
        try:
            _, tx_profit = txbx(node_prices_filled[["lmp"]], x=txbx_x, battery=battery)
        except Exception as e:
            print(f"  - txbx skipped for node {node_id}: {e}")
            tx_profit = float("nan")

        return {
            "pnode_id": int(node_id),
            "arbitrage_profit": float(arb_profit),
            "regulation_profit": float(reg_profit),
            "multi_market_profit": float(mm_profit),
            "pjm_multi_market_profit": float(pjm_profit),
            "txbx_profit": float(tx_profit) if pd.notna(tx_profit) else float("nan"),
            "lmp_missing_5min_steps": int(lmp_missing_steps),
            "reg_missing_5min_steps": int(reg_missing_5min_steps),
        }
    except Exception as e:
        print(f"Node {node_id} failed: {e}")
        return None


def _to_utc_timestamp(ts: "str | pd.Timestamp") -> pd.Timestamp:
    """Parse to UTC Timestamp, localizing naive values to UTC."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _compute_interval_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Per-row forward interval length in hours; last row is 0."""
    dt = index.to_series().shift(-1) - index.to_series()
    return dt.dt.total_seconds().fillna(0.0) / 3600.0


def _count_missing_5min(index: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> int:
    """Count missing 5-minute steps from expected."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return int(len(expected) - len(idx.intersection(expected)))


def _compute_arbitrage_profit(res_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    """Arbitrage revenue from result and price series."""
    idx = res_df.index
    dt_hours = _compute_interval_hours(idx)
    price = price_df.reindex(idx)["lmp"].astype(float)
    net_mw = (res_df["discharge_mw"] - res_df["charge_mw"]).astype(float)
    return float((price * net_mw * dt_hours).fillna(0.0).sum())


def _compute_multi_market_profit(
    res_df: pd.DataFrame, arb_prices: pd.DataFrame, reg_prices: pd.DataFrame
) -> float:
    """Total revenue for reg+arbitrage from result and inputs."""
    idx = res_df.index
    dt_hours = _compute_interval_hours(idx)
    arb_price = arb_prices.reindex(idx)["lmp"].astype(float)
    reg_price = reg_prices.reindex(idx)["mcp"].astype(float)
    net_mw = (res_df["discharge_mw"] - res_df["charge_mw"]).astype(float)
    reg_mw = res_df.get("reg_allocation_mw", pd.Series(0.0, index=idx)).astype(float)
    arb_revenue = (arb_price * net_mw * dt_hours).fillna(0.0)
    reg_revenue = (reg_price * reg_mw * dt_hours).fillna(0.0)
    return float((arb_revenue + reg_revenue).sum())


def compute_pjm_node_objectives(
    start: "str | pd.Timestamp",
    end: "str | pd.Timestamp",
    *,
    battery: BatteryParams = DEFAULT_BATTERY,
    lmp_dir: Optional[Path] = None,
    reg_dir: Optional[Path] = None,
    nodes: Optional[Iterable[int]] = None,
    txbx_x: int = 4,
    parallel_workers: Optional[int] = 12,
    use_processes: bool = True,
) -> pd.DataFrame:
    """Compute per-node objectives from PJM LMP and REG CSVs within a time window.

    Returns df indexed by pnode_id with:
      arbitrage_profit, regulation_profit, multi_market_profit,
      pjm_multi_market_profit, txbx_profit, lmp_missing_5min_steps, reg_missing_5min_steps.
    """
    repo_root = Path(__file__).resolve().parents[1]
    lmp_dir = lmp_dir or (repo_root / "data" / "pjm_lmps")
    reg_dir = reg_dir or (repo_root / "data" / "pjm_reg")

    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    # Read RT LMPs and REG via shared helpers
    lmp_all = _read_lmp_folder_shared(lmp_dir, da=False).rename(
        columns={"lmp_rt": "lmp"}
    )
    reg_all = _read_reg_folder_shared(reg_dir)

    # Filter to window
    lmp_win = lmp_all[
        (lmp_all["datetime_beginning_utc"] >= start_ts)
        & (lmp_all["datetime_beginning_utc"] <= end_ts)
    ].copy()
    reg_win = reg_all[
        (reg_all["datetime_beginning_utc"] >= start_ts)
        & (reg_all["datetime_beginning_utc"] <= end_ts)
    ].copy()

    if len(lmp_win) < 2:
        raise ValueError(
            "Insufficient LMP data in the requested time window — choose a different window."
        )
    if len(reg_win) < 2:
        raise ValueError(
            "Insufficient REG data in the requested time window — choose a different window."
        )

    if nodes is not None:
        nodes_set = set(int(n) for n in nodes)
        lmp_win = lmp_win[lmp_win["pnode_id"].isin(nodes_set)].copy()
        if lmp_win.empty:
            raise ValueError("None of the requested nodes have LMP data in the window.")

    expected_5min = pd.date_range(
        start=start_ts.floor("5min"), end=end_ts.ceil("5min"), freq="5min", tz="UTC"
    )

    reg_series = reg_win.set_index("datetime_beginning_utc").sort_index()[["mcp"]]
    reg_missing_steps = _count_missing_5min(reg_series.index, expected_5min)
    print(f"REG missing 5-min steps in window: {reg_missing_steps}")

    node_groups = list(lmp_win.groupby("pnode_id", sort=True))

    results: list[dict] = []
    if node_groups:
        workers = min(parallel_workers, os.cpu_count()) or 1
        print(f"Using {workers} {'processes' if use_processes else 'threads'} for parallel execution...")
        exec_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        t0 = time.perf_counter()
        with exec_cls(max_workers=max(1, workers)) as ex:
            # Submit at most `workers` futures at a time to avoid building a huge queue of pickled args
            pending = set()
            groups_iter = iter(node_groups)

            # Prime the pool
            for _ in range(min(workers, len(node_groups))):
                try:
                    nid, grp = next(groups_iter)
                except StopIteration:
                    break
                pending.add(
                    ex.submit(
                        _process_node_worker,
                        int(nid),
                        grp,
                        start_ts,
                        end_ts,
                        reg_series,
                        battery,
                        txbx_x,
                        reg_missing_steps,
                    )
                )

            # As each completes, submit the next
            while pending:
                for fut in as_completed(pending, timeout=None):
                    pending.remove(fut)
                    try:
                        rec = fut.result()
                        if rec:
                            results.append(rec)
                    except Exception as e:
                        print(f"Worker failed: {e}")
                    # Refill from iterator
                    try:
                        nid, grp = next(groups_iter)
                    except StopIteration:
                        pass
                    else:
                        pending.add(
                            ex.submit(
                                _process_node_worker,
                                int(nid),
                                grp,
                                start_ts,
                                end_ts,
                                reg_series,
                                battery,
                                txbx_x,
                                reg_missing_steps,
                            )
                        )
                    break  # break inner for-loop to re-evaluate pending

        elapsed = time.perf_counter() - t0
        print(
            f"Processed {len(node_groups)} nodes in {elapsed:.2f}s using {'processes' if use_processes else 'threads'} (workers={workers})."
        )

    if not results:
        raise ValueError(
            "No nodes had sufficient data points to run optimizations in this window."
        )

    out_df = (
        pd.DataFrame(results)
        .set_index("pnode_id")
        .sort_index()[
            [
                "arbitrage_profit",
                "regulation_profit",
                "multi_market_profit",
                "pjm_multi_market_profit",
                "txbx_profit",
                "lmp_missing_5min_steps",
                "reg_missing_5min_steps",
            ]
        ]
    )
    return out_df


def summarize_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize averages and percent improvements across nodes."""
    required = {
        "arbitrage_profit",
        "regulation_profit",
        "multi_market_profit",
        "pjm_multi_market_profit",
        "txbx_profit",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    avg_arb = float(df["arbitrage_profit"].mean())
    avg_reg = float(df["regulation_profit"].mean())
    avg_mm = float(df["multi_market_profit"].mean())
    avg_pjm = float(df["pjm_multi_market_profit"].mean())
    avg_txbx = float(df["txbx_profit"].mean(skipna=True))

    def pct_uplift(new: pd.Series, base: pd.Series) -> pd.Series:
        base_nonzero = base.replace(0, pd.NA)
        return (new - base) / base_nonzero * 100.0

    pct_mm_vs_arb = pct_uplift(
        df["multi_market_profit"], df["arbitrage_profit"]
    ).astype(float)
    pct_pjm_vs_arb = pct_uplift(
        df["pjm_multi_market_profit"], df["arbitrage_profit"]
    ).astype(float)
    pct_pjm_vs_mm = pct_uplift(
        df["pjm_multi_market_profit"], df["multi_market_profit"]
    ).astype(float)
    pct_txbx_vs_arb = pct_uplift(df["txbx_profit"], df["arbitrage_profit"]).astype(
        float
    )
    pct_txbx_vs_mm = pct_uplift(df["txbx_profit"], df["multi_market_profit"]).astype(
        float
    )
    pct_txbx_vs_pjm = pct_uplift(
        df["txbx_profit"], df["pjm_multi_market_profit"]
    ).astype(float)

    return pd.DataFrame(
        {
            "avg_arbitrage_profit": [avg_arb],
            "avg_regulation_profit": [avg_reg],
            "avg_multi_market_profit": [avg_mm],
            "avg_pjm_multi_market_profit": [avg_pjm],
            "avg_txbx_profit": [avg_txbx],
            "avg_pct_uplift_mm_vs_arb": [float(pct_mm_vs_arb.mean(skipna=True))],
            "avg_pct_uplift_pjm_vs_arb": [float(pct_pjm_vs_arb.mean(skipna=True))],
            "avg_pct_uplift_pjm_vs_mm": [float(pct_pjm_vs_mm.mean(skipna=True))],
            "avg_pct_uplift_txbx_vs_arb": [float(pct_txbx_vs_arb.mean(skipna=True))],
            "avg_pct_uplift_txbx_vs_mm": [float(pct_txbx_vs_mm.mean(skipna=True))],
            "avg_pct_uplift_txbx_vs_pjm": [float(pct_txbx_vs_pjm.mean(skipna=True))],
        }
    )


def print_objectives_summary(df: pd.DataFrame) -> None:
    """Print a readable summary derived from summarize_objectives()."""
    s = summarize_objectives(df).iloc[0]
    print("Averages across nodes:")
    print(f"- Arbitrage profit: ${s['avg_arbitrage_profit']:.2f}")
    print(f"- Regulation profit: ${s['avg_regulation_profit']:.2f}")
    print(f"- Multi-market profit: ${s['avg_multi_market_profit']:.2f}")
    print(f"- PJM multi-market profit: ${s['avg_pjm_multi_market_profit']:.2f}")
    print(f"- txbx (x-hours) profit: ${s['avg_txbx_profit']:.2f}")
    print("\nAverage percent improvements (per-node, mean):")
    print(f"- Multi-market vs Arbitrage: {s['avg_pct_uplift_mm_vs_arb']:.2f}%")
    print(
        f"- Multi-market vs Regulation: {s['avg_multi_market_profit'] / s['avg_regulation_profit'] * 100 - 100:.2f}%"
    )
    print(
        f"- PJM Multi-market vs Regulation: {s['avg_pjm_multi_market_profit'] / s['avg_regulation_profit'] * 100 - 100:.2f}%"
    )
    print(f"- PJM Multi-market vs Arbitrage: {s['avg_pct_uplift_pjm_vs_arb']:.2f}%")
    print(f"- PJM Multi-market vs Multi-market: {s['avg_pct_uplift_pjm_vs_mm']:.2f}%")
    print(f"- txbx vs Arbitrage: {s['avg_pct_uplift_txbx_vs_arb']:.2f}%")
    print(f"- txbx vs Multi-market: {s['avg_pct_uplift_txbx_vs_mm']:.2f}%")
    print(f"- txbx vs PJM Multi-market: {s['avg_pct_uplift_txbx_vs_pjm']:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute PJM node objectives between start and end datetimes (UTC)."
    )
    parser.add_argument(
        "--start",
        default="2022-10-16 00:00:00+00:00",
        help="Start datetime, e.g., '2025-09-25 00:00:00+00:00' (UTC recommended)",
    )
    parser.add_argument(
        "--end",
        default="2025-10-13 00:00:00+00:00",
        help="End datetime, e.g., '2025-10-14 00:00:00+00:00' (UTC recommended)",
    )
    args = parser.parse_args()

    start_example = args.start
    end_example = args.end

    df = compute_pjm_node_objectives(start_example, end_example)
    print(df.head())
    print()
    print_objectives_summary(df)

    out_path = Path(
        f"deterministic/output/pjm_node_objectives_{start_example.replace(':', '-')}_to_{end_example.replace(':', '-')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"Saved objectives DataFrame to {out_path}")
