from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    from multi_market_battery import (
        deterministic_reg_arbitrage_battery_opt,  # type: ignore
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


def _to_utc_timestamp(ts: "str | pd.Timestamp") -> pd.Timestamp:
    """Parse to a timezone-aware UTC Timestamp.

    Rules:
    - If naive, interpret as UTC and localize to UTC.
    - If tz-aware, convert to UTC.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _compute_interval_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Return per-row interval length in hours using forward difference.

    Last row has 0 hours because there's no forward interval.
    """
    dt = index.to_series().shift(-1) - index.to_series()
    return dt.dt.total_seconds().fillna(0.0) / 3600.0


def _fill_missing_data(
    df: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    # Ensure UTC-aware index
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx

    # Reindex to the full expected 5-minute grid
    reindexed = df.reindex(expected_index)

    # Decide which columns to operate on
    if columns is None:
        candidates = [c for c in ["lmp", "mcp"] if c in reindexed.columns]
        if candidates:
            columns = candidates
        else:
            # Fallback to numeric columns
            columns = [
                c
                for c in reindexed.columns
                if pd.api.types.is_numeric_dtype(reindexed[c])
            ]

    # 24 hours = 288 five-minute periods
    shift_periods = 24 * 60 // 5
    for col in columns:
        prior = reindexed[col].shift(shift_periods)
        reindexed[col] = reindexed[col].where(~reindexed[col].isna(), prior)

    return reindexed


def _count_missing_5min(index: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> int:
    """Count how many 5-minute steps in expected are missing from index."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    # Ensure both are timezone-aware UTC
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    present = index.intersection(expected)
    return int(len(expected) - len(present))


def _read_lmp_folder(lmp_dir: Path) -> pd.DataFrame:
    """Read all PJM LMP CSVs in a folder into a single DataFrame.

    Expected columns: datetime_beginning_utc, pnode_id, total_lmp_rt
    Returns columns: [datetime_beginning_utc, pnode_id, lmp]
    """
    files = sorted(lmp_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No LMP CSVs found in {lmp_dir}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(
            f,
            parse_dates=["datetime_beginning_utc"],
        )
        # Normalize column names and types
        if "total_lmp_rt" not in df.columns or "pnode_id" not in df.columns:
            raise ValueError(
                f"Expected columns 'total_lmp_rt' and 'pnode_id' in {f.name}"
            )
        df = df.rename(columns={"total_lmp_rt": "lmp"})
        # Localize to UTC (values are UTC per column name; they're naive strings)
        df["datetime_beginning_utc"] = df["datetime_beginning_utc"].dt.tz_localize(
            "UTC"
        )
        # Keep only necessary columns
        df = df[["datetime_beginning_utc", "pnode_id", "lmp"]]
        dfs.append(df)

    all_lmp = pd.concat(dfs, ignore_index=True)
    return all_lmp


def _read_reg_folder(reg_dir: Path) -> pd.DataFrame:
    """Read all PJM regulation CSVs in a folder into a single DataFrame.

    Expected columns include: datetime_beginning_utc, service, mcp
    Returns columns: [datetime_beginning_utc, mcp]
    """
    files = sorted(reg_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No REG CSVs found in {reg_dir}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(
            f,
            parse_dates=["datetime_beginning_utc"],
        )
        if "mcp" not in df.columns or "service" not in df.columns:
            raise ValueError(f"Expected columns 'mcp' and 'service' in {f.name}")
        # Filter to REG service rows
        df = df[df["service"].astype(str).str.upper() == "REG"].copy()
        # Localize to UTC (values are UTC per column name; they're naive strings)
        df["datetime_beginning_utc"] = df["datetime_beginning_utc"].dt.tz_localize(
            "UTC"
        )
        df = df[["datetime_beginning_utc", "mcp"]]
        dfs.append(df)

    all_reg = pd.concat(dfs, ignore_index=True)
    return all_reg


def _compute_arbitrage_profit(res_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    """Compute arbitrage revenue from optimization result and price series.

    res_df: DataFrame with index timestamps and columns charge_mw, discharge_mw
    price_df: DataFrame with index timestamps and column 'lmp'
    """
    idx = res_df.index
    dt_hours = _compute_interval_hours(idx)
    price = price_df.reindex(idx)["lmp"].astype(float)
    net_mw = (res_df["discharge_mw"] - res_df["charge_mw"]).astype(float)
    revenue = (price * net_mw * dt_hours).fillna(0.0)
    return float(revenue.sum())


def _compute_multi_market_profit(
    res_df: pd.DataFrame, arb_prices: pd.DataFrame, reg_prices: pd.DataFrame
) -> float:
    """Compute total revenue for reg+arbitrage from result and input prices."""
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
    parallel_workers: int = 9,
) -> pd.DataFrame:
    """Read PJM LMP and REG CSVs, filter by a time window, and compute per-node objectives.

        Returns a DataFrame indexed by pnode_id with columns:
      - arbitrage_profit
      - regulation_profit
      - multi_market_profit
      - pjm_multi_market_profit
            - txbx_profit (heuristic x-hour strategy; default x=4)

    Errors:
      - Raises ValueError if the requested time window has insufficient data overall
            (e.g., < 2 timestamps for LMP or REG within the window).
    """
    # Resolve default data directories relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    lmp_dir = lmp_dir or (repo_root / "data" / "pjm_lmps")
    reg_dir = reg_dir or (repo_root / "data" / "pjm_reg")

    # Parse time window
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    # Read data
    lmp_all = _read_lmp_folder(lmp_dir)
    reg_all = _read_reg_folder(reg_dir)

    # Filter to time window
    lmp_win = lmp_all[
        (lmp_all["datetime_beginning_utc"] >= start_ts)
        & (lmp_all["datetime_beginning_utc"] <= end_ts)
    ].copy()
    reg_win = reg_all[
        (reg_all["datetime_beginning_utc"] >= start_ts)
        & (reg_all["datetime_beginning_utc"] <= end_ts)
    ].copy()

    # Basic window-level validations
    if len(lmp_win) < 2:
        raise ValueError(
            "Insufficient LMP data in the requested time window — choose a different window."
        )
    if len(reg_win) < 2:
        raise ValueError(
            "Insufficient REG data in the requested time window — choose a different window."
        )

    # Optional node filter
    if nodes is not None:
        nodes_set = set(int(n) for n in nodes)
        lmp_win = lmp_win[lmp_win["pnode_id"].isin(nodes_set)].copy()
        if lmp_win.empty:
            raise ValueError("None of the requested nodes have LMP data in the window.")

    # Prepare expected 5-minute grid for the window (inclusive bounds)
    expected_5min = pd.date_range(
        start=start_ts.floor("5T"), end=end_ts.ceil("5T"), freq="5T", tz="UTC"
    )

    # Prepare REG series once (node-agnostic)
    reg_series = reg_win.set_index("datetime_beginning_utc").sort_index()[["mcp"]]
    # Count REG missing steps
    reg_missing_steps = _count_missing_5min(reg_series.index, expected_5min)
    print(f"REG missing 5-min steps in window: {reg_missing_steps}")

    # Prepare groups for parallel processing
    node_groups = list(lmp_win.groupby("pnode_id", sort=True))

    def _process_node(node_id: int, g: pd.DataFrame) -> Optional[dict]:
        try:
            print(f"Processing node {node_id}...")
            node_prices = (
                g[["datetime_beginning_utc", "lmp"]]
                .dropna()
                .drop_duplicates(subset=["datetime_beginning_utc"])  # one per timestamp
                .set_index("datetime_beginning_utc")
                .sort_index()
            )
            if len(node_prices.index) < 2:
                return None

            # LMP missing steps for this node
            lmp_missing_steps = _count_missing_5min(node_prices.index, expected_5min)
            print(f"  - LMP missing 5-min steps: {lmp_missing_steps}")

            # Fill missing LMP data using same 5-minute slot from 24 hours prior
            node_prices_filled = _fill_missing_data(node_prices[["lmp"]], expected_5min)

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

            # 5) Heuristic txbx (charge at lowest x hours, discharge at highest x hours)
            try:
                _, tx_profit = txbx(
                    node_prices_filled[["lmp"]], x=txbx_x, battery=battery
                )
            except Exception as e:
                print(f"  - txbx skipped for node {node_id}: {e}")
                tx_profit = float("nan")

            return {
                "pnode_id": int(node_id),
                "arbitrage_profit": arb_profit,
                "regulation_profit": reg_profit,
                "multi_market_profit": mm_profit,
                "pjm_multi_market_profit": pjm_profit,
                "txbx_profit": tx_profit,
                "lmp_missing_5min_steps": lmp_missing_steps,
                "reg_missing_5min_steps": reg_missing_steps,
            }
        except Exception as e:
            print(f"Node {node_id} failed: {e}")
            return None

    results: list[dict] = []
    if node_groups:
        with ThreadPoolExecutor(max_workers=max(1, int(parallel_workers))) as ex:
            futures = [
                ex.submit(_process_node, int(nid), grp) for nid, grp in node_groups
            ]
            for fut in as_completed(futures):
                rec = fut.result()
                if rec:
                    results.append(rec)

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
    """Create a compact summary of averages and percent improvements across nodes.

    Percent improvements are computed per-node and then averaged (ignoring div-by-zero).
    Returns a single-row DataFrame with columns:
      - avg_arbitrage_profit
      - avg_regulation_profit
      - avg_multi_market_profit
      - avg_pjm_multi_market_profit
            - avg_txbx_profit
            - avg_pct_uplift_mm_vs_arb
      - avg_pct_uplift_pjm_vs_arb
      - avg_pct_uplift_pjm_vs_mm
            - avg_pct_uplift_txbx_vs_arb
            - avg_pct_uplift_txbx_vs_mm
            - avg_pct_uplift_txbx_vs_pjm
    """
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

    # Averages
    avg_arb = float(df["arbitrage_profit"].mean())
    avg_reg = float(df["regulation_profit"].mean())
    avg_mm = float(df["multi_market_profit"].mean())
    avg_pjm = float(df["pjm_multi_market_profit"].mean())
    avg_txbx = float(df["txbx_profit"].mean(skipna=True))

    # Percent improvements (per-node, skip division by zero)
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

    summary = pd.DataFrame(
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
    return summary


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
    # Example usage (adjust the window to match your data coverage)
    # Note: Provided sample data has LMP in 2020 and REG in 2021+, so an overlapping
    # window is required in your actual dataset for multi-market runs.
    start_example = "2025-06-14 00:00:00+00:00"
    end_example = "2025-10-14 00:00:00+00:00"
    df = compute_pjm_node_objectives(start_example, end_example)
    print(df.head())
    print()
    print_objectives_summary(df)

    out_path = Path(
        f"deterministic/output/pjm_node_objectives_{start_example.replace(':', '-')}_to_{end_example.replace(':', '-')}.csv"
    )
    df.to_csv(out_path)
    print(f"Saved objectives DataFrame to {out_path}")
