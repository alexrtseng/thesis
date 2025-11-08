from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

FIVE_MIN = "5min"
DAY_SHIFT_PERIODS = 24 * 60 // 5  # 288


def read_lmp_folder(lmp_dir: Path, da: bool = False) -> pd.DataFrame:
    lmp_name = "total_lmp_rt" if not da else "total_lmp_da"
    files = sorted(lmp_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No LMP CSVs found in {lmp_dir}")
    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["datetime_beginning_utc"])
        if {lmp_name, "pnode_id"}.difference(df.columns):
            raise ValueError(f"Missing expected RT LMP columns in {f.name}")
        df["datetime_beginning_utc"] = df["datetime_beginning_utc"].dt.tz_localize(
            "UTC"
        )
        df = df.rename(columns={lmp_name: "lmp_rt" if not da else "lmp_da"})[
            ["datetime_beginning_utc", "pnode_id", "lmp_rt" if not da else "lmp_da"]
        ]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def read_reg_folder(reg_dir: Path = Path("data/pjm_reg")) -> pd.DataFrame:
    """
    Load all regulation CSVs, expecting columns:
      - datetime_beginning_utc
      - service
      - mcp
    Filters to rows where service == 'REG'. Returns datetime_beginning_utc (UTC), mcp.
    """
    files = sorted(reg_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No REG CSVs found in {reg_dir}")
    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["datetime_beginning_utc"])
        if {"mcp", "service"}.difference(df.columns):
            raise ValueError(f"Missing expected REG columns in {f.name}")
        df = df[df["service"].astype(str).str.upper() == "REG"].copy()
        df["datetime_beginning_utc"] = df["datetime_beginning_utc"].dt.tz_localize(
            "UTC"
        )
        dfs.append(df[["datetime_beginning_utc", "mcp"]])
    return pd.concat(dfs, ignore_index=True)


def _ensure_utc_index(idx: Iterable) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def fill_missing_5min_slots(
    df: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    columns: Optional[list[str]] = None,
    *,
    prior_day_backfill: bool = True,
    ffill: bool = True,
    bfill: bool = False,
) -> pd.DataFrame:
    """
    Reindex to expected 5-min grid and fill missing numeric data.

    Filling order per column:
      1. Prior-day same-slot value (t - 24h) if prior_day_backfill
      2. Forward fill if ffill
      3. Backward fill if bfill

    Columns auto-inferred if not provided: intersection of ['lmp_rt','lmp_da','mcp'] or all numeric.
    """
    df = df.copy()
    df.index = _ensure_utc_index(df.index)
    out = df.reindex(expected_index)

    if columns is None:
        candidate = [c for c in ["lmp_rt", "lmp_da", "mcp"] if c in out.columns]
        if candidate:
            columns = candidate
        else:
            columns = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]

    if prior_day_backfill:
        for c in columns:
            prior = out[c].shift(DAY_SHIFT_PERIODS)
            out[c] = out[c].where(out[c].notna(), prior)

    if ffill:
        out[columns] = out[columns].ffill()
    if bfill:
        out[columns] = out[columns].bfill()

    return out


def read_rt_and_day_ahead_prices(
    rt_dir: Path,
    da_dir: Path,
    *,
    pnode_id: int,
) -> pd.DataFrame:
    """Read real-time and day-ahead LMP data for a single pnode and return a 5-min grid.

    - Real-time files: expect columns datetime_beginning_utc, pnode_id, total_lmp_rt
    - Day-ahead files: expect columns datetime_beginning_utc, pnode_id, total_lmp_da (hourly)
    - Day-ahead hourly values are linearly interpolated to 5-min resolution.

    Returns DataFrame with columns: time, lmp_rt, lmp_da (time is UTC tz-aware).
    """
    # Read all RT and DA using existing helper (reuse logic by toggling da flag)
    rt_all = read_lmp_folder(rt_dir, da=False)
    da_all = read_lmp_folder(da_dir, da=True)

    # Filter to single pnode
    rt_all = rt_all[rt_all["pnode_id"] == int(pnode_id)]
    da_all = da_all[da_all["pnode_id"] == int(pnode_id)]

    # Merge on timestamp (outer to retain full coverage)
    merged = pd.merge(
        rt_all,
        da_all,
        on=["datetime_beginning_utc", "pnode_id"],
        how="left",
    ).sort_values("datetime_beginning_utc")

    # Build full 5-min index across span (ensure UTC tz-aware and collapse duplicates)
    ts = _ensure_utc_index(merged["datetime_beginning_utc"])  # may contain duplicates
    start = ts.min().floor(FIVE_MIN)
    end = ts.max().ceil(FIVE_MIN)
    full_index = pd.date_range(start, end, freq=FIVE_MIN, tz="UTC")

    # Use a clean index named 'time' and aggregate duplicate timestamps
    merged = (
        merged.assign(time=ts)
        .drop(columns=["datetime_beginning_utc", "pnode_id"], errors="ignore")
        .set_index("time")
        .groupby(level=0)
        .mean(numeric_only=True)
        .reindex(full_index)
    )

    # Interpolate lmp_da from hourly to 5-min (after forward/linear fill prep)
    # Keep original hourly positions; linear interpolation only for gaps.
    if "lmp_da" in merged.columns:
        # No prior-day logic for DA; pure linear over time.
        merged["lmp_da"] = merged["lmp_da"].interpolate(method="time")

    # Real-time may have gaps; fill using prior-day then forward fill
    if "lmp_rt" in merged.columns:
        rt_series = merged["lmp_rt"]
        prior = rt_series.shift(DAY_SHIFT_PERIODS)
        rt_series = rt_series.where(rt_series.notna(), prior)
        rt_series = rt_series.ffill()
        merged["lmp_rt"] = rt_series

    merged.index.name = "time"
    return merged[["lmp_rt", "lmp_da"]]


def read_rt_da_with_weather(
    rt_dir: Path,
    da_dir: Path,
    weather_file: Path,
    *,
    pnode_id: int,
) -> pd.DataFrame:
    """Return 5-min grid of real-time + day-ahead LMPs enriched with weather features.

    Inputs:
      rt_dir: folder containing real-time LMP CSVs (5-min) with total_lmp_rt.
      da_dir: folder containing day-ahead LMP CSVs (hourly) with total_lmp_da.
      weather_file: CSV produced by weather script (hourly) containing 'time' column + weather vars.
      pnode_id: single node id to extract.

    Output DataFrame (index: time UTC 5-min): columns [lmp_rt, lmp_da, <weather columns...>].
    Weather is upsampled from hourly to 5-min using time-based linear interpolation for numeric columns.
    LMP day-ahead is interpolated similarly; real-time filled using prior-day then forward-fill.
    """
    prices = read_rt_and_day_ahead_prices(rt_dir, da_dir, pnode_id=pnode_id)

    # Load weather hourly data
    wdf = pd.read_csv(weather_file, parse_dates=["time"])
    if "time" not in wdf.columns:
        raise ValueError(f"weather_file {weather_file} missing 'time' column")

    # Set index and ensure UTC (assume naive timestamps are UTC)
    wdf["time"] = pd.to_datetime(wdf["time"])
    wdf["time"] = (
        wdf["time"].dt.tz_localize("UTC")
        if wdf["time"].dt.tz is None
        else wdf["time"].dt.tz_convert("UTC")
    )
    wdf = wdf.set_index("time").sort_index()

    # Determine 5-min target index from prices
    target_index = prices.index

    # Reindex weather to 5-min grid (upsample then interpolate numeric columns)
    wdf_5 = wdf.reindex(target_index.union(wdf.index)).sort_index()
    wdf_5 = wdf_5.reindex(target_index)  # final align to price span exactly

    # Interpolate numeric columns
    numeric_cols = [c for c in wdf_5.columns if pd.api.types.is_numeric_dtype(wdf_5[c])]
    for c in numeric_cols:
        wdf_5[c] = wdf_5[c].interpolate(method="time")
        wdf_5[c] = wdf_5[c].ffill().bfill()

    # Merge with price data
    out = prices.join(wdf_5, how="left")
    return out


if __name__ == "__main__":
    # Example usage
    rt_dir = Path("data/pjm_lmps")
    da_dir = Path("data/pjm_lmps_da")
    pnode_id = 2156113094

    price_df = read_rt_and_day_ahead_prices(rt_dir, da_dir, pnode_id=pnode_id)
    print(price_df.head())

    # Example with weather
    weather_path = Path(
        f"data/weather/node_{pnode_id}/weather_2020-11-08_to_2025-11-07.csv"
    )
    if weather_path.exists():
        enriched = read_rt_da_with_weather(
            rt_dir, da_dir, weather_path, pnode_id=pnode_id
        )
        print(enriched.head(10))
        print(enriched.tail(10))
        print(enriched.columns)
