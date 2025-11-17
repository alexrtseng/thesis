"""
Market data visualization utilities.

This script produces publication-quality overlays of:
- Real-time LMP (5-min)
- Day-ahead LMP (hourly interpolated to 5-min)
- Regulation MCP (5-min)

Overlays include a centered 30-minute rolling average (13 five-minute points),
consistent with the smoothing used in deterministic/mpc_test.py.

Outputs:
- Year 2024 overview
- Several selected 2025 days and weeks

Figures are saved under data/plots/.

Usage (default pnode_id=2156113094):
  python -m data.graph_data --pnode 2156113094

Optional args:
  --pnode <id>          Pnode ID to plot
  --outdir <path>       Output directory (default: data/plots)
  --show                Display figures interactively as well
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from data.data_output_functions import (
    read_reg_folder,
    read_rt_and_day_ahead_prices,
)
from forecasting.transforms import AsinhScaler

# Rolling window: 13 x 5-min = ~30 minutes centered
ROLLING_WINDOW = 13


@dataclass
class PlotConfig:
    pnode_id: int = 2156113094
    rt_dir: Path = Path("data/pjm_lmps")
    da_dir: Path = Path("data/pjm_lmps_da")
    reg_dir: Path = Path("data/pjm_reg")
    out_dir: Path = Path("data/plots")
    show: bool = False


def _ensure_utc(ts: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def _as_utc(ts_like) -> pd.Timestamp:
    """Return a UTC-aware Timestamp from a string/datetime/Timestamp.

    - If input is naive -> tz_localize("UTC")
    - If input is tz-aware -> tz_convert("UTC")
    """
    t = pd.Timestamp(ts_like)
    if t.tz is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _compute_roll_centered(s: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return s.rolling(window=window, center=True, min_periods=1).mean()


def _compute_daily_extremes(
    prices: pd.DataFrame,
    year: int,
    rt_col: str = "lmp_rt",
    top_k: int = 3,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Return (top_peak_days, low_peak_days) by daily max RT LMP for given calendar year.

    Filters out days with fewer than 12 observations (~1 hour) to avoid sparse artifacts.
    Returns UTC-midnight timestamps for each selected day.
    """
    if rt_col not in prices.columns:
        return [], []
    subset = prices[prices.index.year == year]
    if subset.empty:
        return [], []
    daily = subset.groupby(subset.index.date)[rt_col]
    stats = daily.agg(["max", "count"]).rename(columns={"max": "peak"})
    stats = stats[stats["count"] >= 12]
    if stats.empty:
        return [], []
    top_days = stats.sort_values("peak", ascending=False).head(top_k).index
    low_days = stats.sort_values("peak", ascending=True).head(top_k).index

    def _to_midnight_utc(d) -> pd.Timestamp:
        return pd.Timestamp(str(d)).tz_localize("UTC")

    return [_to_midnight_utc(d) for d in top_days], [
        _to_midnight_utc(d) for d in low_days
    ]


def _compute_weekly_volatility(
    prices: pd.DataFrame,
    year: int,
    rt_col: str = "lmp_rt",
    top_k: int = 2,
    min_days: int = 4,
) -> tuple[
    list[tuple[pd.Timestamp, pd.Timestamp]], list[tuple[pd.Timestamp, pd.Timestamp]]
]:
    """Return (high_vol_weeks, low_vol_weeks) by std dev of RT LMP within ISO weeks.

    Only include weeks with at least `min_days` worth of data (min_days*288 five-min slots).
    Each tuple is (start_timestamp_utc, end_timestamp_utc_inclusive_last_slot).
    """
    if rt_col not in prices.columns:
        return [], []
    subset = prices[prices.index.year == year]
    if subset.empty:
        return [], []
    iso = subset.index.isocalendar()
    df = subset.copy()
    df["iso_year"] = iso.year
    df["iso_week"] = iso.week
    df = df[df["iso_year"] == year]
    grp = df.groupby(["iso_year", "iso_week"])[rt_col]
    stats = grp.agg(["std", "count"]).rename(columns={"std": "vol"})
    stats = stats[stats["count"] >= min_days * 288]
    if stats.empty:
        return [], []
    top_weeks = stats.sort_values("vol", ascending=False).head(top_k).index.tolist()
    low_weeks = stats.sort_values("vol", ascending=True).head(top_k).index.tolist()

    def week_bounds(iso_year: int, iso_week: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp.fromisocalendar(iso_year, iso_week, 1).tz_localize("UTC")
        end = start + pd.Timedelta(days=7) - pd.Timedelta(minutes=5)
        return start, end

    return [week_bounds(y, w) for y, w in top_weeks], [
        week_bounds(y, w) for y, w in low_weeks
    ]


def _load_prices(config: PlotConfig) -> pd.DataFrame:
    """Load RT and DA prices as a 5-min UTC-indexed DataFrame [lmp_rt, lmp_da]."""
    prices = read_rt_and_day_ahead_prices(
        config.rt_dir, config.da_dir, pnode_id=config.pnode_id
    )
    # Ensure sorted and clean
    prices = prices.sort_index()
    return prices


def _load_reg(config: PlotConfig) -> pd.DataFrame:
    """Load regulation MCP; return aligned Series at 5-min index when joined."""
    reg = read_reg_folder(config.reg_dir)
    reg = reg.rename(columns={"datetime_beginning_utc": "time"})
    reg["time"] = pd.to_datetime(reg["time"], utc=True)
    reg = reg.set_index("time").sort_index()
    return reg[["mcp"]]


def _format_time_axis(ax: plt.Axes, freq_hint: str) -> None:
    import matplotlib.dates as mdates

    if freq_hint == "year":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    elif freq_hint == "week":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    elif freq_hint == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d"))
    ax.grid(True, which="major", axis="both", alpha=0.3)


def _plot_overlay(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    *,
    show: bool = False,
    freq_hint: str = "week",
    y_label: str = "Price ($/MWh)",
) -> None:
    """
    Plot overlay: RT LMP, DA LMP, REG MCP with centered rolling averages.
    Left axis: LMPs; Right axis: REG MCP.
    """
    if df.empty:
        print(f"[WARN] Empty frame for plot '{title}', skipping {out_path.name}")
        return

    # Compute rolling averages
    df = df.copy()
    if "lmp_rt" in df:
        df["lmp_rt_roll"] = _compute_roll_centered(df["lmp_rt"])
    # Day-ahead rolling average intentionally omitted per latest requirement.
    if "mcp" in df:
        df["mcp_roll"] = _compute_roll_centered(df["mcp"])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot LMPs (real-time & day-ahead)
    if "lmp_rt" in df:
        ax.plot(df.index, df["lmp_rt"], color="#1f77b4", alpha=0.30, label="RT LMP")
        ax.plot(
            df.index,
            df["lmp_rt_roll"],
            color="#1f77b4",
            linewidth=1.8,
            label="RT LMP (±30m avg)",
        )
    if "lmp_da" in df:
        ax.plot(df.index, df["lmp_da"], color="#ff7f0e", alpha=0.40, label="DA LMP")

    # Plot Regulation MCP on same axis
    if "mcp" in df:
        ax.plot(df.index, df["mcp"], color="#2ca02c", alpha=0.25, label="REG MCP")
    if "mcp_roll" in df:
        ax.plot(
            df.index,
            df["mcp_roll"],
            color="#2ca02c",
            linewidth=1.8,
            label="REG MCP (±30m avg)",
        )

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right")

    _format_time_axis(ax, freq_hint)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_time_spans(config: PlotConfig, *, analysis_year: int = 2025) -> list[Path]:
    """Produce required plots (static + dynamic extremes) and return list of output paths.

    Dynamic selections:
    - Top 3 and bottom 3 peak days by daily max RT LMP in `analysis_year`.
    - Top 2 and bottom 2 volatility weeks by weekly std dev of RT LMP in `analysis_year`.
    """
    prices = _load_prices(config)  # lmp_rt, lmp_da
    reg = _load_reg(config)  # mcp

    # Align REG to price index for joins later
    # We'll join per slice to limit memory footprint

    outputs: list[Path] = []

    def do_span(start: pd.Timestamp, end: pd.Timestamp, label: str, freq_hint: str):
        # slice inclusive start, exclusive end
        s = _as_utc(start)
        e = _as_utc(end)

        df = prices.loc[s:e].copy()
        reg_slice = reg.loc[s:e]
        df = df.join(reg_slice, how="left")

        # Forward-fill small gaps in reg inside the slice for cleaner visuals
        if "mcp" in df:
            df["mcp"] = df["mcp"].ffill().bfill()

        title = f"PJM Prices Overlay (Pnode {config.pnode_id}) — {label}"
        out_file = (
            config.out_dir
            / f"market_overlay_{label.replace(' ', '_').replace(':', '-')}.png"
        )
        _plot_overlay(df, title, out_file, show=config.show, freq_hint=freq_hint)
        outputs.append(out_file)

        # Also produce an asinh-transformed variant
        if getattr(config, "also_asinh", True):
            df_t = df.copy()
            for col in ("lmp_rt", "lmp_da", "mcp"):
                if col in df_t:
                    df_t[col] = AsinhScaler.transform(df_t[col])
            title_t = f"{title} (asinh scale)"
            out_file_t = out_file.with_name(out_file.stem + "_asinh" + out_file.suffix)
            _plot_overlay(
                df_t,
                title_t,
                out_file_t,
                show=config.show,
                freq_hint=freq_hint,
                y_label="asinh(Price)",
            )
            outputs.append(out_file_t)

    # 2024 full year (static)
    do_span(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-12-31 23:55"),
        "2024_full_year",
        "year",
    )

    # Selected example 2025 days (static)
    selected_days = [
        "2025-01-07",
        "2025-01-14",
        "2025-02-01",
    ]
    for d in selected_days:
        s = pd.Timestamp(d, tz="UTC")
        e = s + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        do_span(s, e, f"day_{s.date()}", "day")

    # Selected example 2025 weeks (static)
    selected_week_starts = [
        "2025-01-01",
        "2025-02-01",
    ]
    for d in selected_week_starts:
        s = pd.Timestamp(d, tz="UTC")
        e = s + pd.Timedelta(days=7) - pd.Timedelta(minutes=5)
        do_span(s, e, f"week_{s.date()}", "week")

    # Dynamic extremes for analysis_year
    top_days, low_days = _compute_daily_extremes(prices, year=analysis_year)
    for i, day_ts in enumerate(top_days, start=1):
        end = day_ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        do_span(day_ts, end, f"top_peak_day_{i}_{day_ts.date()}", "day")
    for i, day_ts in enumerate(low_days, start=1):
        end = day_ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        do_span(day_ts, end, f"low_peak_day_{i}_{day_ts.date()}", "day")

    high_vol_weeks, low_vol_weeks = _compute_weekly_volatility(
        prices, year=analysis_year
    )
    for i, (ws, we) in enumerate(high_vol_weeks, start=1):
        do_span(ws, we, f"high_vol_week_{i}_{ws.date()}", "week")
    for i, (ws, we) in enumerate(low_vol_weeks, start=1):
        do_span(ws, we, f"low_vol_week_{i}_{ws.date()}", "week")

    return outputs


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plot PJM market overlays")
    parser.add_argument(
        "--pnode", type=int, default=PlotConfig.pnode_id, help="Pnode ID"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PlotConfig.out_dir),
        help="Output directory",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Analysis year for dynamic extremes (default: 2025)",
    )
    # Asinh variant plots (enabled by default). Use --no-asinh to disable.
    parser.add_argument(
        "--no-asinh",
        dest="asinh",
        action="store_false",
        help="Disable asinh-transformed companion plots",
    )
    parser.set_defaults(asinh=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = PlotConfig(pnode_id=args.pnode, out_dir=Path(args.outdir), show=args.show)
    # attach runtime flag without changing dataclass signature
    setattr(config, "also_asinh", args.asinh)
    _ = plot_time_spans(config, analysis_year=args.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
