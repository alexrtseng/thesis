import pathlib
import random
import sys
import threading
from typing import Optional, Tuple

import matplotlib
import pandas as pd


def _ensure_backend(show: bool) -> bool:
    """Ensure a safe Matplotlib backend for the current context.

    - If show is False, force headless Agg backend.
    - On macOS, only allow GUI when on the main thread; otherwise force Agg.
    Returns True if GUI display is allowed, False if headless.
    """
    # If not showing, ensure headless rendering
    if not show:
        matplotlib.use("Agg", force=True)
        return False

    # On macOS, GUI backends require main thread
    if (
        sys.platform == "darwin"
        and threading.current_thread() is not threading.main_thread()
    ):
        matplotlib.use("Agg", force=True)
        return False

    # Otherwise, use current/default backend (Jupyter/Qt/OSX etc.)
    return True


def _select_random_day_index(df: pd.DataFrame, seed: int) -> pd.DatetimeIndex:
    """
    Choose a random calendar day from df's index (must be DatetimeIndex),
    return the slice covering that full day (midnight to midnight next day).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    dates = df.index.normalize().unique()
    if len(dates) == 0:
        raise ValueError("Index has no dates to sample.")
    random.seed(seed)
    chosen = random.choice(dates)
    day_start = chosen
    day_end = chosen + pd.Timedelta(days=1)
    mask = (df.index >= day_start) & (df.index < day_end)
    return df.index[mask]


def _select_week_index(df: pd.DataFrame, seed: int) -> pd.DatetimeIndex:
    """
    Choose a start day randomly; return 7-day window.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    dates = df.index.normalize().unique()
    if len(dates) < 7:
        raise ValueError("Not enough days (need at least 7) for a week sample.")
    random.seed(seed + 101)  # offset so day/week picks differ
    # restrict possible starts so we have room for 7 days
    valid_starts = dates[dates <= dates.max() - pd.Timedelta(days=6)]
    chosen = random.choice(valid_starts)
    week_start = chosen
    week_end = chosen + pd.Timedelta(days=7)
    mask = (df.index >= week_start) & (df.index < week_end)
    return df.index[mask]


def _slice_by_day(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    start = date.normalize()
    end = start + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def _slice_by_week(df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    start = start_date.normalize()
    end = start + pd.Timedelta(days=7)
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def plot_opt_vs_perf_samples(
    opt_results: Tuple[pd.DataFrame, dict],
    preds_df: pd.DataFrame,
    day: Optional[pd.Timestamp] = None,
    week_start: Optional[pd.Timestamp] = None,
    seed: int = 42,
    save_dir: Optional[str] = None,
    show: bool = False,
):
    """Generate four plots separating price+horizon curves from net decisions.

    Outputs:
        1) Day Price & Horizons
        2) Week Price & Horizons
        3) Day Net Decisions (Perfect vs Forecast)
        4) Week Net Decisions (Perfect vs Forecast)

    Returns (fig_day_price, fig_week_price, fig_day_net, fig_week_net).
    """
    # Configure backend before importing pyplot
    gui_ok = _ensure_backend(show)
    import matplotlib.pyplot as plt  # noqa: WPS433 (intentional local import after backend set)

    combined_df = opt_results[0]
    forecast_charge_col = "charge_mw_6"
    forecast_discharge_col = "discharge_mw_6"
    price_col = "lmp_24"  # super scuffed but this is actual price

    # Build net series
    combined_df["net_perf_mw"] = (
        combined_df["charge_mw_perf"] - combined_df["discharge_mw_perf"]
    )
    combined_df["net_forecast_mw"] = (
        combined_df[forecast_charge_col] - combined_df[forecast_discharge_col]
    )

    # Decide day/week windows
    if day is None:
        day_index = _select_random_day_index(combined_df, seed)
        day_start = day_index.min().normalize()
    else:
        day_start = pd.to_datetime(day)
        day_index = combined_df.loc[
            (combined_df.index >= day_start)
            & (combined_df.index < day_start + pd.Timedelta(days=1))
        ].index

    if len(day_index) == 0:
        raise ValueError("No data found for selected day.")

    if week_start is None:
        week_index = _select_week_index(combined_df, seed)
        week_start_date = week_index.min().normalize()
    else:
        week_start_date = pd.to_datetime(week_start)
        week_index = combined_df.loc[
            (combined_df.index >= week_start_date)
            & (combined_df.index < week_start_date + pd.Timedelta(days=7))
        ].index

    if len(week_index) == 0:
        raise ValueError("No data found for selected week.")

    # Merge preds_df (actual & horizons) for slicing
    # We'll align on index; no strict join required beyond intersection.
    # Combined decisions may be a subset due to dropna; reindex preds_df to combined_df index for alignment.
    preds_aligned = preds_df.reindex(combined_df.index)

    # Horizons to show
    horizons_to_plot = [1, 3, 6, 12, 24]
    horizon_cols = [f"h_{h}" for h in horizons_to_plot]

    # ---- DAY PRICE & HORIZONS FIGURE ----
    day_decisions = combined_df.loc[day_index]
    day_preds = preds_aligned.loc[day_index]

    fig_day_price, ax_day_price = plt.subplots(figsize=(12, 5))

    ax_day_price.plot(
        day_decisions.index,
        day_decisions[price_col],
        color="tab:blue",
        label="Actual Price",
        linewidth=1.6,
    )

    # Plot horizons (skip if missing)
    horizon_colors = {
        "h_1": "tab:orange",
        # "h_3": "tab:green",
        "h_6": "tab:red",
        "h_12": "tab:purple",
        # "h_24": "tab:brown",
    }
    for col in horizon_cols:
        if col in day_preds.columns and day_preds[col].notna().any():
            ax_day_price.plot(
                day_preds.index,
                day_preds[col],
                label=col,
                color=horizon_colors.get(col, "gray"),
                alpha=0.55,
                linewidth=1.2,
            )

    ax_day_price.set_title(f"Prices & Horizons (Day: {day_start.date()})")
    ax_day_price.set_ylabel("Price ($/MWh)")
    ax_day_price.set_xlabel("Time")
    ax_day_price.grid(alpha=0.35, linewidth=0.5)
    ax_day_price.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
    )
    # leave space on the right for the external legend
    fig_day_price.subplots_adjust(right=0.78)
    fig_day_price.tight_layout()

    # ---- DAY NET DECISIONS FIGURE ----
    fig_day_net, ax_day_net = plt.subplots(figsize=(12, 4))
    ax_day_net.step(
        day_decisions.index,
        day_decisions["net_perf_mw"],
        color="black",
        label="Net MW (Perfect)",
        linewidth=1.4,
        where="post",
    )
    ax_day_net.step(
        day_decisions.index,
        day_decisions["net_forecast_mw"],
        color="tab:cyan",
        label="Net MW (Forecast)",
        linewidth=1.4,
        where="post",
    )
    ax_day_net.set_title(f"Net Decisions (Day: {day_start.date()})")
    ax_day_net.set_ylabel("Net Power (MW)")
    ax_day_net.set_xlabel("Time")
    ax_day_net.grid(alpha=0.35, linewidth=0.5)
    ax_day_net.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
    )
    fig_day_net.tight_layout()

    # ---- WEEK PRICE & HORIZONS FIGURE ----
    week_decisions = combined_df.loc[week_index]
    week_preds = preds_aligned.loc[week_index]

    fig_week_price, ax_week_price = plt.subplots(figsize=(14, 5))

    ax_week_price.plot(
        week_decisions.index,
        week_decisions[price_col],
        color="tab:blue",
        label="Actual Price",
        linewidth=1.3,
    )

    for col in horizon_cols:
        if col in week_preds.columns and week_preds[col].notna().any():
            ax_week_price.plot(
                week_preds.index,
                week_preds[col],
                label=col,
                color=horizon_colors.get(col, "gray"),
                alpha=0.45,
                linewidth=0.9,
            )

    ax_week_price.set_title(f"Prices & Horizons (Week start {week_start_date.date()})")
    ax_week_price.set_ylabel("Price ($/MWh)")
    ax_week_price.set_xlabel("Time")
    ax_week_price.grid(alpha=0.3, linewidth=0.5)
    ax_week_price.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
    )
    fig_week_price.subplots_adjust(right=0.78)
    fig_week_price.tight_layout()

    # ---- WEEK NET DECISIONS FIGURE ----
    fig_week_net, ax_week_net = plt.subplots(figsize=(14, 4))
    ax_week_net.step(
        week_decisions.index,
        week_decisions["net_perf_mw"],
        color="black",
        label="Net MW (Perfect)",
        linewidth=1.1,
        where="post",
    )
    ax_week_net.step(
        week_decisions.index,
        week_decisions["net_forecast_mw"],
        color="tab:cyan",
        label="Net MW (Forecast)",
        linewidth=1.1,
        where="post",
    )
    ax_week_net.set_title(f"Net Decisions (Week start {week_start_date.date()})")
    ax_week_net.set_ylabel("Net Power (MW)")
    ax_week_net.set_xlabel("Time")
    ax_week_net.grid(alpha=0.3, linewidth=0.5)
    ax_week_net.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
    )
    fig_week_net.tight_layout()

    # Optionally save
    if save_dir is not None:
        out_dir = pathlib.Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_day_price_path = out_dir / f"day_prices_{day_start.date()}.png"
        fig_day_net_path = out_dir / f"day_net_{day_start.date()}.png"
        fig_week_price_path = out_dir / f"week_prices_{week_start_date.date()}.png"
        fig_week_net_path = out_dir / f"week_net_{week_start_date.date()}.png"
        fig_day_price.savefig(fig_day_price_path, dpi=150)
        fig_day_net.savefig(fig_day_net_path, dpi=150)
        fig_week_price.savefig(fig_week_price_path, dpi=150)
        fig_week_net.savefig(fig_week_net_path, dpi=150)
        print(f"Saved day price plot to {fig_day_price_path}")
        print(f"Saved day net plot to {fig_day_net_path}")
        print(f"Saved week price plot to {fig_week_price_path}")
        print(f"Saved week net plot to {fig_week_net_path}")

    if gui_ok and show:
        plt.show()
    else:
        # Headless mode – close figures to free resources
        try:
            plt.close("all")
        except Exception:
            pass

    return fig_day_price, fig_week_price, fig_day_net, fig_week_net
