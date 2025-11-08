from __future__ import annotations

import os
import shutil
import tempfile
from typing import List

import matplotlib
import numpy as np
import pandas as pd

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from darts import TimeSeries

import wandb


def _to_series(ts: TimeSeries) -> pd.Series:
    # Returns a pandas Series with a DatetimeIndex
    s = ts.pd_series(copy=False)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s


essential_colors = {
    "actual": "#1f77b4",  # blue
    "pred": "#d62728",  # red
}


def log_random_day_plots(
    actual: TimeSeries,
    predicted: TimeSeries,
    *,
    num_days: int = 3,
    title_prefix: str | None = None,
    wandb_key: str = "val_random_day_plots",
    cleanup: bool = True,
) -> List[str]:
    """Generate and log plots for N random days comparing prediction to actual.

    Saves plots under a temporary directory and logs them to Weights & Biases.

    Parameters
    ----------
    actual : TimeSeries
        Ground truth validation series.
    predicted : TimeSeries
        Predicted series aligned with validation.
    num_days : int
        Number of distinct calendar days to plot.
    title_prefix : str | None
        Optional text to prepend to each plot title.
    wandb_key : str
        Key used when logging the list of images to W&B.

    Returns
    -------
    List[str]
        List of file paths for the saved plot images. If cleanup=True (default),
        the files are deleted before return, so the paths will no longer exist
        on disk and are provided only for reference.
    """
    act = _to_series(actual)
    pred = _to_series(predicted)

    # Align indices and drop any timestamps not present in both
    df = pd.concat([act.rename("actual"), pred.rename("pred")], axis=1).dropna()
    if df.empty:
        return []

    days = pd.Index(df.index.normalize().unique())
    k = min(num_days, len(days))
    if k <= 0:
        return []

    chosen = np.random.choice(np.arange(len(days)), size=k, replace=False)
    tmpdir = tempfile.mkdtemp(prefix="darts_day_plots_")
    saved_paths: List[str] = []

    for idx in chosen:
        day = pd.Timestamp(days[int(idx)])
        start = day
        end = day + pd.Timedelta(days=1)
        day_df = df.loc[(df.index >= start) & (df.index < end)]
        if day_df.empty:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(
            day_df.index,
            day_df["actual"],
            label="Actual",
            color=essential_colors["actual"],
            linewidth=1.5,
        )
        plt.plot(
            day_df.index,
            day_df["pred"],
            label="Predicted",
            color=essential_colors["pred"],
            linewidth=1.5,
            alpha=0.9,
        )
        plt.xlabel("Time")
        plt.ylabel("Price")
        title = f"{title_prefix + ' - ' if title_prefix else ''}{day.date()}"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(tmpdir, f"pred_vs_actual_{day.date()}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved_paths.append(out_path)

    if saved_paths:
        images = [wandb.Image(p, caption=os.path.basename(p)) for p in saved_paths]
        wandb.log({wandb_key: images})

    # Remove the temporary directory and files if requested
    if cleanup:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    return saved_paths
