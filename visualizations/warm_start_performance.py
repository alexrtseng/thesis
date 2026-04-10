import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from deterministic.warm_start_arb_solver import warm_battery_arb_benchmark

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

WARM_COLOR = "#4C78A8"
COLD_COLOR = "#E45756"

OUTPUT_DIR = Path(__file__).parent / "outputs" / "mpc"


def run_scaling_benchmarks(
    horizons: list[int] = [4, 8, 16, 32, 64, 128, 256],
    lf_horizon_hours: int = 0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    pnode_id: int = 2156113094,
):
    if start is None:
        start = pd.Timestamp(year=2024, month=1, day=1, tz="UTC")
    if end is None:
        end = pd.Timestamp(year=2024, month=3, day=1, tz="UTC")

    rows = []
    for hf in horizons:
        print(f"\n--- Running hf_horizon={hf}h ---")
        stats = warm_battery_arb_benchmark(
            hf_horizon_hours=hf,
            lf_horizon_hours=lf_horizon_hours,
            start=start,
            end=end,
            pnode_id=pnode_id,
            use_lf_avg=False,
            require_equivalent_soe=False,
            verbose=False,
        )
        rows.append({
            "hf_horizon_hours": hf,
            "warm_seconds": stats["warm_seconds"],
            "cold_seconds": stats["cold_seconds"],
            "speedup": stats["speedup"],
        })

    df = pd.DataFrame(rows)
    df["warm_delta"] = df["warm_seconds"].diff()
    df["cold_delta"] = df["cold_seconds"].diff()
    df["speedup_delta"] = df["speedup"].diff()

    return df[["hf_horizon_hours", "warm_seconds", "cold_seconds",
               "warm_delta", "cold_delta", "speedup", "speedup_delta"]]


def plot_scaling(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["hf_horizon_hours"], df["warm_seconds"],
            marker="o", color=WARM_COLOR, label="Warm start")
    ax.plot(df["hf_horizon_hours"], df["cold_seconds"],
            marker="s", color=COLD_COLOR, label="Cold start")

    ax.set_xlabel("HF Horizon (hours)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("MPC Runtime vs. Horizon Length", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(df["hf_horizon_hours"])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    png_path = output_dir / "warm_start_scaling.png"
    fig.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved chart: {png_path}")
    plt.show()


def save_csv(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "warm_start_scaling.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    df = run_scaling_benchmarks()

    print("\n=== Scaling Summary ===")
    print(df.to_string(index=False))

    plot_scaling(df)
    save_csv(df)
