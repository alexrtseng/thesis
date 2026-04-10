import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

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

INPUT_DIR = Path(__file__).parent / "input" / "ensemble_tests"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "ensemble"

COLOR_MAP = {6: "#4C78A8", 9: "#F58518"}


def load_data() -> pd.DataFrame:
    dfs = []
    for folder in INPUT_DIR.iterdir():
        if folder.is_dir():
            csvs = list(folder.glob("*.csv"))
            if csvs:
                df = pd.read_csv(csvs[0])
                df["HF Horizon"] = 6 if folder.name.startswith("Six") else 9
                dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Length: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    combined = combined.sort_values("pct_perf", ascending=False)
    print(combined.head(10).to_string())
    return combined


def plot_branches(df: pd.DataFrame):
    df = df.copy()
    df["branches"] = df["num_hf_forecasters"] * df["num_lf_forecasters"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in [6, 9]:
        sub = df[df["HF Horizon"] == horizon].groupby("branches")["pct_perf"].mean().reset_index()
        ax.plot(
            sub["branches"],
            sub["pct_perf"],
            marker="o",
            color=COLOR_MAP[horizon],
            label=f"{horizon}-horizon",
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("Forecast Branches (HF Forecasters * LF Forecasters)")
    ax.set_ylabel("Avg. Percent Perfect Foresight")
    ax.set_title("Impact of Forecast Branches on Performance", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="HF Horizon")
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "ensemble_branches.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_hf_forecasters(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in [6, 9]:
        sub = df[df["HF Horizon"] == horizon].groupby("num_hf_forecasters")["pct_perf"].mean().reset_index()
        ax.plot(
            sub["num_hf_forecasters"],
            sub["pct_perf"],
            marker="o",
            color=COLOR_MAP[horizon],
            label=f"{horizon}-horizon",
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("Number of HF Forecasters")
    ax.set_ylabel("Avg. Percent Perfect Foresight")
    ax.set_title("Impact of HF Forecasters on Performance", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="HF Horizon")
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "ensemble_hf_forecasters.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_lf_forecasters(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in [6, 9]:
        sub = df[df["HF Horizon"] == horizon].groupby("num_lf_forecasters")["pct_perf"].mean().reset_index()
        ax.plot(
            sub["num_lf_forecasters"],
            sub["pct_perf"],
            marker="o",
            color=COLOR_MAP[horizon],
            label=f"{horizon}-horizon",
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("Number of LF Forecasters")
    ax.set_ylabel("Avg. Percent Perfect Foresight")
    ax.set_title("Impact of LF Forecasters on Performance", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="HF Horizon")
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "ensemble_lf_forecasters.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def scatter_branches(df: pd.DataFrame):
    df = df.copy()
    df["branches"] = df["num_hf_forecasters"] * df["num_lf_forecasters"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in [6, 9]:
        sub = df[df["HF Horizon"] == horizon]
        ax.scatter(
            sub["branches"],
            sub["pct_perf"],
            color=COLOR_MAP[horizon],
            label=f"{horizon}-horizon",
            alpha=0.5,
            s=20,
        )

    ax.set_xlabel("Forecast Branches (HF Forecasters * LF Forecasters)")
    ax.set_ylabel("Percent Perfect Foresight")
    ax.set_title("Forecast Branches vs. Performance (Scatter)", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="HF Horizon")
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "scatter_branches.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def scatter_hf_forecasters(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in [6, 9]:
        sub = df[df["HF Horizon"] == horizon]
        ax.scatter(
            sub["num_hf_forecasters"],
            sub["pct_perf"],
            color=COLOR_MAP[horizon],
            label=f"{horizon}-horizon",
            alpha=0.5,
            s=20,
        )

    ax.set_xlabel("Number of HF Forecasters")
    ax.set_ylabel("Percent Perfect Foresight")
    ax.set_title("HF Forecasters vs. Performance (Scatter)", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="HF Horizon")
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "scatter_hf_forecasters.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def scatter_lf_forecasters(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in [6, 9]:
        sub = df[df["HF Horizon"] == horizon]
        ax.scatter(
            sub["num_lf_forecasters"],
            sub["pct_perf"],
            color=COLOR_MAP[horizon],
            label=f"{horizon}-horizon",
            alpha=0.5,
            s=20,
        )

    ax.set_xlabel("Number of LF Forecasters")
    ax.set_ylabel("Percent Perfect Foresight")
    ax.set_title("LF Forecasters vs. Performance (Scatter)", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="HF Horizon")
    ax.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "scatter_lf_forecasters.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = load_data()
    plot_branches(df)
    plot_hf_forecasters(df)
    plot_lf_forecasters(df)
    scatter_branches(df)
    scatter_hf_forecasters(df)
    scatter_lf_forecasters(df)
