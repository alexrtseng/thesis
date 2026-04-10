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

INPUT_PATH = Path(__file__).parent / "input" / "mpc_value_vs_horizon_all.csv"
FIDELITY_INPUT_PATH = Path(__file__).parent / "input" / "mpc_value_vs_fidelity_combined (1).csv"

BATTERY_ORDER = ["Cap 10 MWh", "Cap 8 MWh", "Cap 4 MWh", "Cap 4 MWh (EqSoE)"]

LABEL_MAP = {
    "Cap 10 MWh": "10-hour",
    "Cap 8 MWh": "8-hour",
    "Cap 4 MWh": "4-hour",
    "Cap 4 MWh (EqSoE)": "4-hour**",
}

COLOR_MAP = {
    "Cap 10 MWh": "#F58518",
    "Cap 8 MWh": "#4C78A8",
    "Cap 4 MWh": "#54A24B",
    "Cap 4 MWh (EqSoE)": "#E45756",
}


def plot_value_vs_horizon(input_path: Path = INPUT_PATH):
    df = pd.read_csv(input_path)

    fig, ax = plt.subplots(figsize=(10, 7))

    for battery in BATTERY_ORDER:
        sub = df[df["battery"] == battery].sort_values("hf_hours")
        ax.plot(
            sub["hf_hours"],
            sub["value_$"],
            marker="o",
            color=COLOR_MAP[battery],
            label=LABEL_MAP[battery],
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("MPC High-Frequency Horizon (hours)")
    ax.set_ylabel("Simulated Value ($)")
    ax.set_title("Value Generated vs. MPC Horizon Length", fontsize=16, fontweight="bold", pad=12)
    ax.legend(title="Battery")
    ax.grid(alpha=0.3)

    output_path = Path(__file__).parent / "outputs" / "mpc" / "mpc_horizon.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


FIDELITY_SCENARIO_MAP = {
    "no_avg": "No Averaging",
    "lf_avg": "Rolling Average",
}

FIDELITY_COLOR_MAP = {
    "no_avg": "#4C78A8",
    "lf_avg": "#F58518",
}

FIDELITY_MARKER_MAP = {
    "no_avg": "o",
    "lf_avg": "s",
}


def plot_value_vs_fidelity(input_path: Path = FIDELITY_INPUT_PATH):
    df = pd.read_csv(input_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for scenario in ["no_avg", "lf_avg"]:
        sub = df[df["scenario"] == scenario].sort_values("fidelity_hours")
        ax.plot(
            sub["fidelity_hours"],
            sub["value_$"],
            marker=FIDELITY_MARKER_MAP[scenario],
            color=FIDELITY_COLOR_MAP[scenario],
            label=FIDELITY_SCENARIO_MAP[scenario],
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("High-Fidelity Horizon (hours)")
    ax.set_ylabel("Simulated Value (\\$)")
    ax.set_title("MPC Value vs. High-Fidelity Horizon", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="Scenario", framealpha=0.9)
    ax.grid(alpha=0.3)

    output_path = Path(__file__).parent / "outputs" / "mpc" / "mpc_value_vs_fidelity.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # plot_value_vs_horizon()
    plot_value_vs_fidelity()
