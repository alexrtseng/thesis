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

INPUT_PATH = Path(__file__).parent / "input" / "pjm_node_objectives_2021-10-16 00-00-00+00-00_to_2025-10-14 00-00-00+00-00.csv"

PROFIT_COLS = [
    "arbitrage_profit",
    "regulation_profit",
    "multi_market_profit",
    "pjm_multi_market_profit",
    "txbx_profit",
]

LABEL_MAP = {
    "arbitrage_profit": "Arbitrage",
    "regulation_profit": "Regulation",
    "multi_market_profit": "Multi-market",
    "pjm_multi_market_profit": "PJM multi-market",
    "txbx_profit": "TXBX",
}

COLOR_MAP = {
    "arbitrage_profit": "#4C78A8",
    "regulation_profit": "#F58518",
    "multi_market_profit": "#54A24B",
    "pjm_multi_market_profit": "#E45756",
    "txbx_profit": "#72B7B2",
}


def plot_ranked_profit_bars(input_path: Path = INPUT_PATH):
    df = pd.read_csv(input_path)

    candidate_label_cols = ["pnode_name", "pnode_short_name", "node_name", "name", "pnode_id"]
    node_col = next((c for c in candidate_label_cols if c in df.columns), "pnode_id")

    plot_df = df[[node_col] + PROFIT_COLS].copy()
    plot_df[node_col] = plot_df[node_col].astype(str)
    plot_df = plot_df.sort_values("multi_market_profit", ascending=True).reset_index(drop=True)

    for c in PROFIT_COLS:
        plot_df[c] = plot_df[c] / 1_000_000

    # Build layered non-additive bars: each strategy drawn from the previous profit level
    # to its own, so bar height = max profit across strategies (not their sum)
    intervals = []
    for _, row in plot_df.iterrows():
        sorted_pairs = sorted([(c, row[c]) for c in PROFIT_COLS], key=lambda x: x[1])
        prev = 0.0
        for col, value in sorted_pairs:
            intervals.append({"node": row[node_col], "strategy": col, "bottom": prev, "height": value - prev})
            prev = value
    interval_df = pd.DataFrame(intervals)

    n_nodes = len(plot_df)
    fig, ax = plt.subplots(figsize=(13, max(12, 0.34 * n_nodes + 1.5)))

    for i, node in enumerate(plot_df[node_col]):
        for _, seg in interval_df[interval_df["node"] == node].iterrows():
            ax.barh(i, seg["height"], left=seg["bottom"], color=COLOR_MAP[seg["strategy"]], edgecolor="none", height=0.5)

    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels(plot_df[node_col], fontsize=11)
    ax.set_xlabel("Profit ($ millions)")
    ax.set_ylabel("PJM Node ID")
    ax.set_title("PJM Nodes Ranked by Profit with Strategy Layers", fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[c], label=LABEL_MAP[c]) for c in PROFIT_COLS]
    ax.legend(handles=handles, title="Strategy", loc="lower right")

    output_path = Path(__file__).parent / "outputs" / "deterministic_section" / "ranked_profit_bars.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


def plot_strategy_boxplot(input_path: Path = INPUT_PATH):
    df = pd.read_csv(input_path)

    data = [df[c].dropna() / 1_000_000 for c in PROFIT_COLS]
    labels = [LABEL_MAP[c] for c in PROFIT_COLS]
    colors = [COLOR_MAP[c] for c in PROFIT_COLS]

    fig, ax = plt.subplots(figsize=(10, 7))

    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "white", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1}, capprops={"linewidth": 1}, flierprops={"markersize": 3})

    for i, (patch, col, color) in enumerate(zip(bp["boxes"], PROFIT_COLS, colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        if df[col].std() == 0:
            ax.hlines(df[col].iloc[0] / 1_000_000, i + 0.7, i + 1.3, colors=color, linewidths=2.5, zorder=5)

    ax.set_xticks(range(1, len(PROFIT_COLS) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Profit ($ millions)")
    ax.set_xlabel("Strategy")
    ax.set_title("Profit Distribution by Strategy Across PJM Nodes", fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)

    output_path = Path(__file__).parent / "outputs" / "deterministic_section" / "strategy_boxplot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_ranked_profit_bars()
    plot_strategy_boxplot()
