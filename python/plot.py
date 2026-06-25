# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
# ]
# ///

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(project_root, "rust", "generated", "benchmark_results.csv")
    output_path = os.path.join(project_root, "rust", "generated", "benchmark_plot.png")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Set up styling
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Line styles, markers, and colors
    styles = {
        "system": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "mkl_seq": {"color": "#9467bd", "marker": "p", "linestyle": "--"},
        "mkl_par": {"color": "#9467bd", "marker": "*", "linestyle": "-"},
        "faer_seq": {"color": "#17becf", "marker": "x", "linestyle": "--"},
        "faer_par": {"color": "#17becf", "marker": "d", "linestyle": "-"},
        # Strassen
        "strassen_single": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "strassen_dfs": {"color": "#d95f02", "marker": "^", "linestyle": "-."},
        "strassen_bfs": {"color": "#fdbb84", "marker": "v", "linestyle": ":"},
        "strassen_hybrid": {"color": "#e34a33", "marker": "D", "linestyle": "-"},
        # Grey-Strassen
        "grey_strassen_single": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
        "grey_strassen_dfs": {"color": "#a6761d", "marker": "^", "linestyle": "-."},
        "grey_strassen_bfs": {"color": "#dfc27d", "marker": "v", "linestyle": ":"},
        "grey_strassen_hybrid": {"color": "#543005", "marker": "H", "linestyle": "-"},
        # HK323_15_94
        "hk323_15_94_single": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
        "hk323_15_94_dfs": {"color": "#1b9e77", "marker": "<", "linestyle": "-."},
        "hk323_15_94_bfs": {"color": "#a1d99b", "marker": ">", "linestyle": ":"},
        "hk323_15_94_hybrid": {"color": "#006d2c", "marker": "v", "linestyle": "-"},
        # Smirnov333_23_139
        "smirnov333_23_139_single": {
            "color": "#d62728",
            "marker": "<",
            "linestyle": "--",
        },
        "smirnov333_23_139_dfs": {"color": "#e7298a", "marker": "d", "linestyle": "-."},
        "smirnov333_23_139_bfs": {"color": "#fbb4ae", "marker": "p", "linestyle": ":"},
        "smirnov333_23_139_hybrid": {
            "color": "#980043",
            "marker": ">",
            "linestyle": "-",
        },
    }

    # Filter out very small sizes to focus on regions with algorithm differences
    df = df[df["size"] >= 128]

    # Plot each column except size (converting time to milliseconds)
    for col in df.columns:
        if col == "size":
            continue

        style = styles.get(col, {"marker": "x", "linestyle": ":"})
        ax.plot(
            df["size"],
            df[col] * 1000.0,
            label=col.replace("_", " ").title(),
            linewidth=2,
            markersize=6,
            **style,
        )

    # Configure axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix Size (N x N)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(
        "Execution Time (milliseconds)", fontsize=12, fontweight="bold", labelpad=10
    )
    ax.set_title(
        "Matrix Multiplication Performance Comparison",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Set x-ticks explicitly to size values
    ax.set_xticks(df["size"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#e0e0e0",
        framealpha=0.9,
        fontsize=10,
    )

    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
