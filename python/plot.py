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
        "strassen_single": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "strassen_multithread": {"color": "#ff7f0e", "marker": "D", "linestyle": "-"},
        "grey_strassen_single": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
        "grey_strassen_multithread": {"color": "#8c564b", "marker": "H", "linestyle": "-"},
        "hk323_15_94_single": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
        "hk323_15_94_multithread": {
            "color": "#2ca02c",
            "marker": "v",
            "linestyle": "-",
        },
        "smirnov333_23_139_single": {
            "color": "#d62728",
            "marker": "<",
            "linestyle": "--",
        },
        "smirnov333_23_139_multithread": {
            "color": "#d62728",
            "marker": ">",
            "linestyle": "-",
        },
    }

    # Plot each column except size
    for col in df.columns:
        if col == "size":
            continue

        style = styles.get(col, {"marker": "x", "linestyle": ":"})
        ax.plot(
            df["size"],
            df[col],
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
        "Execution Time (seconds)", fontsize=12, fontweight="bold", labelpad=10
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
