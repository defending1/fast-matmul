# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
# ]
# ///

import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


def format_label(col: str) -> str:
    """Format column names into clean, publication-ready LaTeX labels.

    Args:
        col: The column name from the benchmark CSV.

    Returns:
        A formatted string suitable for LaTeX legend display.
    """
    label_mappings = {
        "mkl_seq": "MKL (Sequential)",
        "mkl_par": "MKL (Parallel)",
        "faer_seq": "faer (Sequential)",
        "faer_par": "faer (Parallel)",
        "strassen_single": "Strassen (Single-threaded)",
        "strassen_dfs": "Strassen (DFS)",
        "strassen_bfs": "Strassen (BFS)",
        "strassen_hybrid": "Strassen (Hybrid)",
        "grey_strassen_single": "Grey-Strassen (Single-threaded)",
        "grey_strassen_dfs": "Grey-Strassen (DFS)",
        "grey_strassen_bfs": "Grey-Strassen (BFS)",
        "grey_strassen_hybrid": "Grey-Strassen (Hybrid)",
    }
    if col in label_mappings:
        return label_mappings[col]

    # Dynamic parsing fallback for algorithms like hk323_15_94 or smirnov333_23_139
    parts = col.split("_")
    if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
        name = (
            parts[0].upper() if parts[0].lower().startswith("hk") else parts[0].title()
        )
        rank = parts[1]
        mults = parts[2]
        suffix = parts[3:]
        suffix_str = " ".join(suffix).lower()
        if suffix_str == "single":
            suffix_str = "Single-threaded"
        elif suffix_str in ("dfs", "bfs", "hybrid"):
            suffix_str = suffix_str.upper()
        else:
            suffix_str = suffix_str.title()
        return f"{name} {rank}/{mults} ({suffix_str})"

    return col.replace("_", " ").title()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(project_root, "rust", "generated", "benchmark_results.csv")
    output_path = os.path.join(project_root, "rust", "generated", "benchmark_plot.pdf")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Set up styling using scienceplots based on LaTeX availability
    import shutil

    latex_installed = (
        shutil.which("latex") is not None or shutil.which("pdflatex") is not None
    )
    if latex_installed:
        plt.style.use(["science", "grid"])
    else:
        plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

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
            label=format_label(col),
            linewidth=1.2,
            markersize=4.5,
            **style,
        )

    # Configure axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)
    ax.set_ylabel(r"Execution Time (ms)", labelpad=10)
    ax.set_title("Matrix Multiplication Performance Comparison", pad=15)

    # Set x-ticks explicitly to size values
    ax.set_xticks(df["size"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Legend placement and formatting
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot successfully saved to: {pdf_output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
