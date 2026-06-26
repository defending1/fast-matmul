# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
# ]
# ///

import os
import sys
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


def plot_csv(csv_path: str, output_path: str) -> None:
    """Generates a performance plot from a benchmark CSV file.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Path to save the output PDF plot.
    """
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
        plt.style.use(["ieee", "grid"])
    else:
        plt.style.use(["ieee", "no-latex", "grid"])
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
    df = df[df["size"] >= 8]

    # Plot each column except size (converting time to Effective GFLOPS)
    for col in df.columns:
        if col == "size":
            continue

        style = styles.get(col, {"marker": "x", "linestyle": ":"})
        flops = 2 * (df["size"] ** 3) - (df["size"] ** 2)
        gflops = flops / (df[col] * 1e9)
        ax.plot(
            df["size"],
            gflops,
            label=format_label(col),
            linewidth=1.2,
            markersize=4.5,
            **style,
        )

    # Configure axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("linear")
    ax.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)
    ax.set_ylabel(r"Effective GFLOPS", labelpad=10)
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
    print(f"Plot successfully saved to: {output_path}")
    plt.close(fig)


def generate_grid_plot(
    csv_path_faer: str, csv_path_dgemm: str, output_path: str
) -> None:
    """Generates a 2x2 grid performance plot.

    Row 1: Faer base results
    Row 2: MKL base results
    Col 1: Sequential algorithms
    Col 2: Parallel algorithms

    Args:
        csv_path_faer: Path to the Faer benchmark CSV results.
        csv_path_dgemm: Path to the MKL/Dgemm benchmark CSV results.
        output_path: Path to save the final PDF grid plot.
    """
    if not os.path.exists(csv_path_faer):
        print(f"Error: Faer CSV file not found at {csv_path_faer}")
        return
    if not os.path.exists(csv_path_dgemm):
        print(f"Error: Dgemm CSV file not found at {csv_path_dgemm}")
        return

    # Load datasets
    df_faer = pd.read_csv(csv_path_faer)
    df_dgemm = pd.read_csv(csv_path_dgemm)

    # Filter out very small sizes to focus on regions with algorithm differences
    df_faer = df_faer[df_faer["size"] >= 8]
    df_dgemm = df_dgemm[df_dgemm["size"] >= 8]

    # Set up styling using scienceplots based on LaTeX availability
    import shutil

    latex_installed = (
        shutil.which("latex") is not None or shutil.which("pdflatex") is not None
    )
    if latex_installed:
        plt.style.use(["ieee", "grid"])
    else:
        plt.style.use(["ieee", "no-latex", "grid"])

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 11), sharex=True, dpi=300)

    # Styles mapping (line styles, markers, and colors)
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

    # Helpers to filter columns
    def is_seq(col: str) -> bool:
        return col.endswith("_seq") or col.endswith("_single")

    def is_par(col: str) -> bool:
        return (
            col.endswith("_par")
            or col.endswith("_dfs")
            or col.endswith("_bfs")
            or col.endswith("_hybrid")
        )

    # Helper to plot on a specific axis
    def plot_on_ax(ax, df, filter_fn, title):
        ax.set_xscale("log", base=2)
        ax.set_yscale("linear")
        ax.set_title(title, pad=10)

        has_lines = False
        for col in df.columns:
            if col == "size" or not filter_fn(col):
                continue

            # Skip columns if they are entirely empty or NaN
            if df[col].isna().all():
                continue

            style = styles.get(col, {"marker": "x", "linestyle": ":"})
            flops = 2 * (df["size"] ** 3) - (df["size"] ** 2)
            gflops = flops / (df[col] * 1e9)
            ax.plot(
                df["size"],
                gflops,
                label=format_label(col),
                linewidth=1.2,
                markersize=4.5,
                **style,
            )
            has_lines = True

        ax.set_xticks(df["size"])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        if has_lines:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Plot Row 1: Faer Base
    plot_on_ax(axs[0, 0], df_faer, is_seq, "Faer Base - Sequential Algorithms")
    plot_on_ax(axs[0, 1], df_faer, is_par, "Faer Base - Parallel Algorithms")

    # Plot Row 2: MKL Base
    plot_on_ax(axs[1, 0], df_dgemm, is_seq, "MKL Base - Sequential Algorithms")
    plot_on_ax(axs[1, 1], df_dgemm, is_par, "MKL Base - Parallel Algorithms")

    # Shared labels
    for ax in axs[1, :]:
        ax.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=8)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Effective GFLOPS", labelpad=8)

    plt.suptitle(
        "Matrix Multiplication Performance Grid Comparison", fontsize=14, y=0.98
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Grid plot successfully saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    """Main execution block to check arguments and plot results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    path_faer = os.path.join(
        project_root, "rust", "generated", "benchmark_results_faer.csv"
    )
    path_dgemm = os.path.join(
        project_root, "rust", "generated", "benchmark_results_dgemm.csv"
    )
    output_grid = os.path.join(project_root, "rust", "generated", "benchmark_plot.pdf")

    # Check if we should plot the 2x2 grid
    if os.path.exists(path_faer) and os.path.exists(path_dgemm):
        print(
            f"Both Faer and MKL results exist. Plotting 2x2 grid comparison -> {output_grid}..."
        )
        generate_grid_plot(path_faer, path_dgemm, output_grid)
    else:
        # Fallback to plotting individual files if they exist
        print(
            "One of the benchmark CSV files is missing. Falling back to individual plots..."
        )
        configs = []
        if len(sys.argv) > 1:
            csv_path = os.path.abspath(sys.argv[1])
            base_name = os.path.basename(csv_path)
            if "results_faer" in base_name:
                out_name = "benchmark_plot_faer.pdf"
            elif "results_dgemm" in base_name:
                out_name = "benchmark_plot_dgemm.pdf"
            else:
                out_name = "benchmark_plot.pdf"
            output_path = os.path.join(os.path.dirname(csv_path), out_name)
            configs.append((csv_path, output_path))
        else:
            path_legacy = os.path.join(
                project_root, "rust", "generated", "benchmark_results.csv"
            )
            if os.path.exists(path_faer):
                out_faer = os.path.join(
                    project_root, "rust", "generated", "benchmark_plot_faer.pdf"
                )
                configs.append((path_faer, out_faer))
            if os.path.exists(path_dgemm):
                out_dgemm = os.path.join(
                    project_root, "rust", "generated", "benchmark_plot_dgemm.pdf"
                )
                configs.append((path_dgemm, out_dgemm))
            if not configs and os.path.exists(path_legacy):
                out_legacy = os.path.join(
                    project_root, "rust", "generated", "benchmark_plot.pdf"
                )
                configs.append((path_legacy, out_legacy))

        if not configs:
            print("Error: No CSV files found to plot.")
            return

        for csv_path, output_path in configs:
            print(f"Plotting {csv_path} -> {output_path}...")
            plot_csv(csv_path, output_path)


if __name__ == "__main__":
    main()
