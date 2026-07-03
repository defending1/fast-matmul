# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
# ]
# ///

import os
import re
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
        "strassen_seq": "Strassen (Sequential)",
        "strassen_dfs": "Strassen (DFS)",
        "strassen_bfs": "Strassen (BFS)",
        "strassen_hybrid": "Strassen (Hybrid)",
        "strassen_par": "Strassen (Parallel)",
        "grey_strassen_single": "Grey-Strassen (Single-threaded)",
        "grey_strassen_seq": "Grey-Strassen (Sequential)",
        "grey_strassen_dfs": "Grey-Strassen (DFS)",
        "grey_strassen_bfs": "Grey-Strassen (BFS)",
        "grey_strassen_hybrid": "Grey-Strassen (Hybrid)",
        "grey_strassen_par": "Grey-Strassen (Parallel)",
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
        if suffix_str in ("single", "seq"):
            suffix_str = "Sequential"
        elif suffix_str in ("par", "parallel", "hybrid"):
            suffix_str = "Parallel"
        elif suffix_str in ("dfs", "bfs"):
            suffix_str = suffix_str.upper()
        else:
            suffix_str = suffix_str.title()
        return f"{name} {rank}/{mults} ({suffix_str})"

    return col.replace("_", " ").title()


BALLARD_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "generated", "benchmarks.txt"
)

# Style for Ballard reference lines
BALLARD_STYLES = {
    "mkl_ballard": {
        "color": "#e41a1c",
        "marker": "X",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "strassen_ballard_1": {
        "color": "#4daf4a",
        "marker": "X",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    "strassen_ballard_2": {
        "color": "#377eb8",
        "marker": "s",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    "strassen_ballard_3": {
        "color": "#984ea3",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 1.5,
    },
}


def parse_matlab_vector(filepath: str, vec_name: str) -> pd.DataFrame:
    """Parse a MATLAB-style vector definition from a text file.

    Expected format:  NAME = [ P Q R steps time_ms ; ... ; ];

    Args:
        filepath: Path to the data file.
        vec_name: Name of the vector (e.g. "MKL_0").

    Returns:
        DataFrame with integer size and time_ms columns, sorted by size.
    """
    with open(filepath) as f:
        text = f.read()

    pattern = re.compile(rf"{re.escape(vec_name)}\s*=\s*\[(.*?)\];", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Vector '{vec_name}' not found in {filepath}")

    rows = []
    for entry in match.group(1).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split()
        if len(parts) >= 5:
            p = float(parts[0])
            time_ms = float(parts[4])
            rows.append({"size": p, "time_ms": time_ms})

    df = pd.DataFrame(rows)
    return df.sort_values("size").reset_index(drop=True)


def plot_ballard_lines(
    ax: plt.Axes, df_ballard_mkl: pd.DataFrame, *df_ballard_strassen: pd.DataFrame
) -> None:
    """Plot Ballard reference lines on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        df_ballard_mkl: MKL Ballard data with size and time_ms columns.
        df_ballard_strassen: One or more Strassen Ballard dataframes
                           (expected order: STRASSEN_1, STRASSEN_2, STRASSEN_3).
    """
    strassen_labels = {
        "strassen_ballard_1": "Strassen 1 (Ballard)",
        "strassen_ballard_2": "Strassen 2 (Ballard)",
        "strassen_ballard_3": "Strassen 3 (Ballard)",
    }
    entries = [("mkl_ballard", df_ballard_mkl)]
    for i, df in enumerate(df_ballard_strassen):
        entries.append((f"strassen_ballard_{i + 1}", df))

    for name, df in entries:
        style = BALLARD_STYLES[name]
        time_s = df["time_ms"] / 1000.0
        n = df["size"]
        gflops = (2 * n**3 - 2 * n**2) / (time_s * 1e9)
        label = "MKL (Ballard)" if name == "mkl_ballard" else strassen_labels[name]
        ax.plot(n, gflops, label=label, markersize=5.0, **style)


def is_parallel(col: str) -> bool:
    """Check if a benchmark column corresponds to a parallel algorithm.

    Args:
        col: The column name from the benchmark CSV.

    Returns:
        True if the column is a parallel execution case, False otherwise.
    """
    return (
        col.endswith("_par")
        or col.endswith("_dfs")
        or col.endswith("_bfs")
        or col.endswith("_hybrid")
    )


def plot_csv(csv_path: str, output_path: str) -> None:
    """Generates performance plots from a benchmark CSV file.

    If the CSV contains the new config columns, it splits and generates plots for all configs.
    Otherwise, it falls back to generating a single plot.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Default fallback path to save the output plot.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    new_format_cols = {"base_choice", "recursion_level", "size_cutoff"}
    if new_format_cols.issubset(df.columns):
        out_dir = os.path.dirname(output_path)
        # Drop duplicates to find unique recursion configs (keeping NaNs intact)
        df_configs = df[["recursion_level", "size_cutoff"]].drop_duplicates()
        for _, row in df_configs.iterrows():
            r_level = row["recursion_level"]
            s_cutoff = row["size_cutoff"]

            # Safe comparison representing NaN or the numeric level/cutoff
            if pd.isna(r_level):
                cond_r = df["recursion_level"].isna()
                r_str = ""
            else:
                cond_r = df["recursion_level"] == r_level
                r_str = f"level_{int(r_level)}"

            if pd.isna(s_cutoff):
                cond_c = df["size_cutoff"].isna()
                c_str = ""
            else:
                cond_c = df["size_cutoff"] == s_cutoff
                c_str = f"cutoff_{int(s_cutoff)}"

            df_config = df[cond_r & cond_c]
            prefix = r_str if r_str else c_str

            for base in ["faer", "dgemm"]:
                df_base = df_config[df_config["base_choice"] == base]
                if df_base.empty:
                    continue

                plot_columns = [col for col in df_base.columns if col not in new_format_cols]
                df_to_plot = df_base[plot_columns]

                suffix = "faer" if base == "faer" else "dgemm"
                out_name = f"benchmark_{prefix}_{suffix}.pdf"
                out_path = os.path.join(out_dir, out_name)

                plot_df_core(df_to_plot, out_path)

            # Plot grid comparison for this config if both base variants exist
            df_faer = df_config[df_config["base_choice"] == "faer"]
            df_dgemm = df_config[df_config["base_choice"] == "dgemm"]

            if not df_faer.empty and not df_dgemm.empty:
                plot_columns = [col for col in df.columns if col not in new_format_cols]
                df_faer_to_plot = df_faer[plot_columns]
                df_dgemm_to_plot = df_dgemm[plot_columns]

                out_grid_name = f"benchmark_{prefix}_grid.pdf"
                out_grid_path = os.path.join(out_dir, out_grid_name)

                generate_grid_plot_core(df_faer_to_plot, df_dgemm_to_plot, out_grid_path)
    else:
        plot_df_core(df, output_path)


def plot_df_core(df: pd.DataFrame, output_path: str) -> None:
    """Core logic to generate a performance plot from a dataframe in both PDF and PNG formats."""
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
        "strassen_seq": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "strassen_dfs": {"color": "#d95f02", "marker": "^", "linestyle": "-."},
        "strassen_bfs": {"color": "#fdbb84", "marker": "v", "linestyle": ":"},
        "strassen_hybrid": {"color": "#e34a33", "marker": "D", "linestyle": "-"},
        "strassen_par": {"color": "#e34a33", "marker": "D", "linestyle": "-"},
        # Grey-Strassen
        "grey_strassen_single": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
        "grey_strassen_seq": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
        "grey_strassen_dfs": {"color": "#a6761d", "marker": "^", "linestyle": "-."},
        "grey_strassen_bfs": {"color": "#dfc27d", "marker": "v", "linestyle": ":"},
        "grey_strassen_hybrid": {"color": "#543005", "marker": "H", "linestyle": "-"},
        "grey_strassen_par": {"color": "#543005", "marker": "H", "linestyle": "-"},
        # HK323_15_94
        "hk323_15_94_single": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
        "hk323_15_94_seq": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
        "hk323_15_94_dfs": {"color": "#1b9e77", "marker": "<", "linestyle": "-."},
        "hk323_15_94_bfs": {"color": "#a1d99b", "marker": ">", "linestyle": ":"},
        "hk323_15_94_hybrid": {"color": "#006d2c", "marker": "v", "linestyle": "-"},
        "hk323_15_94_par": {"color": "#006d2c", "marker": "v", "linestyle": "-"},
        # Smirnov333_23_139
        "smirnov333_23_139_single": {
            "color": "#d62728",
            "marker": "<",
            "linestyle": "--",
        },
        "smirnov333_23_139_seq": {
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
        "smirnov333_23_139_par": {
            "color": "#980043",
            "marker": ">",
            "linestyle": "-",
        },
        "mkl_ballard": BALLARD_STYLES["mkl_ballard"],
        "strassen_ballard_1": BALLARD_STYLES["strassen_ballard_1"],
        "strassen_ballard_2": BALLARD_STYLES["strassen_ballard_2"],
        "strassen_ballard_3": BALLARD_STYLES["strassen_ballard_3"],
    }

    # Parse Ballard reference data
    try:
        df_mkl_ballard = parse_matlab_vector(BALLARD_DATA_PATH, "MKL_0")
        df_strassen_ballard_1 = parse_matlab_vector(BALLARD_DATA_PATH, "STRASSEN_1")
        df_strassen_ballard_2 = parse_matlab_vector(BALLARD_DATA_PATH, "STRASSEN_2")
        df_strassen_ballard_3 = parse_matlab_vector(BALLARD_DATA_PATH, "STRASSEN_3")
        has_ballard = True
    except Exception as e:
        print(f"Warning: Could not parse Ballard data ({e})")
        has_ballard = False

    # Plot each column except size (converting time to Effective GFLOPS)
    num_cores = os.cpu_count() or 1
    for col in df.columns:
        if col == "size":
            continue

        if df[col].isna().all():
            continue

        style = styles.get(col, {"marker": "x", "linestyle": ":"})
        flops = 2 * (df["size"] ** 3) - (df["size"] ** 2)
        gflops = flops / (df[col] * 1e9)
        if is_parallel(col):
            gflops = gflops / num_cores
        ax.plot(
            df["size"],
            gflops,
            label=format_label(col),
            linewidth=1.2,
            markersize=4.5,
            **style,
        )

    # Plot Ballard reference lines (only if sequential algorithms are present)
    if has_ballard:
        has_seq_col = any(
            col != "size" and (col.endswith("_seq") or col.endswith("_single"))
            for col in df.columns
        )
        if has_seq_col:
            plot_ballard_lines(
                ax, df_mkl_ballard, df_strassen_ballard_1, df_strassen_ballard_2, df_strassen_ballard_3
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

    # Save in both PDF and PNG formats
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".pdf":
        alt_path = base + ".png"
    elif ext.lower() == ".png":
        alt_path = base + ".pdf"
    else:
        alt_path = output_path + ".png"

    plt.savefig(alt_path, bbox_inches="tight")
    print(f"Plot successfully saved to: {alt_path}")
    plt.close(fig)


def generate_grid_plot_core(
    df_faer: pd.DataFrame, df_dgemm: pd.DataFrame, output_path: str
) -> None:
    """Generates a 2x2 grid performance plot from dataframes directly."""
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
        "strassen_seq": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "strassen_dfs": {"color": "#d95f02", "marker": "^", "linestyle": "-."},
        "strassen_bfs": {"color": "#fdbb84", "marker": "v", "linestyle": ":"},
        "strassen_hybrid": {"color": "#e34a33", "marker": "D", "linestyle": "-"},
        "strassen_par": {"color": "#e34a33", "marker": "D", "linestyle": "-"},
        # Grey-Strassen
        "grey_strassen_single": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
        "grey_strassen_seq": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
        "grey_strassen_dfs": {"color": "#a6761d", "marker": "^", "linestyle": "-."},
        "grey_strassen_bfs": {"color": "#dfc27d", "marker": "v", "linestyle": ":"},
        "grey_strassen_hybrid": {"color": "#543005", "marker": "H", "linestyle": "-"},
        "grey_strassen_par": {"color": "#543005", "marker": "H", "linestyle": "-"},
        # HK323_15_94
        "hk323_15_94_single": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
        "hk323_15_94_seq": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
        "hk323_15_94_dfs": {"color": "#1b9e77", "marker": "<", "linestyle": "-."},
        "hk323_15_94_bfs": {"color": "#a1d99b", "marker": ">", "linestyle": ":"},
        "hk323_15_94_hybrid": {"color": "#006d2c", "marker": "v", "linestyle": "-"},
        "hk323_15_94_par": {"color": "#006d2c", "marker": "v", "linestyle": "-"},
        # Smirnov333_23_139
        "smirnov333_23_139_single": {
            "color": "#d62728",
            "marker": "<",
            "linestyle": "--",
        },
        "smirnov333_23_139_seq": {
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
        "smirnov333_23_139_par": {
            "color": "#980043",
            "marker": ">",
            "linestyle": "-",
        },
        "mkl_ballard": BALLARD_STYLES["mkl_ballard"],
        "strassen_ballard_1": BALLARD_STYLES["strassen_ballard_1"],
        "strassen_ballard_2": BALLARD_STYLES["strassen_ballard_2"],
        "strassen_ballard_3": BALLARD_STYLES["strassen_ballard_3"],
    }

    # Parse Ballard reference data
    try:
        df_mkl_ballard = parse_matlab_vector(BALLARD_DATA_PATH, "MKL_0")
        df_strassen_ballard_1 = parse_matlab_vector(BALLARD_DATA_PATH, "STRASSEN_1")
        df_strassen_ballard_2 = parse_matlab_vector(BALLARD_DATA_PATH, "STRASSEN_2")
        df_strassen_ballard_3 = parse_matlab_vector(BALLARD_DATA_PATH, "STRASSEN_3")
        has_ballard = True
    except Exception as e:
        print(f"Warning: Could not parse Ballard data ({e})")
        has_ballard = False

    # Helpers to filter columns
    def is_seq(col: str) -> bool:
        return col.endswith("_seq") or col.endswith("_single")

    def is_par(col: str) -> bool:
        return is_parallel(col)

    # Helper to plot on a specific axis
    num_cores = os.cpu_count() or 1

    def plot_on_ax(ax, df, filter_fn, sequential=False):
        ax.set_facecolor("none")
        ax.set_xscale("log", base=2)
        ax.set_yscale("linear")

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
            if is_par(col):
                gflops = gflops / num_cores
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

        if has_ballard and sequential:
            plot_ballard_lines(
                ax, df_mkl_ballard, df_strassen_ballard_1, df_strassen_ballard_2, df_strassen_ballard_3
            )

        if has_lines:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Plot Row 1: Faer Base
    plot_on_ax(axs[0, 0], df_faer, is_seq, sequential=True)
    plot_on_ax(axs[0, 1], df_faer, is_par, sequential=False)

    # Plot Row 2: MKL Base
    plot_on_ax(axs[1, 0], df_dgemm, is_seq, sequential=True)
    plot_on_ax(axs[1, 1], df_dgemm, is_par, sequential=False)

    # Shared labels
    for ax in axs[1, :]:
        ax.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=8)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Effective GFLOPS", labelpad=8)

    plt.suptitle(
        "Matrix Multiplication Performance Grid Comparison", fontsize=14, y=0.98
    )
    plt.tight_layout()
    # Add spacing between subplots (less space between rows) and leave room at top/bottom/sides
    fig.subplots_adjust(
        hspace=0.12, wspace=0.32, top=0.88, bottom=0.08, left=0.08, right=0.92
    )

    # Force a draw to resolve final positions of all labels and layout bounds
    fig.canvas.draw()

    # Helper to get tight bounding box in figure fraction coordinates
    def get_tight_pos(ax):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        return bbox.transformed(fig.transFigure.inverted())

    # Get tight positions (which include axis ticks, xlabel, and ylabel)
    box00 = get_tight_pos(axs[0, 0])
    box10 = get_tight_pos(axs[1, 0])
    box01 = get_tight_pos(axs[0, 1])
    box11 = get_tight_pos(axs[1, 1])

    # Column 0: Sequential Algorithms bounding box
    x0_seq = min(box00.x0, box10.x0)
    x1_seq = max(box00.x1, box10.x1)
    y0_seq = min(box00.y0, box10.y0)
    y1_seq = max(box00.y1, box10.y1)

    # Column 1: Parallel Algorithms bounding box
    x0_par = min(box01.x0, box11.x0)
    x1_par = max(box01.x1, box11.x1)
    y0_par = min(box01.y0, box11.y0)
    y1_par = max(box01.y1, box11.y1)

    # Draw borders for the two columns in different light colors (border only, no fill)
    from matplotlib.patches import FancyBboxPatch

    pad_val = 0.015

    rect_seq = FancyBboxPatch(
        (x0_seq - pad_val, y0_seq - pad_val),
        (x1_seq - x0_seq) + 2 * pad_val,
        (y1_seq - y0_seq) + 2 * pad_val,
        boxstyle="square,pad=0.0",
        edgecolor="#64748b",  # Light slate border
        facecolor="none",  # No inner background fill
        linewidth=1.5,
        transform=fig.transFigure,
        zorder=1,
    )
    fig.patches.append(rect_seq)

    rect_par = FancyBboxPatch(
        (x0_par - pad_val, y0_par - pad_val),
        (x1_par - x0_par) + 2 * pad_val,
        (y1_par - y0_par) + 2 * pad_val,
        boxstyle="square,pad=0.0",
        edgecolor="#a78bfa",  # Light purple border
        facecolor="none",  # No inner background fill
        linewidth=1.5,
        transform=fig.transFigure,
        zorder=1,
    )
    fig.patches.append(rect_par)

    # Sequential and parallel algorithms on the top, centered above the column bounding border
    y_col_title = max(y1_seq, y1_par) + pad_val + 0.01
    center_x_seq = (x0_seq + x1_seq) / 2.0
    center_x_par = (x0_par + x1_par) / 2.0

    fig.text(
        center_x_seq,
        y_col_title,
        "Sequential Algorithms",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#1e293b",
    )
    fig.text(
        center_x_par,
        y_col_title,
        "Parallel Algorithms",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#1e293b",
    )

    # Centered titles for each row, positioned in the horizontal gap between column borders,
    # and vertically centered on each row's plotting axes.
    pos00 = axs[0, 0].get_position()
    pos10 = axs[1, 0].get_position()

    center_x_row = ((x1_seq + pad_val) + (x0_par - pad_val)) / 2.0

    y_faer_center = (pos00.y0 + pos00.y1) / 2.0
    fig.text(
        center_x_row,
        y_faer_center,
        "Faer Base",
        ha="center",
        va="center",
        fontsize=12,
        color="#1e293b",
        zorder=10,
    )

    y_mkl_center = (pos10.y0 + pos10.y1) / 2.0
    fig.text(
        center_x_row,
        y_mkl_center,
        "MKL Base",
        ha="center",
        va="center",
        fontsize=12,
        color="#1e293b",
        zorder=10,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Grid plot successfully saved to: {output_path}")

    # Save in both PDF and PNG formats
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".pdf":
        alt_path = base + ".png"
    elif ext.lower() == ".png":
        alt_path = base + ".pdf"
    else:
        alt_path = output_path + ".png"

    plt.savefig(alt_path, bbox_inches="tight")
    print(f"Grid plot successfully saved to: {alt_path}")
    plt.close(fig)


def main() -> None:
    """Main execution block to check arguments and plot results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Determine input path
    if len(sys.argv) > 1:
        csv_path = os.path.abspath(sys.argv[1])
    else:
        csv_path = os.path.join(
            project_root, "rust", "generated", "csv", "benchmark_results.csv"
        )

    if os.path.exists(csv_path):
        # Determine fallback output path name based on the CSV path
        base_name = os.path.basename(csv_path)
        if "results" in base_name:
            out_name = base_name.replace("results", "plot").replace(".csv", ".pdf")
        else:
            out_name = os.path.splitext(base_name)[0] + ".pdf"

        # Reorganize: if csv_path is in generated/csv/, output to generated/plots/
        csv_dir = os.path.abspath(os.path.dirname(csv_path))
        if os.path.basename(csv_dir) == "csv" and os.path.basename(os.path.dirname(csv_dir)) == "generated":
            output_path = os.path.join(os.path.dirname(csv_dir), "plots", out_name)
        else:
            output_path = os.path.join(csv_dir, out_name)

        print(f"Plotting {csv_path} -> {output_path}...")
        plot_csv(csv_path, output_path)
    else:
        print(f"Error: CSV file not found at {csv_path}")


if __name__ == "__main__":
    main()
