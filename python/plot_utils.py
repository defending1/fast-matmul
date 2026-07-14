"""Common utilities and styling definitions for benchmark plots.

This module provides shared styling configurations, MATLAB vector parsing,
Ballard data loading, GFLOPS computation functions, LaTeX styling configuration,
and plot saving utilities used across the fast-matmul visualization scripts.
"""

import os
import re
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


def detect_latex() -> bool:
    """Check if latex/pdflatex command line tools are available.

    Returns:
        True if latex is installed, False otherwise.
    """
    return (
        shutil.which("latex") is not None or shutil.which("pdflatex") is not None
    )


def setup_matplotlib_style(use_latex: bool = None) -> bool:
    """Setup Matplotlib style using scienceplots.

    Args:
        use_latex: Force enable/disable LaTeX. If None, auto-detects.

    Returns:
        A boolean indicating whether LaTeX usage was successfully enabled.
    """
    latex_active = detect_latex() if use_latex is None else use_latex

    if latex_active:
        plt.style.use(["ieee", "grid"])
    else:
        plt.style.use(["ieee", "no-latex", "grid"])

    # Base RC parameter updates for premium font typography and size
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
            "text.usetex": latex_active,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    return latex_active


# Ballard baseline styles dictionary
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

# Main algorithm line styles, markers, and colors
ALGORITHM_STYLES = {
    "system": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "mkl_seq": {"color": "#9467bd", "marker": "p", "linestyle": "--"},
    "mkl_par": {"color": "#9467bd", "marker": "p", "linestyle": "--"},
    "faer_seq": {"color": "#17becf", "marker": "d", "linestyle": "-"},
    "faer_par": {"color": "#17becf", "marker": "d", "linestyle": "-"},
    # Strassen
    "strassen_single": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "strassen_seq": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "strassen_dfs": {"color": "#d95f02", "marker": "s", "linestyle": "-."},
    "strassen_bfs": {"color": "#fdbb84", "marker": "s", "linestyle": ":"},
    "strassen_hybrid": {"color": "#e34a33", "marker": "s", "linestyle": "-"},
    "strassen_par": {"color": "#e34a33", "marker": "s", "linestyle": "-"},
    # Grey-Strassen
    "grey_strassen_single": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
    "grey_strassen_seq": {"color": "#8c564b", "marker": "h", "linestyle": "--"},
    "grey_strassen_dfs": {"color": "#a6761d", "marker": "h", "linestyle": "-."},
    "grey_strassen_bfs": {"color": "#dfc27d", "marker": "h", "linestyle": ":"},
    "grey_strassen_hybrid": {"color": "#543005", "marker": "h", "linestyle": "-"},
    "grey_strassen_par": {"color": "#543005", "marker": "h", "linestyle": "-"},
    # HK323_15_94
    "hk323_15_94_single": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
    "hk323_15_94_seq": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
    "hk323_15_94_dfs": {"color": "#1b9e77", "marker": "^", "linestyle": "-."},
    "hk323_15_94_bfs": {"color": "#a1d99b", "marker": "^", "linestyle": ":"},
    "hk323_15_94_hybrid": {"color": "#006d2c", "marker": "^", "linestyle": "-"},
    "hk323_15_94_par": {"color": "#006d2c", "marker": "^", "linestyle": "-"},
    # Smirnov333_23_139
    "smirnov333_23_139_single": {"color": "#d62728", "marker": "v", "linestyle": "--"},
    "smirnov333_23_139_seq": {"color": "#d62728", "marker": "v", "linestyle": "--"},
    "smirnov333_23_139_dfs": {"color": "#e7298a", "marker": "v", "linestyle": "-."},
    "smirnov333_23_139_bfs": {"color": "#fbb4ae", "marker": "v", "linestyle": ":"},
    "smirnov333_23_139_hybrid": {"color": "#980043", "marker": "v", "linestyle": "-"},
    "smirnov333_23_139_par": {"color": "#980043", "marker": "v", "linestyle": "-"},
    # Reference baselines
    "mkl_ballard": BALLARD_STYLES["mkl_ballard"],
    "strassen_ballard_1": BALLARD_STYLES["strassen_ballard_1"],
    "strassen_ballard_2": BALLARD_STYLES["strassen_ballard_2"],
    "strassen_ballard_3": BALLARD_STYLES["strassen_ballard_3"],
}

# Unified grid plot color styling map
RUST_GRID_COLORS = {
    "seq_dgemm": "#ff7f0e",
    "seq_faer": "#e34a33",
    "par_dfs_dgemm": "#ff7f0e",
    "par_bfs_dgemm": "#8c564b",
    "par_hybrid_dgemm": "#2ca02c",
    "par_dfs_faer": "#e34a33",
    "par_bfs_faer": "#02818a",
    "par_hybrid_faer": "#bcbd22",
}

BALLARD_GRID_COLORS = {
    "seq": "#e41a1c",
    "dfs": "#4daf4a",
    "bfs": "#377eb8",
    "hybrid": "#984ea3",
}


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


def parse_matlab_vector(filepath: str, vec_name: str) -> pd.DataFrame | None:
    """Parse a MATLAB-style vector definition from a text file.

    Expected format:  NAME = [ P Q R steps time_ms ; ... ; ];

    Args:
        filepath: Path to the data file.
        vec_name: Name of the vector (e.g. "MKL_0").

    Returns:
        DataFrame with size and time_ms columns, sorted by size, or None if file not found.
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        text = f.read()

    pattern = re.compile(rf"{re.escape(vec_name)}\s*=\s*\[(.*?)\];", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None

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


def load_ballard_data(project_root: str, mode_suffix: str) -> dict | None:
    """Loads Ballard reference data for a specific mode suffix.

    Args:
        project_root: The root directory of the project.
        mode_suffix: The suffix indicating parallelization mode ('seq', 'dfs', etc.).

    Returns:
        A dictionary containing 'mkl' (DataFrame) and 'strassen' (list of DataFrames),
        or None if parsing fails.
    """
    path = os.path.join(
        project_root,
        "generated",
        "csv",
        f"run_{mode_suffix}" if mode_suffix in ("seq", "par") else "run_par",
        f"benchmarks_{mode_suffix}.txt",
    )
    if not os.path.exists(path):
        # Alternative search paths inside benchmarks/generated/
        path = os.path.join(
            project_root,
            "benchmarks",
            "generated",
            f"benchmarks_{mode_suffix}.txt",
        )
        if not os.path.exists(path):
            if mode_suffix == "seq":
                fallback = os.path.join(project_root, "benchmarks", "generated", "benchmarks.txt")
                if os.path.exists(fallback):
                    path = fallback
                else:
                    return None
            else:
                return None

    try:
        df_mkl = parse_matlab_vector(path, "MKL_0")
        df_strassen_1 = parse_matlab_vector(path, "STRASSEN_1")
        df_strassen_2 = parse_matlab_vector(path, "STRASSEN_2")
        df_strassen_3 = parse_matlab_vector(path, "STRASSEN_3")
        return {
            "mkl": df_mkl,
            "strassen": [df_strassen_1, df_strassen_2, df_strassen_3],
        }
    except Exception as e:
        print(f"Warning: Could not parse Ballard data for '{mode_suffix}' from {path} ({e})")
        return None


def calculate_gflops_rust(size: pd.Series, time_seconds: pd.Series, is_parallel: bool = False, num_cores: int = 1) -> pd.Series:
    """Compute Effective GFLOPS from execution time for Rust algorithms.

    Timings represent total execution time. Normalizes by CPU core count for parallel runs.

    Args:
        size: Matrix sizes (N).
        time_seconds: Timings in seconds.
        is_parallel: Normalizes the GFLOPS metric per core.
        num_cores: CPU core count.

    Returns:
        A pandas Series of computed GFLOPS.
    """
    flops = 2 * (size ** 3) - (size ** 2)
    gflops = flops / (time_seconds * 1e9)
    if is_parallel:
        gflops = gflops / float(num_cores)
    return gflops


def calculate_gflops_ballard(size: pd.Series, time_ms: pd.Series, is_parallel: bool = False, num_cores: int = 1) -> pd.Series:
    """Compute Effective GFLOPS from execution time for Ballard reference data.

    Timings are in milliseconds. Normalizes by CPU core count for parallel runs.

    Args:
        size: Matrix sizes (N).
        time_ms: Ballard timing in milliseconds.
        is_parallel: Normalizes the GFLOPS metric per core.
        num_cores: CPU core count.

    Returns:
        A pandas Series of computed GFLOPS.
    """
    time_s = time_ms / 1000.0
    flops = 2 * (size ** 3) - 2 * (size ** 2)
    gflops = flops / (time_s * 1e9)
    if is_parallel:
        gflops = gflops / float(num_cores)
    return gflops


def save_plot(fig: plt.Figure, output_path: str, dpi: int = 300) -> None:
    """Save the Matplotlib figure to the filesystem as PDF only.

    Args:
        fig: The Matplotlib Figure instance.
        output_path: Absolute or relative filepath (typically ending in .pdf).
        dpi: Dots per inch resolution.
    """
    base, ext = os.path.splitext(output_path)
    if ext.lower() != ".pdf":
        output_path = base + ".pdf"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"Plot saved successfully to: {output_path}")


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


def plot_ballard_lines(
    ax: plt.Axes,
    df_ballard_mkl: pd.DataFrame,
    strassen_dataframes: list,
    mode_label: str = "",
    num_cores: int = 1,
) -> None:
    """Plot Ballard reference lines on a given axis.

    Args:
        ax: Matplotlib axis to plot on.
        df_ballard_mkl: MKL Ballard data with size and time_ms columns.
        strassen_dataframes: List of (level, df) tuples for Strassen Ballard data.
        mode_label: Optional label suffix to append to the legend (e.g. 'DFS', 'BFS', 'Hybrid').
        num_cores: Number of CPU cores to divide by in the parallel case.
    """
    suffix = f" {mode_label}" if mode_label else ""
    entries = []

    # Check if MKL (Ballard) was already plotted on this axis to avoid duplicates
    mkl_already_plotted = False
    for line in ax.get_lines():
        lbl = line.get_label()
        if lbl and "MKL" in lbl and "Ballard" in lbl:
            mkl_already_plotted = True
            break

    if df_ballard_mkl is not None and not mkl_already_plotted:
        entries.append(("mkl_ballard", df_ballard_mkl, f"MKL{suffix} (Ballard)"))

    for level, df in strassen_dataframes:
        if df is not None:
            entries.append(
                (f"strassen_ballard_{level}", df, f"Strassen {level}{suffix} (Ballard)")
            )

    for name, df, label in entries:
        style = dict(BALLARD_STYLES.get(name, BALLARD_STYLES.get("strassen_ballard_1")))

        # Customize style based on parallel mode label to distinguish BFS, DFS, Hybrid
        if name != "mkl_ballard":
            if mode_label == "DFS":
                style["linestyle"] = "-."
                style["marker"] = "^"
            elif mode_label == "BFS":
                style["linestyle"] = ":"
                style["marker"] = "v"
            elif mode_label == "Hybrid":
                style["linestyle"] = "-"
                style["marker"] = "D"

        gflops = calculate_gflops_ballard(
            df["size"], df["time_ms"], is_parallel=(num_cores > 1), num_cores=num_cores
        )
        ax.plot(df["size"], gflops, label=label, markersize=5.0, **style)


