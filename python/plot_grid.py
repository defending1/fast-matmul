#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
#   "numpy",
# ]
# ///

"""Benchmark plotting script for matrix multiplication algorithms.
Generates 2-row grid plots (Row 1: Cutoffs, Row 2: Levels) comparing MKL,
faer, and Strassen variants in sequential or parallel modes.
Also generates a 4x3 Ballard comparison plot.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import shutil

# Set style
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
    }
)


def parse_matlab_vector(filepath, vec_name):
    """Parses matlab-style vectors of size/time data from reference files.

    Args:
        filepath: Absolute path to the text file containing matlab-like output.
        vec_name: The name of the vector/variable in the matlab file.

    Returns:
        A sorted pandas DataFrame with 'size' and 'time_ms' columns, or None if not found.
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


def plot_mode_grid(project_root, mode):
    """Generates and saves a 2-row performance comparison grid plot for the specified mode.
    Does not display Ballard reference lines. Enforces tight y-limits starting at 0.

    Args:
        project_root: The root directory of the project.
        mode: The plotting mode, either 'sequential' ('seq') or 'parallel' ('par').
    """
    is_seq = mode in ("sequential", "seq")
    csv_dir = "run_seq" if is_seq else "run_par"

    csv_path = os.path.join(
        project_root, "generated", "csv", csv_dir, "benchmark_results.csv"
    )
    base_csv_path = os.path.join(
        project_root, "generated", "csv", csv_dir, "benchmark_results_base.csv"
    )

    if not os.path.exists(csv_path) or not os.path.exists(base_csv_path):
        print(f"Error: Missing input CSVs at {csv_path} or {base_csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_base = pd.read_csv(base_csv_path)

    # Core count for parallel GFLOPS normalization
    num_cores = os.cpu_count() or 1
    norm_factor = 1.0 if is_seq else float(num_cores)
    if not is_seq:
        print(f"Normalizing parallel results using {num_cores} cores.")

    # Legend font size
    legend_fontsize = 9.5 if is_seq else 8.5
    plt.rcParams.update({"legend.fontsize": legend_fontsize})

    # Create figure
    fig, axs = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True, dpi=300)

    configs = [
        # (row, col, title, is_cutoff, value)
        (0, 0, "Cutoff = 256", True, 256),
        (0, 1, "Cutoff = 512", True, 512),
        (0, 2, "Cutoff = 1024", True, 1024),
        (0, 3, "Cutoff = 2048", True, 2048),
        (1, 0, "Level = 1", False, 1),
        (1, 1, "Level = 2", False, 2),
        (1, 2, "Level = 3", False, 3),
    ]

    legend_handles = {}
    max_gflops = 0.0

    for row, col, title, is_cutoff, value in configs:
        ax = axs[row, col]
        ax.set_facecolor("none")
        ax.set_xscale("log", base=2)
        ax.set_xlim(left=2)
        ax.set_yscale("linear")
        ax.set_title(title, pad=10)

        n_base = df_base["size"]
        mkl_flops = 2 * n_base**3 - n_base**2

        if is_seq:
            # 1. Plot MKL Base (Sequential)
            gflops_mkl = mkl_flops / (df_base["mkl_seq"] * 1e9)
            if not gflops_mkl.empty:
                max_gflops = max(max_gflops, float(gflops_mkl.max()))
            (l_mkl,) = ax.plot(
                n_base,
                gflops_mkl,
                label="MKL dgemm",
                color="#9467bd",
                marker="o",
                linestyle="--",
                linewidth=1.2,
                markersize=4.5,
            )
            legend_handles["MKL dgemm"] = l_mkl

            # 2. Plot faer Base (Sequential) - uses upward triangle marker
            gflops_faer = mkl_flops / (df_base["faer_seq"] * 1e9)
            if not gflops_faer.empty:
                max_gflops = max(max_gflops, float(gflops_faer.max()))
            (l_faer,) = ax.plot(
                n_base,
                gflops_faer,
                label="faer (Sequential)",
                color="#17becf",
                marker="^",
                linestyle="--",
                linewidth=1.2,
                markersize=4.5,
            )
            legend_handles["faer (Sequential)"] = l_faer
        else:
            # 1. Plot MKL Base (Parallel, normalized)
            if "mkl_par" in df_base.columns and not df_base["mkl_par"].isna().all():
                gflops_mkl = mkl_flops / (df_base["mkl_par"] * 1e9) / norm_factor
                if not gflops_mkl.empty:
                    max_gflops = max(max_gflops, float(gflops_mkl.max()))
                (l_mkl,) = ax.plot(
                    n_base,
                    gflops_mkl,
                    label="MKL dgemm",
                    color="#9467bd",
                    marker="o",
                    linestyle="--",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["MKL dgemm"] = l_mkl

            # 2. Plot faer Base (Parallel, normalized) - uses upward triangle marker
            if "faer_par" in df_base.columns and not df_base["faer_par"].isna().all():
                gflops_faer = mkl_flops / (df_base["faer_par"] * 1e9) / norm_factor
                if not gflops_faer.empty:
                    max_gflops = max(max_gflops, float(gflops_faer.max()))
                (l_faer,) = ax.plot(
                    n_base,
                    gflops_faer,
                    label="faer (Parallel)",
                    color="#17becf",
                    marker="^",
                    linestyle="--",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["faer (Parallel)"] = l_faer

        # 3. Filter Strassen results
        if is_cutoff:
            df_config = df[
                (df["size_cutoff"] == value) & (df["recursion_level"].isna())
            ]
        else:
            df_config = df[
                (df["recursion_level"] == value) & (df["size_cutoff"].isna())
            ]

        df_dgemm = df_config[df_config["base_choice"] == "dgemm"]
        df_faer_strassen = df_config[df_config["base_choice"] == "faer"]

        if is_seq:
            # Strassen dgemm base sequential - uses circle marker
            if not df_dgemm.empty:
                flops_dg = 2 * df_dgemm["size"] ** 3 - df_dgemm["size"] ** 2
                gflops_dg = flops_dg / (df_dgemm["strassen_seq"] * 1e9)
                if not gflops_dg.empty:
                    max_gflops = max(max_gflops, float(gflops_dg.max()))
                (l_strassen_dg,) = ax.plot(
                    df_dgemm["size"],
                    gflops_dg,
                    label="Strassen (dgemm base)",
                    color="#ff7f0e",
                    marker="o",
                    linestyle="-",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["Strassen (dgemm base)"] = l_strassen_dg

            # Strassen faer base sequential - uses upward triangle marker
            if not df_faer_strassen.empty:
                flops_fs = (
                    2 * df_faer_strassen["size"] ** 3 - df_faer_strassen["size"] ** 2
                )
                gflops_fs = flops_fs / (df_faer_strassen["strassen_seq"] * 1e9)
                if not gflops_fs.empty:
                    max_gflops = max(max_gflops, float(gflops_fs.max()))
                (l_strassen_fs,) = ax.plot(
                    df_faer_strassen["size"],
                    gflops_fs,
                    label="Strassen (faer base)",
                    color="#e34a33",
                    marker="^",
                    linestyle="-",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["Strassen (faer base)"] = l_strassen_fs
        else:
            # Plot Rust dgemm base Strassen curves (DFS, BFS, Hybrid) - use circle markers
            if not df_dgemm.empty:
                flops_dg = 2 * df_dgemm["size"] ** 3 - df_dgemm["size"] ** 2

                if (
                    "strassen_dfs" in df_dgemm.columns
                    and not df_dgemm["strassen_dfs"].isna().all()
                ):
                    gflops_dfs = (
                        flops_dg / (df_dgemm["strassen_dfs"] * 1e9) / norm_factor
                    )
                    if not gflops_dfs.empty:
                        max_gflops = max(max_gflops, float(gflops_dfs.max()))
                    (l_dfs_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_dfs,
                        label="Strassen DFS (dgemm base)",
                        color="#ff7f0e",
                        marker="o",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen DFS (dgemm base)"] = l_dfs_dg

                if (
                    "strassen_bfs" in df_dgemm.columns
                    and not df_dgemm["strassen_bfs"].isna().all()
                ):
                    gflops_bfs = (
                        flops_dg / (df_dgemm["strassen_bfs"] * 1e9) / norm_factor
                    )
                    if not gflops_bfs.empty:
                        max_gflops = max(max_gflops, float(gflops_bfs.max()))
                    (l_bfs_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_bfs,
                        label="Strassen BFS (dgemm base)",
                        color="#8c564b",
                        marker="o",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen BFS (dgemm base)"] = l_bfs_dg

                if (
                    "strassen_hybrid" in df_dgemm.columns
                    and not df_dgemm["strassen_hybrid"].isna().all()
                ):
                    gflops_hybrid = (
                        flops_dg / (df_dgemm["strassen_hybrid"] * 1e9) / norm_factor
                    )
                    if not gflops_hybrid.empty:
                        max_gflops = max(max_gflops, float(gflops_hybrid.max()))
                    (l_hybrid_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_hybrid,
                        label="Strassen Hybrid (dgemm base)",
                        color="#2ca02c",
                        marker="o",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen Hybrid (dgemm base)"] = l_hybrid_dg

            # Plot Rust faer base Strassen curves (DFS, BFS, Hybrid) - use triangle markers (up, down, left)
            if not df_faer_strassen.empty:
                flops_fs = (
                    2 * df_faer_strassen["size"] ** 3 - df_faer_strassen["size"] ** 2
                )

                if (
                    "strassen_dfs" in df_faer_strassen.columns
                    and not df_faer_strassen["strassen_dfs"].isna().all()
                ):
                    gflops_dfs_fs = (
                        flops_fs
                        / (df_faer_strassen["strassen_dfs"] * 1e9)
                        / norm_factor
                    )
                    if not gflops_dfs_fs.empty:
                        max_gflops = max(max_gflops, float(gflops_dfs_fs.max()))
                    (l_dfs_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_dfs_fs,
                        label="Strassen DFS (faer base)",
                        color="#e34a33",
                        marker="^",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen DFS (faer base)"] = l_dfs_fs

                if (
                    "strassen_bfs" in df_faer_strassen.columns
                    and not df_faer_strassen["strassen_bfs"].isna().all()
                ):
                    gflops_bfs_fs = (
                        flops_fs
                        / (df_faer_strassen["strassen_bfs"] * 1e9)
                        / norm_factor
                    )
                    if not gflops_bfs_fs.empty:
                        max_gflops = max(max_gflops, float(gflops_bfs_fs.max()))
                    (l_bfs_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_bfs_fs,
                        label="Strassen BFS (faer base)",
                        color="#02818a",
                        marker="v",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen BFS (faer base)"] = l_bfs_fs

                if (
                    "strassen_hybrid" in df_faer_strassen.columns
                    and not df_faer_strassen["strassen_hybrid"].isna().all()
                ):
                    gflops_hybrid_fs = (
                        flops_fs
                        / (df_faer_strassen["strassen_hybrid"] * 1e9)
                        / norm_factor
                    )
                    if not gflops_hybrid_fs.empty:
                        max_gflops = max(max_gflops, float(gflops_hybrid_fs.max()))
                    (l_hybrid_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_hybrid_fs,
                        label="Strassen Hybrid (faer base)",
                        color="#bcbd22",
                        marker="<",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen Hybrid (faer base)"] = l_hybrid_fs

        ax.set_xticks(df_base["size"])
        ax.tick_params("x", labelbottom=True, rotation=30, rotation_mode="xtick")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Set dynamic tight y-limits across subplots to reduce blank space
    if max_gflops > 0:
        for r in range(2):
            for c in range(4):
                axs[r, c].set_ylim(0, max_gflops * 1.05)

    # Add labels on outer plots
    for col in range(4):
        axs[1, col].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)
    axs[1, 3].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)

    for row in range(2):
        ylabel = "Effective GFLOPS" if is_seq else "Effective GFLOPS / core"
        axs[row, 0].set_ylabel(ylabel, labelpad=10)

    # Legend axis configuration
    legend_ax = axs[1, 3]
    legend_ax.axis("off")

    if is_seq:
        sorted_labels = [
            "MKL dgemm",
            "faer (Sequential)",
            "Strassen (dgemm base)",
            "Strassen (faer base)",
        ]
        ncol = 1
    else:
        sorted_labels = [
            "MKL dgemm",
            "faer (Parallel)",
            "Strassen DFS (dgemm base)",
            "Strassen BFS (dgemm base)",
            "Strassen Hybrid (dgemm base)",
            "Strassen DFS (faer base)",
            "Strassen BFS (faer base)",
            "Strassen Hybrid (faer base)",
        ]
        ncol = 2

    handles = [legend_handles[lbl] for lbl in sorted_labels if lbl in legend_handles]
    labels = [lbl for lbl in sorted_labels if lbl in legend_handles]

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cbd5e1",
        ncol=ncol,
    )

    title_suffix = "Row 1: Cutoffs | Row 2: Levels"
    title_text = (
        f"Sequential Matrix Multiplication Performance Comparison\n{title_suffix}"
        if is_seq
        else f"Parallel Matrix Multiplication Performance Comparison (Normalized per Core)\n{title_suffix}"
    )
    plt.suptitle(
        title_text,
        fontsize=15,
        y=0.98,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        hspace=0.25, wspace=0.15, top=0.90, bottom=0.08, left=0.06, right=0.96
    )

    # Save outputs
    out_dir_plots = os.path.join(project_root, "generated", "plots")
    out_name = "sequential_grid_plot" if is_seq else "parallel_grid_plot"
    pdf_path_plots = os.path.join(out_dir_plots, f"{out_name}.pdf")

    os.makedirs(out_dir_plots, exist_ok=True)
    plt.savefig(pdf_path_plots, bbox_inches="tight")

    print(f"{mode.capitalize()} grid plot saved successfully:")
    print(f"  - {pdf_path_plots}")
    plt.close(fig)


def plot_compare_ballard(project_root):
    """Generates and saves a 4x3 grid plot comparing Rust Strassen to Ballard references.
    Enforces row-specific tight y-limits starting at 0 to minimize blank space.
    Rows: Sequential, DFS, BFS, Hybrid. Columns: Level 1, Level 2, Level 3.
    A single common legend is placed below the 3 subplots of each row (centered horizontally).
    """
    csv_seq_dir = "run_seq"
    csv_par_dir = "run_par"

    # CSV Paths
    csv_seq_path = os.path.join(
        project_root, "generated", "csv", csv_seq_dir, "benchmark_results.csv"
    )
    base_seq_path = os.path.join(
        project_root, "generated", "csv", csv_seq_dir, "benchmark_results_base.csv"
    )
    csv_par_path = os.path.join(
        project_root, "generated", "csv", csv_par_dir, "benchmark_results.csv"
    )
    base_par_path = os.path.join(
        project_root, "generated", "csv", csv_par_dir, "benchmark_results_base.csv"
    )

    if not all(
        os.path.exists(p)
        for p in [csv_seq_path, base_seq_path, csv_par_path, base_par_path]
    ):
        print("Error: Missing input CSVs for Ballard comparison plot.")
        return

    df_seq = pd.read_csv(csv_seq_path)
    df_base_seq = pd.read_csv(base_seq_path)
    df_par = pd.read_csv(csv_par_path)
    df_base_par = pd.read_csv(base_par_path)

    # Core count for parallel GFLOPS normalization
    num_cores = os.cpu_count() or 1
    print(f"Normalizing parallel results using {num_cores} cores.")

    # Load reference Ballard files
    ballard_seq_path = os.path.join(
        project_root, "generated", "csv", "run_seq", "benchmarks_seq.txt"
    )
    if not os.path.exists(ballard_seq_path):
        ballard_seq_path = os.path.join(
            project_root, "benchmarks", "generated", "benchmarks_seq.txt"
        )

    ballard_dfs_path = os.path.join(
        project_root, "generated", "csv", "run_par", "benchmarks_dfs.txt"
    )
    ballard_bfs_path = os.path.join(
        project_root, "generated", "csv", "run_par", "benchmarks_bfs.txt"
    )
    ballard_hybrid_path = os.path.join(
        project_root, "generated", "csv", "run_par", "benchmarks_hybrid.txt"
    )

    if not os.path.exists(ballard_dfs_path):
        ballard_dfs_path = os.path.join(
            project_root, "benchmarks", "generated", "benchmarks_dfs.txt"
        )
    if not os.path.exists(ballard_bfs_path):
        ballard_bfs_path = os.path.join(
            project_root, "benchmarks", "generated", "benchmarks_bfs.txt"
        )
    if not os.path.exists(ballard_hybrid_path):
        ballard_hybrid_path = os.path.join(
            project_root, "benchmarks", "generated", "benchmarks_hybrid.txt"
        )

    # Parse Ballard Sequential
    ballard_seq_levels = [
        parse_matlab_vector(ballard_seq_path, f"STRASSEN_{l}") for l in (1, 2, 3)
    ]
    # Parse Ballard Parallel DFS, BFS, Hybrid
    ballard_dfs_levels = [
        parse_matlab_vector(ballard_dfs_path, f"STRASSEN_{l}") for l in (1, 2, 3)
    ]
    ballard_bfs_levels = [
        parse_matlab_vector(ballard_bfs_path, f"STRASSEN_{l}") for l in (1, 2, 3)
    ]
    ballard_hybrid_levels = [
        parse_matlab_vector(ballard_hybrid_path, f"STRASSEN_{l}") for l in (1, 2, 3)
    ]

    # Create 4x3 figure with row-specific shared y-axis (sharey='row')
    # Use standard LaTeX font sizes (8pt base) and size for A4 width (5.8 inches with 1.2-inch margins)
    custom_rc = {
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.title_fontsize": 7.5,
    }
    with plt.rc_context(custom_rc):
        fig, axs = plt.subplots(
            4, 3, figsize=(5.8, 8.0), sharex=True, sharey="row", dpi=300
        )

        max_gflops_seq = 0.0
        max_gflops_dfs = 0.0
        max_gflops_bfs = 0.0
        max_gflops_hybrid = 0.0

        # Define colors, markers and styles matching cohesive palette
        # Rust lines
        rust_colors = {
            "seq_dgemm": "#ff7f0e",
            "seq_faer": "#e34a33",
            "par_dfs_dgemm": "#ff7f0e",
            "par_bfs_dgemm": "#8c564b",
            "par_hybrid_dgemm": "#2ca02c",
            "par_dfs_faer": "#e34a33",
            "par_bfs_faer": "#02818a",
            "par_hybrid_faer": "#bcbd22",
        }
        # Ballard lines
        ballard_colors = {
            "seq": "#e41a1c",
            "dfs": "#4daf4a",
            "bfs": "#377eb8",
            "hybrid": "#984ea3",
        }

        # Loop over Level = 1, 2, 3 (cols 0, 1, 2)
        for col in range(3):
            level = col + 1

            # ==========================================
            # --- Row 0: Sequential ---
            # ==========================================
            ax_seq = axs[0, col]
            ax_seq.set_facecolor("none")
            ax_seq.set_xscale("log", base=2)
            ax_seq.set_xlim(left=2)
            ax_seq.set_yscale("linear")
            ax_seq.set_title(
                f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4
            )

            df_config_seq = df_seq[
                (df_seq["recursion_level"] == float(level))
                & (df_seq["size_cutoff"].isna())
            ]
            df_dg_seq = df_config_seq[df_config_seq["base_choice"] == "dgemm"]
            df_fs_seq = df_config_seq[df_config_seq["base_choice"] == "faer"]

            # Rust Sequential dgemm base - solid line
            if not df_dg_seq.empty:
                flops = 2 * df_dg_seq["size"] ** 3 - df_dg_seq["size"] ** 2
                gflops = flops / (df_dg_seq["strassen_seq"] * 1e9)
                if not gflops.empty:
                    max_gflops_seq = max(max_gflops_seq, float(gflops.max()))
                ax_seq.plot(
                    df_dg_seq["size"],
                    gflops,
                    label="Rust Strassen (dgemm)",
                    color=rust_colors["seq_dgemm"],
                    marker="o",
                    linestyle="-",
                    linewidth=1.0,
                    markersize=3.5,
                )

            # Rust Sequential faer base - solid line
            if not df_fs_seq.empty:
                flops = 2 * df_fs_seq["size"] ** 3 - df_fs_seq["size"] ** 2
                gflops = flops / (df_fs_seq["strassen_seq"] * 1e9)
                if not gflops.empty:
                    max_gflops_seq = max(max_gflops_seq, float(gflops.max()))
                ax_seq.plot(
                    df_fs_seq["size"],
                    gflops,
                    label="Rust Strassen (faer)",
                    color=rust_colors["seq_faer"],
                    marker="^",
                    linestyle="-",
                    linewidth=1.0,
                    markersize=3.5,
                )

            # Ballard Sequential Reference - dotted line
            b_seq = ballard_seq_levels[col]
            if b_seq is not None:
                time_s = b_seq["time_ms"] / 1000.0
                n_b = b_seq["size"]
                gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9)
                if not gflops_b.empty:
                    max_gflops_seq = max(max_gflops_seq, float(gflops_b.max()))
                ax_seq.plot(
                    n_b,
                    gflops_b,
                    label="Ballard Strassen",
                    color=ballard_colors["seq"],
                    marker="X",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # ==========================================
            # --- Row 1: DFS ---
            # ==========================================
            ax_dfs = axs[1, col]
            ax_dfs.set_facecolor("none")
            ax_dfs.set_xscale("log", base=2)
            ax_dfs.set_xlim(left=2)
            ax_dfs.set_yscale("linear")
            ax_dfs.set_title(
                f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4
            )

            df_config_par = df_par[
                (df_par["recursion_level"] == float(level))
                & (df_par["size_cutoff"].isna())
            ]
            df_dg_par = df_config_par[df_config_par["base_choice"] == "dgemm"]
            df_fs_par = df_config_par[df_config_par["base_choice"] == "faer"]

            # Rust DFS dgemm base - solid line
            if not df_dg_par.empty:
                flops = 2 * df_dg_par["size"] ** 3 - df_dg_par["size"] ** 2
                if (
                    "strassen_dfs" in df_dg_par.columns
                    and not df_dg_par["strassen_dfs"].isna().all()
                ):
                    gflops = flops / (df_dg_par["strassen_dfs"] * 1e9) / num_cores
                    if not gflops.empty:
                        max_gflops_dfs = max(max_gflops_dfs, float(gflops.max()))
                    ax_dfs.plot(
                        df_dg_par["size"],
                        gflops,
                        label="Rust Strassen DFS (dgemm)",
                        color=rust_colors["par_dfs_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            # Rust DFS faer base - solid line
            if not df_fs_par.empty:
                flops = 2 * df_fs_par["size"] ** 3 - df_fs_par["size"] ** 2
                if (
                    "strassen_dfs" in df_fs_par.columns
                    and not df_fs_par["strassen_dfs"].isna().all()
                ):
                    gflops = flops / (df_fs_par["strassen_dfs"] * 1e9) / num_cores
                    if not gflops.empty:
                        max_gflops_dfs = max(max_gflops_dfs, float(gflops.max()))
                    ax_dfs.plot(
                        df_fs_par["size"],
                        gflops,
                        label="Rust Strassen DFS (faer)",
                        color=rust_colors["par_dfs_faer"],
                        marker="^",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            # Ballard DFS reference - dotted line
            b_dfs = ballard_dfs_levels[col]
            if b_dfs is not None:
                time_s = b_dfs["time_ms"] / 1000.0
                n_b = b_dfs["size"]
                gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
                if not gflops_b.empty:
                    max_gflops_dfs = max(max_gflops_dfs, float(gflops_b.max()))
                ax_dfs.plot(
                    n_b,
                    gflops_b,
                    label="Ballard DFS",
                    color=ballard_colors["dfs"],
                    marker="X",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # ==========================================
            # --- Row 2: BFS ---
            # ==========================================
            ax_bfs = axs[2, col]
            ax_bfs.set_facecolor("none")
            ax_bfs.set_xscale("log", base=2)
            ax_bfs.set_xlim(left=2)
            ax_bfs.set_yscale("linear")
            ax_bfs.set_title(
                f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4
            )

            # Rust BFS dgemm base - solid line
            if not df_dg_par.empty:
                flops = 2 * df_dg_par["size"] ** 3 - df_dg_par["size"] ** 2
                if (
                    "strassen_bfs" in df_dg_par.columns
                    and not df_dg_par["strassen_bfs"].isna().all()
                ):
                    gflops = flops / (df_dg_par["strassen_bfs"] * 1e9) / num_cores
                    if not gflops.empty:
                        max_gflops_bfs = max(max_gflops_bfs, float(gflops.max()))
                    ax_bfs.plot(
                        df_dg_par["size"],
                        gflops,
                        label="Rust Strassen BFS (dgemm)",
                        color=rust_colors["par_bfs_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            # Rust BFS faer base - solid line
            if not df_fs_par.empty:
                flops = 2 * df_fs_par["size"] ** 3 - df_fs_par["size"] ** 2
                if (
                    "strassen_bfs" in df_fs_par.columns
                    and not df_fs_par["strassen_bfs"].isna().all()
                ):
                    gflops = flops / (df_fs_par["strassen_bfs"] * 1e9) / num_cores
                    if not gflops.empty:
                        max_gflops_bfs = max(max_gflops_bfs, float(gflops.max()))
                    ax_bfs.plot(
                        df_fs_par["size"],
                        gflops,
                        label="Rust Strassen BFS (faer)",
                        color=rust_colors["par_bfs_faer"],
                        marker="v",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            # Ballard BFS reference - dotted line
            b_bfs = ballard_bfs_levels[col]
            if b_bfs is not None:
                time_s = b_bfs["time_ms"] / 1000.0
                n_b = b_bfs["size"]
                gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
                if not gflops_b.empty:
                    max_gflops_bfs = max(max_gflops_bfs, float(gflops_b.max()))
                ax_bfs.plot(
                    n_b,
                    gflops_b,
                    label="Ballard BFS",
                    color=ballard_colors["bfs"],
                    marker="s",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # ==========================================
            # --- Row 3: Hybrid ---
            # ==========================================
            ax_hybrid = axs[3, col]
            ax_hybrid.set_facecolor("none")
            ax_hybrid.set_xscale("log", base=2)
            ax_hybrid.set_xlim(left=2)
            ax_hybrid.set_yscale("linear")
            ax_hybrid.set_title(
                f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4
            )

            # Rust Hybrid dgemm base - solid line
            if not df_dg_par.empty:
                flops = 2 * df_dg_par["size"] ** 3 - df_dg_par["size"] ** 2
                if (
                    "strassen_hybrid" in df_dg_par.columns
                    and not df_dg_par["strassen_hybrid"].isna().all()
                ):
                    gflops = flops / (df_dg_par["strassen_hybrid"] * 1e9) / num_cores
                    if not gflops.empty:
                        max_gflops_hybrid = max(max_gflops_hybrid, float(gflops.max()))
                    ax_hybrid.plot(
                        df_dg_par["size"],
                        gflops,
                        label="Rust Strassen Hybrid (dgemm)",
                        color=rust_colors["par_hybrid_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            # Rust Hybrid faer base - solid line
            if not df_fs_par.empty:
                flops = 2 * df_fs_par["size"] ** 3 - df_fs_par["size"] ** 2
                if (
                    "strassen_hybrid" in df_fs_par.columns
                    and not df_fs_par["strassen_hybrid"].isna().all()
                ):
                    gflops = flops / (df_fs_par["strassen_hybrid"] * 1e9) / num_cores
                    if not gflops.empty:
                        max_gflops_hybrid = max(max_gflops_hybrid, float(gflops.max()))
                    ax_hybrid.plot(
                        df_fs_par["size"],
                        gflops,
                        label="Rust Strassen Hybrid (faer)",
                        color=rust_colors["par_hybrid_faer"],
                        marker="<",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            # Ballard Hybrid reference - dotted line
            b_hybrid = ballard_hybrid_levels[col]
            if b_hybrid is not None:
                time_s = b_hybrid["time_ms"] / 1000.0
                n_b = b_hybrid["size"]
                gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
                if not gflops_b.empty:
                    max_gflops_hybrid = max(max_gflops_hybrid, float(gflops_b.max()))
                ax_hybrid.plot(
                    n_b,
                    gflops_b,
                    label="Ballard Hybrid",
                    color=ballard_colors["hybrid"],
                    marker="D",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            ax_seq.set_xticks(df_base_seq["size"])
            ax_seq.tick_params(
                "x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5.5
            )
            ax_seq.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax_dfs.set_xticks(df_base_par["size"])
            ax_dfs.tick_params(
                "x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5.5
            )
            ax_dfs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax_bfs.set_xticks(df_base_par["size"])
            ax_bfs.tick_params(
                "x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5.5
            )
            ax_bfs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax_hybrid.set_xticks(df_base_par["size"])
            ax_hybrid.tick_params(
                "x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5.5
            )
            ax_hybrid.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Set dynamic row-specific tight y-limits across subplots to reduce blank space
        if max_gflops_seq > 0:
            for c in range(3):
                axs[0, c].set_ylim(0, max_gflops_seq * 1.05)
        if max_gflops_dfs > 0:
            for c in range(3):
                axs[1, c].set_ylim(0, max_gflops_dfs * 1.05)
        if max_gflops_bfs > 0:
            for c in range(3):
                axs[2, c].set_ylim(0, max_gflops_bfs * 1.05)
        if max_gflops_hybrid > 0:
            for c in range(3):
                axs[3, c].set_ylim(0, max_gflops_hybrid * 1.05)

        # Place a common legend centered horizontally below the subplots for each row
        for r in range(4):
            handles, labels = axs[r, 1].get_legend_handles_labels()
            # For the last row (r=3), we push the legend further down to accommodate the x-axis labels
            anchor_y = -0.44 if r == 3 else -0.28
            axs[r, 1].legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, anchor_y),
                ncol=3,
                frameon=True,
                framealpha=0.9,
                edgecolor="#cbd5e1",
            )

        axs[0, 0].set_ylabel("[SEQ] Effective GFLOPS", labelpad=10)
        axs[1, 0].set_ylabel("[DFS] Effective GFLOPS / core", labelpad=10)
        axs[2, 0].set_ylabel("[BFS] Effective GFLOPS / core", labelpad=10)
        axs[3, 0].set_ylabel("[HYBRID] Effective GFLOPS / core", labelpad=10)

        plt.suptitle(
            "Strassen Matrix Multiplication: Rust vs Ballard Reference Comparison",
            fontsize=10.0,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        fig.subplots_adjust(
            hspace=0.85, wspace=0.22, top=0.92, bottom=0.07, left=0.13, right=0.96
        )

        # Save outputs
        out_dir_plots = os.path.join(project_root, "generated", "plots")
        report_figures_dir = os.path.join(project_root, "report", "figures")
        os.makedirs(out_dir_plots, exist_ok=True)
        os.makedirs(report_figures_dir, exist_ok=True)

        paths_to_save = [
            os.path.join(out_dir_plots, "compare_ballard.pdf"),
            os.path.join(out_dir_plots, "compare_ballard.png"),
            os.path.join(report_figures_dir, "compare_ballard.pdf"),
            os.path.join(report_figures_dir, "compare_ballard.png"),
        ]

        for path in paths_to_save:
            plt.savefig(path, bbox_inches="tight")
            print(f"Saved Ballard plot: {path}")

        plt.close(fig)


def main():
    """Main entry point parsing arguments and invoking grid plots."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Unified plotting script for sequential and parallel grid performance plots."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "sequential",
            "parallel",
            "both",
            "seq",
            "par",
            "compare_ballard",
            "ballard",
        ],
        default="both",
        help="Plotting mode: 'sequential' ('seq'), 'parallel' ('par'), 'compare_ballard' ('ballard'), or 'both' (default).",
    )
    args = parser.parse_args()

    modes_to_run = []
    if args.mode in ("sequential", "seq", "both"):
        modes_to_run.append("sequential")
    if args.mode in ("parallel", "par", "both"):
        modes_to_run.append("parallel")
    if args.mode in ("compare_ballard", "ballard", "both"):
        modes_to_run.append("compare_ballard")

    for mode in modes_to_run:
        if mode == "compare_ballard":
            plot_compare_ballard(project_root)
        else:
            plot_mode_grid(project_root, mode)


if __name__ == "__main__":
    main()
