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
Also generates a 2x3 Ballard comparison plot.
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
    Does not display Ballard reference lines.

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

    for row, col, title, is_cutoff, value in configs:
        ax = axs[row, col]
        ax.set_facecolor("none")
        ax.set_xscale("log", base=2)
        ax.set_yscale("linear")
        ax.set_title(title, pad=10)

        n_base = df_base["size"]
        mkl_flops = 2 * n_base**3 - n_base**2

        if is_seq:
            # 1. Plot MKL Base (Sequential)
            gflops_mkl = mkl_flops / (df_base["mkl_seq"] * 1e9)
            (l_mkl,) = ax.plot(
                n_base,
                gflops_mkl,
                label="MKL (Sequential)",
                color="#9467bd",
                marker="p",
                linestyle="--",
                linewidth=1.2,
                markersize=4.5,
            )
            legend_handles["MKL (Sequential)"] = l_mkl

            # 2. Plot faer Base (Sequential)
            gflops_faer = mkl_flops / (df_base["faer_seq"] * 1e9)
            (l_faer,) = ax.plot(
                n_base,
                gflops_faer,
                label="faer (Sequential)",
                color="#17becf",
                marker="d",
                linestyle="-",
                linewidth=1.2,
                markersize=4.5,
            )
            legend_handles["faer (Sequential)"] = l_faer
        else:
            # 1. Plot MKL Base (Parallel, normalized)
            if "mkl_par" in df_base.columns and not df_base["mkl_par"].isna().all():
                gflops_mkl = mkl_flops / (df_base["mkl_par"] * 1e9) / norm_factor
                (l_mkl,) = ax.plot(
                    n_base,
                    gflops_mkl,
                    label="MKL (Parallel)",
                    color="#9467bd",
                    marker="p",
                    linestyle="--",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["MKL (Parallel)"] = l_mkl

            # 2. Plot faer Base (Parallel, normalized)
            if "faer_par" in df_base.columns and not df_base["faer_par"].isna().all():
                gflops_faer = mkl_flops / (df_base["faer_par"] * 1e9) / norm_factor
                (l_faer,) = ax.plot(
                    n_base,
                    gflops_faer,
                    label="faer (Parallel)",
                    color="#17becf",
                    marker="d",
                    linestyle="-",
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
            if not df_dgemm.empty:
                flops_dg = 2 * df_dgemm["size"] ** 3 - df_dgemm["size"] ** 2
                gflops_dg = flops_dg / (df_dgemm["strassen_seq"] * 1e9)
                (l_strassen_dg,) = ax.plot(
                    df_dgemm["size"],
                    gflops_dg,
                    label="Strassen (dgemm base)",
                    color="#ff7f0e",
                    marker="s",
                    linestyle="--",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["Strassen (dgemm base)"] = l_strassen_dg

            if not df_faer_strassen.empty:
                flops_fs = 2 * df_faer_strassen["size"] ** 3 - df_faer_strassen["size"] ** 2
                gflops_fs = flops_fs / (df_faer_strassen["strassen_seq"] * 1e9)
                (l_strassen_fs,) = ax.plot(
                    df_faer_strassen["size"],
                    gflops_fs,
                    label="Strassen (faer base)",
                    color="#e34a33",
                    marker="o",
                    linestyle="-",
                    linewidth=1.2,
                    markersize=4.5,
                )
                legend_handles["Strassen (faer base)"] = l_strassen_fs
        else:
            # Plot Rust dgemm base Strassen curves (DFS, BFS, Hybrid)
            if not df_dgemm.empty:
                flops_dg = 2 * df_dgemm["size"] ** 3 - df_dgemm["size"] ** 2

                if "strassen_dfs" in df_dgemm.columns and not df_dgemm["strassen_dfs"].isna().all():
                    gflops_dfs = flops_dg / (df_dgemm["strassen_dfs"] * 1e9) / norm_factor
                    (l_dfs_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_dfs,
                        label="Strassen DFS (dgemm base)",
                        color="#ff7f0e",
                        marker="s",
                        linestyle="--",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen DFS (dgemm base)"] = l_dfs_dg

                if "strassen_bfs" in df_dgemm.columns and not df_dgemm["strassen_bfs"].isna().all():
                    gflops_bfs = flops_dg / (df_dgemm["strassen_bfs"] * 1e9) / norm_factor
                    (l_bfs_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_bfs,
                        label="Strassen BFS (dgemm base)",
                        color="#8c564b",
                        marker="v",
                        linestyle="--",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen BFS (dgemm base)"] = l_bfs_dg

                if "strassen_hybrid" in df_dgemm.columns and not df_dgemm["strassen_hybrid"].isna().all():
                    gflops_hybrid = flops_dg / (df_dgemm["strassen_hybrid"] * 1e9) / norm_factor
                    (l_hybrid_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_hybrid,
                        label="Strassen Hybrid (dgemm base)",
                        color="#2ca02c",
                        marker="^",
                        linestyle="--",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen Hybrid (dgemm base)"] = l_hybrid_dg

            # Plot Rust faer base Strassen curves (DFS, BFS, Hybrid)
            if not df_faer_strassen.empty:
                flops_fs = 2 * df_faer_strassen["size"] ** 3 - df_faer_strassen["size"] ** 2

                if "strassen_dfs" in df_faer_strassen.columns and not df_faer_strassen["strassen_dfs"].isna().all():
                    gflops_dfs_fs = flops_fs / (df_faer_strassen["strassen_dfs"] * 1e9) / norm_factor
                    (l_dfs_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_dfs_fs,
                        label="Strassen DFS (faer base)",
                        color="#e34a33",
                        marker="o",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen DFS (faer base)"] = l_dfs_fs

                if "strassen_bfs" in df_faer_strassen.columns and not df_faer_strassen["strassen_bfs"].isna().all():
                    gflops_bfs_fs = flops_fs / (df_faer_strassen["strassen_bfs"] * 1e9) / norm_factor
                    (l_bfs_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_bfs_fs,
                        label="Strassen BFS (faer base)",
                        color="#02818a",
                        marker="D",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen BFS (faer base)"] = l_bfs_fs

                if "strassen_hybrid" in df_faer_strassen.columns and not df_faer_strassen["strassen_hybrid"].isna().all():
                    gflops_hybrid_fs = flops_fs / (df_faer_strassen["strassen_hybrid"] * 1e9) / norm_factor
                    (l_hybrid_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_hybrid_fs,
                        label="Strassen Hybrid (faer base)",
                        color="#bcbd22",
                        marker="*",
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4.5,
                    )
                    legend_handles["Strassen Hybrid (faer base)"] = l_hybrid_fs

        ax.set_xticks(df_base["size"])
        ax.tick_params("x", labelbottom=True, rotation=30, rotation_mode="xtick")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

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
            "MKL (Sequential)",
            "faer (Sequential)",
            "Strassen (dgemm base)",
            "Strassen (faer base)",
        ]
        ncol = 1
    else:
        sorted_labels = [
            "MKL (Parallel)",
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

    title_suffix = "Row 1: Cutoffs | Row 2: Recursion Levels"
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
    """Generates and saves a 2x3 grid plot comparing Rust Strassen to Ballard references."""
    csv_seq_dir = "run_seq"
    csv_par_dir = "run_par"

    # CSV Paths
    csv_seq_path = os.path.join(project_root, "generated", "csv", csv_seq_dir, "benchmark_results.csv")
    base_seq_path = os.path.join(project_root, "generated", "csv", csv_seq_dir, "benchmark_results_base.csv")
    csv_par_path = os.path.join(project_root, "generated", "csv", csv_par_dir, "benchmark_results.csv")
    base_par_path = os.path.join(project_root, "generated", "csv", csv_par_dir, "benchmark_results_base.csv")

    if not all(os.path.exists(p) for p in [csv_seq_path, base_seq_path, csv_par_path, base_par_path]):
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
    ballard_seq_path = os.path.join(project_root, "generated", "csv", "run_seq", "benchmarks_seq.txt")
    if not os.path.exists(ballard_seq_path):
        ballard_seq_path = os.path.join(project_root, "benchmarks", "generated", "benchmarks_seq.txt")

    ballard_dfs_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmarks_dfs.txt")
    ballard_bfs_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmarks_bfs.txt")
    ballard_hybrid_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmarks_hybrid.txt")

    if not os.path.exists(ballard_dfs_path):
        ballard_dfs_path = os.path.join(project_root, "benchmarks", "generated", "benchmarks_dfs.txt")
    if not os.path.exists(ballard_bfs_path):
        ballard_bfs_path = os.path.join(project_root, "benchmarks", "generated", "benchmarks_bfs.txt")
    if not os.path.exists(ballard_hybrid_path):
        ballard_hybrid_path = os.path.join(project_root, "benchmarks", "generated", "benchmarks_hybrid.txt")

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

    # Create 2x3 figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True, dpi=300)

    legend_handles = {}

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

        # --- Row 0: Sequential ---
        ax_seq = axs[0, col]
        ax_seq.set_facecolor("none")
        ax_seq.set_xscale("log", base=2)
        ax_seq.set_yscale("linear")
        # Subplot title
        ax_seq.set_title(f"Level = {level}", pad=10)

        # Rust Sequential (dgemm and faer base)
        # Filter for this recursion level
        df_config_seq = df_seq[(df_seq["recursion_level"] == float(level)) & (df_seq["size_cutoff"].isna())]
        df_dg_seq = df_config_seq[df_config_seq["base_choice"] == "dgemm"]
        df_fs_seq = df_config_seq[df_config_seq["base_choice"] == "faer"]

        if not df_dg_seq.empty:
            flops = 2 * df_dg_seq["size"] ** 3 - df_dg_seq["size"] ** 2
            gflops = flops / (df_dg_seq["strassen_seq"] * 1e9)
            (l,) = ax_seq.plot(
                df_dg_seq["size"], gflops,
                label="Rust Strassen (dgemm base)", color=rust_colors["seq_dgemm"],
                marker="s", linestyle="--", linewidth=1.2, markersize=4.5
            )
            legend_handles["Rust Strassen (dgemm base)"] = l

        if not df_fs_seq.empty:
            flops = 2 * df_fs_seq["size"] ** 3 - df_fs_seq["size"] ** 2
            gflops = flops / (df_fs_seq["strassen_seq"] * 1e9)
            (l,) = ax_seq.plot(
                df_fs_seq["size"], gflops,
                label="Rust Strassen (faer base)", color=rust_colors["seq_faer"],
                marker="o", linestyle="-", linewidth=1.2, markersize=4.5
            )
            legend_handles["Rust Strassen (faer base)"] = l

        # Ballard Sequential Reference
        b_seq = ballard_seq_levels[col]
        if b_seq is not None:
            time_s = b_seq["time_ms"] / 1000.0
            n_b = b_seq["size"]
            gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9)
            (l,) = ax_seq.plot(
                n_b, gflops_b,
                label="Ballard Strassen (Sequential)", color=ballard_colors["seq"],
                marker="X", linestyle=":", linewidth=1.5, markersize=5.0
            )
            legend_handles["Ballard Strassen (Sequential)"] = l

        ax_seq.set_xticks(df_base_seq["size"])
        ax_seq.tick_params("x", labelbottom=True, rotation=30, rotation_mode="xtick")
        ax_seq.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # --- Row 1: Parallel ---
        ax_par = axs[1, col]
        ax_par.set_facecolor("none")
        ax_par.set_xscale("log", base=2)
        ax_par.set_yscale("linear")

        # Rust Parallel (DFS, BFS, Hybrid; dgemm and faer bases)
        df_config_par = df_par[(df_par["recursion_level"] == float(level)) & (df_par["size_cutoff"].isna())]
        df_dg_par = df_config_par[df_config_par["base_choice"] == "dgemm"]
        df_fs_par = df_config_par[df_config_par["base_choice"] == "faer"]

        # Rust dgemm base parallel curves
        if not df_dg_par.empty:
            flops = 2 * df_dg_par["size"] ** 3 - df_dg_par["size"] ** 2
            if "strassen_dfs" in df_dg_par.columns and not df_dg_par["strassen_dfs"].isna().all():
                gflops = flops / (df_dg_par["strassen_dfs"] * 1e9) / num_cores
                (l,) = ax_par.plot(
                    df_dg_par["size"], gflops,
                    label="Rust Strassen DFS (dgemm base)", color=rust_colors["par_dfs_dgemm"],
                    marker="s", linestyle="--", linewidth=1.2, markersize=4.5
                )
                legend_handles["Rust Strassen DFS (dgemm base)"] = l

            if "strassen_bfs" in df_dg_par.columns and not df_dg_par["strassen_bfs"].isna().all():
                gflops = flops / (df_dg_par["strassen_bfs"] * 1e9) / num_cores
                (l,) = ax_par.plot(
                    df_dg_par["size"], gflops,
                    label="Rust Strassen BFS (dgemm base)", color=rust_colors["par_bfs_dgemm"],
                    marker="v", linestyle="--", linewidth=1.2, markersize=4.5
                )
                legend_handles["Rust Strassen BFS (dgemm base)"] = l

            if "strassen_hybrid" in df_dg_par.columns and not df_dg_par["strassen_hybrid"].isna().all():
                gflops = flops / (df_dg_par["strassen_hybrid"] * 1e9) / num_cores
                (l,) = ax_par.plot(
                    df_dg_par["size"], gflops,
                    label="Rust Strassen Hybrid (dgemm base)", color=rust_colors["par_hybrid_dgemm"],
                    marker="^", linestyle="--", linewidth=1.2, markersize=4.5
                )
                legend_handles["Rust Strassen Hybrid (dgemm base)"] = l

        # Rust faer base parallel curves
        if not df_fs_par.empty:
            flops = 2 * df_fs_par["size"] ** 3 - df_fs_par["size"] ** 2
            if "strassen_dfs" in df_fs_par.columns and not df_fs_par["strassen_dfs"].isna().all():
                gflops = flops / (df_fs_par["strassen_dfs"] * 1e9) / num_cores
                (l,) = ax_par.plot(
                    df_fs_par["size"], gflops,
                    label="Rust Strassen DFS (faer base)", color=rust_colors["par_dfs_faer"],
                    marker="o", linestyle="-", linewidth=1.2, markersize=4.5
                )
                legend_handles["Rust Strassen DFS (faer base)"] = l

            if "strassen_bfs" in df_fs_par.columns and not df_fs_par["strassen_bfs"].isna().all():
                gflops = flops / (df_fs_par["strassen_bfs"] * 1e9) / num_cores
                (l,) = ax_par.plot(
                    df_fs_par["size"], gflops,
                    label="Rust Strassen BFS (faer base)", color=rust_colors["par_bfs_faer"],
                    marker="D", linestyle="-", linewidth=1.2, markersize=4.5
                )
                legend_handles["Rust Strassen BFS (faer base)"] = l

            if "strassen_hybrid" in df_fs_par.columns and not df_fs_par["strassen_hybrid"].isna().all():
                gflops = flops / (df_fs_par["strassen_hybrid"] * 1e9) / num_cores
                (l,) = ax_par.plot(
                    df_fs_par["size"], gflops,
                    label="Rust Strassen Hybrid (faer base)", color=rust_colors["par_hybrid_faer"],
                    marker="*", linestyle="-", linewidth=1.2, markersize=4.5
                )
                legend_handles["Rust Strassen Hybrid (faer base)"] = l

        # Ballard Parallel DFS, BFS, Hybrid references
        b_dfs = ballard_dfs_levels[col]
        if b_dfs is not None:
            time_s = b_dfs["time_ms"] / 1000.0
            n_b = b_dfs["size"]
            gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
            (l,) = ax_par.plot(
                n_b, gflops_b,
                label="Ballard Strassen DFS (C Parallel)", color=ballard_colors["dfs"],
                marker="X", linestyle=":", linewidth=1.5, markersize=5.0
            )
            legend_handles["Ballard Strassen DFS (C Parallel)"] = l

        b_bfs = ballard_bfs_levels[col]
        if b_bfs is not None:
            time_s = b_bfs["time_ms"] / 1000.0
            n_b = b_bfs["size"]
            gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
            (l,) = ax_par.plot(
                n_b, gflops_b,
                label="Ballard Strassen BFS (C Parallel)", color=ballard_colors["bfs"],
                marker="s", linestyle=":", linewidth=1.5, markersize=5.0
            )
            legend_handles["Ballard Strassen BFS (C Parallel)"] = l

        b_hybrid = ballard_hybrid_levels[col]
        if b_hybrid is not None:
            time_s = b_hybrid["time_ms"] / 1000.0
            n_b = b_hybrid["size"]
            gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
            (l,) = ax_par.plot(
                n_b, gflops_b,
                label="Ballard Strassen Hybrid (C Parallel)", color=ballard_colors["hybrid"],
                marker="D", linestyle=":", linewidth=1.5, markersize=5.0
            )
            legend_handles["Ballard Strassen Hybrid (C Parallel)"] = l

        ax_par.set_xticks(df_base_par["size"])
        ax_par.tick_params("x", labelbottom=True, rotation=30, rotation_mode="xtick")
        ax_par.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Labels on outer plots
    for col in range(3):
        axs[1, col].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)

    axs[0, 0].set_ylabel("Effective GFLOPS", labelpad=10)
    axs[1, 0].set_ylabel("Effective GFLOPS / core", labelpad=10)

    # Reconfigure legend font size for 2x3 layout
    plt.rcParams.update({"legend.fontsize": 8.0})

    # Sort labels for the legend
    sorted_labels = [
        "Rust Strassen (dgemm base)",
        "Rust Strassen (faer base)",
        "Ballard Strassen (Sequential)",
        "Rust Strassen DFS (dgemm base)",
        "Rust Strassen BFS (dgemm base)",
        "Rust Strassen Hybrid (dgemm base)",
        "Rust Strassen DFS (faer base)",
        "Rust Strassen BFS (faer base)",
        "Rust Strassen Hybrid (faer base)",
        "Ballard Strassen DFS (C Parallel)",
        "Ballard Strassen BFS (C Parallel)",
        "Ballard Strassen Hybrid (C Parallel)",
    ]

    handles = [legend_handles[lbl] for lbl in sorted_labels if lbl in legend_handles]
    labels = [lbl for lbl in sorted_labels if lbl in legend_handles]

    # Place the legend at the bottom of the figure
    fig.legend(
        handles, labels, loc="lower center", frameon=True, framealpha=0.9,
        edgecolor="#cbd5e1", ncol=4, bbox_to_anchor=(0.5, 0.01)
    )

    plt.suptitle(
        "Strassen Matrix Multiplication: Rust vs Ballard Reference Comparison\nRow 1: Sequential | Row 2: Parallel (Normalized per Core)",
        fontsize=14,
        y=0.97,
    )

    plt.tight_layout()
    # Adjust to make room for suptitle at top and legend at bottom
    fig.subplots_adjust(
        hspace=0.20, wspace=0.15, top=0.88, bottom=0.18, left=0.08, right=0.95
    )

    # Save outputs
    out_dir_plots = os.path.join(project_root, "generated", "plots")
    out_name = "compare_ballard"
    pdf_path_plots = os.path.join(out_dir_plots, f"{out_name}.pdf")

    os.makedirs(out_dir_plots, exist_ok=True)
    plt.savefig(pdf_path_plots, bbox_inches="tight")

    print(f"Compare Ballard grid plot saved successfully:")
    print(f"  - {pdf_path_plots}")
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
        choices=["sequential", "parallel", "both", "seq", "par", "compare_ballard", "ballard"],
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
