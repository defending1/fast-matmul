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
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import shutil

import plot_utils


def plot_mode_grid(project_root: str, mode: str, par_dir: str = "run_par", backend_filter: str = None) -> None:
    """Generates a performance comparison grid plot for the specified mode.

    Does not display Ballard reference lines. Enforces tight y-limits starting at 0.

    Args:
        project_root: The root directory of the project.
        mode: The plotting mode, either 'sequential' ('seq') or 'parallel' ('par').
        par_dir: Directory name under generated/csv/ for parallel results (default: 'run_par').
        backend_filter: Filter for backend: 'faer', 'dgemm', or None (default).
    """
    is_seq = mode in ("sequential", "seq")
    csv_dir = "run_seq" if is_seq else par_dir

    if is_seq:
        cutoff_path = os.path.join(project_root, "generated", "csv", "run_seq", "benchmark_results_cutoff.csv")
        levels_path = os.path.join(project_root, "generated", "csv", "run_seq", "benchmark_results_levels.csv")
        base_csv_path = os.path.join(project_root, "generated", "csv", "run_seq", "benchmark_results_base.csv")
    else:
        # Load parallel levels from run_par and parallel cutoffs from run_par2
        cutoff_path = os.path.join(project_root, "generated", "csv", "run_par2", "benchmark_results_cutoff.csv")
        levels_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results_levels.csv")
        base_csv_path = os.path.join(project_root, "generated", "csv", par_dir, "benchmark_results_base.csv")

    if not os.path.exists(base_csv_path):
        base_csv_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results_base.csv")
        if not os.path.exists(base_csv_path):
            base_csv_path = os.path.join(project_root, "generated", "csv", "run_par2", "benchmark_results_base.csv")

    if not os.path.exists(base_csv_path):
        print(f"Error: Missing input CSV at {base_csv_path}")
        return

    dfs = []
    if os.path.exists(cutoff_path):
        try:
            dfs.append(pd.read_csv(cutoff_path))
        except Exception as e:
            print(f"Warning: Failed to load {cutoff_path}: {e}")
    if os.path.exists(levels_path):
        try:
            dfs.append(pd.read_csv(levels_path))
        except Exception as e:
            print(f"Warning: Failed to load {levels_path}: {e}")

    if not dfs:
        std_csv_path = os.path.join(project_root, "generated", "csv", csv_dir, "benchmark_results.csv")
        if os.path.exists(std_csv_path):
            try:
                dfs.append(pd.read_csv(std_csv_path))
            except Exception as e:
                print(f"Warning: Failed to load {std_csv_path}: {e}")

    if not dfs:
        print(f"Error: Missing input Strassen CSVs at {cutoff_path} or {levels_path}")
        return

    df = pd.concat(dfs, ignore_index=True)
    df_base = pd.read_csv(base_csv_path)

    num_cores = os.cpu_count() or 1
    norm_factor = 1.0 if is_seq else float(num_cores)
    if not is_seq:
        print(f"Normalizing parallel results using {num_cores} cores.")

    plot_utils.setup_matplotlib_style()
    latex_active = plot_utils.detect_latex()
    custom_rc = {
        "text.usetex": latex_active,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.titlesize": 9.5,
        "axes.titlesize": 8,
    }

    nrows, ncols = 2, 4
    figsize = (5.8, 3.8)

    if is_seq:
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
    else:
        configs = [
            # (row, col, title, is_cutoff, value)
            (0, 0, "Cutoff = 2048", True, 2048),
            (0, 1, "Cutoff = 4096", True, 4096),
            (0, 2, "Cutoff = 8192", True, 8192),
            (0, 3, "Cutoff = 16384", True, 16384),
            (1, 0, "Level = 1", False, 1),
            (1, 1, "Level = 2", False, 2),
            (1, 2, "Level = 3", False, 3),
        ]

    with plt.rc_context(custom_rc):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey="row", dpi=300)

        legend_handles = {}
        max_gflops_row = [0.0, 0.0]

        for row, col, title, is_cutoff, value in configs:
            ax = axs[row, col]
            ax.set_facecolor("none")
            ax.set_xscale("log", base=2)
            ax.set_xlim(left=2, right=65536)
            ax.set_yscale("linear")
            ax.set_title(title, pad=4)
            ax.grid(False) # Remove grid inside plot

            n_base = df_base["size"]

            if is_seq:
                # 1. Plot MKL Base (Sequential)
                if backend_filter not in ("faer", "strassen_only"):
                    gflops_mkl = plot_utils.calculate_gflops_rust(n_base, df_base["mkl_seq"])
                    if not gflops_mkl.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_mkl.max()))
                    (l_mkl,) = ax.plot(
                        n_base,
                        gflops_mkl,
                        label="MKL dgemm",
                        color="#9467bd",
                        marker="o",
                        linestyle="--",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["MKL dgemm"] = l_mkl

                # 2. Plot faer Base (Sequential)
                if backend_filter not in ("dgemm", "strassen_only"):
                    gflops_faer = plot_utils.calculate_gflops_rust(n_base, df_base["faer_seq"])
                    if not gflops_faer.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_faer.max()))
                    (l_faer,) = ax.plot(
                        n_base,
                        gflops_faer,
                        label="faer (Sequential)",
                        color="#17becf",
                        marker="^",
                        linestyle="--",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["faer (Sequential)"] = l_faer
            else:
                # 1. Plot MKL Base (Parallel, normalized)
                if (
                    backend_filter not in ("faer", "strassen_only")
                    and "mkl_par" in df_base.columns
                    and not df_base["mkl_par"].isna().all()
                ):
                    gflops_mkl = plot_utils.calculate_gflops_rust(n_base, df_base["mkl_par"], is_parallel=True, num_cores=num_cores)
                    if not gflops_mkl.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_mkl.max()))
                    (l_mkl,) = ax.plot(
                        n_base,
                        gflops_mkl,
                        label="MKL dgemm",
                        color="#9467bd",
                        marker="o",
                        linestyle="--",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["MKL dgemm"] = l_mkl

                # 2. Plot faer Base (Parallel, normalized)
                if (
                    backend_filter not in ("dgemm", "strassen_only")
                    and "faer_par" in df_base.columns
                    and not df_base["faer_par"].isna().all()
                ):
                    gflops_faer = plot_utils.calculate_gflops_rust(n_base, df_base["faer_par"], is_parallel=True, num_cores=num_cores)
                    if not gflops_faer.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_faer.max()))
                    (l_faer,) = ax.plot(
                        n_base,
                        gflops_faer,
                        label="faer (Parallel)",
                        color="#17becf",
                        marker="^",
                        linestyle="--",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["faer (Parallel)"] = l_faer

            # 3. Filter Strassen results
            if is_cutoff:
                df_config = df[(df["size_cutoff"] == value) & (df["recursion_level"].isna())]
            else:
                df_config = df[(df["recursion_level"] == float(value)) & (df["size_cutoff"].isna())]

            df_dgemm = df_config[df_config["base_choice"] == "dgemm"]
            df_faer_strassen = df_config[df_config["base_choice"] == "faer"]

            if is_seq:
                # Plot Rust dgemm base Strassen curves (Sequential)
                if backend_filter != "faer" and not df_dgemm.empty:
                    if "strassen_seq" in df_dgemm.columns and not df_dgemm["strassen_seq"].isna().all():
                        gflops_seq = plot_utils.calculate_gflops_rust(df_dgemm["size"], df_dgemm["strassen_seq"])
                        if not gflops_seq.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_seq.max()))
                        (l_seq_dg,) = ax.plot(
                            df_dgemm["size"],
                            gflops_seq,
                            label="Strassen (dgemm base)",
                            color=plot_utils.RUST_GRID_COLORS["seq_dgemm"],
                            marker="o",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen (dgemm base)"] = l_seq_dg

                # Plot Rust faer base Strassen curves (Sequential)
                if backend_filter != "dgemm" and not df_faer_strassen.empty:
                    if "strassen_seq" in df_faer_strassen.columns and not df_faer_strassen["strassen_seq"].isna().all():
                        gflops_seq_fs = plot_utils.calculate_gflops_rust(
                            df_faer_strassen["size"], df_faer_strassen["strassen_seq"]
                        )
                        if not gflops_seq_fs.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_seq_fs.max()))
                        (l_seq_fs,) = ax.plot(
                            df_faer_strassen["size"],
                            gflops_seq_fs,
                            label="Strassen (faer base)",
                            color=plot_utils.RUST_GRID_COLORS["seq_faer"],
                            marker="^",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen (faer base)"] = l_seq_fs
            else:
                # Plot Rust dgemm base Strassen curves (Parallel - DFS, BFS, Hybrid)
                if backend_filter != "faer" and not df_dgemm.empty:
                    if "strassen_dfs" in df_dgemm.columns and not df_dgemm["strassen_dfs"].isna().all():
                        gflops_dfs = plot_utils.calculate_gflops_rust(
                            df_dgemm["size"], df_dgemm["strassen_dfs"], is_parallel=True, num_cores=num_cores
                        )
                        if not gflops_dfs.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_dfs.max()))
                        (l_dfs_dg,) = ax.plot(
                            df_dgemm["size"],
                            gflops_dfs,
                            label="Strassen DFS (dgemm base)",
                            color=plot_utils.RUST_GRID_COLORS["par_dfs_dgemm"],
                            marker="o",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen DFS (dgemm base)"] = l_dfs_dg

                    if "strassen_bfs" in df_dgemm.columns and not df_dgemm["strassen_bfs"].isna().all():
                        gflops_bfs = plot_utils.calculate_gflops_rust(
                            df_dgemm["size"], df_dgemm["strassen_bfs"], is_parallel=True, num_cores=num_cores
                        )
                        if not gflops_bfs.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_bfs.max()))
                        (l_bfs_dg,) = ax.plot(
                            df_dgemm["size"],
                            gflops_bfs,
                            label="Strassen BFS (dgemm base)",
                            color=plot_utils.RUST_GRID_COLORS["par_bfs_dgemm"],
                            marker="o",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen BFS (dgemm base)"] = l_bfs_dg

                    if "strassen_hybrid" in df_dgemm.columns and not df_dgemm["strassen_hybrid"].isna().all():
                        gflops_hybrid = plot_utils.calculate_gflops_rust(
                            df_dgemm["size"], df_dgemm["strassen_hybrid"], is_parallel=True, num_cores=num_cores
                        )
                        if not gflops_hybrid.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_hybrid.max()))
                        (l_hybrid_dg,) = ax.plot(
                            df_dgemm["size"],
                            gflops_hybrid,
                            label="Strassen Hybrid (dgemm base)",
                            color=plot_utils.RUST_GRID_COLORS["par_hybrid_dgemm"],
                            marker="o",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen Hybrid (dgemm base)"] = l_hybrid_dg

                # Plot Rust faer base Strassen curves (Parallel - DFS, BFS, Hybrid)
                if backend_filter != "dgemm" and not df_faer_strassen.empty:
                    if "strassen_dfs" in df_faer_strassen.columns and not df_faer_strassen["strassen_dfs"].isna().all():
                        gflops_dfs_fs = plot_utils.calculate_gflops_rust(
                            df_faer_strassen["size"], df_faer_strassen["strassen_dfs"], is_parallel=True, num_cores=num_cores
                        )
                        if not gflops_dfs_fs.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_dfs_fs.max()))
                        (l_dfs_fs,) = ax.plot(
                            df_faer_strassen["size"],
                            gflops_dfs_fs,
                            label="Strassen DFS (faer base)",
                            color=plot_utils.RUST_GRID_COLORS["par_dfs_faer"],
                            marker="^",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen DFS (faer base)"] = l_dfs_fs

                    if "strassen_bfs" in df_faer_strassen.columns and not df_faer_strassen["strassen_bfs"].isna().all():
                        gflops_bfs_fs = plot_utils.calculate_gflops_rust(
                            df_faer_strassen["size"], df_faer_strassen["strassen_bfs"], is_parallel=True, num_cores=num_cores
                        )
                        if not gflops_bfs_fs.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_bfs_fs.max()))
                        (l_bfs_fs,) = ax.plot(
                            df_faer_strassen["size"],
                            gflops_bfs_fs,
                            label="Strassen BFS (faer base)",
                            color=plot_utils.RUST_GRID_COLORS["par_bfs_faer"],
                            marker="v",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen BFS (faer base)"] = l_bfs_fs

                    if "strassen_hybrid" in df_faer_strassen.columns and not df_faer_strassen["strassen_hybrid"].isna().all():
                        gflops_hybrid_fs = plot_utils.calculate_gflops_rust(
                            df_faer_strassen["size"], df_faer_strassen["strassen_hybrid"], is_parallel=True, num_cores=num_cores
                        )
                        if not gflops_hybrid_fs.empty:
                            max_gflops_row[row] = max(max_gflops_row[row], float(gflops_hybrid_fs.max()))
                        (l_hybrid_fs,) = ax.plot(
                            df_faer_strassen["size"],
                            gflops_hybrid_fs,
                            label="Strassen Hybrid (faer base)",
                            color=plot_utils.RUST_GRID_COLORS["par_hybrid_faer"],
                            marker="<",
                            linestyle="-",
                            linewidth=0.8,
                            markersize=3.0,
                        )
                        legend_handles["Strassen Hybrid (faer base)"] = l_hybrid_fs

            ax.set_xticks(df_base["size"])
            ax.tick_params("x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Set dynamic tight y-limits across subplots per row to reduce blank space
        for r in range(2):
            if max_gflops_row[r] > 0:
                for c in range(4):
                    axs[r, c].set_ylim(0, max_gflops_row[r] * 1.05)

        # Add labels on outer plots
        for col in range(4):
            axs[1, col].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=4)
        axs[1, 3].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=4)

        for row in range(2):
            ylabel = "Effective GFLOPS" if is_seq else "Effective GFLOPS / core"
            axs[row, 0].set_ylabel(ylabel, labelpad=4)

        # Legend axis configuration
        legend_ax = axs[1, 3]
        legend_ax.axis("off")

        if is_seq:
            if backend_filter == "faer":
                sorted_labels = ["faer (Sequential)", "Strassen (faer base)"]
                ncol = 1
            elif backend_filter == "dgemm":
                sorted_labels = ["MKL dgemm", "Strassen (dgemm base)"]
                ncol = 1
            elif backend_filter == "strassen_only":
                sorted_labels = ["Strassen (dgemm base)", "Strassen (faer base)"]
                ncol = 1
            else:
                sorted_labels = ["MKL dgemm", "faer (Sequential)", "Strassen (dgemm base)", "Strassen (faer base)"]
                ncol = 1
        else:
            if backend_filter == "faer":
                sorted_labels = [
                    "faer (Parallel)",
                    "Strassen DFS (faer base)",
                    "Strassen BFS (faer base)",
                    "Strassen Hybrid (faer base)",
                ]
                ncol = 1
            elif backend_filter == "dgemm":
                sorted_labels = [
                    "MKL dgemm",
                    "Strassen DFS (dgemm base)",
                    "Strassen BFS (dgemm base)",
                    "Strassen Hybrid (dgemm base)",
                ]
                ncol = 1
            elif backend_filter == "strassen_only":
                sorted_labels = [
                    "Strassen DFS (dgemm base)",
                    "Strassen BFS (dgemm base)",
                    "Strassen Hybrid (dgemm base)",
                    "Strassen DFS (faer base)",
                    "Strassen BFS (faer base)",
                    "Strassen Hybrid (faer base)",
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
            loc="center left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#cbd5e1",
            ncol=ncol,
            bbox_to_anchor=(-0.15, 0.5) if ncol == 2 else (0.05, 0.5),
        )

        # Remove suptitle for both sequential and parallel modes to follow report formatting standards, and use uniform margins
        fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.92, bottom=0.15, left=0.09, right=0.97)

        # Save outputs
        out_dir_plots = os.path.join(project_root, "generated", "plots")
        report_figures_dir = os.path.join(project_root, "report", "figures")
        out_name = "sequential_grid_plot" if is_seq else "parallel_grid_plot"
        if backend_filter:
            out_name = f"{out_name}_{backend_filter}"

        for d in (out_dir_plots, report_figures_dir):
            os.makedirs(d, exist_ok=True)
            pdf_path_plots = os.path.join(d, f"{out_name}.pdf")
            plot_utils.save_plot(fig, pdf_path_plots)

        plt.close(fig)


def plot_cutoff_grid(project_root: str, par_dir: str = "run_par2", backend_filter: str = None) -> None:
    """Generates a 2-row performance comparison grid plot showing all 7 size cutoffs.

    Args:
        project_root: The root directory of the project.
        par_dir: Directory name under generated/csv/ for parallel results (default: 'run_par2').
        backend_filter: Filter for backend: 'faer', 'dgemm', or None (default).
    """
    cutoff_path = os.path.join(project_root, "generated", "csv", par_dir, "benchmark_results_cutoff.csv")
    base_csv_path = os.path.join(project_root, "generated", "csv", par_dir, "benchmark_results_base.csv")

    if not os.path.exists(base_csv_path):
        base_csv_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results_base.csv")
        if not os.path.exists(base_csv_path):
            base_csv_path = os.path.join(project_root, "generated", "csv", "run_par2", "benchmark_results_base.csv")

    if not os.path.exists(cutoff_path) or not os.path.exists(base_csv_path):
        print(f"Error: Missing input CSVs at {cutoff_path} or {base_csv_path}")
        return

    df = pd.read_csv(cutoff_path)
    df_base = pd.read_csv(base_csv_path)

    num_cores = os.cpu_count() or 1
    print(f"Normalizing parallel results using {num_cores} cores.")

    plot_utils.setup_matplotlib_style()
    latex_active = plot_utils.detect_latex()
    custom_rc = {
        "text.usetex": latex_active,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.titlesize": 9.5,
        "axes.titlesize": 8,
    }

    with plt.rc_context(custom_rc):
        # Create figure with shared y-axis per row, adapted for A4 page width
        fig, axs = plt.subplots(2, 4, figsize=(5.8, 3.8), sharex=True, sharey="row", dpi=300)

        configs = [
            # (row, col, title, value)
            (0, 0, "Cutoff = 256", 256),
            (0, 1, "Cutoff = 512", 512),
            (0, 2, "Cutoff = 1024", 1024),
            (0, 3, "Cutoff = 2048", 2048),
            (1, 0, "Cutoff = 4096", 4096),
            (1, 1, "Cutoff = 8192", 8192),
            (1, 2, "Cutoff = 16384", 16384),
        ]

        legend_handles = {}
        max_gflops_row = [0.0, 0.0]

        for row, col, title, value in configs:
            ax = axs[row, col]
            ax.set_facecolor("none")
            ax.set_xscale("log", base=2)
            ax.set_xlim(left=2, right=65536)
            ax.set_yscale("linear")
            ax.set_title(title, pad=4)
            ax.grid(False)

            n_base = df_base["size"]

            # 1. Plot MKL Base (Parallel, normalized)
            if (
                backend_filter not in ("faer", "strassen_only")
                and "mkl_par" in df_base.columns
                and not df_base["mkl_par"].isna().all()
            ):
                gflops_mkl = plot_utils.calculate_gflops_rust(n_base, df_base["mkl_par"], is_parallel=True, num_cores=num_cores)
                if not gflops_mkl.empty:
                    max_gflops_row[row] = max(max_gflops_row[row], float(gflops_mkl.max()))
                (l_mkl,) = ax.plot(
                    n_base,
                    gflops_mkl,
                    label="MKL dgemm",
                    color="#9467bd",
                    marker="o",
                    linestyle="--",
                    linewidth=0.8,
                    markersize=3.0,
                )
                legend_handles["MKL dgemm"] = l_mkl

            # 2. Plot faer Base (Parallel, normalized)
            if (
                backend_filter not in ("dgemm", "strassen_only")
                and "faer_par" in df_base.columns
                and not df_base["faer_par"].isna().all()
            ):
                gflops_faer = plot_utils.calculate_gflops_rust(n_base, df_base["faer_par"], is_parallel=True, num_cores=num_cores)
                if not gflops_faer.empty:
                    max_gflops_row[row] = max(max_gflops_row[row], float(gflops_faer.max()))
                (l_faer,) = ax.plot(
                    n_base,
                    gflops_faer,
                    label="faer (Parallel)",
                    color="#17becf",
                    marker="^",
                    linestyle="--",
                    linewidth=0.8,
                    markersize=3.0,
                )
                legend_handles["faer (Parallel)"] = l_faer

            # 3. Filter Strassen results
            df_config = df[(df["size_cutoff"] == value) & (df["recursion_level"].isna())]

            df_dgemm = df_config[df_config["base_choice"] == "dgemm"]
            df_faer_strassen = df_config[df_config["base_choice"] == "faer"]

            # Plot Rust dgemm base Strassen curves (DFS, BFS, Hybrid)
            if backend_filter != "faer" and not df_dgemm.empty:
                if "strassen_dfs" in df_dgemm.columns and not df_dgemm["strassen_dfs"].isna().all():
                    gflops_dfs = plot_utils.calculate_gflops_rust(
                        df_dgemm["size"], df_dgemm["strassen_dfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops_dfs.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_dfs.max()))
                    (l_dfs_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_dfs,
                        label="Strassen DFS (dgemm base)",
                        color=plot_utils.RUST_GRID_COLORS["par_dfs_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["Strassen DFS (dgemm base)"] = l_dfs_dg

                if "strassen_bfs" in df_dgemm.columns and not df_dgemm["strassen_bfs"].isna().all():
                    gflops_bfs = plot_utils.calculate_gflops_rust(
                        df_dgemm["size"], df_dgemm["strassen_bfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops_bfs.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_bfs.max()))
                    (l_bfs_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_bfs,
                        label="Strassen BFS (dgemm base)",
                        color=plot_utils.RUST_GRID_COLORS["par_bfs_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["Strassen BFS (dgemm base)"] = l_bfs_dg

                if "strassen_hybrid" in df_dgemm.columns and not df_dgemm["strassen_hybrid"].isna().all():
                    gflops_hybrid = plot_utils.calculate_gflops_rust(
                        df_dgemm["size"], df_dgemm["strassen_hybrid"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops_hybrid.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_hybrid.max()))
                    (l_hybrid_dg,) = ax.plot(
                        df_dgemm["size"],
                        gflops_hybrid,
                        label="Strassen Hybrid (dgemm base)",
                        color=plot_utils.RUST_GRID_COLORS["par_hybrid_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["Strassen Hybrid (dgemm base)"] = l_hybrid_dg

            # Plot Rust faer base Strassen curves (DFS, BFS, Hybrid)
            if backend_filter != "dgemm" and not df_faer_strassen.empty:
                if "strassen_dfs" in df_faer_strassen.columns and not df_faer_strassen["strassen_dfs"].isna().all():
                    gflops_dfs_fs = plot_utils.calculate_gflops_rust(
                        df_faer_strassen["size"], df_faer_strassen["strassen_dfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops_dfs_fs.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_dfs_fs.max()))
                    (l_dfs_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_dfs_fs,
                        label="Strassen DFS (faer base)",
                        color=plot_utils.RUST_GRID_COLORS["par_dfs_faer"],
                        marker="^",
                        linestyle="-",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["Strassen DFS (faer base)"] = l_dfs_fs

                if "strassen_bfs" in df_faer_strassen.columns and not df_faer_strassen["strassen_bfs"].isna().all():
                    gflops_bfs_fs = plot_utils.calculate_gflops_rust(
                        df_faer_strassen["size"], df_faer_strassen["strassen_bfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops_bfs_fs.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_bfs_fs.max()))
                    (l_bfs_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_bfs_fs,
                        label="Strassen BFS (faer base)",
                        color=plot_utils.RUST_GRID_COLORS["par_bfs_faer"],
                        marker="v",
                        linestyle="-",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["Strassen BFS (faer base)"] = l_bfs_fs

                if "strassen_hybrid" in df_faer_strassen.columns and not df_faer_strassen["strassen_hybrid"].isna().all():
                    gflops_hybrid_fs = plot_utils.calculate_gflops_rust(
                        df_faer_strassen["size"], df_faer_strassen["strassen_hybrid"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops_hybrid_fs.empty:
                        max_gflops_row[row] = max(max_gflops_row[row], float(gflops_hybrid_fs.max()))
                    (l_hybrid_fs,) = ax.plot(
                        df_faer_strassen["size"],
                        gflops_hybrid_fs,
                        label="Strassen Hybrid (faer base)",
                        color=plot_utils.RUST_GRID_COLORS["par_hybrid_faer"],
                        marker="<",
                        linestyle="-",
                        linewidth=0.8,
                        markersize=3.0,
                    )
                    legend_handles["Strassen Hybrid (faer base)"] = l_hybrid_fs

            ax.set_xticks(df_base["size"])
            ax.tick_params("x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Set dynamic tight y-limits across subplots per row to reduce blank space
        for r in range(2):
            if max_gflops_row[r] > 0:
                for c in range(4):
                    axs[r, c].set_ylim(0, max_gflops_row[r] * 1.05)

        # Add labels on outer plots
        for col in range(4):
            axs[1, col].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=4)
        axs[1, 3].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=4)

        for row in range(2):
            axs[row, 0].set_ylabel("Effective GFLOPS / core", labelpad=4)

        # Legend axis configuration
        legend_ax = axs[1, 3]
        legend_ax.axis("off")

        if backend_filter == "faer":
            sorted_labels = [
                "faer (Parallel)",
                "Strassen DFS (faer base)",
                "Strassen BFS (faer base)",
                "Strassen Hybrid (faer base)",
            ]
            ncol = 1
        elif backend_filter == "dgemm":
            sorted_labels = [
                "MKL dgemm",
                "Strassen DFS (dgemm base)",
                "Strassen BFS (dgemm base)",
                "Strassen Hybrid (dgemm base)",
            ]
            ncol = 1
        elif backend_filter == "strassen_only":
            sorted_labels = [
                "Strassen DFS (dgemm base)",
                "Strassen BFS (dgemm base)",
                "Strassen Hybrid (dgemm base)",
                "Strassen DFS (faer base)",
                "Strassen BFS (faer base)",
                "Strassen Hybrid (faer base)",
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
            loc="center left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#cbd5e1",
            ncol=ncol,
            bbox_to_anchor=(-0.15, 0.5) if ncol == 2 else (0.05, 0.5),
        )

        title_text = "Parallel Matrix Multiplication Performance Comparison (Normalized per Core) - All Cutoffs"
        if backend_filter == "strassen_only":
            title_text = f"{title_text} (Strassen Variants)"
        elif backend_filter:
            title_text = f"{title_text} ({backend_filter.upper()} Backend)"

        # plt.suptitle(title_text, y=0.98)
        fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.88, bottom=0.15, left=0.09, right=0.97)

        # Save outputs
        out_dir_plots = os.path.join(project_root, "generated", "plots")
        report_figures_dir = os.path.join(project_root, "report", "figures")
        out_name = "parallel_cutoff_grid_plot"
        if backend_filter:
            out_name = f"{out_name}_{backend_filter}"

        for d in (out_dir_plots, report_figures_dir):
            os.makedirs(d, exist_ok=True)
            pdf_path_plots = os.path.join(d, f"{out_name}.pdf")
            plot_utils.save_plot(fig, pdf_path_plots)
        plt.close(fig)


def plot_compare_ballard(project_root: str, par_dir: str = "run_par") -> None:
    """Generates and saves a 4x3 grid plot comparing Rust Strassen to Ballard references.

    Enforces row-specific tight y-limits starting at 0 to minimize blank space.
    Rows: Sequential, DFS, BFS, Hybrid. Columns: Level 1, Level 2, Level 3.
    A single common legend is placed below the 3 subplots of each row (centered horizontally).

    Args:
        project_root: The root directory of the project.
        par_dir: Directory name under generated/csv/ for parallel results (default: 'run_par').
    """
    csv_seq_dir = "run_seq"
    csv_par_dir = par_dir

    seq_cutoff_path = os.path.join(project_root, "generated", "csv", csv_seq_dir, "benchmark_results_cutoff.csv")
    seq_levels_path = os.path.join(project_root, "generated", "csv", csv_seq_dir, "benchmark_results_levels.csv")
    base_seq_path = os.path.join(project_root, "generated", "csv", csv_seq_dir, "benchmark_results_base.csv")

    seq_dfs = []
    if os.path.exists(seq_cutoff_path):
        seq_dfs.append(pd.read_csv(seq_cutoff_path))
    if os.path.exists(seq_levels_path):
        seq_dfs.append(pd.read_csv(seq_levels_path))
    if not seq_dfs:
        seq_std_path = os.path.join(project_root, "generated", "csv", csv_seq_dir, "benchmark_results.csv")
        if os.path.exists(seq_std_path):
            seq_dfs.append(pd.read_csv(seq_std_path))

    par_cutoff_path = os.path.join(project_root, "generated", "csv", "run_par2", "benchmark_results_cutoff.csv")
    par_levels_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results_levels.csv")
    base_par_path = os.path.join(project_root, "generated", "csv", csv_par_dir, "benchmark_results_base.csv")

    if not os.path.exists(base_par_path):
        base_par_path = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results_base.csv")
        if not os.path.exists(base_par_path):
            base_par_path = os.path.join(project_root, "generated", "csv", "run_par2", "benchmark_results_base.csv")

    par_dfs = []
    if os.path.exists(par_cutoff_path):
        par_dfs.append(pd.read_csv(par_cutoff_path))
    if os.path.exists(par_levels_path):
        par_dfs.append(pd.read_csv(par_levels_path))
    if not par_dfs:
        par_std_path = os.path.join(project_root, "generated", "csv", csv_par_dir, "benchmark_results.csv")
        if os.path.exists(par_std_path):
            par_dfs.append(pd.read_csv(par_std_path))

    if not seq_dfs or not par_dfs or not os.path.exists(base_seq_path) or not os.path.exists(base_par_path):
        print("Error: Missing input CSVs for Ballard comparison plot.")
        return

    df_seq = pd.concat(seq_dfs, ignore_index=True)
    df_base_seq = pd.read_csv(base_seq_path)
    df_par = pd.concat(par_dfs, ignore_index=True)
    df_base_par = pd.read_csv(base_par_path)

    num_cores = os.cpu_count() or 1
    print(f"Normalizing parallel results using {num_cores} cores.")

    # Load reference Ballard files
    ballard_seq_levels = []
    ballard_dfs_levels = []
    ballard_bfs_levels = []
    ballard_hybrid_levels = []

    data_seq = plot_utils.load_ballard_data(project_root, "seq")
    data_dfs = plot_utils.load_ballard_data(project_root, "dfs")
    data_bfs = plot_utils.load_ballard_data(project_root, "bfs")
    data_hybrid = plot_utils.load_ballard_data(project_root, "hybrid")

    for l_idx in range(3):
        ballard_seq_levels.append(data_seq["strassen"][l_idx] if data_seq else None)
        ballard_dfs_levels.append(data_dfs["strassen"][l_idx] if data_dfs else None)
        ballard_bfs_levels.append(data_bfs["strassen"][l_idx] if data_bfs else None)
        ballard_hybrid_levels.append(data_hybrid["strassen"][l_idx] if data_hybrid else None)

    plot_utils.setup_matplotlib_style()
    latex_active = plot_utils.detect_latex()
    custom_rc = {
        "text.usetex": latex_active,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.title_fontsize": 7.5,
    }
    with plt.rc_context(custom_rc):
        fig, axs = plt.subplots(4, 3, figsize=(5.8, 8.0), sharex=True, sharey="row", dpi=300)

        max_gflops_seq = 0.0
        max_gflops_dfs = 0.0
        max_gflops_bfs = 0.0
        max_gflops_hybrid = 0.0

        for col in range(3):
            level = col + 1

            # --- Row 0: Sequential ---
            ax_seq = axs[0, col]
            ax_seq.set_facecolor("none")
            ax_seq.set_xscale("log", base=2)
            ax_seq.set_xlim(left=2, right=65536)
            ax_seq.set_yscale("linear")
            ax_seq.set_title(f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4)
            ax_seq.grid(False)

            df_config_seq = df_seq[(df_seq["recursion_level"] == float(level)) & (df_seq["size_cutoff"].isna())]
            df_dg_seq = df_config_seq[df_config_seq["base_choice"] == "dgemm"]
            df_fs_seq = df_config_seq[df_config_seq["base_choice"] == "faer"]

            if not df_dg_seq.empty:
                gflops = plot_utils.calculate_gflops_rust(df_dg_seq["size"], df_dg_seq["strassen_seq"])
                if not gflops.empty:
                    max_gflops_seq = max(max_gflops_seq, float(gflops.max()))
                ax_seq.plot(
                    df_dg_seq["size"],
                    gflops,
                    label="Rust Strassen (dgemm)",
                    color=plot_utils.RUST_GRID_COLORS["seq_dgemm"],
                    marker="o",
                    linestyle="-",
                    linewidth=1.0,
                    markersize=3.5,
                )

            if not df_fs_seq.empty:
                gflops = plot_utils.calculate_gflops_rust(df_fs_seq["size"], df_fs_seq["strassen_seq"])
                if not gflops.empty:
                    max_gflops_seq = max(max_gflops_seq, float(gflops.max()))
                ax_seq.plot(
                    df_fs_seq["size"],
                    gflops,
                    label="Rust Strassen (faer)",
                    color=plot_utils.RUST_GRID_COLORS["seq_faer"],
                    marker="^",
                    linestyle="-",
                    linewidth=1.0,
                    markersize=3.5,
                )

            b_seq = ballard_seq_levels[col]
            if b_seq is not None:
                gflops_b = plot_utils.calculate_gflops_ballard(b_seq["size"], b_seq["time_ms"])
                if not gflops_b.empty:
                    max_gflops_seq = max(max_gflops_seq, float(gflops_b.max()))
                ax_seq.plot(
                    b_seq["size"],
                    gflops_b,
                    label="Ballard Strassen",
                    color=plot_utils.BALLARD_GRID_COLORS["seq"],
                    marker="X",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # --- Row 1: DFS ---
            ax_dfs = axs[1, col]
            ax_dfs.set_facecolor("none")
            ax_dfs.set_xscale("log", base=2)
            ax_dfs.set_xlim(left=2, right=65536)
            ax_dfs.set_yscale("linear")
            ax_dfs.set_title(f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4)
            ax_dfs.grid(False)

            df_config_par = df_par[(df_par["recursion_level"] == float(level)) & (df_par["size_cutoff"].isna())]
            df_dg_par = df_config_par[df_config_par["base_choice"] == "dgemm"]
            df_fs_par = df_config_par[df_config_par["base_choice"] == "faer"]

            if not df_dg_par.empty:
                if "strassen_dfs" in df_dg_par.columns and not df_dg_par["strassen_dfs"].isna().all():
                    gflops = plot_utils.calculate_gflops_rust(
                        df_dg_par["size"], df_dg_par["strassen_dfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops.empty:
                        max_gflops_dfs = max(max_gflops_dfs, float(gflops.max()))
                    ax_dfs.plot(
                        df_dg_par["size"],
                        gflops,
                        label="Rust Strassen DFS (dgemm)",
                        color=plot_utils.RUST_GRID_COLORS["par_dfs_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            if not df_fs_par.empty:
                if "strassen_dfs" in df_fs_par.columns and not df_fs_par["strassen_dfs"].isna().all():
                    gflops = plot_utils.calculate_gflops_rust(
                        df_fs_par["size"], df_fs_par["strassen_dfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops.empty:
                        max_gflops_dfs = max(max_gflops_dfs, float(gflops.max()))
                    ax_dfs.plot(
                        df_fs_par["size"],
                        gflops,
                        label="Rust Strassen DFS (faer)",
                        color=plot_utils.RUST_GRID_COLORS["par_dfs_faer"],
                        marker="^",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            b_dfs = ballard_dfs_levels[col]
            if b_dfs is not None:
                gflops_b = plot_utils.calculate_gflops_ballard(
                    b_dfs["size"], b_dfs["time_ms"], is_parallel=True, num_cores=num_cores
                )
                if not gflops_b.empty:
                    max_gflops_dfs = max(max_gflops_dfs, float(gflops_b.max()))
                ax_dfs.plot(
                    b_dfs["size"],
                    gflops_b,
                    label="Ballard DFS",
                    color=plot_utils.BALLARD_GRID_COLORS["dfs"],
                    marker="X",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # --- Row 2: BFS ---
            ax_bfs = axs[2, col]
            ax_bfs.set_facecolor("none")
            ax_bfs.set_xscale("log", base=2)
            ax_bfs.set_xlim(left=2, right=65536)
            ax_bfs.set_yscale("linear")
            ax_bfs.set_title(f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4)
            ax_bfs.grid(False)

            if not df_dg_par.empty:
                if "strassen_bfs" in df_dg_par.columns and not df_dg_par["strassen_bfs"].isna().all():
                    gflops = plot_utils.calculate_gflops_rust(
                        df_dg_par["size"], df_dg_par["strassen_bfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops.empty:
                        max_gflops_bfs = max(max_gflops_bfs, float(gflops.max()))
                    ax_bfs.plot(
                        df_dg_par["size"],
                        gflops,
                        label="Rust Strassen BFS (dgemm)",
                        color=plot_utils.RUST_GRID_COLORS["par_bfs_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            if not df_fs_par.empty:
                if "strassen_bfs" in df_fs_par.columns and not df_fs_par["strassen_bfs"].isna().all():
                    gflops = plot_utils.calculate_gflops_rust(
                        df_fs_par["size"], df_fs_par["strassen_bfs"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops.empty:
                        max_gflops_bfs = max(max_gflops_bfs, float(gflops.max()))
                    ax_bfs.plot(
                        df_fs_par["size"],
                        gflops,
                        label="Rust Strassen BFS (faer)",
                        color=plot_utils.RUST_GRID_COLORS["par_bfs_faer"],
                        marker="v",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            b_bfs = ballard_bfs_levels[col]
            if b_bfs is not None:
                gflops_b = plot_utils.calculate_gflops_ballard(
                    b_bfs["size"], b_bfs["time_ms"], is_parallel=True, num_cores=num_cores
                )
                if not gflops_b.empty:
                    max_gflops_bfs = max(max_gflops_bfs, float(gflops_b.max()))
                ax_bfs.plot(
                    b_bfs["size"],
                    gflops_b,
                    label="Ballard BFS",
                    color=plot_utils.BALLARD_GRID_COLORS["bfs"],
                    marker="s",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # --- Row 3: Hybrid ---
            ax_hybrid = axs[3, col]
            ax_hybrid.set_facecolor("none")
            ax_hybrid.set_xscale("log", base=2)
            ax_hybrid.set_xlim(left=2, right=65536)
            ax_hybrid.set_yscale("linear")
            ax_hybrid.set_title(f"Level = {level}", fontsize=8.0, fontweight="normal", pad=4)
            ax_hybrid.grid(False)

            if not df_dg_par.empty:
                if "strassen_hybrid" in df_dg_par.columns and not df_dg_par["strassen_hybrid"].isna().all():
                    gflops = plot_utils.calculate_gflops_rust(
                        df_dg_par["size"], df_dg_par["strassen_hybrid"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops.empty:
                        max_gflops_hybrid = max(max_gflops_hybrid, float(gflops.max()))
                    ax_hybrid.plot(
                        df_dg_par["size"],
                        gflops,
                        label="Rust Strassen Hybrid (dgemm)",
                        color=plot_utils.RUST_GRID_COLORS["par_hybrid_dgemm"],
                        marker="o",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            if not df_fs_par.empty:
                if "strassen_hybrid" in df_fs_par.columns and not df_fs_par["strassen_hybrid"].isna().all():
                    gflops = plot_utils.calculate_gflops_rust(
                        df_fs_par["size"], df_fs_par["strassen_hybrid"], is_parallel=True, num_cores=num_cores
                    )
                    if not gflops.empty:
                        max_gflops_hybrid = max(max_gflops_hybrid, float(gflops.max()))
                    ax_hybrid.plot(
                        df_fs_par["size"],
                        gflops,
                        label="Rust Strassen Hybrid (faer)",
                        color=plot_utils.RUST_GRID_COLORS["par_hybrid_faer"],
                        marker="<",
                        linestyle="-",
                        linewidth=1.0,
                        markersize=3.5,
                    )

            b_hybrid = ballard_hybrid_levels[col]
            if b_hybrid is not None:
                gflops_b = plot_utils.calculate_gflops_ballard(
                    b_hybrid["size"], b_hybrid["time_ms"], is_parallel=True, num_cores=num_cores
                )
                if not gflops_b.empty:
                    max_gflops_hybrid = max(max_gflops_hybrid, float(gflops_b.max()))
                ax_hybrid.plot(
                    b_hybrid["size"],
                    gflops_b,
                    label="Ballard Hybrid",
                    color=plot_utils.BALLARD_GRID_COLORS["hybrid"],
                    marker="D",
                    linestyle=":",
                    linewidth=1.2,
                    markersize=4.0,
                )

            # Standard formats for X axis ticks
            ax_seq.set_xticks(df_base_seq["size"])
            ax_seq.tick_params("x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5)
            ax_seq.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax_dfs.set_xticks(df_base_par["size"])
            ax_dfs.tick_params("x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5)
            ax_dfs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax_bfs.set_xticks(df_base_par["size"])
            ax_bfs.tick_params("x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5)
            ax_bfs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            ax_hybrid.set_xticks(df_base_par["size"])
            ax_hybrid.tick_params("x", labelbottom=True, rotation=70, rotation_mode="xtick", labelsize=5)
            ax_hybrid.get_xaxis().set_major_formatter(plt.ScalarFormatter())

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

        for r in range(4):
            handles, labels = axs[r, 1].get_legend_handles_labels()
            anchor_y = -0.32 if r == 3 else -0.22
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

        # plt.suptitle(
        #     "Strassen Matrix Multiplication: Rust vs Ballard Reference Comparison",
        #     fontsize=10.0,
        #     fontweight="bold",
        #     y=0.98,
        # )

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.58, wspace=0.22, top=0.92, bottom=0.08, left=0.13, right=0.96)

        out_dir_plots = os.path.join(project_root, "generated", "plots")
        report_figures_dir = os.path.join(project_root, "report", "figures")

        for d in (out_dir_plots, report_figures_dir):
            plot_utils.save_plot(fig, os.path.join(d, "compare_ballard.pdf"))

        plt.close(fig)


def main() -> None:
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
            "all",
            "seq",
            "par",
            "compare_ballard",
            "ballard",
            "cutoff_grid",
        ],
        default="all",
        help="Plotting mode: 'sequential' ('seq'), 'parallel' ('par'), 'compare_ballard' ('ballard'), 'cutoff_grid', 'both', or 'all' (default).",
    )
    parser.add_argument(
        "--par-dir",
        default="run_par",
        help="Directory name under generated/csv/ for parallel results (default: 'run_par').",
    )
    args = parser.parse_args()

    modes_to_run = []
    if args.mode in ("sequential", "seq", "both", "all"):
        modes_to_run.append("sequential")
    if args.mode in ("parallel", "par", "both", "all"):
        modes_to_run.append("parallel")
    if args.mode in ("compare_ballard", "ballard", "both", "all"):
        modes_to_run.append("compare_ballard")
    if args.mode in ("cutoff_grid", "all", "both"):
        modes_to_run.append("cutoff_grid")

    for mode in modes_to_run:
        if mode == "compare_ballard":
            plot_compare_ballard(project_root, par_dir=args.par_dir)
        elif mode == "parallel":
            plot_mode_grid(project_root, "parallel", par_dir=args.par_dir, backend_filter="faer")
            plot_mode_grid(project_root, "parallel", par_dir=args.par_dir, backend_filter="dgemm")
            plot_mode_grid(project_root, "parallel", par_dir=args.par_dir, backend_filter="strassen_only")
        elif mode == "sequential":
            plot_mode_grid(project_root, "sequential")
        elif mode == "cutoff_grid":
            plot_cutoff_grid(project_root, par_dir="run_par2", backend_filter="faer")
            plot_cutoff_grid(project_root, par_dir="run_par2", backend_filter="dgemm")
            plot_cutoff_grid(project_root, par_dir="run_par2", backend_filter="strassen_only")
        else:
            plot_mode_grid(project_root, mode, par_dir=args.par_dir)


if __name__ == "__main__":
    main()
