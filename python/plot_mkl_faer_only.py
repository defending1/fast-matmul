# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
#   "numpy",
# ]
# ///

"""Benchmark plotting script for comparing MKL and faer implementations.

Generates a side-by-side comparative GFLOPS visualization for sequential
and parallel execution modes, incorporating reference Ballard metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

import plot_utils


def main() -> None:
    """Loads baseline results, generates comparison subplots, and saves plots."""
    import argparse
    parser = argparse.ArgumentParser(description="Plot MKL vs faer comparison.")
    parser.add_argument(
        "--par-dir",
        default="run_par",
        help="Directory name under generated/csv/ for parallel results (default: 'run_par')."
    )
    args = parser.parse_args()
    par_dir = args.par_dir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 1. Paths for Sequential Data
    seq_base_csv_path = os.path.join(project_root, "generated", "csv", "run_seq", "benchmark_results_base.csv")
    seq_ballard_path = os.path.join(project_root, "generated", "csv", "run_seq", "benchmarks_seq.txt")

    # 2. Paths for Parallel Data
    par_base_csv_path = os.path.join(project_root, "generated", "csv", par_dir, "benchmark_results_base.csv")
    par_c_path = os.path.join(project_root, "generated", "csv", par_dir, "benchmarks_dfs.txt")
    if not os.path.exists(par_c_path):
        par_c_path = os.path.join(project_root, "benchmarks", "generated", "benchmarks_dfs.txt")

    num_cores = os.cpu_count() or 1

    if not os.path.exists(seq_base_csv_path):
        print(f"Error: Missing sequential base CSV at {seq_base_csv_path}")
        return
    if not os.path.exists(par_base_csv_path):
        print(f"Error: Missing parallel base CSV at {par_base_csv_path}")
        return

    # Load data
    df_base_seq = pd.read_csv(seq_base_csv_path)
    df_base_par = pd.read_csv(par_base_csv_path)
    ballard_mkl_seq = plot_utils.parse_matlab_vector(seq_ballard_path, "MKL_0")
    ballard_mkl_par = plot_utils.parse_matlab_vector(par_c_path, "MKL_0")

    # Establish LaTeX font environment
    plot_utils.setup_matplotlib_style()

    # Create figure (side-by-side subplots) adapted for LaTeX text width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=300)

    # ==================== SUBPLOT 1: SEQUENTIAL ====================
    ax1.set_facecolor("none")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("linear")

    n_base_seq = df_base_seq["size"]

    if "mkl_seq" in df_base_seq.columns and not df_base_seq["mkl_seq"].isna().all():
        gflops_mkl = plot_utils.calculate_gflops_rust(n_base_seq, df_base_seq["mkl_seq"])
        ax1.plot(
            n_base_seq,
            gflops_mkl,
            label="Rust MKL dgemm (sequential)",
            color="#9467bd",
            marker="o",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    if "faer_seq" in df_base_seq.columns and not df_base_seq["faer_seq"].isna().all():
        gflops_faer = plot_utils.calculate_gflops_rust(n_base_seq, df_base_seq["faer_seq"])
        ax1.plot(
            n_base_seq,
            gflops_faer,
            label="Rust faer (sequential)",
            color="#17becf",
            marker="^",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    if ballard_mkl_seq is not None:
        gflops_b = plot_utils.calculate_gflops_ballard(ballard_mkl_seq["size"], ballard_mkl_seq["time_ms"])
        ax1.plot(
            ballard_mkl_seq["size"],
            gflops_b,
            label="Ballard MKL dgemm (sequential)",
            color="#e41a1c",
            marker="X",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    ticks_to_show = [4, 16, 64, 256, 1024, 4096, 16384, 65536]
    ax1.set_xticks(ticks_to_show)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.tick_params(axis="x", labelbottom=True, rotation=0)
    ax1.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=6)
    ax1.set_ylabel("Effective GFLOPS", labelpad=6)
    ax1.set_title("Sequential Execution", pad=8)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.5, edgecolor="none")

    # ==================== SUBPLOT 2: PARALLEL ====================
    ax2.set_facecolor("none")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("linear")

    n_base_par = df_base_par["size"]

    if "mkl_par" in df_base_par.columns and not df_base_par["mkl_par"].isna().all():
        gflops_mkl_par = plot_utils.calculate_gflops_rust(
            n_base_par, df_base_par["mkl_par"], is_parallel=True, num_cores=num_cores
        )
        ax2.plot(
            n_base_par,
            gflops_mkl_par,
            label="Rust MKL dgemm (parallel)",
            color="#9467bd",
            marker="o",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    if "faer_par" in df_base_par.columns and not df_base_par["faer_par"].isna().all():
        gflops_faer_par = plot_utils.calculate_gflops_rust(
            n_base_par, df_base_par["faer_par"], is_parallel=True, num_cores=num_cores
        )
        ax2.plot(
            n_base_par,
            gflops_faer_par,
            label="Rust faer (parallel)",
            color="#17becf",
            marker="^",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    if ballard_mkl_par is not None:
        gflops_b_par = plot_utils.calculate_gflops_ballard(
            ballard_mkl_par["size"], ballard_mkl_par["time_ms"], is_parallel=True, num_cores=num_cores
        )
        ax2.plot(
            ballard_mkl_par["size"],
            gflops_b_par,
            label="Ballard MKL dgemm (parallel)",
            color="#e41a1c",
            marker="X",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    ax2.set_xticks(ticks_to_show)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.tick_params(axis="x", labelbottom=True, rotation=0)
    ax2.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=6)
    ax2.set_ylabel("Effective GFLOPS / core", labelpad=6)
    ax2.set_title("Parallel Execution (Normalized per Core)", pad=8)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.5, edgecolor="none")

    plt.tight_layout()

    out_dir_plots = os.path.join(project_root, "generated", "plots")
    report_figures_dir = os.path.join(project_root, "report", "figures")

    for directory in (out_dir_plots, report_figures_dir):
        plot_utils.save_plot(fig, os.path.join(directory, "mkl_faer_comparison.pdf"))

    plt.close(fig)


if __name__ == "__main__":
    main()
