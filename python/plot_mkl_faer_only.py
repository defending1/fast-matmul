# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
#   "numpy",
# ]
# ///

"""Benchmark plotting script for comparing MKL and faer sequential/parallel implementations side-by-side.

This script parses the base benchmark CSV files and C reference data for both sequential and parallel modes,
and plots the performance (Effective GFLOPS vs Matrix Size) comparing Rust MKL, Rust faer, and C MKL.
"""

import os
import re
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
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
        "text.usetex": latex_installed,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
    }
)

def parse_matlab_vector(filepath, vec_name):
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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 1. Paths for Sequential Data
    seq_base_csv_path = os.path.join(
        project_root, "generated", "csv", "run_seq", "benchmark_results_base.csv"
    )
    seq_ballard_path = os.path.join(
        project_root, "generated", "csv", "run_seq", "benchmarks_seq.txt"
    )

    # 2. Paths for Parallel Data
    par_base_csv_path = os.path.join(
        project_root, "generated", "csv", "run_par", "benchmark_results_base.csv"
    )
    par_c_path = os.path.join(
        project_root, "generated", "csv", "run_par", "benchmarks_dfs.txt"
    )

    # Core count for parallel GFLOPS normalization
    num_cores = os.cpu_count() or 1

    # Check that input CSV files exist
    if not os.path.exists(seq_base_csv_path):
        print(f"Error: Missing sequential base CSV at {seq_base_csv_path}")
        return
    if not os.path.exists(par_base_csv_path):
        print(f"Error: Missing parallel base CSV at {par_base_csv_path}")
        return

    # Load data
    df_base_seq = pd.read_csv(seq_base_csv_path)
    df_base_par = pd.read_csv(par_base_csv_path)
    ballard_mkl_seq = parse_matlab_vector(seq_ballard_path, "MKL_0")
    ballard_mkl_par = parse_matlab_vector(par_c_path, "MKL_0")

    # Create figure (side-by-side subplots) adapted for LaTeX text width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=300)

    # ==================== SUBPLOT 1: SEQUENTIAL ====================
    ax1.set_facecolor("none")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("linear")

    n_base_seq = df_base_seq["size"]
    flops_base_seq = 2 * n_base_seq**3 - n_base_seq**2
    
    # Rust MKL Sequential
    if "mkl_seq" in df_base_seq.columns and not df_base_seq["mkl_seq"].isna().all():
        gflops_mkl = flops_base_seq / (df_base_seq["mkl_seq"] * 1e9)
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

    # Rust faer Sequential
    if "faer_seq" in df_base_seq.columns and not df_base_seq["faer_seq"].isna().all():
        gflops_faer = flops_base_seq / (df_base_seq["faer_seq"] * 1e9)
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

    # C MKL Sequential (Ballard)
    if ballard_mkl_seq is not None:
        time_s = ballard_mkl_seq["time_ms"] / 1000.0
        n_b = ballard_mkl_seq["size"]
        gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9)
        ax1.plot(
            n_b,
            gflops_b,
            label="Ballard MKL dgemm (sequential)",
            color="#e41a1c",
            marker="X",
            linestyle="--",
            linewidth=1.5,
            markersize=5.5,
        )

    # Use a clean subset of power-of-two ticks to avoid overlapping text
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
    flops_base_par = 2 * n_base_par**3 - n_base_par**2
    
    # Rust MKL Parallel
    if "mkl_par" in df_base_par.columns and not df_base_par["mkl_par"].isna().all():
        gflops_mkl_par = flops_base_par / (df_base_par["mkl_par"] * 1e9) / num_cores
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

    # Rust faer Parallel
    if "faer_par" in df_base_par.columns and not df_base_par["faer_par"].isna().all():
        gflops_faer_par = flops_base_par / (df_base_par["faer_par"] * 1e9) / num_cores
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

    # C MKL Parallel (Ballard)
    if ballard_mkl_par is not None:
        time_s = ballard_mkl_par["time_ms"] / 1000.0
        n_b = ballard_mkl_par["size"]
        gflops_b_par = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9) / num_cores
        ax2.plot(
            n_b,
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

    # In LaTeX documents, titles belong in the caption. No suptitle is added.
    plt.tight_layout()

    # Save to multiple targets
    out_dir_plots = os.path.join(project_root, "generated", "plots")
    report_figures_dir = os.path.join(project_root, "report", "figures")
    os.makedirs(out_dir_plots, exist_ok=True)
    os.makedirs(report_figures_dir, exist_ok=True)

    paths_to_save = [
        os.path.join(out_dir_plots, "mkl_faer_comparison.pdf"),
        os.path.join(out_dir_plots, "mkl_faer_comparison.png"),
        os.path.join(report_figures_dir, "mkl_faer_seq_comparison.pdf"),
        os.path.join(report_figures_dir, "mkl_faer_seq_comparison.png"),
    ]

    for path in paths_to_save:
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)

if __name__ == "__main__":
    main()
