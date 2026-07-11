# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
#   "numpy",
# ]
# ///

"""Benchmark plotting script for comparing MKL and faer sequential implementations.

This script parses the base benchmark CSV file and Ballard reference data,
and plots the performance (Effective GFLOPS vs Matrix Size) comparing
Rust MKL sequential, Rust faer sequential, and the Ballard MKL reference line.
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
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,
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

    base_csv_path = os.path.join(
        project_root, "generated", "csv", "benchmark_results_base.csv"
    )
    ballard_path = os.path.join(
        project_root, "benchmarks", "generated", "benchmarks_seq.txt"
    )

    if not os.path.exists(base_csv_path):
        print(f"Error: Missing input base CSV at {base_csv_path}")
        return

    df_base = pd.read_csv(base_csv_path)

    # Load Ballard seq reference data
    ballard_mkl = parse_matlab_vector(ballard_path, "MKL_0")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.set_facecolor("none")
    ax.set_xscale("log", base=2)
    ax.set_yscale("linear")

    # 1. Plot MKL Base (Sequential)
    n_base = df_base["size"]
    flops_base = 2 * n_base**3 - n_base**2
    
    if "mkl_seq" in df_base.columns and not df_base["mkl_seq"].isna().all():
        gflops_mkl = flops_base / (df_base["mkl_seq"] * 1e9)
        ax.plot(
            n_base,
            gflops_mkl,
            label="MKL (Sequential)",
            color="#9467bd",
            marker="p",
            linestyle="--",
            linewidth=1.2,
            markersize=4.5,
        )

    # 2. Plot faer Base (Sequential)
    if "faer_seq" in df_base.columns and not df_base["faer_seq"].isna().all():
        gflops_faer = flops_base / (df_base["faer_seq"] * 1e9)
        ax.plot(
            n_base,
            gflops_faer,
            label="faer (Sequential)",
            color="#17becf",
            marker="d",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
        )

    # 3. Plot Ballard MKL (Sequential)
    if ballard_mkl is not None:
        time_s = ballard_mkl["time_ms"] / 1000.0
        n_b = ballard_mkl["size"]
        gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9)
        ax.plot(
            n_b,
            gflops_b,
            label="MKL (Ballard)",
            color="#e41a1c",
            marker="X",
            linestyle="--",
            linewidth=1.5,
            markersize=5.0,
        )

    ax.set_xticks(df_base["size"])
    ax.tick_params("x", labelbottom=True, rotation=30, rotation_mode="xtick")
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)
    ax.set_ylabel("Effective GFLOPS", labelpad=10)
    ax.set_title("Sequential Matrix Multiplication Performance Comparison\nMKL, faer, and MKL (Ballard)", pad=15)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cbd5e1"
    )

    plt.tight_layout()

    out_dir_plots = os.path.join(project_root, "generated", "plots")
    os.makedirs(out_dir_plots, exist_ok=True)

    pdf_path = os.path.join(out_dir_plots, "mkl_faer_seq_comparison.pdf")
    png_path = os.path.join(out_dir_plots, "mkl_faer_seq_comparison.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")

    print(f"Comparison plot saved successfully:")
    print(f"  - {pdf_path}")
    print(f"  - {png_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
