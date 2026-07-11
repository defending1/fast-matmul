#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scienceplots",
#   "numpy",
# ]
# ///

"""Benchmark plotting script for sequential matrix multiplication algorithms.
Generates a 2-row grid plot:
- Row 1: Cutoffs [256, 512, 1024, 2048]
- Row 2: Levels [1, 2, 3]
All subplots are sequential only and compare MKL, faer, and Strassen variants.
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
        "axes.titlesize": 12,
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

    csv_path = os.path.join(
        project_root, "generated", "csv", "run1", "benchmark_results.csv"
    )
    base_csv_path = os.path.join(
        project_root, "generated", "csv", "run1", "benchmark_results_base.csv"
    )
    ballard_path = os.path.join(
        project_root, "benchmarks", "generated", "benchmarks_seq.txt"
    )

    if not os.path.exists(csv_path) or not os.path.exists(base_csv_path):
        print(f"Error: Missing input CSVs at {csv_path} or {base_csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_base = pd.read_csv(base_csv_path)

    # Load Ballard seq reference data
    ballard_mkl = parse_matlab_vector(ballard_path, "MKL_0")
    ballard_strassen_1 = parse_matlab_vector(ballard_path, "STRASSEN_1")
    ballard_strassen_2 = parse_matlab_vector(ballard_path, "STRASSEN_2")
    ballard_strassen_3 = parse_matlab_vector(ballard_path, "STRASSEN_3")
    ballard_strassen_levels = [
        ballard_strassen_1,
        ballard_strassen_2,
        ballard_strassen_3,
    ]

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

        # 1. Plot MKL Base (Sequential)
        n_base = df_base["size"]
        mkl_flops = 2 * n_base**3 - n_base**2
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

        # 4. Plot Ballard Reference lines if loaded
        if ballard_mkl is not None:
            time_s = ballard_mkl["time_ms"] / 1000.0
            n_b = ballard_mkl["size"]
            gflops_b = (2 * n_b**3 - 2 * n_b**2) / (time_s * 1e9)
            (l_b_mkl,) = ax.plot(
                n_b,
                gflops_b,
                label="MKL (Ballard)",
                color="#e41a1c",
                marker="X",
                linestyle="--",
                linewidth=1.5,
                markersize=5.0,
            )
            legend_handles["MKL (Ballard)"] = l_b_mkl

        if not is_cutoff:
            level = int(value)
            b_strassen = ballard_strassen_levels[level - 1]
            if b_strassen is not None:
                time_s = b_strassen["time_ms"] / 1000.0
                n_bs = b_strassen["size"]
                gflops_bs = (2 * n_bs**3 - 2 * n_bs**2) / (time_s * 1e9)

                colors = ["#4daf4a", "#377eb8", "#984ea3"]
                markers = ["X", "s", "D"]
                (l_b_str,) = ax.plot(
                    n_bs,
                    gflops_bs,
                    label=f"Strassen L{level} (Ballard)",
                    color=colors[level - 1],
                    marker=markers[level - 1],
                    linestyle=":",
                    linewidth=1.5,
                    markersize=5.0,
                )
                legend_handles[f"Strassen L{level} (Ballard)"] = l_b_str

        ax.set_xticks(df_base["size"])
        ax.tick_params("x", labelbottom=True, rotation=30, rotation_mode="xtick")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Add labels on outer plots
    for col in range(4):
        axs[1, col].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)
    axs[1, 3].set_xlabel(r"Matrix Size ($N \times N$)", labelpad=10)

    for row in range(2):
        axs[row, 0].set_ylabel("Effective GFLOPS", labelpad=10)

    # Legend axis configuration
    legend_ax = axs[1, 3]
    legend_ax.axis("off")

    sorted_labels = [
        "MKL (Sequential)",
        "faer (Sequential)",
        "Strassen (dgemm base)",
        "Strassen (faer base)",
        "MKL (Ballard)",
        "Strassen L1 (Ballard)",
        "Strassen L2 (Ballard)",
        "Strassen L3 (Ballard)",
    ]
    handles = [legend_handles[lbl] for lbl in sorted_labels if lbl in legend_handles]
    labels = [lbl for lbl in sorted_labels if lbl in legend_handles]

    legend_ax.legend(
        handles, labels, loc="center", frameon=True, framealpha=0.9, edgecolor="#cbd5e1"
    )

    plt.suptitle(
        "Sequential Matrix Multiplication Performance Comparison\nRow 1: Cutoffs | Row 2: Recursion Levels",
        fontsize=15,
        y=0.98,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        hspace=0.25, wspace=0.15, top=0.90, bottom=0.08, left=0.06, right=0.96
    )

    # Save outputs
    out_dir_run1 = os.path.join(project_root, "generated", "csv", "run1")
    out_dir_plots = os.path.join(project_root, "generated", "plots")

    pdf_path_run1 = os.path.join(out_dir_run1, "sequential_grid_plot.pdf")
    png_path_run1 = os.path.join(out_dir_run1, "sequential_grid_plot.png")
    pdf_path_plots = os.path.join(out_dir_plots, "sequential_grid_plot.pdf")
    png_path_plots = os.path.join(out_dir_plots, "sequential_grid_plot.png")

    os.makedirs(out_dir_plots, exist_ok=True)

    plt.savefig(pdf_path_run1, bbox_inches="tight")
    plt.savefig(png_path_run1, bbox_inches="tight")
    plt.savefig(pdf_path_plots, bbox_inches="tight")
    plt.savefig(png_path_plots, bbox_inches="tight")

    print(f"Sequential grid plots saved successfully:")
    print(f"  - {pdf_path_run1}")
    print(f"  - {png_path_run1}")
    print(f"  - {pdf_path_plots}")
    print(f"  - {png_path_plots}")
    plt.close(fig)


if __name__ == "__main__":
    main()
