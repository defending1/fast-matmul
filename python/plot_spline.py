# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scipy",
#   "scienceplots",
# ]
# ///

"""Benchmark plotting script to fit a Cubic Spline to baseline GFLOPS data.

Plots the continuous spline fit alongside the computed spline derivative.
Formatted for academic papers and LaTeX document insertion.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import plot_utils


def main() -> None:
    """Executes spline fitting and saves the plot."""
    if len(sys.argv) < 2:
        print("Error: Missing CSV path argument.")
        print("Usage: uv run python/plot_spline.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    sizes = df["size"].values
    gflops = df["gflops"].values if "gflops" in df.columns else None

    # Calculate GFLOPS if not present but raw timings are
    if gflops is None:
        col = next((c for c in ["mkl_seq", "faer_seq"] if c in df.columns), None)
        if col is not None:
            gflops = plot_utils.calculate_gflops_rust(df["size"], df[col]).values
        else:
            print("Error: Could not locate GFLOPS or timing columns in CSV.")
            sys.exit(1)

    # Fit Cubic Spline using scipy
    cs = CubicSpline(sizes, gflops)
    cs_derivative = cs.derivative()

    # Generate smooth points for continuous curve visualization
    sizes_smooth = np.logspace(np.log2(sizes.min()), np.log2(sizes.max()), 500, base=2)
    gflops_smooth = cs(sizes_smooth)
    derivatives_smooth = cs_derivative(sizes_smooth)

    # Configure Matplotlib style for IEEE/LaTeX formatting
    plot_utils.setup_matplotlib_style()

    # Compact figure size (5.8 x 3.8 inches) optimal for paper layout margins
    fig, ax1 = plt.subplots(figsize=(5.8, 3.8), dpi=300)
    ax1.set_facecolor("none")
    ax1.grid(True, which="both", ls="--", color="gray", alpha=0.15)

    # Primary Y-Axis: GFLOPS Performance
    color_gflops = "#6366f1"  # Premium Indigo
    ax1.set_xlabel(r"Matrix Dimension ($N$)", labelpad=6)
    ax1.set_ylabel("Performance (GFLOPS)", color=color_gflops, labelpad=6)
    curve_gflops, = ax1.plot(sizes_smooth, gflops_smooth, color=color_gflops, lw=2.0, label="Fitted Spline")
    scatter_gflops = ax1.scatter(
        sizes, gflops, color=color_gflops, edgecolors="#4f46e5", s=40, zorder=5, label="Effective GFLOPS"
    )
    ax1.tick_params(axis="y", labelcolor=color_gflops)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Secondary Y-Axis: Spline Derivative
    ax2 = ax1.twinx()
    color_deriv = "#f43f5e"  # Premium Rose Red
    ax2.set_ylabel(r"Spline Derivative ($d\mathrm{GFLOPS}/dN$)", color=color_deriv, labelpad=6)
    curve_deriv, = ax2.plot(sizes_smooth, derivatives_smooth, color=color_deriv, lw=2.0, ls="--", label="Derivative")
    scatter_deriv = ax2.scatter(
        sizes, cs_derivative(sizes), color=color_deriv, edgecolors="#e11d48", s=40, zorder=5, label="Discrete Derivatives"
    )
    ax2.tick_params(axis="y", labelcolor=color_deriv)

    # Note: Figure title is omitted to comply with LaTeX figure caption standards.
    lines = [curve_gflops, scatter_gflops, curve_deriv, scatter_deriv]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e8f0")

    plt.tight_layout()

    # Determine saving path based on CSV directory location
    csv_dir = os.path.abspath(os.path.dirname(csv_path))
    if os.path.basename(csv_dir) == "csv" and os.path.basename(os.path.dirname(csv_dir)) == "generated":
        output_dir = os.path.join(os.path.dirname(csv_dir), "plots")
    else:
        output_dir = csv_dir

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "base_matmul_spline.pdf")
    plot_utils.save_plot(fig, output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
