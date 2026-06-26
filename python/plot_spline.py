# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Check command line arguments
if len(sys.argv) < 2:
    print("Error: Missing CSV path argument.")
    print("Usage: uv run python/plot_spline.py <csv_path>")
    sys.exit(1)

csv_path = sys.argv[1]
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    sys.exit(1)

# Load data from the CSV file
df = pd.read_csv(csv_path)
N = df["size"].values
gflops = df["gflops"].values

# Fit Cubic Spline using scipy
cs = CubicSpline(N, gflops)
cs_derivative = cs.derivative()

# Generate smooth points for plotting the continuous curves
N_smooth = np.logspace(np.log2(2), np.log2(1024), 500, base=2)
gflops_smooth = cs(N_smooth)
derivatives_smooth = cs_derivative(N_smooth)

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

# Configure modern aesthetic grid and font
plt.rcParams['font.family'] = 'sans-serif'
ax1.grid(True, which="both", ls="--", color="gray", alpha=0.15)

# Left Y-Axis: GFLOPS Performance
color1 = '#6366f1'  # Premium Indigo
ax1.set_xlabel('Matrix Dimension ($N$)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Performance (GFLOPS)', color=color1, fontsize=12, fontweight='bold', labelpad=10)
curve1, = ax1.plot(N_smooth, gflops_smooth, color=color1, lw=2.5, label='Fitted GFLOPS Spline')
scatter1 = ax1.scatter(N, gflops, color=color1, edgecolors='#4f46e5', s=60, zorder=5, label='Effective GFLOPS')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log', base=2)
ax1.set_xticks(N)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Right Y-Axis: Spline Derivative
ax2 = ax1.twinx()
color2 = '#f43f5e'  # Premium Rose Red
ax2.set_ylabel(r'Spline Derivative ($d\mathrm{GFLOPS}/dN$)', color=color2, fontsize=12, fontweight='bold', labelpad=10)
curve2, = ax2.plot(N_smooth, derivatives_smooth, color=color2, lw=2.5, ls='--', label='Spline Derivative')
scatter2 = ax2.scatter(N, cs_derivative(N), color=color2, edgecolors='#e11d48', s=60, zorder=5, label='Computed Derivatives')
ax2.tick_params(axis='y', labelcolor=color2)

# Title and Legend configuration
plt.title('Baseline Matrix Multiplication GFLOPS Performance & Spline Derivative', fontsize=14, fontweight='bold', pad=15)
lines = [curve1, scatter1, curve2, scatter2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', shadow=False)

plt.tight_layout()

# Save image in the same directory as the input CSV file
output_dir = os.path.dirname(os.path.abspath(csv_path))
output_path = os.path.join(output_dir, "base_matmul_spline.png")
plt.savefig(output_path, dpi=300)
print(f"Successfully generated spline plot at: {output_path}")
