#!/bin/bash
#SBATCH --job-name=gpu-matmul-32k
#SBATCH --partition=gpu                          # Directs the job to the GPU nodes (gpu0[1-4])
#SBATCH --nodes=1                                # Request exactly 1 compute node
#SBATCH --ntasks=1                               # Execute a single task
#SBATCH --gres=gpu:1                             # Request 1 NVIDIA A40 GPU (48GB VRAM)
#SBATCH --cpus-per-task=8                        # Allocate 8 CPU cores for host coordination
#SBATCH --mem=64G                                # Allocate 64GB of Host RAM (for f64 matrices)
#SBATCH --time=00:30:00                          # 30-minute limit (execution takes ~2-30 seconds)
#SBATCH --output=gpu_matmul_%j.log

# Exit on errors
set -e

# Load compiler and CUDA environment modules
module purge
module load gcc/12.2.0
# Depending on local installation, load CUDA toolkit module if present:
module load cuda/12.0 2>/dev/null || true

# Print Slurm metadata
echo "=== Toeplitz GPU Job Information ==="
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated GPU device: $CUDA_VISIBLE_DEVICES"
echo "Host RAM requested: $SLURM_MEM_PER_NODE"
echo "====================================="

# Optional: compile your code targeting the AMD EPYC architecture of the GPU nodes
# (EPYC 7763 uses znver3)
export CFLAGS="-O3 -march=znver3 -mtune=znver3"
export CXXFLAGS="-O3 -march=znver3 -mtune=znver3"

# Run your compiled GPU-accelerated Matrix Multiplication binary
# ./build/gpu_matmul_bin -size 32768
