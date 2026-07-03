#!/bin/bash
#SBATCH --job-name=fast-matmul-seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cl2
#SBATCH --time=08:30:00
#SBATCH --mem=128G
#SBATCH --output=benchmark_seq_%j.log
#SBATCH --exclusive

# Ensure we exit on any error
set -e

# Source Spack, activate, concretize, and install the project's Spack environment
source /software/spack/share/spack/setup-env.sh
spack env activate -d rust

module load gcc/13.2.0
module load intel-oneapi-compilers/2025.0.4-gcc-11.4.0
module load intel-oneapi-mpi/2021.14.1-oneapi-2025.0.4
module load intel-oneapi-mkl/2024.2.2-intel-oneapi-mpi-2021.14.1-oneapi-2025.0.4

# Print environment information for debugging
echo "=== Slurm Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "MKLROOT: $MKLROOT"
echo ""

# Enforce sequential execution for underlying BLAS/OpenMP libraries
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Parse command-line argument: rust, c, or both (default: both)
TARGET_PROJECT=${1:-both}
echo "Target project to execute: ${TARGET_PROJECT}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run Rust project if selected
if [ "${TARGET_PROJECT}" = "rust" ] || [ "${TARGET_PROJECT}" = "both" ]; then
echo "============================================="
echo "=== Running Rust Benchmarks ==="
echo "============================================="
cd "${PROJECT_ROOT}/rust"
./generated/bin/bench_broadwell --sequential --full
fi

# Run C/C++ project if selected
if [ "${TARGET_PROJECT}" = "c" ] || [ "${TARGET_PROJECT}" = "both" ]; then
echo "============================================="
echo "=== Running C/C++ Benchmarks ==="
echo "============================================="
cd "${PROJECT_ROOT}"
./build/matmul_benchmarks -square_test 1
fi

echo "Done!"
