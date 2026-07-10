#!/bin/bash
#SBATCH --job-name=fast-matmul-seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=08:30:00
#SBATCH --mem=128G
#SBATCH --output=benchmark_seq_%j.log

# Ensure we exit on any error
set -e

# Source Spack and load Environment Modules (avoiding local environment activation block for MKL)
source /software/spack/share/spack/setup-env.sh

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

# Parse command-line arguments:
# $1 - project to execute: rust, c, or both (default: both)
# $2 - CPU architecture: broadwell, znver3, native (default: detect)
TARGET_PROJECT=${1:-both}
TARGET_ARCH=${2:-detect}

echo "Target project to execute: ${TARGET_PROJECT}"
echo "Target architecture parameter: ${TARGET_ARCH}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/fast-matmul}"
cd "${PROJECT_ROOT}"

# Resolve architecture: use parameter if provided, otherwise detect automatically
if [ "${TARGET_ARCH}" != "detect" ]; then
ARCH="${TARGET_ARCH}"
else
# Detect target architecture automatically based on system CPU
if grep -q "EPYC" /proc/cpuinfo; then
ARCH="znver3"
elif grep -q "Xeon" /proc/cpuinfo; then
ARCH="broadwell"
else
ARCH="native"
fi
fi

# Compile/Pre-setup if COMPILE=1 or COMPILE=yes
if [ "${COMPILE}" = "1" ] || [ "${COMPILE}" = "yes" ]; then
echo "============================================="
echo "=== Compiling / Pre-setting up Project ==="
echo "============================================="
"${PROJECT_ROOT}/scripts/pre-setup.sh" "${ARCH}"
fi

# Run Rust project if selected
if [ "${TARGET_PROJECT}" = "rust" ] || [ "${TARGET_PROJECT}" = "both" ]; then
echo "============================================="
echo "=== Running Rust Benchmarks ==="
echo "============================================="
cd "${PROJECT_ROOT}/rust"

# Fallback to native if the specific architecture binary is not found
if [ ! -f "./generated/bin/bench_${ARCH}" ]; then
echo "Warning: ./generated/bin/bench_${ARCH} not found. Falling back to native."
ARCH="native"
fi

# Set sequential/parallel flags depending on Slurm allocation
if [ "${SLURM_CPUS_PER_TASK:-1}" -gt 1 ]; then
MODE_FLAG="--par"
else
MODE_FLAG="--seq"
fi

echo "Executing: ./generated/bin/bench_${ARCH} ${MODE_FLAG} --full"
./generated/bin/bench_${ARCH} ${MODE_FLAG} --full

echo "Merging Rust benchmark results..."
cd "${PROJECT_ROOT}"
python3 scripts/merge_results.py
fi

# Run C/C++ project if selected
if [ "${TARGET_PROJECT}" = "c" ] || [ "${TARGET_PROJECT}" = "both" ]; then
echo "============================================="
echo "=== Running C/C++ Benchmarks ==="
echo "============================================="
cd "${PROJECT_ROOT}"
./build/matmul_benchmarks -square_test 1 -full

echo "Merging C/C++ benchmark results..."
python3 scripts/merge_c_results.py
fi

echo "Done!"
