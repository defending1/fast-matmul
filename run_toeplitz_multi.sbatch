#!/bin/bash
#SBATCH --job-name=gpu-matmul-32k
#SBATCH --partition=gpu                          # Directs the job to the GPU nodes (gpu0[1-4])
#SBATCH --nodes=1                                # Request exactly 1 compute node
#SBATCH --ntasks=1                               # Execute a single task
#SBATCH --gres=gpu:1                             # Request 1 NVIDIA A40 GPU (48GB VRAM)
#SBATCH --cpus-per-task=16                        # Allocate 16 CPU cores for host coordination
#SBATCH --mem=64G                                # Allocate 64GB of Host RAM (for f64 matrices)
#SBATCH --time=8-00:30:00
#SBATCH --output=fast_matmul_multi_%j.log
#SBATCH --exclusive                              # Exclusive node allocation for clean benchmarking

# Ensure we exit on any error
set -e

# Source Spack and load Environment Modules (avoiding local environment activation block for MKL)
source /software/spack/share/spack/setup-env.sh

# Load compiler, MKL, and CUDA environment modules
module load gcc/13.2.0
module load intel-oneapi-compilers/2025.0.4-gcc-11.4.0
module load intel-oneapi-mpi/2021.14.1-oneapi-2025.0.4
module load intel-oneapi-mkl/2024.2.2-intel-oneapi-mpi-2021.14.1-oneapi-2025.0.4
module load cuda/12.0 2>/dev/null || true

# Print Slurm metadata and environment information for debugging
echo "=== Slurm Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPU device: $CUDA_VISIBLE_DEVICES"
echo "Host RAM requested: $SLURM_MEM_PER_NODE"
echo "MKLROOT: $MKLROOT"
echo "====================================="
echo ""

# Set threading variables to match requested CPUs per task for host libraries (MKL, OpenMP)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Parse command-line arguments:
# $1 - Matrix size (default: 32768)
# $2 - CPU architecture: znver3, broadwell, native (default: detect)
MATRIX_SIZE=${1:-32768}
TARGET_ARCH=${2:-detect}

echo "Matrix size: ${MATRIX_SIZE}"
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

echo "Resolved architecture: ${ARCH}"

# Compile compiler flags targeting the resolved CPU architecture of the node
export CFLAGS="-O3 -march=${ARCH} -mtune=${ARCH}"
export CXXFLAGS="-O3 -march=${ARCH} -mtune=${ARCH}"


# Run Rust project if selected
if [ "${TARGET_PROJECT}" = "rust" ] || [ "${TARGET_PROJECT}" = "both" ]; then
echo "============================================="
echo "=== Running Rust Benchmarks ==="
echo "============================================="
cd "${PROJECT_ROOT}/rust"

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
fi


# Run C/C++ project if selected
if [ "${TARGET_PROJECT}" = "c" ] || [ "${TARGET_PROJECT}" = "both" ]; then
echo "============================================="
echo "=== Running C/C++ Benchmarks ==="
echo "============================================="
cd "${PROJECT_ROOT}"

echo "--- Running DFS Parallel Benchmark ---"
./build/matmul_bench_dfs -square_test_par 1 -full

echo "--- Running BFS Parallel Benchmark ---"
./build/matmul_bench_bfs -square_test_par 1 -full

echo "--- Running HYBRID Parallel Benchmark ---"
./build/matmul_bench_hybrid -square_test_par 1 -full
fi

echo "Done!"
