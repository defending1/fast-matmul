#!/bin/bash
#SBATCH --job-name=cl_c-matmul
#SBATCH --partition=cl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=cl2:1
#SBATCH --mem=250G
#SBATCH --time=3-02:00:00
#SBATCH --output=generated/logs/cl_c_matmul_%j.log

# Ensure we exit on any error
set -e

# Source Spack and load Environment Modules (avoiding local environment activation block for MKL)
source /software/spack/share/spack/setup-env.sh

module load gcc/13.2.0
module load intel-oneapi-compilers/2025.0.4-gcc-11.4.0
module load intel-oneapi-mpi/2021.14.1-oneapi-2025.0.4
module load intel-oneapi-mkl/2024.2.2-intel-oneapi-mpi-2021.14.1-oneapi-2025.0.4

# Set threading variables to match requested CPUs per task for host libraries (MKL, OpenMP)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/fast-matmul}"
cd "${PROJECT_ROOT}"

MODE=$1         # "seq", "dfs", "bfs", or "hybrid"
MATRIX_SIZE=$2  # matrix size

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/generated/logs"

if [ "${MODE}" = "seq" ] || [ "${MODE}" = "sequential" ]; then
echo "Running sequential benchmark for size ${MATRIX_SIZE}"
./build/matmul_benchmarks -square_test 1 -size "${MATRIX_SIZE}" -full
elif [ "${MODE}" = "dfs" ]; then
echo "Running DFS parallel benchmark for size ${MATRIX_SIZE}"
./build/matmul_bench_dfs -square_test_par 1 -size "${MATRIX_SIZE}" -full
elif [ "${MODE}" = "bfs" ]; then
echo "Running BFS parallel benchmark for size ${MATRIX_SIZE}"
./build/matmul_bench_bfs -square_test_par 1 -size "${MATRIX_SIZE}" -full
elif [ "${MODE}" = "hybrid" ]; then
echo "Running HYBRID parallel benchmark for size ${MATRIX_SIZE}"
./build/matmul_bench_hybrid -square_test_par 1 -size "${MATRIX_SIZE}" -full
else
# Fallback to original behavior if only one argument is provided (for backward compatibility)
if [ -z "$MATRIX_SIZE" ]; then
MATRIX_SIZE=$MODE
echo "Running sequential benchmark for size ${MATRIX_SIZE} (fallback)"
./build/matmul_benchmarks -square_test 1 -size "${MATRIX_SIZE}" -full
else
echo "Error: Unknown mode ${MODE}"
exit 1
fi
fi

echo "Job completed successfully!"
