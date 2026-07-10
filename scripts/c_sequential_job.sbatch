#!/bin/bash
#SBATCH --job-name=c-matmul
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=generated/logs/c_matmul_%j.log

# Ensure we exit on any error
set -e

# Source Spack and load Environment Modules (avoiding local environment activation block for MKL)
source /software/spack/share/spack/setup-env.sh

# Load compiler, MKL, and CUDA environment modules
module load gcc/13.2.0
module load intel-oneapi-compilers/2025.0.4-gcc-11.4.0
module load intel-oneapi-mpi/2021.14.1-oneapi-2025.0.4
module load intel-oneapi-mkl/2024.2.2-intel-oneapi-mpi-2021.14.1-oneapi-2025.0.4

# Set threading variables to match requested CPUs per task for host libraries (MKL, OpenMP)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/fast-matmul}"
cd "${PROJECT_ROOT}"
export RUN_FOLDER=$(python3 python/resolve_run_dir.py)

MATRIX_SIZE=$1 # e.g., 1024

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/generated/logs"

echo "Running sequential benchmark for size ${MATRIX_SIZE}"
./build/matmul_benchmarks -square_test 1 -size "${MATRIX_SIZE}" -full

echo "Running DFS parallel benchmark for size ${MATRIX_SIZE}"
./build/matmul_bench_dfs -square_test_par 1 -size "${MATRIX_SIZE}" -full

echo "Running BFS parallel benchmark for size ${MATRIX_SIZE}"
./build/matmul_bench_bfs -square_test_par 1 -size "${MATRIX_SIZE}" -full

echo "Running HYBRID parallel benchmark for size ${MATRIX_SIZE}"
./build/matmul_bench_hybrid -square_test_par 1 -size "${MATRIX_SIZE}" -full

echo "Merging C/C++ benchmark results..."
python3 scripts/merge_c_results.py

echo "Job completed successfully!"

# Copy output log file to the generated logs folder
if [ -n "${SLURM_JOB_ID}" ] && [ -f "${PROJECT_ROOT}/generated/logs/c_matmul_${SLURM_JOB_ID}.log" ]; then
  sync
  cp "${PROJECT_ROOT}/generated/logs/c_matmul_${SLURM_JOB_ID}.log" "${PROJECT_ROOT}/generated/log/${RUN_FOLDER}/"
fi
