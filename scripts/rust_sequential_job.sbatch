#!/bin/bash
#SBATCH --job-name=rust-matmul
#SBATCH --partition=cl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=cl2:1
#SBATCH --mem=250G
#SBATCH --time=3-02:00:00
#SBATCH --output=generated/logs/cl_rmatmul_%j.log

# Ensure we exit on any error
set -e

# Source Spack and load Environment Modules (avoiding local environment activation block for MKL)
source /software/spack/share/spack/setup-env.sh

module load gcc/11.4.0
module load intel-oneapi-compilers/2025.0.4-gcc-11.4.0
module load intel-oneapi-mpi/2021.14.1-oneapi-2025.0.4
module load intel-oneapi-mkl/2024.2.2-intel-oneapi-mpi-2021.14.1-oneapi-2025.0.4

# Set threading variables to match requested CPUs per task for host libraries (MKL, OpenMP)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/fast-matmul}"
cd "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}/rust"

PARAM_TYPE=$1  # "base", "cutoff", or "level"
PARAM_VAL=$2   # e.g., 256 or 1 (ignored for "base")
MATRIX_SIZE=$3 # e.g., 1024
ARCH="znver3"
MODE_FLAG=${4:-"--par"}

# Execute the specific benchmark
if [ "${PARAM_TYPE}" = "base" ]; then
echo "Running baseline benchmark with size ${MATRIX_SIZE} on ${ARCH}"
./generated/bin/bench_${ARCH} ${MODE_FLAG} --matmul base --size ${MATRIX_SIZE}
elif [ "${PARAM_TYPE}" = "cutoff" ]; then
echo "Running Strassen benchmark with cutoff ${PARAM_VAL} and size ${MATRIX_SIZE} on ${ARCH}"
./generated/bin/bench_${ARCH} ${MODE_FLAG} --matmul strassen --cutoff ${PARAM_VAL} --size ${MATRIX_SIZE}
elif [ "${PARAM_TYPE}" = "level" ]; then
echo "Running Strassen benchmark with level ${PARAM_VAL} and size ${MATRIX_SIZE} on ${ARCH}"
./generated/bin/bench_${ARCH} ${MODE_FLAG} --matmul strassen --level ${PARAM_VAL} --size ${MATRIX_SIZE}
else
echo "Error: Unknown parameter type ${PARAM_TYPE}"
exit 1
fi

echo "Job completed successfully!"
