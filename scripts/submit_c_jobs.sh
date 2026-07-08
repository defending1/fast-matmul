#!/bin/bash
# Script to compile and submit multiple C/C++ benchmark jobs on Toeplitz.
# It iterates over matrix sizes N, submitting an sbatch job for each.

# Ensure we exit on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

ARCH="znver3"

# Parse command-line arguments:
# --no-compile: Skip compiling before submitting jobs (default: compiles znver3)
COMPILE_PROJECT=yes
for arg in "$@"; do
    if [ "$arg" = "--no-compile" ]; then
        COMPILE_PROJECT=no
    fi
done

if [ "${COMPILE_PROJECT}" = "yes" ]; then
    echo "=================================================="
    echo "Compiling benchmarks once targeting ${ARCH}..."
    echo "=================================================="
    "${PROJECT_ROOT}/scripts/pre-setup.sh" "${ARCH}"
else
    echo "Skipping compilation as requested by --no-compile."
fi

# Sizes N = [2, 4, 8, ..., 32768] (2^1 to 2^15)
sizes=()
for ((i=1; i<=15; i++)); do
    sizes+=($((1 << i)))
done

echo "Submitting batch jobs for matrix sizes: ${sizes[*]}"
echo "--------------------------------------------------"

job_count=0

# Iterate over sizes N
for size in "${sizes[@]}"; do
    echo "Submitting job: size=${size}"
    sbatch "${PROJECT_ROOT}/scripts/c_job.sbatch" "${size}"
    job_count=$((job_count + 1))
done

echo "--------------------------------------------------"
echo "Successfully submitted ${job_count} jobs to the GPU partition!"
echo "Use 'squeue -u \$USER' to monitor your jobs."
echo "=================================================="
