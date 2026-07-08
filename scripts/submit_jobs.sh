#!/bin/bash
# Script to compile and submit multiple GPU benchmark jobs on Toeplitz.
# It iterates over cutoffs, levels, and matrix sizes N, submitting an sbatch job for each.

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

# Define the parameter spaces to iterate over
cutoffs=(256 512 1024 2048)
levels=(1 2 3)

# Sizes N = [2, 4, 8, ..., 32768] (2^1 to 2^15)
sizes=()
for ((i=1; i<=15; i++)); do
    sizes+=($((1 << i)))
done

echo "Submitting batch jobs for cutoffs: ${cutoffs[*]}"
echo "Submitting batch jobs for levels: ${levels[*]}"
echo "Submitting batch jobs for matrix sizes: ${sizes[*]}"
echo "--------------------------------------------------"

job_count=0

# Iterate over cutoffs and sizes N
for cutoff in "${cutoffs[@]}"; do
    for size in "${sizes[@]}"; do
        echo "Submitting job: cutoff=${cutoff}, size=${size}"
        sbatch "${PROJECT_ROOT}/scripts/gpu_job.sbatch" cutoff "${cutoff}" "${size}"
        job_count=$((job_count + 1))
    done
done

# Iterate over levels and sizes N
for level in "${levels[@]}"; do
    for size in "${sizes[@]}"; do
        echo "Submitting job: level=${level}, size=${size}"
        sbatch "${PROJECT_ROOT}/scripts/gpu_job.sbatch" level "${level}" "${size}"
        job_count=$((job_count + 1))
    done
done


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
echo "Once all jobs are complete, run 'python3 scripts/merge_results.py' to merge CSV files and generate plots."
echo "=================================================="
