#!/bin/bash
# Script to compile and submit multiple GPU benchmark jobs on Toeplitz.
# It iterates over cutoffs, levels, and matrix sizes N, submitting an sbatch job for each.

# Ensure we exit on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/generated/logs"



ARCH="broadwell"

# Parse command-line arguments:
# --no-compile: Skip compiling before submitting jobs (default: compiles znver3)
# --seq / --sequential: Launch sequential benchmark jobs (default)
# --par / --parallel: Launch parallel benchmark jobs
COMPILE_PROJECT=yes
RUN_MODE="seq" # Default run mode

for arg in "$@"; do
    if [ "$arg" = "--no-compile" ]; then
        COMPILE_PROJECT=no
    elif [ "$arg" = "--seq" ] || [ "$arg" = "--sequential" ]; then
        RUN_MODE="seq"
    elif [ "$arg" = "--par" ] || [ "$arg" = "--parallel" ]; then
        RUN_MODE="par"
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

# Configure CPU counts and flags based on the execution mode
if [ "${RUN_MODE}" = "seq" ]; then
    CPUS_PER_TASK=1
    RUST_FLAG="--seq"
else
    CPUS_PER_TASK=${CPUS_PER_TASK:-16}
    RUST_FLAG="--par"
fi

# Define the parameter spaces to iterate over
cutoffs=(256 512 1024 2048)
levels=(1 2 3)

# Sizes N = [2, 4, 8, ..., 32768] (2^1 to 2^15)
sizes=()
for ((i=1; i<=16; i++)); do
    sizes+=($((1 << i)))
done

echo "Submitting batch jobs (mode: ${RUN_MODE})"
echo "Submitting batch jobs for cutoffs: ${cutoffs[*]}"
echo "Submitting batch jobs for levels: ${levels[*]}"
echo "Submitting batch jobs for matrix sizes: ${sizes[*]}"
echo "Using CPU count per task: ${CPUS_PER_TASK}"
echo "--------------------------------------------------"

job_count=0

# Iterate over sizes N for baseline Rust benchmarks
for size in "${sizes[@]}"; do
    echo "Submitting baseline job: size=${size}"
    sbatch --cpus-per-task="${CPUS_PER_TASK}" "${PROJECT_ROOT}/scripts/rust_sequential_job.sbatch" base "_" "${size}" "${RUST_FLAG}"
    job_count=$((job_count + 1))
done

# Iterate over cutoffs and sizes N for Strassen benchmarks
for cutoff in "${cutoffs[@]}"; do
    for size in "${sizes[@]}"; do
        echo "Submitting Strassen job: cutoff=${cutoff}, size=${size}"
        sbatch --cpus-per-task="${CPUS_PER_TASK}" "${PROJECT_ROOT}/scripts/rust_sequential_job.sbatch" cutoff "${cutoff}" "${size}" "${RUST_FLAG}"
        job_count=$((job_count + 1))
    done
done

# Iterate over levels and sizes N for Strassen benchmarks
for level in "${levels[@]}"; do
    for size in "${sizes[@]}"; do
        echo "Submitting Strassen job: level=${level}, size=${size}"
        sbatch --cpus-per-task="${CPUS_PER_TASK}" "${PROJECT_ROOT}/scripts/rust_sequential_job.sbatch" level "${level}" "${size}" "${RUST_FLAG}"
        job_count=$((job_count + 1))
    done
done

# Submit C benchmark jobs
if [ "${RUN_MODE}" = "seq" ]; then
    # Iterate over sizes N for C sequential benchmarks
    for size in "${sizes[@]}"; do
        echo "Submitting C job: mode=seq, size=${size}"
        sbatch --cpus-per-task="${CPUS_PER_TASK}" "${PROJECT_ROOT}/scripts/c_sequential_job.sbatch" "seq" "${size}"
        job_count=$((job_count + 1))
    done
else
    # Iterate over parallel modes and sizes N for C parallel benchmarks
    for mode in dfs bfs hybrid; do
        for size in "${sizes[@]}"; do
            echo "Submitting C job: mode=${mode}, size=${size}"
            sbatch --cpus-per-task="${CPUS_PER_TASK}" "${PROJECT_ROOT}/scripts/c_sequential_job.sbatch" "${mode}" "${size}"
            job_count=$((job_count + 1))
        done
    done
fi

echo "--------------------------------------------------"
echo "Successfully submitted ${job_count} ${RUN_MODE} jobs to the GPU partition!"
echo "Use 'squeue -u \$USER' to monitor your jobs."
echo "=================================================="
