#!/bin/bash
# Pre-setup script to load the Spack and Environment Module compiling toolchain,
# and run the target-specific compiler for the benchmarks.
#
# Usage: ./scripts/pre-setup.sh [architecture]
# Example: ./scripts/pre-setup.sh broadwell

set -e

# Target CPU architecture defaults to broadwell if not specified
TARGET_ARCH=${1:-broadwell}

echo "=================================================="
echo "Starting pre-setup compilation for: ${TARGET_ARCH}"
echo "=================================================="

# Sourcing Spack env
SPACK_ROOT=""
if [ -f "/software/spack/share/spack/setup-env.sh" ]; then
    SPACK_ROOT="/software/spack"
elif [ -f "$HOME/spack/share/spack/setup-env.sh" ]; then
    SPACK_ROOT="$HOME/spack"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

if [ -n "${SPACK_ROOT}" ]; then
    echo "Sourcing Spack environment: ${SPACK_ROOT}/share/spack/setup-env.sh"
    # Sourcing via dot command for portability
    . "${SPACK_ROOT}/share/spack/setup-env.sh"
    
    if [ -d "${PROJECT_ROOT}/rust" ]; then
        echo "Activating Spack environment in ${PROJECT_ROOT}/rust..."
        spack env activate -d "${PROJECT_ROOT}/rust"
    fi
else
    echo "Warning: Spack installation not found. Skipping Spack activation."
fi

# Load required compiler and MKL modules if the module command exists
if command -v module &> /dev/null || [ -n "$(type -t module)" ]; then
    echo "Loading Environment Modules..."
    module load gcc/13.2.0 2>/dev/null || true
    module load intel-oneapi-compilers/2025.0.4-gcc-11.4.0 2>/dev/null || true
    module load intel-oneapi-mpi/2021.14.1-oneapi-2025.0.4 2>/dev/null || true
    module load intel-oneapi-mkl/2024.2.2-intel-oneapi-mpi-2021.14.1-oneapi-2025.0.4 2>/dev/null || true
else
    echo "Warning: 'module' command not found. Skipping Environment Module loads."
fi

# Load Anaconda and activate Python 2.7 environment for C++ algorithm codegen
if command -v conda &> /dev/null; then
    echo "Activating Conda Python 2.7 environment..."
    eval "$(conda shell.bash hook)"
    conda activate py27 2>/dev/null || echo "Warning: Could not activate conda env 'py27'."
elif command -v module &> /dev/null || [ -n "$(type -t module)" ]; then
    echo "Attempting to load Anaconda module..."
    module load anaconda3/2024.02 2>/dev/null || true
    if command -v conda &> /dev/null; then
        echo "Activating Conda Python 2.7 environment..."
        eval "$(conda shell.bash hook)"
        conda activate py27 2>/dev/null || echo "Warning: Could not activate conda env 'py27'."
    else
        echo "Warning: Conda command still not found after loading Anaconda module."
    fi
else
    echo "Warning: Conda not found. Skipping Python 2.7 environment activation."
fi

# Run the target-specific compile script
"${PROJECT_ROOT}/scripts/compile.sh" "${TARGET_ARCH}"

echo "=================================================="
echo "Pre-setup compilation successfully completed!"
echo "=================================================="
