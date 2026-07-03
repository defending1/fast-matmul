#!/bin/bash
# Scripts to clean and compile the benchmarks targeting a specific CPU microarchitecture
# (e.g., broadwell, znver3, native).
#
# Usage: ./scripts/compile.sh [architecture]
# Example: ./scripts/compile.sh broadwell

set -e

# Target CPU architecture defaults to native if not specified
TARGET_ARCH=${1:-native}

echo "=================================================="
echo "Compiling project targeting: ${TARGET_ARCH}"
echo "=================================================="

# Determine project root and navigate to the rust directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}/rust"

# Set target-specific flags for both C and Rust compilation
export TARGET_CPU="${TARGET_ARCH}"
export RUSTFLAGS="-C target-cpu=${TARGET_ARCH}"

echo "Cleaning previous build artifacts..."
cargo clean

echo "Building release benchmarks..."
cargo build --release --benches

# Copy the compiled binaries to a stable directory under generated/bin
BIN_DIR="generated/bin"
mkdir -p "${BIN_DIR}"

# Find the latest compiled binary for benchmark
BENCH_BIN=$(ls -t target/release/deps/bench-* 2>/dev/null | grep -v '\.d$' | head -n 1 || true)
if [ -n "${BENCH_BIN}" ]; then
    cp "${BENCH_BIN}" "${BIN_DIR}/bench_${TARGET_ARCH}"
    echo "Saved optimized benchmark binary to: rust/${BIN_DIR}/bench_${TARGET_ARCH}"
else
    echo "Warning: Could not find compiled benchmark binary."
fi

# Find the latest compiled binary for curves
CURVES_BIN=$(ls -t target/release/deps/base_matmul_curves-* 2>/dev/null | grep -v '\.d$' | head -n 1 || true)
if [ -n "${CURVES_BIN}" ]; then
    cp "${CURVES_BIN}" "${BIN_DIR}/base_matmul_curves_${TARGET_ARCH}"
    echo "Saved optimized base curves binary to: rust/${BIN_DIR}/base_matmul_curves_${TARGET_ARCH}"
else
    echo "Warning: Could not find compiled base curves binary."
fi

echo "=================================================="
echo "Compiling C/C++ project targets..."
echo "=================================================="

# Navigate to codegen directory and generate C++ algorithms
cd "${PROJECT_ROOT}/codegen"
echo "Generating C/C++ algorithms (adds_type=0)..."
bash gen_all_algorithms.sh 0

# Navigate to project root and compile C++ targets
cd "${PROJECT_ROOT}"
echo "Cleaning previous C/C++ build artifacts..."
make clean

echo "Building C/C++ strassen and matmul_benchmarks..."
make OPT="-O3 -march=${TARGET_ARCH} -mtune=${TARGET_ARCH}" strassen matmul_benchmarks

echo "=================================================="
echo "Compilation complete!"
echo "=================================================="
