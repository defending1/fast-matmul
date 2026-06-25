#!/bin/bash
# setup.sh - Automated installation script for the fast-matmul Rust project.
#
# This script automates:
#   1. Configuring the Rust toolchain to use nightly (project override).
#   2. Locating or downloading Spack.
#   3. Bootstrapping GCC 14.3.0 inside Spack.
#   4. Activating, concretizing, and installing the Spack environment.
#   5. Compiling and running tests in the Rust project.

set -e

# Determine directory paths
RUST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUST_DIR"

echo "=== Fast MatMul Rust Project Setup ==="
echo "Directory: $RUST_DIR"
echo ""

# ----------------------------------------------------
# 1. Rust Environment Verification
# ----------------------------------------------------
echo ">>> Checking Rustup and compiler..."
if ! command -v rustup &> /dev/null; then
    echo "Error: 'rustup' is not installed."
    echo "Please install rustup from https://rustup.rs/ before running this script."
    exit 1
fi

echo "Installing and setting Rust nightly toolchain override for this project..."
rustup toolchain install nightly
rustup override set nightly
echo "Active Rust version:"
rustc --version
echo ""

# ----------------------------------------------------
# 2. Spack Installation / Location
# ----------------------------------------------------
echo ">>> Checking Spack package manager..."
SPACK_ROOT_DIR="$HOME/spack"
SPACK_SETUP_PATH="$SPACK_ROOT_DIR/share/spack/setup-env.sh"

if [ -f "$SPACK_SETUP_PATH" ]; then
    echo "Found Spack installation at $SPACK_ROOT_DIR."
else
    if command -v spack &> /dev/null; then
        echo "Spack is already in PATH."
    else
        echo "Spack not found. Cloning Spack repository to $SPACK_ROOT_DIR..."
        git clone -b releases/v0.22 https://github.com/spack/spack.git "$SPACK_ROOT_DIR"
    fi
fi

# Source Spack environment
if [ -f "$SPACK_SETUP_PATH" ]; then
    echo "Sourcing Spack environment: $SPACK_SETUP_PATH"
    # Sourcing via dot command for portability
    . "$SPACK_SETUP_PATH"
fi

# Verify Spack is available in the shell
if ! command -v spack &> /dev/null; then
    echo "Error: Spack command could not be resolved."
    exit 1
fi
echo ""

# ----------------------------------------------------
# 3. Spack Bootstrapping
# ----------------------------------------------------
echo ">>> Finding external system compilers and utilities..."
spack external find

# Check if gcc@14.3.0 is configured
echo "Checking for gcc@14.3.0 in Spack..."
if spack find gcc@14.3.0 &>/dev/null || spack compiler list | grep -q "gcc@14.3.0"; then
    echo "gcc@14.3.0 compiler is already configured in Spack."
else
    echo "gcc@14.3.0 not found. Installing gcc@14.3.0 via Spack..."
    echo "Note: Building GCC from source may take a significant amount of time."
    spack install gcc@14.3.0 languages=fortran,c,c++
    echo "Registering the new compiler..."
    spack compiler find
fi
echo ""

# ----------------------------------------------------
# 4. Spack Environment Installation
# ----------------------------------------------------
echo ">>> Activating Spack environment in $RUST_DIR..."
# Activate the environment (using . command to ensure variables propagate)
. "$SPACK_SETUP_PATH"
spack env activate -d .

echo "Concretizing Spack dependencies (MKL, OpenBLAS, LLVM, etc.)..."
spack concretize -f

echo "Installing Spack packages..."
spack install -v
echo ""

# ----------------------------------------------------
# 5. Rust Compilation and Correctness Tests
# ----------------------------------------------------
echo ">>> Building Rust project targets (Release mode)..."
cargo build --release --all-targets

echo ">>> Running unit and integration tests..."
cargo test

echo ""
echo "=================================================================="
echo "Setup completed successfully!"
echo "To run the benchmarks, make sure you activate Spack in your shell:"
echo "  source ~/spack/share/spack/setup-env.sh"
echo "  spack env activate -d $RUST_DIR"
echo "  cargo bench --bench bench"
echo "=================================================================="
