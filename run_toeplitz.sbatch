#!/bin/bash
#SBATCH --job-name=fast-matmul-seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cl2
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --output=benchmark_seq_%j.log

# Ensure we exit on any error
set -e

# Load pre-installed GCC and Intel MKL modules already on the Toeplitz cluster
module load gcc/12.2.0
module load intel-oneapi-mkl/2022.1.0-oneapi-2022.1.0

# Print environment information for debugging
echo "=== Slurm Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "MKLROOT: $MKLROOT"
echo ""

# Enforce sequential execution for underlying BLAS/OpenMP libraries
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Ensure cargo/rustup are in the PATH and configure nightly toolchain
export PATH="$HOME/.cargo/bin:$PATH"
echo "Setting up Rust nightly toolchain..."
rustup toolchain install nightly
rustup override set nightly
rustc --version

# Navigate to the rust directory where Cargo.toml is located
cd rust

# Compile the benchmarks in release mode first
echo "Building benchmarks..."
cargo build --release --bench bench

# Run the sequential benchmarks up to size 2^15
echo "Running sequential benchmarks..."
cargo bench --bench bench -- --sequential --full

echo "Done!"
