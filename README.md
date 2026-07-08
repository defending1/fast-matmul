## Rust fork

My project contributions can be found in the directories `./rust` and
`./report`.

### Installation (rust project)

To run the automated installation:

```bash
cd rust
./setup.sh
```

Then run

```bash
cargo bench
```

to run benchmarks and generate plots.

### Dependencies

- **Rust nightly compiler:** Get [rustup](https://rustup.rs/) and switch to
  nightly:
  ```bash
  rustup default nightly
  ```
- **Spack**: This project depends on Spack package manager for running
  comparison with Intel's
  [MKL dgemm](https://www.intel.com/content/www/us/en/docs/onemkl/tutorial-c/2021-4/multiplying-matrices-using-dgemm.html)
  library.

### Running

Benchmark algorithms and generate plots:

```bash
cargo bench --bench bench

Options:
  --full 
    When this argument is passed, full benchmark runs up to machine's physical limit for matrix
    storage.
  -- SIZE
    Benchmarks only a specific size
```

## Instructions for running on Toeplitz cluster

To run benchmarks on the Mathematics Department's Toeplitz cluster:

### 1. Pre-compilation

To avoid slow compilation times and network request latency during batch jobs
(on compute nodes), you should compile the project **once** on the login node or
an interactive node for your target microarchitecture.

We provide a pre-setup wrapper script that sources Spack, loads the appropriate
Environment Modules (compilers/MKL), activates the Python 2.7 Conda environment
(for C++ codegen), and compiles both C/C++ and Rust targets.

Run the pre-setup script from the project root:

```bash
./scripts/pre-setup.sh [architecture]
```

Where `[architecture]` is the target CPU microarchitecture of the compute nodes:

- `broadwell`: For the Xeon nodes (`cl1` / `cl2` partitions)
- `znver3`: For the AMD EPYC nodes (`gpu` partition)
- `native`: Compiles for the current host CPU (default)

This script will output:

- Highly optimized Rust benchmark binaries to
  `rust/generated/bin/bench_${architecture}` and
  `rust/generated/bin/base_matmul_curves_${architecture}`.
- C++ binaries to `build/strassen` and `build/matmul_benchmarks`.

### 2. Running Spawned Batch Jobs via SLURM

Rather than running all matrix sizes sequentially in a single monolithic job
(which risks timeout or memory exhaustion), we spawn a separate batch job for
each matrix size.

We provide two scripts to manage job submission for Rust and C++ benchmarks:

#### A. Spawning Rust Benchmark Jobs

To spawn benchmark jobs for all matrix sizes ($2^1, 2^2, \dots, 2^{15}$) for the
Rust project:

```bash
./scripts/submit_rust_jobs.sh
```

- **Options**:
  - Pass `--no-compile` to skip the compilation step if the project is already
    compiled (e.g., if you ran `pre-setup.sh` beforehand):
    ```bash
    ./scripts/submit_gpu_jobs.sh --no-compile
    ```

#### B. Spawning C/C++ Benchmark Jobs

To spawn benchmark jobs for all matrix sizes ($2^1, 2^2, \dots, 2^{15}$) for the
C/C++ project:

```bash
./scripts/submit_c_jobs.sh
```

- **Options**:
  - Pass `--no-compile` to skip the compilation step:
    ```bash
    ./scripts/submit_c_jobs.sh --no-compile
    ```

The jobs are submitted to the SLURM `gpu` partition, allocating 16 CPU cores per
task (`cpus-per-task=16`), a GPU node, and up to 64GB of RAM per matrix size.
Log files are saved to `generated/logs/` (e.g.,
`generated/logs/gpu_matmul_%j.log` and `generated/logs/c_matmul_%j.log`).

#### C. Merging C++ Benchmark Results

After all C++ benchmark jobs are complete, you can merge and convert the results
into CSV format by running:

```bash
python3 scripts/merge_c_results.py
```

This script will look for all output text files under `benchmarks/generated/`
and output:

- `benchmarks/generated/benchmarks_merged.csv` (Long/Tidy format)
- `benchmarks/generated/benchmarks_merged_wide.csv` (Wide format with algorithms
  and modes as columns)

## Fast matrix multiplication

Austin R. Benson and Grey Ballard

This software contains implementations of fast matrix multiplication algorithms
for sequential and shared-memory parallel environments.

To cite this work, please use:

Austin R. Benson and Grey Ballard. "A framework for practical parallel fast
matrix multiplication". In Proceedings of the 20th ACM SIGPLAN Symposium on
Principles and Practice of Parallel Programming (PPoPP), 2015.

An extended version of the paper is available on
[arxiv](http://arxiv.org/pdf/1409.2908v1.pdf).

For references to APA algorithms, see

Grey Ballard, Jack Weissenberger, and Luoping Zhang. "Accelerating Neural
Network Training using Arbitrary Precision Approximating Matrix Multiplication
Algorithms". In Proceedings of the 50th International Conference on Parallel
Processing Workshops (ICPP-W), 2021.

## License

Copyright 2014 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000
with Sandia Corporation, the U.S. Government retains certain rights in this
software.

This software is released under the BSD 2-Clause license. Please see the LICENSE
file.

## Setup

The code requires:

- Intel MKL
- Compiler supporting C++11 and OpenMP

The Makefile depends on an included file that specifies the compiler and the
run-time mode. You must specify this file in the first line of the Makefile. For
an example, see the file `make.incs/make.inc.edison`, which contains the
information for running on NERSC's Edison machine. The `MODE` variable specifies
sequential or parallel mode. The `DEFINES` variable can specify the type of
parallelism if running in parallel mode. The `DEFINES` variable also specifies
the naming convention for BLAS routines. By default, names are considered to be
like "dgemm". However, by defining BLAS_POST, names are considered to be like
"dgemm_", i.e., the routines have a trailing underscore. The `MKL_ROOT` variable
must be set for your machine.

We did most testing using the Intel compiler (icpc). Depending on the version of
g++, the OpenMP task constructs can be different and the hybrid shared-memory
parallel code may crash. Sequential mode, DFS parallel, and BFS parallel should
work with g++.

## Building examples

First, use the code generator to generate the algorithms:

    cd codegen
    bash gen_all_algorithms.sh 0

Some simple codes that use the fast algorithms are in the `examples` directory.
For example, you can build and run the (4, 3, 3) algorithm:

    make fast433
    ./build/fast433

## Building tests

We now assume that all of the algorithms have been gernated with the code
generator (see above). The tests are built and run with:

    make matmul_tests
    ./build/matmul_tests -all 1

The tests are just for correctness of the algorithms, not for performance. You
should see output like:

    STRASSEN_1: 257, 500, 55
    Maximum relative difference: 3.87253e-15

This test runs one step of Strassen's algorithm, multiplying a 257 x 500 matrix
with a 500 x 55 matrix. The maximum relative difference is an error measure:

    max_{ij} |C_{ij} - D_{ij}| / |C_{ij}|,

where C is the result computed with the fast algorithm and D is the result
computed with the classical algorithm. For all of the exact fast algorithms, the
error should be around 1e-14 or 1e-15. The approximate algorithms (e.g., Bini's)
have larger error. Typically, additional recursive steps leads to a larger
error.

## Building with different parallel methods

The BFS, DFS, and HYBRID parallel algorithms are compile-time options. In your
make include file in the `make.incs` directory, to use DFS:

    MODE=openmp
    DEFINES += -D_PARALLEL_=1

Switch the `_PARALLEL_` define to 2 for BFS or 3 for HYBRID. For an example, run

    make fast424
    ./build/fast424

## DGEMM curve benchmarks

Build and run the benchmark for the dgemm curves:

    make dgemm_curves
    ./build/dgemm_curves 1  # N x N x N
    ./build/dgemm_curves 2  # N x 800 x N
    ./build/dgemm_curves 3  # N x 800 x 800

The output is a semi-colon separated list, where each item loooks like:

    M K N num_trials time;

The M, K, and N terms specify the matrix dimensions: M x K multiplied by K x N.
The time is in milliseconds and is the total time to run num_trials multiplies.
For example,

    1200 800 1200 5 104.87;

means that it took 104.87 milliseconds to multiply a 1200 x 800 matrix by a 800
x 1200 matrix five times. To build with parallelism enabled, you need to define
the `_PARALLEL_` (see `make.incs/make.inc.linux`). To run without dynamic
threads (i.e., mkl_set_dynamic(0)), append a second argument, e.g.:

./build/dgemm_curves 1 1 # Square timings without dynamic thread allocation

## Fast algorithms benchmarks

Build benchmarking code for all of the fast algorithms:

    make matmul_benchmarks

The build takes a while because we are compiling all of the algorithms. To run a
small test to benchmark MKL against Strassen with one, two, and three levels of
recursion:

    ./build/matmul_benchmarks -square_test 1

The output format is specified in `data/README.md`.

To run all of the benchmarks for the tall-and-skinny matrix multiplied by a
small square matrix (N x k x k for fixed k):

    ./build/matmul_benchmarks -ts_square_like 1
