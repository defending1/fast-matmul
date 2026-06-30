mod export_helper;
mod util;

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion};
use faer::Mat;
use fast_matmul::cp::CP;
use fast_matmul::matmul::{BaseMatMul, MatMul, ParallelismMode};
use rand::Rng;

/// A struct for running benchmarks on various matrix multiplication algorithms.
pub struct Benchmark {
    run_sequential: bool,
    run_parallel: bool,
    run_plot: bool,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new(true, true, false)
    }
}

impl Benchmark {
    /// Creates a new `Benchmark` instance.
    pub fn new(run_sequential: bool, run_parallel: bool, run_plot: bool) -> Self {
        Self {
            run_sequential,
            run_parallel,
            run_plot,
        }
    }

    /// Adjusts Criterion group sampling parameters based on matrix size.
    fn configure_group_for_size(group: &mut BenchmarkGroup<WallTime>, size: usize) {
        let (samples, warmup_ms, measure_ms) = match size {
            ..=16 => (30, 50, 100),
            17..=64 => (20, 100, 200),
            65..=256 => (10, 200, 500),
            257..=1024 => (10, 500, 1000),
            _ => (10, 100, 200),
        };
        group.sample_size(samples);
        group.warm_up_time(std::time::Duration::from_millis(warmup_ms));
        group.measurement_time(std::time::Duration::from_millis(measure_ms));
    }

    /// Allocates a random `size × size` matrix.
    fn random_matrix(size: usize, rng: &mut impl Rng) -> Mat<f64> {
        Mat::from_fn(size, size, |_, _| rng.gen_range(-1.0..1.0))
    }

    /// Helper to register a benchmark with Criterion.
    fn register_bench<F, O>(group: &mut BenchmarkGroup<WallTime>, name: &str, size: usize, mut f: F)
    where
        F: FnMut() -> O,
    {
        group.bench_with_input(BenchmarkId::new(name, size), &size, move |bench, &_| {
            bench.iter(&mut f);
        });
    }

    /// Registers the MKL sequential and parallel benchmarks for one matrix size.
    fn bench_mkl(
        &self,
        group: &mut BenchmarkGroup<WallTime>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
    ) {
        if self.run_sequential {
            Self::register_bench(group, "MKL-Sequential", size, || {
                util::base_matmul(a, b, false, BaseMatMul::Dgemm)
            });
        }
        if self.run_parallel {
            Self::register_bench(group, "MKL-Parallel", size, || {
                util::base_matmul(a, b, true, BaseMatMul::Dgemm)
            });
        }
    }

    /// Registers the Faer sequential and parallel benchmarks for one matrix size.
    fn bench_faer(
        &self,
        group: &mut BenchmarkGroup<WallTime>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
    ) {
        if self.run_sequential {
            Self::register_bench(group, "Faer-Sequential", size, || {
                util::base_matmul(a, b, false, BaseMatMul::Faer)
            });
        }
        if self.run_parallel {
            Self::register_bench(group, "Faer-Parallel", size, || {
                util::base_matmul(a, b, true, BaseMatMul::Faer)
            });
        }
    }

    /// Registers sequential and parallel CP benchmarks for one algorithm, matrix size, and base matrix multiplication choice.
    #[allow(clippy::too_many_arguments)]
    fn bench_cp(
        &self,
        group: &mut BenchmarkGroup<WallTime>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
        algo: &str,
        mm: &MatMul<'_>,
        base_choice: BaseMatMul,
    ) {
        let suffix = match base_choice {
            BaseMatMul::Faer => "Faer",
            BaseMatMul::Dgemm => "Dgemm",
        };
        if self.run_sequential {
            Self::register_bench(
                group,
                &format!("{}-{}/Sequential", algo, suffix),
                size,
                || mm.cp_matmul(a, b, ParallelismMode::Sequential, base_choice),
            );
        }
        if self.run_parallel {
            Self::register_bench(group, &format!("{}-{}/DFS", algo, suffix), size, || {
                mm.cp_matmul(a, b, ParallelismMode::Dfs, base_choice)
            });
            Self::register_bench(group, &format!("{}-{}/BFS", algo, suffix), size, || {
                mm.cp_matmul(a, b, ParallelismMode::Bfs, base_choice)
            });
            Self::register_bench(group, &format!("{}-{}/Hybrid", algo, suffix), size, || {
                mm.cp_matmul(a, b, ParallelismMode::Hybrid, base_choice)
            });
        }
    }

    /// Checks if the matrix size is supported by the machine's memory and limits.
    ///
    /// Returns `Ok(())` if the size is supported, or an `Err(String)` containing
    /// a descriptive message of why it is not supported.
    pub fn check_size_supported(&self, size: usize) -> Result<(), String> {
        if size == 0 {
            return Err("Matrix size must be greater than 0.".to_string());
        }

        // 1. Check for arithmetic overflow in size calculations
        let elements = size.checked_mul(size).ok_or_else(|| {
            format!(
                "Matrix size {}x{} would overflow usize elements count.",
                size, size
            )
        })?;

        let bytes_per_matrix = elements
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                format!(
                    "Matrix size {}x{} would overflow memory byte count.",
                    size, size
                )
            })?;

        // Rust's allocator limit is isize::MAX
        if bytes_per_matrix > isize::MAX as usize {
            return Err(format!(
                "Matrix size {}x{} requires {} bytes, which exceeds Rust's maximum allocation limit of {} bytes.",
                size,
                size,
                bytes_per_matrix,
                isize::MAX
            ));
        }

        // Estimate total peak memory required for the benchmark at this size.
        // We run multiple algorithms (MKL, Faer, Strassen single/multi-thread).
        // Strassen in parallel mode with Rayon has the highest peak memory overhead.
        // Let's estimate peak memory overhead as:
        // Input matrices (A, B) + output matrix (C) + concurrent workspace.
        // A safe factor for Strassen parallel is (3 + T * 1.5) where T is the thread count.
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let multiplier = 3.0 + (num_threads as f64).min(7.0) * 1.5;
        let estimated_required_bytes = (bytes_per_matrix as f64 * multiplier) as u64;

        if let Some(avail_bytes) = get_available_memory() {
            // Keep a safety buffer: 10% of available memory or at least 256MB free
            let safety_buffer = (avail_bytes / 10).max(256 * 1024 * 1024);
            if estimated_required_bytes + safety_buffer > avail_bytes {
                return Err(format!(
                    "Matrix size {}x{} requires estimated {} MB of memory (with safety buffer), but only {} MB is available.",
                    size,
                    size,
                    (estimated_required_bytes + safety_buffer) / (1024 * 1024),
                    avail_bytes / (1024 * 1024)
                ));
            }
        }

        Ok(())
    }

    /// Runs benchmarks using the programmatic Criterion API and exports the results to CSV
    /// using the specified base matrix multiplication choice.
    ///
    /// Stops running benchmarks if a matrix size exceeds the machine's memory capacity,
    /// printing a clean warning, and still exporting results for the sizes that completed.
    pub fn run(
        &self,
        sizes: &[usize],
        algorithms: &[&str],
        filename: &str,
        base_choice: BaseMatMul,
    ) -> Result<(), std::io::Error> {
        println!("Running programmatic Criterion benchmarks...");

        let mut c = Criterion::default();
        let mut group = c.benchmark_group("Matrix Multiplication");
        let mut rng = rand::thread_rng();

        let cps: Vec<(&str, CP)> = algorithms
            .iter()
            .map(|&algo| {
                println!("Loading CP decomposition for '{}'...", algo);
                (algo, CP::load(algo))
            })
            .collect();

        let mut successful_sizes = Vec::new();

        for &size in sizes {
            if let Err(e) = self.check_size_supported(size) {
                println!("\n--- Gracefully Stopping Benchmarks ---");
                println!("Reason: {}", e);
                println!(
                    "Writing results for completed sizes {:?} and exiting...",
                    successful_sizes
                );
                break;
            }

            println!("Benchmarking size {}x{} with Criterion...", size, size);
            Self::configure_group_for_size(&mut group, size);

            let a = Self::random_matrix(size, &mut rng);
            let b = Self::random_matrix(size, &mut rng);

            self.bench_mkl(&mut group, &a, &b, size);

            self.bench_faer(&mut group, &a, &b, size);

            for &(algo, ref cp) in &cps {
                let mm = MatMul::with_cp(cp);
                self.bench_cp(&mut group, &a, &b, size, algo, &mm, base_choice);
            }
            successful_sizes.push(size);
        }

        group.finish();

        if !successful_sizes.is_empty() {
            export_helper::export_results_to_csv(
                &successful_sizes,
                algorithms,
                filename,
                base_choice,
                self.run_plot,
            )?;
        } else {
            println!("No matrix sizes were benchmarked.");
        }

        Ok(())
    }
}

/// Helper function to parse `/proc/meminfo` and return the available memory in bytes.
/// If not on Linux or if it fails, returns `None`.
fn get_available_memory() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if line.to_ascii_lowercase().starts_with("memavailable:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(kb) = parts.get(1).and_then(|s| s.parse::<u64>().ok()) {
                return Some(kb * 1024); // Convert kB to bytes
            }
        }
    }
    None
}

/// Entry point for running the matrix multiplication benchmarks.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let plot_only = args.iter().any(|arg| arg == "--plot-only" || arg == "-p");

    let full = args.iter().any(|arg| arg == "--full");

    let has_seq_flag = args
        .iter()
        .any(|arg| arg == "--sequential" || arg == "--seq");
    let has_par_flag = args.iter().any(|arg| arg == "--parallel" || arg == "--par");

    // Default: if neither flag is specified, we run both
    let run_sequential = has_seq_flag || !has_par_flag;
    let run_parallel = has_par_flag || !has_seq_flag;
    let run_plot = args.iter().any(|arg| arg == "--plot");

    // Default limit is 2^10 (1024). Under --full, we run up to 2^20 (1,048,576),
    // which will dynamically check system memory and stop before exceeding limits.
    let n_limit = if full { 20 } else { 11 };
    let sizes: Vec<usize> = (1..=n_limit).map(|n| 1usize << n).collect(); // 2, 4, ..., 2^N

    let csv_file_faer = "generated/benchmark_results_faer.csv";
    let csv_file_dgemm = "generated/benchmark_results_dgemm.csv";
    let algorithms = &["strassen"];

    if plot_only {
        println!("Plot-only mode: Regenerating CSV results from cached Criterion data...");
        if let Err(e) = export_helper::export_results_to_csv(
            &sizes,
            algorithms,
            csv_file_faer,
            BaseMatMul::Faer,
            true,
        ) {
            eprintln!("Failed to export Faer CSV: {:?}", e);
        } else {
            println!("Faer CSV results successfully updated from cache.");
        }
        if let Err(e) = export_helper::export_results_to_csv(
            &sizes,
            algorithms,
            csv_file_dgemm,
            BaseMatMul::Dgemm,
            true,
        ) {
            eprintln!("Failed to export Dgemm CSV: {:?}", e);
        } else {
            println!("Dgemm CSV results successfully updated from cache.");
        }
    } else {
        println!("\n--- Running Matrix Multiplication Benchmarks ---");
        let bench = Benchmark::new(run_sequential, run_parallel, run_plot);

        println!("\n--- Benchmark Set 1/2: Using Faer Base MatMul ---");
        if let Err(e) = bench.run(&sizes, algorithms, csv_file_faer, BaseMatMul::Faer) {
            eprintln!("Failed to run Faer benchmarks: {:?}", e);
        } else {
            println!(
                "Faer benchmark results successfully written to {}",
                csv_file_faer
            );
        }

        println!("\n--- Benchmark Set 2/2: Using MKL/Dgemm Base MatMul ---");
        if let Err(e) = bench.run(&sizes, algorithms, csv_file_dgemm, BaseMatMul::Dgemm) {
            eprintln!("Failed to run Dgemm benchmarks: {:?}", e);
        } else {
            println!(
                "Dgemm benchmark results successfully written to {}",
                csv_file_dgemm
            );
        }
    }
}
