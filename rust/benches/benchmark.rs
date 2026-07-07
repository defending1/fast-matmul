mod check_memory_helper;
mod export_helper;
mod job_helper;
mod util;

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion};
use faer::Mat;
use fast_matmul::cp::CP;
use fast_matmul::matmul::{BaseMatMul, MatMul, ParallelismMode, RecursionLimit};
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

    /// Configures Criterion group for single-shot sampling.
    ///
    /// Every benchmark calls the function exactly once per sample.  Ten samples
    /// are collected (Criterion's minimum).  Warmup time is scaled with matrix
    /// size so that CPU caches and thermal state are stabilised before
    /// measurement begins, without wasting minutes on huge matrices.
    fn configure_group_for_size(group: &mut BenchmarkGroup<WallTime>, size: usize) {
        let (samples, warmup_ms, measure_ms) = match size {
            ..=16 => (10, 50, 100),
            17..=64 => (10, 100, 200),
            65..=256 => (10, 200, 500),
            257..=1024 => (10, 500, 1000),
            _ => (10, 1000, 2000),
        };
        group.sample_size(samples);
        group.warm_up_time(std::time::Duration::from_millis(warmup_ms));
        group.measurement_time(std::time::Duration::from_millis(measure_ms));
    }

    /// Computes the total number of benchmarks to run based on sizes, configurations, targets, and algorithms.
    fn compute_total_benchmarks(
        &self,
        sizes_len: usize,
        configs: &[(RecursionLimit, String, String)],
        targets: &[(BaseMatMul, &str, &str)],
        algorithms: &[&str],
    ) -> (usize, usize) {
        let mut per_size = 0;
        if self.run_sequential {
            per_size += 2; // Dgemm-Sequential, Faer-Sequential
        }
        if self.run_parallel {
            per_size += 2; // Dgemm-Parallel, Faer-Parallel
        }
        for (limit, _, _) in configs {
            for _ in targets {
                for _ in algorithms {
                    if self.run_sequential {
                        per_size += 1;
                    }
                    if self.run_parallel {
                        per_size += 2; // DFS, BFS
                        if let RecursionLimit::Depth(_) = limit {
                            per_size += 1; // Hybrid
                        }
                    }
                }
            }
        }
        (sizes_len * per_size, per_size)
    }

    /// Allocates a random `size × size` matrix.
    fn random_matrix(size: usize, rng: &mut impl Rng) -> Mat<f64> {
        Mat::from_fn(size, size, |_, _| rng.gen_range(-1.0..1.0))
    }

    /// Registers a single-shot benchmark with Criterion, printing a progress line.
    ///
    /// Each Criterion sample times exactly one invocation of `f`.  Using
    /// `iter_custom` with a single call prevents Criterion from looping the
    /// function thousands of times, which would be prohibitive for large
    /// matrix sizes.  The mean of the 10 single-shot samples is written to
    /// Criterion's JSON output and picked up by `export_helper`.
    ///
    /// `counter` is a shared mutable index that is incremented on each call,
    /// and `total` is the pre-computed total number of benchmarks in this run.
    fn register_bench<F, O>(
        group: &mut BenchmarkGroup<WallTime>,
        name: &str,
        size: usize,
        counter: &mut usize,
        total: usize,
        mut f: F,
    ) where
        F: FnMut() -> O,
    {
        *counter += 1;
        println!(
            "  Benchmark {} of {}: {} (size {}x{})",
            counter, total, name, size, size
        );
        group.bench_with_input(BenchmarkId::new(name, size), &size, move |bench, &_| {
            bench.iter(&mut f);
        });
    }

    /// Registers the base sequential and parallel benchmarks for one matrix size.
    #[allow(clippy::too_many_arguments)]
    fn bench_base(
        &self,
        group: &mut BenchmarkGroup<WallTime>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
        base_choice: BaseMatMul,
        counter: &mut usize,
        total: usize,
    ) {
        let name = match base_choice {
            BaseMatMul::Faer => "Faer",
            BaseMatMul::Dgemm => "MKL",
        };
        if self.run_sequential {
            Self::register_bench(
                group,
                &format!("{}-Sequential", name),
                size,
                counter,
                total,
                || util::base_matmul(a, b, false, base_choice),
            );
        }
        if self.run_parallel {
            Self::register_bench(
                group,
                &format!("{}-Parallel", name),
                size,
                counter,
                total,
                || util::base_matmul(a, b, true, base_choice),
            );
        }
    }

    /// Registers sequential and parallel CP benchmarks for one algorithm, matrix size, and base matrix multiplication choice.
    ///
    /// # Arguments
    /// * `group` - The Criterion benchmark group.
    /// * `a` - The left operand matrix.
    /// * `b` - The right operand matrix.
    /// * `size` - The matrix dimension size.
    /// * `algo` - The name of the algorithm.
    /// * `mm` - The `MatMul` instance.
    /// * `base_choice` - The base matrix multiplication choice.
    /// * `recursion_limit` - The recursion limit choice.
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
        recursion_limit: RecursionLimit,
        counter: &mut usize,
        total: usize,
    ) {
        let suffix = match base_choice {
            BaseMatMul::Faer => "Faer",
            BaseMatMul::Dgemm => "Dgemm",
        };
        let config_suffix = match recursion_limit {
            RecursionLimit::Depth(level) => format!("level_{}", level),
            RecursionLimit::Cutoff(cutoff) => format!("cutoff_{}", cutoff),
        };
        if self.run_sequential {
            Self::register_bench(
                group,
                &format!("{}-{}-{}/Sequential", algo, suffix, config_suffix),
                size,
                counter,
                total,
                || {
                    mm.cp_matmul(
                        a,
                        b,
                        ParallelismMode::Sequential,
                        base_choice,
                        recursion_limit,
                    )
                },
            );
        }
        if self.run_parallel {
            Self::register_bench(
                group,
                &format!("{}-{}-{}/DFS", algo, suffix, config_suffix),
                size,
                counter,
                total,
                || mm.cp_matmul(a, b, ParallelismMode::Dfs, base_choice, recursion_limit),
            );
            Self::register_bench(
                group,
                &format!("{}-{}-{}/BFS", algo, suffix, config_suffix),
                size,
                counter,
                total,
                || mm.cp_matmul(a, b, ParallelismMode::Bfs, base_choice, recursion_limit),
            );
            if let RecursionLimit::Depth(_) = recursion_limit {
                Self::register_bench(
                    group,
                    &format!("{}-{}-{}/Hybrid", algo, suffix, config_suffix),
                    size,
                    counter,
                    total,
                    || mm.cp_matmul(a, b, ParallelismMode::Hybrid, base_choice, recursion_limit),
                );
            }
        }
    }

    /// Runs benchmarks using the programmatic Criterion API and exports the results to CSV
    /// using the specified base matrix multiplication choice and recursion limit.
    ///
    /// Stops running benchmarks if a matrix size exceeds the machine's memory capacity,
    /// printing a clean warning, and still exporting results for the sizes that completed.
    ///
    /// # Arguments
    /// * `sizes` - A slice of matrix dimension sizes to benchmark.
    /// * `algorithms` - A slice of algorithm names.
    /// * `filename` - The output CSV filename.
    /// * `base_choice` - The base matrix multiplication choice.
    /// * `recursion_limit` - The recursion limit choice.
    pub fn run(
        &self,
        sizes: &[usize],
        algorithms: &[&str],
        filename: &str,
        configs: &[(RecursionLimit, String, String)],
        targets: &[(BaseMatMul, &str, &str)],
    ) -> Result<(), std::io::Error> {
        // --- Compute total benchmark count upfront for progress display ---
        let (total, per_size) = self.compute_total_benchmarks(sizes.len(), configs, targets, algorithms);

        println!(
            "Running {} benchmarks across {} sizes ({} per size)...",
            total,
            sizes.len(),
            per_size
        );

        // Criterion configured without HTML plots; CSV export is handled separately.
        let mut c = Criterion::default().without_plots();
        let mut group = c.benchmark_group("Matrix Multiplication");
        let mut rng = rand::thread_rng();

        let cps: Vec<(&str, CP)> = algorithms
            .iter()
            .map(|&algo| {
                println!("Loading CP decomposition for '{}'...", algo);
                (algo, CP::load(algo))
            })
            .collect();

        let mut counter: usize = 0;
        let mut successful_sizes = Vec::new();

        for &size in sizes {
            if let Err(e) = check_memory_helper::CheckMemoryHelper::check_size_supported(size) {
                println!("\n--- Gracefully Stopping Benchmarks ---");
                println!("Reason: {}", e);
                println!(
                    "Writing results for completed sizes {:?} and exiting...",
                    successful_sizes
                );
                break;
            }

            println!("\nSize {}x{}:", size, size);
            Self::configure_group_for_size(&mut group, size);

            let a = Self::random_matrix(size, &mut rng);
            let b = Self::random_matrix(size, &mut rng);

            // Run baseline benchmarks exactly once per size
            self.bench_base(
                &mut group,
                &a,
                &b,
                size,
                BaseMatMul::Dgemm,
                &mut counter,
                total,
            );
            self.bench_base(
                &mut group,
                &a,
                &b,
                size,
                BaseMatMul::Faer,
                &mut counter,
                total,
            );

            // Run CP decomposition algorithms for all configs
            for (limit, _, _) in configs {
                for &(base_choice, _, _) in targets {
                    for &(algo, ref cp) in &cps {
                        let mm = MatMul::with_cp(cp);
                        self.bench_cp(
                            &mut group,
                            &a,
                            &b,
                            size,
                            algo,
                            &mm,
                            base_choice,
                            *limit,
                            &mut counter,
                            total,
                        );
                    }
                }
            }
            successful_sizes.push(size);

            // Checkpoint-save: export the results computed so far (without running plot script)
            for (limit, _, _) in configs {
                for &(base_choice, _, _) in targets {
                    export_helper::export_results_to_csv(
                        &successful_sizes,
                        algorithms,
                        filename,
                        base_choice,
                        *limit,
                        false,
                    )?;
                }
            }
        }

        group.finish();

        if !successful_sizes.is_empty() {
            // Write final results for all configs, only running the plot script on the last configuration
            let last_idx = configs.len() * targets.len() - 1;
            let mut idx = 0;
            for (limit, _, _) in configs {
                for &(base_choice, _, _) in targets {
                    let is_last = idx == last_idx;
                    export_helper::export_results_to_csv(
                        &successful_sizes,
                        algorithms,
                        filename,
                        base_choice,
                        *limit,
                        self.run_plot && is_last,
                    )?;
                    idx += 1;
                }
            }
        } else {
            println!("No matrix sizes were benchmarked.");
        }

        Ok(())
    }
}

/// Entry point for running the matrix multiplication benchmarks.
fn main() {
    job_helper::handle_job_dependent_execution();

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

    // Running with --full will dynamically check system memory and stop before exceeding limits.
    let n_limit = if full { 15 } else { 13 };
    let sizes: Vec<usize> = (1..=n_limit).map(|n| 1usize << n).collect(); // 2, 4, ..., 2^N

    let cutoffs = [256, 512, 1024, 2048];
    let recursion_levels = [1, 2, 3];
    let algorithms = &["strassen"];

    let targets = [
        (BaseMatMul::Faer, "Faer", "faer"),
        (BaseMatMul::Dgemm, "Dgemm", "dgemm"),
    ];

    let configs = {
        let mut list = Vec::new();
        for &cutoff in &cutoffs {
            list.push((
                RecursionLimit::Cutoff(cutoff),
                format!("benchmark_cutoff_{}", cutoff),
                format!("cutoff: {}", cutoff),
            ));
        }
        for &level in &recursion_levels {
            list.push((
                RecursionLimit::Depth(level),
                format!("benchmark_level_{}", level),
                format!("level: {}", level),
            ));
        }
        list
    };

    let out_file = {
        let job_id = std::env::var("SLURM_JOB_ID")
            .or_else(|_| std::env::var("PBS_JOBID"))
            .or_else(|_| std::env::var("RUN_ID"));
        let root = util::get_project_root();
        let csv_dir = root.join("generated").join("csv");
        let path = if let Ok(id) = job_id {
            csv_dir.join(format!("benchmark_results_{}.csv", id))
        } else {
            csv_dir.join("benchmark_results.csv")
        };
        path.to_string_lossy().to_string()
    };

    if plot_only {
        println!("Plot-only mode: Regenerating CSV results from cached Criterion data...");
        for (limit, _file_prefix, label) in &configs {
            for &(base, name, _suffix) in &targets {
                if let Err(e) = export_helper::export_results_to_csv(
                    &sizes, algorithms, &out_file, base, *limit, true,
                ) {
                    eprintln!("Failed to export {} CSV for {}: {:?}", name, label, e);
                } else {
                    println!(
                        "{} CSV results ({}) successfully updated from cache.",
                        name, label
                    );
                }
            }
        }
    } else {
        println!("\n--- Running Matrix Multiplication Benchmarks ---");
        let bench = Benchmark::new(run_sequential, run_parallel, run_plot);

        if let Err(e) = bench.run(&sizes, algorithms, &out_file, &configs, &targets) {
            eprintln!("Failed to run benchmarks: {:?}", e);
        } else {
            println!(
                "All benchmark results successfully written to {}",
                out_file
            );
        }
    }

    // Clean up job-dependent clone on exit if we are running the clone
    job_helper::cleanup_job_dependent_execution();
}
