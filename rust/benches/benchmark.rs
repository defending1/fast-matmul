mod check_memory_helper;
mod export_helper;
mod job_helper;
mod util;

use faer::Mat;
use fast_matmul::cp::CP;
use fast_matmul::matmul::{BaseMatMul, MatMul, ParallelismMode, RecursionLimit};
use rand::Rng;
use std::collections::HashMap;

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
    ///
    /// # Arguments
    /// * `run_sequential` - Whether to benchmark sequential implementations.
    /// * `run_parallel` - Whether to benchmark parallel implementations.
    /// * `run_plot` - Whether to generate performance plots.
    pub fn new(run_sequential: bool, run_parallel: bool, run_plot: bool) -> Self {
        Self {
            run_sequential,
            run_parallel,
            run_plot,
        }
    }

    /// Returns the warmup and measurement time in milliseconds for a given size.
    ///
    /// # Arguments
    /// * `size` - The matrix size dimension.
    fn get_timings_for_size(size: usize) -> (u64, u64) {
        match size {
            ..=16 => (50, 100),
            17..=64 => (100, 200),
            65..=256 => (200, 500),
            257..=1024 => (500, 1000),
            _ => (1000, 2000),
        }
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

    /// Measures and registers a single benchmark, recording its timing.
    #[allow(clippy::too_many_arguments)]
    fn register_bench<F, O>(
        new_timings: &mut HashMap<String, HashMap<usize, f64>>,
        name: &str,
        size: usize,
        counter: &mut usize,
        total: usize,
        warmup_ms: u64,
        measure_ms: u64,
        f: F,
    ) where
        F: FnMut() -> O,
    {
        *counter += 1;
        println!(
            "  Benchmark {} of {}: {} (size {}x{})",
            counter, total, name, size, size
        );
        let time_s = util::run_benchmark_minstant(warmup_ms, measure_ms, f);
        new_timings
            .entry(name.to_string())
            .or_default()
            .insert(size, time_s);
    }

    /// Registers the base sequential and parallel benchmarks for one matrix size.
    #[allow(clippy::too_many_arguments)]
    fn bench_base(
        &self,
        new_timings: &mut HashMap<String, HashMap<usize, f64>>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
        base_choice: BaseMatMul,
        counter: &mut usize,
        total: usize,
        warmup_ms: u64,
        measure_ms: u64,
    ) {
        let name = match base_choice {
            BaseMatMul::Faer => "Faer",
            BaseMatMul::Dgemm => "MKL",
        };
        if self.run_sequential {
            Self::register_bench(
                new_timings,
                &format!("{}-Sequential", name),
                size,
                counter,
                total,
                warmup_ms,
                measure_ms,
                || util::base_matmul(a, b, false, base_choice),
            );
        }
        if self.run_parallel {
            Self::register_bench(
                new_timings,
                &format!("{}-Parallel", name),
                size,
                counter,
                total,
                warmup_ms,
                measure_ms,
                || util::base_matmul(a, b, true, base_choice),
            );
        }
    }

    /// Registers sequential and parallel CP benchmarks for one algorithm, matrix size, and base matrix multiplication choice.
    #[allow(clippy::too_many_arguments)]
    fn bench_cp(
        &self,
        new_timings: &mut HashMap<String, HashMap<usize, f64>>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
        algo: &str,
        mm: &MatMul<'_>,
        base_choice: BaseMatMul,
        recursion_limit: RecursionLimit,
        counter: &mut usize,
        total: usize,
        warmup_ms: u64,
        measure_ms: u64,
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
                new_timings,
                &format!("{}-{}-{}_Sequential", algo, suffix, config_suffix),
                size,
                counter,
                total,
                warmup_ms,
                measure_ms,
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
                new_timings,
                &format!("{}-{}-{}_DFS", algo, suffix, config_suffix),
                size,
                counter,
                total,
                warmup_ms,
                measure_ms,
                || mm.cp_matmul(a, b, ParallelismMode::Dfs, base_choice, recursion_limit),
            );
            Self::register_bench(
                new_timings,
                &format!("{}-{}-{}_BFS", algo, suffix, config_suffix),
                size,
                counter,
                total,
                warmup_ms,
                measure_ms,
                || mm.cp_matmul(a, b, ParallelismMode::Bfs, base_choice, recursion_limit),
            );
            if let RecursionLimit::Depth(_) = recursion_limit {
                Self::register_bench(
                    new_timings,
                    &format!("{}-{}-{}_Hybrid", algo, suffix, config_suffix),
                    size,
                    counter,
                    total,
                    warmup_ms,
                    measure_ms,
                    || mm.cp_matmul(a, b, ParallelismMode::Hybrid, base_choice, recursion_limit),
                );
            }
        }
    }

    /// Runs benchmarks and exports the results to CSV.
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

        let mut new_timings = HashMap::new();
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
            let (warmup_ms, measure_ms) = Self::get_timings_for_size(size);

            let a = Self::random_matrix(size, &mut rng);
            let b = Self::random_matrix(size, &mut rng);

            // Run baseline benchmarks exactly once per size
            self.bench_base(
                &mut new_timings,
                &a,
                &b,
                size,
                BaseMatMul::Dgemm,
                &mut counter,
                total,
                warmup_ms,
                measure_ms,
            );
            self.bench_base(
                &mut new_timings,
                &a,
                &b,
                size,
                BaseMatMul::Faer,
                &mut counter,
                total,
                warmup_ms,
                measure_ms,
            );

            // Run CP decomposition algorithms for all configs
            for (limit, _, _) in configs {
                for &(base_choice, _, _) in targets {
                    for &(algo, ref cp) in &cps {
                        let mm = MatMul::with_cp(cp);
                        self.bench_cp(
                            &mut new_timings,
                            &a,
                            &b,
                            size,
                            algo,
                            &mm,
                            base_choice,
                            *limit,
                            &mut counter,
                            total,
                            warmup_ms,
                            measure_ms,
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
                        &new_timings,
                    )?;
                }
            }
        }

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
                        &new_timings,
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

    // Parameter parsing
    let mut param_cutoffs = Vec::new();
    let mut param_levels = Vec::new();
    let mut param_sizes = Vec::new();
    let mut param_out = None;

    let mut idx_arg = 0;
    while idx_arg < args.len() {
        if (args[idx_arg] == "--cutoff" || args[idx_arg] == "--cutoffs") && idx_arg + 1 < args.len() {
            for s in args[idx_arg + 1].split(',') {
                if let Ok(v) = s.trim().parse::<usize>() {
                    param_cutoffs.push(v);
                }
            }
            idx_arg += 2;
        } else if (args[idx_arg] == "--level" || args[idx_arg] == "--levels") && idx_arg + 1 < args.len() {
            for s in args[idx_arg + 1].split(',') {
                if let Ok(v) = s.trim().parse::<usize>() {
                    param_levels.push(v);
                }
            }
            idx_arg += 2;
        } else if (args[idx_arg] == "--size" || args[idx_arg] == "--sizes") && idx_arg + 1 < args.len() {
            for s in args[idx_arg + 1].split(',') {
                if let Ok(v) = s.trim().parse::<usize>() {
                    param_sizes.push(v);
                }
            }
            idx_arg += 2;
        } else if (args[idx_arg] == "--out" || args[idx_arg] == "-o") && idx_arg + 1 < args.len() {
            param_out = Some(args[idx_arg + 1].clone());
            idx_arg += 2;
        } else {
            idx_arg += 1;
        }
    }

    let is_param_mode = !param_cutoffs.is_empty() || !param_levels.is_empty() || !param_sizes.is_empty();

    // Running with --full will dynamically check system memory and stop before exceeding limits.
    let sizes: Vec<usize> = if !param_sizes.is_empty() {
        param_sizes
    } else {
        let n_limit = if full { 15 } else { 13 };
        (1..=n_limit).map(|n| 1usize << n).collect()
    };

    let cutoffs = [256, 512, 1024, 2048];
    let recursion_levels = [1, 2, 3];
    let algorithms = &["strassen"];

    let targets = [
        (BaseMatMul::Faer, "Faer", "faer"),
        (BaseMatMul::Dgemm, "Dgemm", "dgemm"),
    ];

    let configs = if is_param_mode {
        let mut list = Vec::new();
        for &cutoff in &param_cutoffs {
            list.push((
                RecursionLimit::Cutoff(cutoff),
                format!("benchmark_cutoff_{}", cutoff),
                format!("cutoff: {}", cutoff),
            ));
        }
        for &level in &param_levels {
            list.push((
                RecursionLimit::Depth(level),
                format!("benchmark_level_{}", level),
                format!("level: {}", level),
            ));
        }
        if list.is_empty() {
            // Default configs if cutoff and level are both unspecified
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
        }
        list
    } else {
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

    let out_file = if let Some(out) = param_out {
        out
    } else {
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
        println!("Plot-only mode: Regenerating CSV results from cached data...");
        let empty_timings = HashMap::new();
        for (limit, _file_prefix, label) in &configs {
            for &(base, name, _suffix) in &targets {
                if let Err(e) = export_helper::export_results_to_csv(
                    &sizes,
                    algorithms,
                    &out_file,
                    base,
                    *limit,
                    true,
                    &empty_timings,
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
