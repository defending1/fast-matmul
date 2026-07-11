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



    /// Allocates a random `size × size` matrix.
    fn random_matrix(size: usize, rng: &mut impl Rng) -> Mat<f64> {
        Mat::from_fn(size, size, |_, _| rng.gen_range(-1.0..1.0))
    }

    /// Measures and registers a single benchmark, recording its timing.
    fn register_bench<F, O>(
        new_timings: &mut HashMap<String, HashMap<usize, f64>>,
        name: &str,
        size: usize,
        warmup_ms: u64,
        measure_ms: u64,
        f: F,
    ) where
        F: FnMut() -> O,
    {
        println!(
            "  Benchmark: {} (size {}x{})",
            name, size, size
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
                warmup_ms,
                measure_ms,
                || mm.cp_matmul(a, b, ParallelismMode::Dfs, base_choice, recursion_limit),
            );
            Self::register_bench(
                new_timings,
                &format!("{}-{}-{}_BFS", algo, suffix, config_suffix),
                size,
                warmup_ms,
                measure_ms,
                || mm.cp_matmul(a, b, ParallelismMode::Bfs, base_choice, recursion_limit),
            );
            if let RecursionLimit::Depth(_) = recursion_limit {
                Self::register_bench(
                    new_timings,
                    &format!("{}-{}-{}_Hybrid", algo, suffix, config_suffix),
                    size,
                    warmup_ms,
                    measure_ms,
                    || mm.cp_matmul(a, b, ParallelismMode::Hybrid, base_choice, recursion_limit),
                );
            }
        }
    }
    /// Runs benchmarks and exports the results to CSV.
    ///
    /// # Arguments
    /// * `size` - The matrix size dimension.
    /// * `algorithms` - List of algorithms.
    /// * `filename` - The output CSV filename.
    /// * `config` - A single configuration tuple, optional if matmul_mode is "base".
    /// * `targets` - Base target matrix multiplication algorithms.
    /// * `matmul_mode` - The benchmark mode: "base" or "strassen".
    pub fn run(
        &self,
        size: usize,
        algorithms: &[&str],
        filename: &str,
        config: Option<(RecursionLimit, String, String)>,
        targets: &[(BaseMatMul, &str, &str)],
        matmul_mode: &str,
    ) -> Result<(), std::io::Error> {
        println!(
            "Running benchmarks for size {}x{}...",
            size,
            size
        );

        let mut new_timings = HashMap::new();
        let mut rng = rand::thread_rng();

        let cps: Vec<(&str, CP)> = if matmul_mode == "base" {
            Vec::new()
        } else {
            algorithms
                .iter()
                .map(|&algo| {
                    println!("Loading CP decomposition for '{}'...", algo);
                    (algo, CP::load(algo))
                })
                .collect()
        };

        if let Err(e) = check_memory_helper::CheckMemoryHelper::check_size_supported(size) {
            println!("\n--- Gracefully Stopping Benchmarks ---");
            println!("Reason: {}", e);
            return Ok(());
        }

        println!("\nSize {}x{}:", size, size);
        let (warmup_ms, measure_ms) = Self::get_timings_for_size(size);

        let a = Self::random_matrix(size, &mut rng);
        let b = Self::random_matrix(size, &mut rng);

        if matmul_mode == "base" {
            // Run baseline benchmarks exactly once per size
            self.bench_base(
                &mut new_timings,
                &a,
                &b,
                size,
                BaseMatMul::Dgemm,
                warmup_ms,
                measure_ms,
            );
            self.bench_base(
                &mut new_timings,
                &a,
                &b,
                size,
                BaseMatMul::Faer,
                warmup_ms,
                measure_ms,
            );
        } else {
            let (limit, _, _) = config.as_ref().unwrap();
            // Run CP decomposition algorithms
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
                        warmup_ms,
                        measure_ms,
                    );
                }
            }
        }

        // Export results
        if matmul_mode == "base" {
            export_helper::export_base_results_to_csv(
                size,
                filename,
                &new_timings,
            )?;
        } else {
            let (limit, _, _) = config.as_ref().unwrap();
            let last_idx = targets.len() - 1;
            for (idx, &(base_choice, _, _)) in targets.iter().enumerate() {
                let is_last = idx == last_idx;
                export_helper::export_results_to_csv(
                    &[size],
                    algorithms,
                    filename,
                    base_choice,
                    *limit,
                    self.run_plot && is_last,
                    &new_timings,
                )?;
            }
        }

        Ok(())
    }
}

/// Entry point for running the matrix multiplication benchmarks.
fn main() {
    job_helper::handle_job_dependent_execution();
    let args: Vec<String> = std::env::args().collect();

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
    let mut matmul_mode = "strassen".to_string();

    let mut idx_arg = 0;
    while idx_arg < args.len() {
        if args[idx_arg] == "--matmul" && idx_arg + 1 < args.len() {
            matmul_mode = args[idx_arg + 1].clone();
            idx_arg += 2;
        } else if (args[idx_arg] == "--cutoff" || args[idx_arg] == "--cutoffs") && idx_arg + 1 < args.len() {
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

    if matmul_mode == "base" {
        if param_sizes.is_empty() {
            eprintln!("Error: Base benchmark must be run with --size.");
            std::process::exit(1);
        }
    } else if matmul_mode == "strassen" {
        let is_param_mode = (!param_cutoffs.is_empty() || !param_levels.is_empty()) && !param_sizes.is_empty();
        if !is_param_mode {
            eprintln!("Error: Strassen benchmark must be run in parameter mode. Provide --size and either --cutoff or --level.");
            std::process::exit(1);
        }
    } else {
        eprintln!("Error: Unknown matmul mode '{}'. Supported modes: base, strassen", matmul_mode);
        std::process::exit(1);
    }

    let size = param_sizes[0];
    let config = if matmul_mode == "base" {
        None
    } else if !param_cutoffs.is_empty() {
        let cutoff = param_cutoffs[0];
        Some((
            RecursionLimit::Cutoff(cutoff),
            format!("benchmark_cutoff_{}", cutoff),
            format!("cutoff: {}", cutoff),
        ))
    } else {
        let level = param_levels[0];
        Some((
            RecursionLimit::Depth(level),
            format!("benchmark_level_{}", level),
            format!("level: {}", level),
        ))
    };

    let algorithms = &["strassen"];

    let targets = [
        (BaseMatMul::Faer, "Faer", "faer"),
        (BaseMatMul::Dgemm, "Dgemm", "dgemm"),
    ];

    let out_file = if let Some(out) = param_out {
        out
    } else {
        let job_id = std::env::var("SLURM_JOB_ID")
            .or_else(|_| std::env::var("PBS_JOBID"))
            .or_else(|_| std::env::var("RUN_ID"));
        let root = util::get_project_root();
        let csv_dir = root.join("generated").join("csv");
        let path = if let Ok(id) = job_id {
            if matmul_mode == "base" {
                csv_dir.join(format!("benchmark_results_base_{}.csv", id))
            } else {
                csv_dir.join(format!("benchmark_results_{}.csv", id))
            }
        } else {
            if matmul_mode == "base" {
                csv_dir.join("benchmark_results_base.csv")
            } else {
                csv_dir.join("benchmark_results.csv")
            }
        };
        path.to_string_lossy().to_string()
    };

    println!("\n--- Running Matrix Multiplication Benchmarks ---");
    let bench = Benchmark::new(run_sequential, run_parallel, run_plot);

    if let Err(e) = bench.run(size, algorithms, &out_file, config, &targets, &matmul_mode) {
        eprintln!("Failed to run benchmarks: {:?}", e);
    } else {
        println!(
            "All benchmark results successfully written to {}",
            out_file
        );
    }

    // Clean up job-dependent clone on exit if we are running the clone
    job_helper::cleanup_job_dependent_execution();
}
