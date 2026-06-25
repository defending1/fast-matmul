use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion};
use faer::Mat;
use fast_matmul::cp::CP;
use fast_matmul::matmul::{MatMul, ParallelismMode};
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// A struct for running benchmarks on various matrix multiplication algorithms.
pub struct Benchmark;

/// Mapping from a benchmark name folder to its corresponding CSV header.
struct ColumnMapping {
    header: String,
    folder: String,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl Benchmark {
    /// Creates a new `Benchmark` instance.
    pub fn new() -> Self {
        Self
    }

    /// Helper to read a single point estimate of the mean from Criterion's JSON files, converting it to seconds.
    fn get_criterion_time(folder_name: &str, size: usize) -> Option<f64> {
        let path = Path::new("target/criterion/Matrix Multiplication")
            .join(folder_name)
            .join(size.to_string())
            .join("new/estimates.json");
        if !path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(&path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        let nanoseconds = json.get("mean")?.get("point_estimate")?.as_f64()?;
        Some(nanoseconds / 1_000_000_000.0)
    }

    /// Reads existing benchmark CSV results to avoid overwriting unrelated cached data.
    fn read_existing_csv(filename: &str) -> HashMap<(usize, String), f64> {
        let mut map = HashMap::new();
        let content = match std::fs::read_to_string(filename) {
            Ok(c) => c,
            Err(_) => return map,
        };
        let mut lines = content.lines();
        let header_line = match lines.next() {
            Some(h) => h,
            None => return map,
        };
        let headers: Vec<String> = header_line
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        for line in lines {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.is_empty() || parts[0].trim().is_empty() {
                continue;
            }
            let size: usize = match parts[0].trim().parse() {
                Ok(s) => s,
                Err(_) => continue,
            };
            for (i, part) in parts.iter().enumerate().skip(1) {
                if let Some(val) = (i < headers.len())
                    .then(|| part.trim().parse::<f64>().ok())
                    .flatten()
                {
                    map.insert((size, headers[i].clone()), val);
                }
            }
        }
        map
    }

    /// Exports Criterion results from target/criterion to a CSV file.
    /// Preserves existing data in the CSV if the benchmarks weren't re-run in the current session.
    pub fn export_results_to_csv(
        sizes: &[usize],
        algorithms: &[&str],
        filename: &str,
    ) -> Result<(), std::io::Error> {
        let existing = Self::read_existing_csv(filename);

        let mut mappings = vec![
            ColumnMapping {
                header: "mkl_seq".to_string(),
                folder: "MKL-Sequential".to_string(),
            },
            ColumnMapping {
                header: "mkl_par".to_string(),
                folder: "MKL-Parallel".to_string(),
            },
            ColumnMapping {
                header: "faer_seq".to_string(),
                folder: "Faer-Sequential".to_string(),
            },
            ColumnMapping {
                header: "faer_par".to_string(),
                folder: "Faer-Parallel".to_string(),
            },
        ];

        for &algo in algorithms {
            let clean = algo.replace(['-', '.'], "_");
            mappings.push(ColumnMapping {
                header: format!("{}_single", clean),
                folder: format!("{}_Single-Thread", algo),
            });
            mappings.push(ColumnMapping {
                header: format!("{}_dfs", clean),
                folder: format!("{}_DFS", algo),
            });
            mappings.push(ColumnMapping {
                header: format!("{}_bfs", clean),
                folder: format!("{}_BFS", algo),
            });
            mappings.push(ColumnMapping {
                header: format!("{}_hybrid", clean),
                folder: format!("{}_Hybrid", algo),
            });
        }

        // Only export sizes that have at least one valid measurement (either in Criterion files or existing CSV)
        let mut active_sizes = Vec::new();
        for &size in sizes {
            let mut has_data = false;
            for col in &mappings {
                if Self::get_criterion_time(&col.folder, size).is_some()
                    || existing.contains_key(&(size, col.header.clone()))
                {
                    has_data = true;
                    break;
                }
            }
            if has_data {
                active_sizes.push(size);
            }
        }

        if let Some(parent) = Path::new(filename).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = File::create(filename)?;

        write!(file, "size")?;
        for col in &mappings {
            write!(file, ",{}", col.header)?;
        }
        writeln!(file)?;

        for &size in &active_sizes {
            write!(file, "{}", size)?;
            for col in &mappings {
                let time_val = Self::get_criterion_time(&col.folder, size)
                    .or_else(|| existing.get(&(size, col.header.clone())).copied());

                if let Some(t) = time_val {
                    write!(file, ",{:.9}", t)?;
                } else {
                    write!(file, ",")?;
                }
            }
            writeln!(file)?;
        }

        println!("Successfully wrote benchmark CSV output to: {}", filename);

        let plot_script = if std::path::Path::new("python/plot.py").exists() {
            "python/plot.py"
        } else if std::path::Path::new("../python/plot.py").exists() {
            "../python/plot.py"
        } else {
            "python/plot.py"
        };

        println!("Generating plot automatically using '{}'...", plot_script);
        let plot_status = std::process::Command::new("uv")
            .args(["run", plot_script])
            .status();
        match plot_status {
            Ok(status) if status.success() => {
                println!("Plot generated successfully.");
            }
            Ok(status) => {
                eprintln!("Plot generation failed with status: {:?}", status);
            }
            Err(e) => {
                eprintln!("Failed to execute plot script: {:?}", e);
            }
        }

        Ok(())
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

    /// Run the baseline sequential or parallel matrix multiplication.
    fn base_matmul(a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        let mut c = Mat::zeros(a.nrows(), b.ncols());
        let par = if multithreaded {
            faer::get_global_parallelism()
        } else {
            faer::Par::Seq
        };
        faer::linalg::matmul::matmul(
            c.as_mut(),
            faer::Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            1.0,
            par,
        );
        c
    }

    /// Registers the MKL sequential and parallel benchmarks for one matrix size.
    fn bench_mkl(group: &mut BenchmarkGroup<WallTime>, a: &Mat<f64>, b: &Mat<f64>, size: usize) {
        group.bench_with_input(
            BenchmarkId::new("MKL-Sequential", size),
            &size,
            |bench, &_| {
                fast_matmul::mkl::mkl_set_threads(1);
                bench.iter(|| fast_matmul::mkl::mkl_matmul(a, b));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("MKL-Parallel", size),
            &size,
            |bench, &_| {
                fast_matmul::mkl::mkl_set_threads(0);
                bench.iter(|| fast_matmul::mkl::mkl_matmul(a, b));
            },
        );
    }

    /// Registers the Faer sequential and parallel benchmarks for one matrix size.
    fn bench_faer(group: &mut BenchmarkGroup<WallTime>, a: &Mat<f64>, b: &Mat<f64>, size: usize) {
        group.bench_with_input(
            BenchmarkId::new("Faer-Sequential", size),
            &size,
            |bench, &_| {
                bench.iter(|| Self::base_matmul(a, b, false));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Faer-Parallel", size),
            &size,
            |bench, &_| {
                bench.iter(|| Self::base_matmul(a, b, true));
            },
        );
    }

    /// Registers single-thread and all parallel CP benchmarks for one algorithm and matrix size.
    fn bench_cp(
        group: &mut BenchmarkGroup<WallTime>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
        algo: &str,
        mm: &MatMul<'_>,
    ) {
        group.bench_with_input(
            BenchmarkId::new(format!("{}/Single-Thread", algo), size),
            &size,
            |bench, &_| bench.iter(|| mm.cp_matmul_single_thread(a, b)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}/DFS", algo), size),
            &size,
            |bench, &_| bench.iter(|| mm.cp_matmul(a, b, ParallelismMode::Dfs)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}/BFS", algo), size),
            &size,
            |bench, &_| bench.iter(|| mm.cp_matmul(a, b, ParallelismMode::Bfs)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}/Hybrid", algo), size),
            &size,
            |bench, &_| bench.iter(|| mm.cp_matmul(a, b, ParallelismMode::Hybrid)),
        );
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
                size, size, bytes_per_matrix, isize::MAX
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
                    size, size, (estimated_required_bytes + safety_buffer) / (1024 * 1024), avail_bytes / (1024 * 1024)
                ));
            }
        }

        Ok(())
    }

    /// Runs benchmarks using the programmatic Criterion API and exports the results to CSV.
    ///
    /// Stops running benchmarks if a matrix size exceeds the machine's memory capacity,
    /// printing a clean warning, and still exporting results for the sizes that completed.
    pub fn run(
        &self,
        sizes: &[usize],
        algorithms: &[&str],
        filename: &str,
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

            Self::bench_mkl(&mut group, &a, &b, size);

            Self::bench_faer(&mut group, &a, &b, size);

            for &(algo, ref cp) in &cps {
                let mm = MatMul::with_cp(cp);
                Self::bench_cp(&mut group, &a, &b, size, algo, &mm);
            }
            successful_sizes.push(size);
        }

        group.finish();

        if !successful_sizes.is_empty() {
            Self::export_results_to_csv(&successful_sizes, algorithms, filename)?;
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

    // Default limit is 2^10 (1024). Under --full, we run up to 2^20 (1,048,576),
    // which will dynamically check system memory and stop before exceeding limits.
    let n_limit = if full { 20 } else { 11 };
    let sizes: Vec<usize> = (1..=n_limit).map(|n| 1usize << n).collect(); // 2, 4, ..., 2^N
    let csv_file = "generated/benchmark_results.csv";
    let algorithms = &["strassen", "grey-strassen"];

    if plot_only {
        println!("Plot-only mode: Regenerating CSV results from cached Criterion data...");
        if let Err(e) = Benchmark::export_results_to_csv(&sizes, algorithms, csv_file) {
            eprintln!("Failed to export CSV: {:?}", e);
        } else {
            println!("CSV results successfully updated from cache.");
        }
    } else {
        println!("\n--- Running Matrix Multiplication Benchmarks ---");
        let bench = Benchmark::new();
        if let Err(e) = bench.run(&sizes, algorithms, csv_file) {
            eprintln!("Failed to write benchmarks to CSV: {:?}", e);
        } else {
            println!("Benchmark results successfully written to {}", csv_file);
        }
    }
}
