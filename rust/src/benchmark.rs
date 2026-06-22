use crate::cp::CP;
use crate::matmul::MatMul;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use faer::Mat;
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// A struct for running benchmarks on various matrix multiplication algorithms.
pub struct Benchmark;

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
                header: format!("{}_multithread", clean),
                folder: format!("{}_Multi-Thread", algo),
            });
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

        for &size in sizes {
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
        Ok(())
    }

    /// Adjusts Criterion group sampling parameters based on matrix size.
    fn configure_group_for_size(group: &mut BenchmarkGroup<WallTime>, size: usize) {
        let (samples, warmup_ms, measure_ms) = match size {
            ..=16 => (30, 50, 100),
            17..=64 => (20, 100, 200),
            65..=256 => (10, 200, 500),
            _ => (10, 500, 1000),
        };
        group.sample_size(samples);
        group.warm_up_time(std::time::Duration::from_millis(warmup_ms));
        group.measurement_time(std::time::Duration::from_millis(measure_ms));
    }

    /// Allocates a random `size × size` matrix.
    fn random_matrix(size: usize, rng: &mut impl Rng) -> Mat<f64> {
        Mat::from_fn(size, size, |_, _| rng.gen_range(-1.0..1.0))
    }

    /// Registers the MKL sequential and parallel benchmarks for one matrix size.
    fn bench_mkl(group: &mut BenchmarkGroup<WallTime>, a: &Mat<f64>, b: &Mat<f64>, size: usize) {
        group.bench_with_input(BenchmarkId::new("MKL-Sequential", size), &size, |bench, &_| {
            crate::mkl::mkl_set_threads(1);
            bench.iter(|| crate::mkl::mkl_matmul(a, b));
        });
        group.bench_with_input(BenchmarkId::new("MKL-Parallel", size), &size, |bench, &_| {
            crate::mkl::mkl_set_threads(0);
            bench.iter(|| crate::mkl::mkl_matmul(a, b));
        });
    }

    /// Registers the Faer sequential and parallel benchmarks for one matrix size.
    fn bench_faer(
        group: &mut BenchmarkGroup<WallTime>,
        a: &Mat<f64>,
        b: &Mat<f64>,
        size: usize,
        mm: &MatMul<'_>,
    ) {
        group.bench_with_input(BenchmarkId::new("Faer-Sequential", size), &size, |bench, &_| {
            bench.iter(|| mm.base_matmul(a, b, false));
        });
        group.bench_with_input(BenchmarkId::new("Faer-Parallel", size), &size, |bench, &_| {
            bench.iter(|| mm.base_matmul(a, b, true));
        });
    }

    /// Registers single-thread and multi-thread CP benchmarks for one algorithm and matrix size.
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
            BenchmarkId::new(format!("{}/Multi-Thread", algo), size),
            &size,
            |bench, &_| bench.iter(|| mm.cp_matmul(a, b)),
        );
    }

    /// Runs benchmarks using the programmatic Criterion API and exports the results to CSV.
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

        for &size in sizes {
            println!("Benchmarking size {}x{} with Criterion...", size, size);
            Self::configure_group_for_size(&mut group, size);

            let a = Self::random_matrix(size, &mut rng);
            let b = Self::random_matrix(size, &mut rng);

            Self::bench_mkl(&mut group, &a, &b, size);

            let mm_base = MatMul::new();
            Self::bench_faer(&mut group, &a, &b, size, &mm_base);

            for &(algo, ref cp) in &cps {
                let mm = MatMul::with_cp(cp);
                Self::bench_cp(&mut group, &a, &b, size, algo, &mm);
            }
        }

        group.finish();

        Self::export_results_to_csv(sizes, algorithms, filename)?;

        Ok(())
    }
}
