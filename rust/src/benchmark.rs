use crate::cp::CP;
use crate::matmul::MatMul;
use criterion::{BenchmarkId, Criterion};
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

        // Write header: size,mkl,faer,faer_multi,algo1_single,algo1_multithread,...
        write!(file, "size")?;
        for col in &mappings {
            write!(file, ",{}", col.header)?;
        }
        writeln!(file)?;

        // Write data rows
        for &size in sizes {
            write!(file, "{}", size)?;
            for col in &mappings {
                let time_val = if let Some(t) = Self::get_criterion_time(&col.folder, size) {
                    Some(t)
                } else {
                    existing.get(&(size, col.header.clone())).copied()
                };

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

        // Load CP decompositions for all requested algorithms
        let mut cps = Vec::new();
        for &algo in algorithms {
            println!("Loading CP decomposition for '{}'...", algo);
            let cp = CP::load(algo);
            cps.push((algo, cp));
        }

        for &size in sizes {
            println!("Benchmarking size {}x{} with Criterion...", size, size);

            // Dynamically adjust parameters based on matrix size to avoid excessive execution time
            if size <= 16 {
                group.sample_size(30);
                group.warm_up_time(std::time::Duration::from_millis(50));
                group.measurement_time(std::time::Duration::from_millis(100));
            } else if size <= 64 {
                group.sample_size(20);
                group.warm_up_time(std::time::Duration::from_millis(100));
                group.measurement_time(std::time::Duration::from_millis(200));
            } else if size <= 256 {
                group.sample_size(10);
                group.warm_up_time(std::time::Duration::from_millis(200));
                group.measurement_time(std::time::Duration::from_millis(500));
            } else {
                group.sample_size(10);
                group.warm_up_time(std::time::Duration::from_millis(500));
                group.measurement_time(std::time::Duration::from_millis(1000));
            }

            let mut a = Mat::<f64>::zeros(size, size);
            let mut b = Mat::<f64>::zeros(size, size);
            for r in 0..size {
                for c in 0..size {
                    a[(r, c)] = rng.gen_range(-1.0..1.0);
                    b[(r, c)] = rng.gen_range(-1.0..1.0);
                }
            }

            // Intel MKL MatMul (Sequential)
            group.bench_with_input(
                BenchmarkId::new("MKL-Sequential", size),
                &size,
                |bench, &_size| {
                    crate::mkl::mkl_set_threads(1);
                    bench.iter(|| crate::mkl::mkl_matmul(&a, &b));
                },
            );

            // 3. Intel MKL MatMul (Parallel)
            group.bench_with_input(
                BenchmarkId::new("MKL-Parallel", size),
                &size,
                |bench, &_size| {
                    crate::mkl::mkl_set_threads(0);
                    bench.iter(|| crate::mkl::mkl_matmul(&a, &b));
                },
            );

            let mm_base = MatMul::new();

            // Faer MatMul (Sequential)
            group.bench_with_input(
                BenchmarkId::new("Faer-Sequential", size),
                &size,
                |bench, &_size| {
                    bench.iter(|| mm_base.base_matmul(&a, &b, false));
                },
            );

            // Faer MatMul (Parallel)
            group.bench_with_input(
                BenchmarkId::new("Faer-Parallel", size),
                &size,
                |bench, &_size| {
                    bench.iter(|| mm_base.base_matmul(&a, &b, true));
                },
            );

            // CP MatMul for each algorithm
            for &(algo, ref cp) in &cps {
                let mm = MatMul::with_cp(cp);

                // Single-thread
                group.bench_with_input(
                    BenchmarkId::new(format!("{}/Single-Thread", algo), size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| mm.cp_matmul_single_thread(&a, &b));
                    },
                );

                // Multi-thread
                group.bench_with_input(
                    BenchmarkId::new(format!("{}/Multi-Thread", algo), size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| mm.cp_matmul(&a, &b));
                    },
                );
            }
        }

        group.finish();

        // Export data to CSV
        Self::export_results_to_csv(sizes, algorithms, filename)?;

        Ok(())
    }
}
