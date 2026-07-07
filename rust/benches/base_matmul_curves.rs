mod util;

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use faer::Mat;
use fast_matmul::matmul::BaseMatMul;
use rand::Rng;

/// Helper function to generate a random matrix of double precision floats.
fn random_matrix(rows: usize, cols: usize) -> Mat<f64> {
    let mut rng = rand::thread_rng();
    Mat::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0))
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

/// Registers the curves benchmarks for all shapes, threading modes, and libraries with Criterion.
fn bench_base_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("base_matmul_curves");
    const N: i32 = 11;
    let n_vals: Vec<usize> = (1..=N).map(|n| 1usize << n).collect();

    for &n in &n_vals {
        configure_group_for_size(&mut group, n);

        // 1. SQUARE: N x N x N
        {
            let a = random_matrix(n, n);
            let b = random_matrix(n, n);

            group.bench_with_input(
                BenchmarkId::new("Square/Faer-Sequential", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| util::base_matmul(&a, &b, false, BaseMatMul::Faer));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("Square/Faer-Parallel", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| util::base_matmul(&a, &b, true, BaseMatMul::Faer));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("Square/Dgemm-Sequential", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| util::base_matmul(&a, &b, false, BaseMatMul::Dgemm));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("Square/Dgemm-Parallel", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| util::base_matmul(&a, &b, true, BaseMatMul::Dgemm));
                },
            );
        }
    }

    group.finish();
}

fn main() {
    // 1. Run spline fitting and derivative calculation
    println!("\n==================================================");
    println!("RUNNING SPLINE INTERPOLATION & DERIVATIVE ANALYSIS");
    println!("==================================================");
    let root = util::get_project_root();
    let csv_path_buf = root.join("generated").join("csv").join("base_matmul_spline.csv");
    let csv_path = csv_path_buf.to_str().unwrap_or("generated/csv/base_matmul_spline.csv");

    const N: i32 = 11;
    // Generate sizes dynamically: 2, 4, 8, ..., 2^N
    let n_vals: Vec<f64> = (1..=N).map(|n| (1usize << n) as f64).collect();

    match util::fit_and_differentiate_spline(csv_path, &n_vals) {
        Ok((gflops, derivatives)) => {
            // 2. Call python script to generate plot
            let plot_script = if std::path::Path::new("python/plot_spline.py").exists() {
                "python/plot_spline.py"
            } else if std::path::Path::new("../python/plot_spline.py").exists() {
                "../python/plot_spline.py"
            } else {
                "python/plot_spline.py"
            };

            let absolute_csv = std::path::Path::new(csv_path)
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(csv_path));

            println!(
                "Generating spline plot automatically using '{}' for '{}'...",
                plot_script, csv_path
            );
            let plot_status = std::process::Command::new("uv")
                .args(["run", plot_script, &absolute_csv.to_string_lossy()])
                .status();
            match plot_status {
                Ok(status) if status.success() => {
                    println!("Spline plot generated successfully.");
                }
                Ok(status) => {
                    eprintln!("Spline plot generation failed with status: {:?}", status);
                }
                Err(e) => {
                    eprintln!("Failed to execute spline plot script: {:?}", e);
                }
            }
            println!("\nResults:");
            println!("{:<10} {:<20} {:<20}", "N", "GFLOPS", "dGFLOPS/dN");
            for i in 0..n_vals.len() {
                println!(
                    "{:<10.0} {:<20.6} {:<20.6}",
                    n_vals[i], gflops[i], derivatives[i]
                );
            }

            if let Some(k) = derivatives
                .iter()
                .enumerate()
                .position(|(i, &d)| n_vals[i] > 32.0 && d < 0.15)
            {
                println!(
                    "\nMinimum index k such that derivative[k] < 0.15 (for N > 32): {} (N = {}, derivative = {:.6})",
                    k, n_vals[k], derivatives[k]
                );
            } else {
                println!("\nNo index k found such that derivative[k] < 0.15 (for N > 32)");
            }
        }
        Err(e) => {
            eprintln!("Error running spline interpolation: {:?}", e);
        }
    }
    println!("==================================================\n");

    // 3. Run Criterion benchmarks
    let mut c = Criterion::default().configure_from_args();
    bench_base_matmul(&mut c);
    c.final_summary();
}
