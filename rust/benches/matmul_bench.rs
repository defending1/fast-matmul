use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::Mat;
use fast_matmul::benchmark::Benchmark;
use fast_matmul::cp::CP;
use fast_matmul::matmul::MatMul;
use rand::Rng;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication");

    let sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512];
    let mut rng = rand::thread_rng();

    // Define CP algorithms to benchmark
    let algorithms = [
        "strassen",
        "grey-strassen",
        "hk323-15-94",
        "smirnov333-23-139",
    ];

    // Load CP decompositions and instantiate MatMul helpers
    let cps: Vec<CP> = algorithms.iter().map(|&name| CP::load(name)).collect();
    let mm_runs: Vec<(&str, MatMul)> = algorithms
        .iter()
        .zip(&cps)
        .map(|(&name, cp)| (name, MatMul::with_cp(cp)))
        .collect();

    for &size in &sizes {
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

        // 1. Classic/System MatMul (faer * operator)
        group.bench_with_input(
            BenchmarkId::new("System/Faer", size),
            &size,
            |bench, &_size| {
                bench.iter(|| &a * &b);
            },
        );

        // 2. Intel MKL MatMul (Sequential)
        group.bench_with_input(
            BenchmarkId::new("MKL-Sequential", size),
            &size,
            |bench, &_size| {
                fast_matmul::mkl::mkl_set_threads(1);
                bench.iter(|| fast_matmul::mkl::mkl_matmul(&a, &b));
            },
        );

        // 3. Intel MKL MatMul (Parallel)
        group.bench_with_input(
            BenchmarkId::new("MKL-Parallel", size),
            &size,
            |bench, &_size| {
                fast_matmul::mkl::mkl_set_threads(0);
                bench.iter(|| fast_matmul::mkl::mkl_matmul(&a, &b));
            },
        );

        // 3. CP MatMul (both single-thread and multi-thread)
        for &(name, ref mm) in &mm_runs {
            group.bench_with_input(
                BenchmarkId::new(format!("{}/Single-Thread", name), size),
                &size,
                |bench, &_size| {
                    bench.iter(|| mm.cp_matmul_single_thread(&a, &b));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{}/Multi-Thread", name), size),
                &size,
                |bench, &_size| {
                    bench.iter(|| mm.cp_matmul(&a, &b));
                },
            );
        }
    }

    group.finish();

    // Automatically export the results from the target/criterion directory to the CSV file
    let csv_file = "generated/benchmark_results.csv";
    if let Err(e) = Benchmark::export_results_to_csv(&sizes, &algorithms, csv_file) {
        eprintln!("Failed to auto-export Criterion results to CSV: {:?}", e);
    }
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
