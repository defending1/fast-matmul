mod base_matmul;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
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
        ..=16 => (10, 50, 100),
        17..=64 => (10, 100, 200),
        65..=256 => (10, 200, 500),
        257..=1024 => (10, 500, 1000),
        _ => (10, 100, 200),
    };
    group.sample_size(samples);
    group.warm_up_time(std::time::Duration::from_millis(warmup_ms));
    group.measurement_time(std::time::Duration::from_millis(measure_ms));
}

/// Registers the curves benchmarks for all shapes, threading modes, and libraries with Criterion.
fn bench_dgemm_curves(c: &mut Criterion) {
    let mut group = c.benchmark_group("dgemm_curves");
    let n_vals: Vec<usize> = (1..=10).map(|n| 1usize << n).collect();

    for &n in &n_vals {
        configure_group_for_size(&mut group, n);
        {
            let a = random_matrix(n, n);
            let b = random_matrix(n, n);

            group.bench_with_input(
                BenchmarkId::new("Square/Faer-Sequential", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| base_matmul::base_matmul(&a, &b, false, BaseMatMul::Faer));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_dgemm_curves);
criterion_main!(benches);
