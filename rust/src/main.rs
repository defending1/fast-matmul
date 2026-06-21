use faer::{Col, Mat};
use fast_matmul::benchmark::Benchmark;
use fast_matmul::matmul::MatMul;
use rand::Rng;

fn standard_block_matmul() {
    println!("--- Running Runtime Verification with Random Matrices ---");
    let m = 3;
    let n = 4;
    let p = 5;

    let mm = MatMul::new();

    println!("Matrix Multiplication Tensor (m=2, n=2, p=2) Front Slices:\n");

    let x = mm.matmul(2, 2, 2);
    let (shape_i, shape_j, shape_k) = x.dim();

    for k in 0..shape_k {
        println!("X{} =", k + 1);
        for i in 0..shape_i {
            print!("  ");
            for j in 0..shape_j {
                print!("{:>3}", x[[i, j, k]] as i32);
            }
            println!();
        }
        println!();
    }

    println!("Testing dimensions: m={}, n={}, p={}", m, n, p);
    let tensor = mm.matmul(m, n, p);

    let mut rng = rand::thread_rng();

    let vec_a_raw: Vec<f64> = (0..(m * n)).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let vec_b_raw: Vec<f64> = (0..(n * p)).map(|_| rng.gen_range(-5.0..5.0)).collect();

    // Convert to faer matrices
    let a = Mat::from_fn(m, n, |r, c| vec_a_raw[c * m + r]);
    let b = Mat::from_fn(n, p, |r, c| vec_b_raw[c * n + r]);

    // Vectorizations in column-major
    let vec_a = Col::from_fn(m * n, |i| vec_a_raw[i]);
    let vec_b = Col::from_fn(n * p, |i| vec_b_raw[i]);

    let res_tensor = mm.evaluate_tensor_product(&tensor, &vec_a, &vec_b);
    let res_standard = mm.standard_matmul_vec_wt(&a, &b);

    let mut max_diff = 0.0;
    for i in 0..res_tensor.nrows() {
        let diff = (res_tensor[i] - res_standard[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!(
        "Verification completed. Maximum difference between tensor and standard: {:.6e}",
        max_diff
    );
    if max_diff < 1e-12 {
        println!("Result: SUCCESS! The tensor represents the matrix multiplication correctly.");
    } else {
        println!("Result: FAILURE! The results differ significantly.");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let plot_only = args.iter().any(|arg| arg == "--plot-only" || arg == "-p");

    let sizes: Vec<usize> = (1..=9).map(|n| 1usize << n).collect(); // 2, 4, 8, 16, 32, 64, 128, 256, 512
    let csv_file = "generated/benchmark_results.csv";
    let algorithms = &[
        "strassen",
        "grey-strassen",
        "hk323-15-94",
        "smirnov333-23-139",
    ];

    if plot_only {
        println!("Plot-only mode: Regenerating CSV results from cached Criterion data...");
        if let Err(e) = Benchmark::export_results_to_csv(&sizes, algorithms, csv_file) {
            eprintln!("Failed to export CSV: {:?}", e);
        } else {
            println!("CSV results successfully updated from cache.");
        }
    } else {
        standard_block_matmul();

        println!("\n--- Running Matrix Multiplication Benchmarks ---");
        let bench = Benchmark::new();
        if let Err(e) = bench.run(&sizes, algorithms, csv_file) {
            eprintln!("Failed to write benchmarks to CSV: {:?}", e);
        } else {
            println!("Benchmark results successfully written to {}", csv_file);
        }
    }
}
