use fast_matmul::benchmark::Benchmark;
use fast_matmul::cp::CP;
use fast_matmul::matmul::MatMul;
use ndarray::{Array1, Array2};
use rand::Rng;

fn standard_block_matmul(mm: &MatMul, m: usize, n: usize, p: usize) {
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

    let vec_a: Vec<f64> = (0..(m * n)).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let vec_b: Vec<f64> = (0..(n * p)).map(|_| rng.gen_range(-5.0..5.0)).collect();

    // Convert to ndarray matrices (transposed load to achieve column-major layout)
    let a_t = Array2::from_shape_vec((n, m), vec_a.clone()).unwrap();
    let a = a_t.t().to_owned();

    let b_t = Array2::from_shape_vec((p, n), vec_b.clone()).unwrap();
    let b = b_t.t().to_owned();

    // Vectorizations in column-major
    let nd_vec_a = Array1::from_vec(vec_a);
    let nd_vec_b = Array1::from_vec(vec_b);

    let res_tensor = mm.evaluate_tensor_product(&tensor, &nd_vec_a, &nd_vec_b);
    let res_standard = mm.standard_matmul_vec_wt(&a, &b);

    let mut max_diff = 0.0;
    for i in 0..res_tensor.len() {
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
    // Pre-load CP matrices to avoid disk I/O and initialization overhead during matrix multiplication
    let _ = CP::get_strassen();

    println!("--- Running Runtime Verification with Random Matrices ---");
    let m = 3;
    let n = 4;
    let p = 5;

    let mm = MatMul::new();
    standard_block_matmul(&mm, m, n, p);

    println!("\n--- Running Matrix Multiplication Benchmarks ---");
    let sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let csv_file = "generated/benchmark_results.csv";
    
    let bench = Benchmark::new(&mm);
    if let Err(e) = bench.run(&sizes, csv_file) {
        eprintln!("Failed to write benchmarks to CSV: {:?}", e);
    } else {
        println!("Benchmark results successfully written to {}", csv_file);
    }
}
