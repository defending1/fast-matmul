use fast_matmul::cp::CP;
use fast_matmul::matmul::MatMul;
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    // Pre-load CP matrices to avoid disk I/O and initialization overhead during matrix multiplication
    let _ = CP::get_strassen();

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

    println!("--- Running Runtime Verification with Random Matrices ---");
    let mut rng = rand::thread_rng();
    let m = 3;
    let n = 4;
    let p = 5;

    println!("Testing dimensions: m={}, n={}, p={}", m, n, p);
    let tensor = mm.matmul(m, n, p);

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

    println!("\n--- Running CP MatMul Verification ---");
    let c_cp = mm.cp_matmul(&a, &b);
    let c_classical = a.dot(&b);
    let mut cp_diff = 0.0;
    for i in 0..m {
        for j in 0..p {
            let diff = (c_cp[[i, j]] - c_classical[[i, j]]).abs();
            if diff > cp_diff {
                cp_diff = diff;
            }
        }
    }
    println!(
        "CP MatMul completed. Maximum difference between CP and classical: {:.6e}",
        cp_diff
    );
    if cp_diff < 1e-10 {
        println!("Result: SUCCESS! CP matrix multiplication is correct.");
    } else {
        println!("Result: FAILURE! CP results differ from classical.");
    }

    println!("\n--- Running Matrix Multiplication Benchmarks ---");
    let sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let csv_file = "generated/benchmark_results.csv";
    if let Err(e) = benchmark_matmul(&sizes, csv_file) {
        eprintln!("Failed to write benchmarks to CSV: {:?}", e);
    } else {
        println!("Benchmark results successfully written to {}", csv_file);
    }
}

/// Computes C = A * B using classical matrix multiplication (using ndarray's built-in dot).
fn classic_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// Benchmarks classic_matmul, cp_matmul_single_thread, and cp_matmul,
/// printing the elapsed times to the console and saving them to a CSV file.
fn benchmark_matmul(sizes: &[usize], filename: &str) -> Result<(), std::io::Error> {
    use std::fs::File;
    use std::io::Write;
    use std::time::Instant;

    if let Some(parent) = std::path::Path::new(filename).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(filename)?;
    writeln!(
        file,
        "size,classic_matmul,cp_matmul_single_thread,cp_matmul"
    )?;

    let mut rng = rand::thread_rng();
    let mm = MatMul::new();

    for &size in sizes {
        println!("\nBenchmarking size {}x{}...", size, size);

        let mut a = Array2::zeros((size, size));
        let mut b = Array2::zeros((size, size));
        for val in a.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }
        for val in b.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }

        // 1. Classic MatMul
        let start = Instant::now();
        let _c_classic = classic_matmul(&a, &b);
        let duration_classic = start.elapsed().as_secs_f64();
        println!("  classic_matmul:               {:.6} s", duration_classic);

        // 2. CP MatMul Single-Thread
        let start = Instant::now();
        let _c_cp_single = mm.cp_matmul_single_thread(&a, &b);
        let duration_cp_single = start.elapsed().as_secs_f64();
        println!(
            "  cp_matmul_single_thread: {:.6} s",
            duration_cp_single
        );

        // 3. CP MatMul (Parallel/Multi-Thread)
        let start = Instant::now();
        let _c_cp_parallel = mm.cp_matmul(&a, &b);
        let duration_cp_parallel = start.elapsed().as_secs_f64();
        println!(
            "  cp_matmul:              {:.6} s",
            duration_cp_parallel
        );

        writeln!(
            file,
            "{},{},{},{}",
            size, duration_classic, duration_cp_single, duration_cp_parallel
        )?;
    }

    Ok(())
}
