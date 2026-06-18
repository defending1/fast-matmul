mod matmul;

use crate::matmul::{evaluate_tensor_product, matmul, standard_matmul_vec_wt, strassen_matmul};
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("Matrix Multiplication Tensor (m=2, n=2, p=2) Front Slices:\n");

    let x = matmul(2, 2, 2);
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
    let tensor = matmul(m, n, p);

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

    let res_tensor = evaluate_tensor_product(&tensor, &nd_vec_a, &nd_vec_b);
    let res_standard = standard_matmul_vec_wt(&a, &b);

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

    println!("\n--- Running Strassen MatMul Verification ---");
    let c_strassen = strassen_matmul(&a, &b);
    let c_classical = a.dot(&b);
    let mut strassen_diff = 0.0;
    for i in 0..m {
        for j in 0..p {
            let diff = (c_strassen[[i, j]] - c_classical[[i, j]]).abs();
            if diff > strassen_diff {
                strassen_diff = diff;
            }
        }
    }
    println!(
        "Strassen MatMul completed. Maximum difference between Strassen and classical: {:.6e}",
        strassen_diff
    );
    if strassen_diff < 1e-10 {
        println!("Result: SUCCESS! Strassen matrix multiplication is correct.");
    } else {
        println!("Result: FAILURE! Strassen results differ from classical.");
    }
}
