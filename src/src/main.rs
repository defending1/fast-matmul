mod matmul;

use crate::matmul::{evaluate_tensor_product, matmul, standard_matmul_vec_wt, Tensor3D};
use rand::Rng;

fn print_x(x: Tensor3D) {
    let (shape_i, shape_j, shape_k) = x.shape;
    for k in 0..shape_k {
        println!("X{} =", k + 1);
        for i in 0..shape_i {
            print!("  ");
            for j in 0..shape_j {
                print!("{:>3}", x.get(i, j, k) as i32);
            }
            println!();
        }
        println!();
    }
}

fn main() {
    println!("Matrix Multiplication Tensor (m=2, n=2, p=2) Front Slices:\n");

    let x = matmul(2, 2, 2);
    print_x(x);

    println!("--- Running Runtime Verification with Random Matrices ---");
    let mut rng = rand::thread_rng();
    let m = 3;
    let n = 4;
    let p = 3;

    println!("Testing dimensions: m={}, n={}, p={}", m, n, p);
    let tensor = matmul(m, n, p);

    let vec_u: Vec<f64> = (0..(m * n)).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let vec_v: Vec<f64> = (0..(n * p)).map(|_| rng.gen_range(-5.0..5.0)).collect();

    let res_tensor = evaluate_tensor_product(&tensor, &vec_u, &vec_v);
    let res_standard = standard_matmul_vec_wt(m, n, p, &vec_u, &vec_v);

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
