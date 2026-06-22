use faer::Mat;
use fast_matmul::mkl::{mkl_matmul, mkl_set_threads};

#[test]
fn test_mkl_matmul_correctness() {
    println!("Setting threads to 1...");
    mkl_set_threads(1);
    println!("Threads set to 1 successfully.");

    let a = Mat::from_fn(2, 2, |r, c| (r + c) as f64);
    let b = Mat::from_fn(2, 2, |r, c| (r as f64) - (c as f64));

    println!("Calling mkl_matmul...");
    let c = mkl_matmul(&a, &b);
    println!("mkl_matmul returned successfully.");

    let expected = &a * &b;
    for r in 0..2 {
        for c_idx in 0..2 {
            assert!((c[(r, c_idx)] - expected[(r, c_idx)]).abs() < 1e-9);
        }
    }
    println!("Assertion passed.");

    println!("Setting threads to 0 (parallel)...");
    mkl_set_threads(0);
    println!("Threads set to parallel successfully.");

    let c_par = mkl_matmul(&a, &b);
    println!("Parallel mkl_matmul returned successfully.");
    for r in 0..2 {
        for c_idx in 0..2 {
            assert!((c_par[(r, c_idx)] - expected[(r, c_idx)]).abs() < 1e-9);
        }
    }
}
