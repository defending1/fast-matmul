use fast_matmul::matmul::MatMul;
use faer::{Mat, Col};
use rand::Rng;

#[test]
fn test_example_2x2_slices() {
    let mm = MatMul::new();
    let x = mm.matmul(2, 2, 2);

    // Assert tensor shape is 4x4x4
    assert_eq!(x.dim(), (4, 4, 4));

    let expected_x1 = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let expected_x2 = [
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let expected_x3 = [
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    let expected_x4 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(
                x[[i, j, 0]],
                expected_x1[i * 4 + j],
                "Mismatch at slice 0, index ({}, {})",
                i,
                j
            );
            assert_eq!(
                x[[i, j, 1]],
                expected_x2[i * 4 + j],
                "Mismatch at slice 1, index ({}, {})",
                i,
                j
            );
            assert_eq!(
                x[[i, j, 2]],
                expected_x3[i * 4 + j],
                "Mismatch at slice 2, index ({}, {})",
                i,
                j
            );
            assert_eq!(
                x[[i, j, 3]],
                expected_x4[i * 4 + j],
                "Mismatch at slice 3, index ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_matmul_tensor_correctness() {
    let mut rng = rand::thread_rng();
    let mm = MatMul::new();

    let test_cases = vec![
        (2, 2, 2),
        (3, 2, 4),
        (4, 3, 2),
        (5, 5, 5),
        (1, 4, 3),
        (3, 1, 3),
    ];

    for &(m, n, p) in &test_cases {
        let x = mm.matmul(m, n, p);

        let vec_a: Vec<f64> = (0..(m * n)).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let vec_b: Vec<f64> = (0..(n * p)).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let a = Mat::from_fn(m, n, |r, c| vec_a[c * m + r]);
        let b = Mat::from_fn(n, p, |r, c| vec_b[c * n + r]);

        let nd_vec_a = Col::from_fn(m * n, |i| vec_a[i]);
        let nd_vec_b = Col::from_fn(n * p, |i| vec_b[i]);

        let res_tensor = mm.evaluate_tensor_product(&x, &nd_vec_a, &nd_vec_b);
        let res_standard = mm.standard_matmul_vec_wt(&a, &b);

        assert_eq!(res_tensor.nrows(), res_standard.nrows());
        for idx in 0..res_tensor.nrows() {
            let diff = (res_tensor[idx] - res_standard[idx]).abs();
            assert!(
                diff < 1e-12,
                "Large difference at idx {} for dims ({}, {}, {}): tensor = {}, standard = {}",
                idx,
                m,
                n,
                p,
                res_tensor[idx],
                res_standard[idx]
            );
        }
    }
}
