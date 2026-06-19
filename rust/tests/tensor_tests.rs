use fast_matmul::matmul::{evaluate_tensor_product, matmul, standard_matmul_vec_wt};
use ndarray::{Array1, Array2};
use rand::Rng;

#[test]
fn test_example_2x2_slices() {
    let x = matmul(2, 2, 2);

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

    let test_cases = vec![
        (2, 2, 2),
        (3, 2, 4),
        (4, 3, 2),
        (5, 5, 5),
        (1, 4, 3),
        (3, 1, 3),
    ];

    for &(m, n, p) in &test_cases {
        let x = matmul(m, n, p);

        let vec_a: Vec<f64> = (0..(m * n)).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let vec_b: Vec<f64> = (0..(n * p)).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let a_t = Array2::from_shape_vec((n, m), vec_a.clone()).unwrap();
        let a = a_t.t().to_owned();

        let b_t = Array2::from_shape_vec((p, n), vec_b.clone()).unwrap();
        let b = b_t.t().to_owned();

        let nd_vec_a = Array1::from_vec(vec_a);
        let nd_vec_b = Array1::from_vec(vec_b);

        let res_tensor = evaluate_tensor_product(&x, &nd_vec_a, &nd_vec_b);
        let res_standard = standard_matmul_vec_wt(&a, &b);

        assert_eq!(res_tensor.len(), res_standard.len());
        for idx in 0..res_tensor.len() {
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
