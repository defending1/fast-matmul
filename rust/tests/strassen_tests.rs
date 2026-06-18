use fast_matmul::matmul::{strassen_matmul, strassen_matmul_single_thread, pad_matrices};
use ndarray::Array2;
use rand::Rng;

#[test]
fn test_strassen_matmul_correctness() {
    let mut rng = rand::thread_rng();

    let test_cases = vec![
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (5, 5, 5),
        (8, 8, 8),
        (2, 4, 3),
        (5, 3, 4),
        (1, 5, 2),
        (3, 4, 1),
        (6, 6, 6),
    ];

    for &(m, n, p) in &test_cases {
        let mut a = Array2::zeros((m, n));
        for val in a.iter_mut() {
            *val = rng.gen_range(-10.0..10.0);
        }
        let mut b = Array2::zeros((n, p));
        for val in b.iter_mut() {
            *val = rng.gen_range(-10.0..10.0);
        }

        let c_strassen = strassen_matmul(&a, &b);
        let c_classical = a.dot(&b);

        assert_eq!(c_strassen.dim(), (m, p));
        for i in 0..m {
            for j in 0..p {
                let diff = (c_strassen[[i, j]] - c_classical[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at ({}, {}) for shape ({}, {}, {}): Strassen = {}, Classical = {}",
                    i,
                    j,
                    m,
                    n,
                    p,
                    c_strassen[[i, j]],
                    c_classical[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_strassen_matmul_single_thread_correctness() {
    let mut rng = rand::thread_rng();

    let test_cases = vec![
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (5, 5, 5),
        (8, 8, 8),
        (2, 4, 3),
        (5, 3, 4),
        (1, 5, 2),
        (3, 4, 1),
        (6, 6, 6),
    ];

    for &(m, n, p) in &test_cases {
        let mut a = Array2::zeros((m, n));
        for val in a.iter_mut() {
            *val = rng.gen_range(-10.0..10.0);
        }
        let mut b = Array2::zeros((n, p));
        for val in b.iter_mut() {
            *val = rng.gen_range(-10.0..10.0);
        }

        let c_strassen = strassen_matmul_single_thread(&a, &b);
        let c_classical = a.dot(&b);

        assert_eq!(c_strassen.dim(), (m, p));
        for i in 0..m {
            for j in 0..p {
                let diff = (c_strassen[[i, j]] - c_classical[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at ({}, {}) for shape ({}, {}, {}): Strassen (single thread) = {}, Classical = {}",
                    i,
                    j,
                    m,
                    n,
                    p,
                    c_strassen[[i, j]],
                    c_classical[[i, j]]
                );
            }
        }
    }
}

#[test]
fn test_pad_matrices() {
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let (a_pad, b_pad, need_padding, next_m, next_n, next_p) = pad_matrices(&a, &b);
    assert!(need_padding);
    assert_eq!(next_m, 2);
    assert_eq!(next_n, 4); // 3 is padded to 4
    assert_eq!(next_p, 2);
    assert_eq!(a_pad.dim(), (2, 4));
    assert_eq!(b_pad.dim(), (4, 2));

    let a_even = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b_even = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (_, _, need_padding_even, _, _, _) = pad_matrices(&a_even, &b_even);
    assert!(!need_padding_even);
}

#[test]
fn test_strassen_power_of_two_correctness() {
    let mut rng = rand::thread_rng();
    for n in 1..=9 {
        let size = 1 << n;
        let mut a = Array2::zeros((size, size));
        let mut b = Array2::zeros((size, size));
        for val in a.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }
        for val in b.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }

        let c_strassen = strassen_matmul(&a, &b);
        let c_classical = a.dot(&b);

        assert_eq!(c_strassen.dim(), (size, size));
        for i in 0..size {
            for j in 0..size {
                let diff = (c_strassen[[i, j]] - c_classical[[i, j]]).abs();
                assert!(
                    diff < 1e-9,
                    "Mismatch at ({}, {}) for size 2^{}: Strassen = {}, Classical = {}",
                    i,
                    j,
                    n,
                    c_strassen[[i, j]],
                    c_classical[[i, j]]
                );
            }
        }
    }
}
