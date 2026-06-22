use faer::Mat;
use fast_matmul::matmul::MatMul;
use rand::Rng;

#[test]
fn test_strassen_matmul_correctness() {
    let mut rng = rand::thread_rng();
    let mm = MatMul::new();

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
        let mut a = Mat::<f64>::zeros(m, n);
        for r in 0..m {
            for c in 0..n {
                a[(r, c)] = rng.gen_range(-10.0..10.0);
            }
        }
        let mut b = Mat::<f64>::zeros(n, p);
        for r in 0..n {
            for c in 0..p {
                b[(r, c)] = rng.gen_range(-10.0..10.0);
            }
        }

        let c_strassen = mm.cp_matmul(&a, &b);
        let c_classical = &a * &b;

        assert_eq!((c_strassen.nrows(), c_strassen.ncols()), (m, p));
        for i in 0..m {
            for j in 0..p {
                let diff = (c_strassen[(i, j)] - c_classical[(i, j)]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at ({}, {}) for shape ({}, {}, {}): Strassen = {}, Classical = {}",
                    i,
                    j,
                    m,
                    n,
                    p,
                    c_strassen[(i, j)],
                    c_classical[(i, j)]
                );
            }
        }
    }
}

#[test]
fn test_strassen_matmul_single_thread_correctness() {
    let mut rng = rand::thread_rng();
    let mm = MatMul::new();

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
        let mut a = Mat::<f64>::zeros(m, n);
        for r in 0..m {
            for c in 0..n {
                a[(r, c)] = rng.gen_range(-10.0..10.0);
            }
        }
        let mut b = Mat::<f64>::zeros(n, p);
        for r in 0..n {
            for c in 0..p {
                b[(r, c)] = rng.gen_range(-10.0..10.0);
            }
        }

        let c_strassen = mm.cp_matmul_single_thread(&a, &b);
        let c_classical = &a * &b;

        assert_eq!((c_strassen.nrows(), c_strassen.ncols()), (m, p));
        for i in 0..m {
            for j in 0..p {
                let diff = (c_strassen[(i, j)] - c_classical[(i, j)]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at ({}, {}) for shape ({}, {}, {}): Strassen (single thread) = {}, Classical = {}",
                    i,
                    j,
                    m,
                    n,
                    p,
                    c_strassen[(i, j)],
                    c_classical[(i, j)]
                );
            }
        }
    }
}

#[test]
fn test_strassen_power_of_two_correctness() {
    let mut rng = rand::thread_rng();
    let mm = MatMul::new();
    for n in 1..=9 {
        let size = 1 << n;
        println!("Testing power of two size: {}", size);
        let mut a = Mat::<f64>::zeros(size, size);
        let mut b = Mat::<f64>::zeros(size, size);
        for r in 0..size {
            for c in 0..size {
                a[(r, c)] = rng.gen_range(-1.0..1.0);
                b[(r, c)] = rng.gen_range(-1.0..1.0);
            }
        }

        let c_strassen = mm.cp_matmul(&a, &b);
        let c_classical = &a * &b;

        assert_eq!((c_strassen.nrows(), c_strassen.ncols()), (size, size));
        for i in 0..size {
            for j in 0..size {
                let diff = (c_strassen[(i, j)] - c_classical[(i, j)]).abs();
                assert!(
                    diff < 1e-9,
                    "Mismatch at ({}, {}) for size 2^{}: Strassen = {}, Classical = {}",
                    i,
                    j,
                    n,
                    c_strassen[(i, j)],
                    c_classical[(i, j)]
                );
            }
        }
    }
}
