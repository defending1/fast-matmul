use faer::Mat;
use fast_matmul::cp::CP;
use fast_matmul::matmul::{BaseMatMul, MatMul, ParallelismMode, RecursionLimit};
use rand::Rng;

#[test]
fn test_cp_matmul_correctness() {
    let mut rng = rand::thread_rng();

    // List of algorithms to test, representing various shapes (M, N, P)
    let algorithms = vec![
        "classical222-8-24",  // 2x2x2
        "grey322-11-50",      // 3x2x2
        "grey332-15-103",     // 3x3x2
        "classical333-27-81", // 3x3x3
    ];

    for algo_name in algorithms {
        println!("Testing generic algorithm: {}", algo_name);
        let cp = CP::load(algo_name);
        let mm = MatMul::with_cp(&cp);

        // Test unpadded case (exact base dimensions)
        let m = cp.m;
        let n = cp.n;
        let p = cp.p;

        let mut a = Mat::<f64>::zeros(m, n);
        for r in 0..m {
            for c in 0..n {
                a[(r, c)] = rng.gen_range(-5.0..5.0);
            }
        }
        let mut b = Mat::<f64>::zeros(n, p);
        for r in 0..n {
            for c in 0..p {
                b[(r, c)] = rng.gen_range(-5.0..5.0);
            }
        }

        let modes = [
            ParallelismMode::Dfs,
            ParallelismMode::Bfs,
            ParallelismMode::Hybrid,
            ParallelismMode::Sequential,
        ];
        let bases = [BaseMatMul::Faer, BaseMatMul::Dgemm];

        for &mode in &modes {
            for &base in &bases {
                let limit = if mode == ParallelismMode::Hybrid {
                    RecursionLimit::Depth(5)
                } else {
                    RecursionLimit::Cutoff(1)
                };
                let c_fast = mm.cp_matmul(&a, &b, mode, base, limit);
                let c_classical = &a * &b;

                assert_eq!((c_fast.nrows(), c_fast.ncols()), (m, p));
                for i in 0..m {
                    for j in 0..p {
                        let diff = (c_fast[(i, j)] - c_classical[(i, j)]).abs();
                        assert!(
                            diff < 1e-10,
                            "Mismatch for algorithm {} with mode {:?} at ({}, {}): fast = {}, classical = {}",
                            algo_name,
                            mode,
                            i,
                            j,
                            c_fast[(i, j)],
                            c_classical[(i, j)]
                        );
                    }
                }
            }
        }

        // Test padded case (base dimensions * 2 + odd offset to force padding)
        let m_pad = cp.m * 2 + 1;
        let n_pad = cp.n * 2 + 1;
        let p_pad = cp.p * 2 + 1;

        let mut a_pad = Mat::<f64>::zeros(m_pad, n_pad);
        for r in 0..m_pad {
            for c in 0..n_pad {
                a_pad[(r, c)] = rng.gen_range(-5.0..5.0);
            }
        }
        let mut b_pad = Mat::<f64>::zeros(n_pad, p_pad);
        for r in 0..n_pad {
            for c in 0..p_pad {
                b_pad[(r, c)] = rng.gen_range(-5.0..5.0);
            }
        }

        for &mode in &modes {
            for &base in &bases {
                let limit = if mode == ParallelismMode::Hybrid {
                    RecursionLimit::Depth(1)
                } else {
                    RecursionLimit::Cutoff(1)
                };
                let c_fast_pad = mm.cp_matmul(&a_pad, &b_pad, mode, base, limit);
                let c_classical_pad = &a_pad * &b_pad;

                assert_eq!((c_fast_pad.nrows(), c_fast_pad.ncols()), (m_pad, p_pad));
                for i in 0..m_pad {
                    for j in 0..p_pad {
                        let diff = (c_fast_pad[(i, j)] - c_classical_pad[(i, j)]).abs();
                        assert!(
                            diff < 1e-10,
                            "Mismatch with padding for algorithm {} with mode {:?} at ({}, {}): fast = {}, classical = {}",
                            algo_name,
                            mode,
                            i,
                            j,
                            c_fast_pad[(i, j)],
                            c_classical_pad[(i, j)]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_recursion_limits() {
    let mut rng = rand::thread_rng();
    let cp = CP::load("classical222-8-24"); // Strassen 2x2x2
    let mm = MatMul::with_cp(&cp);

    // Let's use a size of 8x8.
    let size = 8;
    let mut a = Mat::<f64>::zeros(size, size);
    let mut b = Mat::<f64>::zeros(size, size);
    for r in 0..size {
        for c in 0..size {
            a[(r, c)] = rng.gen_range(-1.0..1.0);
            b[(r, c)] = rng.gen_range(-1.0..1.0);
        }
    }

    let c_classical = &a * &b;

    let limits = [
        RecursionLimit::Depth(0),
        RecursionLimit::Depth(1),
        RecursionLimit::Depth(2),
        RecursionLimit::Depth(3),
        RecursionLimit::Cutoff(8),
        RecursionLimit::Cutoff(4),
        RecursionLimit::Cutoff(2),
        RecursionLimit::Cutoff(1),
    ];

    for &limit in &limits {
        let c_fast = mm.cp_matmul(&a, &b, ParallelismMode::Sequential, BaseMatMul::Faer, limit);

        assert_eq!((c_fast.nrows(), c_fast.ncols()), (size, size));
        for i in 0..size {
            for j in 0..size {
                let diff = (c_fast[(i, j)] - c_classical[(i, j)]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch for recursion limit {:?} at ({}, {}): fast = {}, classical = {}",
                    limit,
                    i,
                    j,
                    c_fast[(i, j)],
                    c_classical[(i, j)]
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "Hybrid parallelism mode is only supported with RecursionLimit::Depth")]
fn test_hybrid_mode_cutoff_panic() {
    let cp = CP::load("classical222-8-24");
    let mm = MatMul::with_cp(&cp);
    let a = Mat::<f64>::zeros(cp.m, cp.n);
    let b = Mat::<f64>::zeros(cp.n, cp.p);
    mm.cp_matmul(
        &a,
        &b,
        ParallelismMode::Hybrid,
        BaseMatMul::Faer,
        RecursionLimit::Cutoff(1),
    );
}

#[test]
fn test_hybrid_mode_correctness_and_threads() {
    let mut rng = rand::thread_rng();
    let cp = CP::load("classical222-8-24");
    let mm = MatMul::with_cp(&cp);

    let size = 8;
    let mut a = Mat::<f64>::zeros(size, size);
    let mut b = Mat::<f64>::zeros(size, size);
    for r in 0..size {
        for c in 0..size {
            a[(r, c)] = rng.gen_range(-1.0..1.0);
            b[(r, c)] = rng.gen_range(-1.0..1.0);
        }
    }

    let c_classical = &a * &b;

    let thread_counts = [1, 2, 4, 7, 8, 16];
    for &threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        pool.install(|| {
            let c_fast = mm.cp_matmul(
                &a,
                &b,
                ParallelismMode::Hybrid,
                BaseMatMul::Faer,
                RecursionLimit::Depth(2),
            );

            assert_eq!((c_fast.nrows(), c_fast.ncols()), (size, size));
            for i in 0..size {
                for j in 0..size {
                    let diff = (c_fast[(i, j)] - c_classical[(i, j)]).abs();
                    assert!(
                        diff < 1e-10,
                        "Mismatch in Hybrid mode with {} threads at ({}, {}): diff = {}",
                        threads,
                        i,
                        j,
                        diff
                    );
                }
            }
        });
    }
}
