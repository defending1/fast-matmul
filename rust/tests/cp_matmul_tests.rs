use fast_matmul::cp::CP;
use fast_matmul::matmul::MatMul;
use ndarray::Array2;
use rand::Rng;

#[test]
fn test_cp_matmul_correctness() {
    let mut rng = rand::thread_rng();

    // List of algorithms to test, representing various shapes (M, N, P)
    let algorithms = vec![
        "classical222-8-24", // 2x2x2
        "grey322-11-50",     // 3x2x2
        "grey332-15-103",    // 3x3x2
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

        let mut a = Array2::zeros((m, n));
        for val in a.iter_mut() {
            *val = rng.gen_range(-5.0..5.0);
        }
        let mut b = Array2::zeros((n, p));
        for val in b.iter_mut() {
            *val = rng.gen_range(-5.0..5.0);
        }

        let c_fast = mm.cp_matmul(&a, &b);
        let c_classical = a.dot(&b);

        assert_eq!(c_fast.dim(), (m, p));
        for i in 0..m {
            for j in 0..p {
                let diff = (c_fast[[i, j]] - c_classical[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch for algorithm {} at ({}, {}): fast = {}, classical = {}",
                    algo_name, i, j, c_fast[[i, j]], c_classical[[i, j]]
                );
            }
        }

        // Test padded case (base dimensions * 2 + odd offset to force padding)
        let m_pad = cp.m * 2 + 1;
        let n_pad = cp.n * 2 + 1;
        let p_pad = cp.p * 2 + 1;

        let mut a_pad = Array2::zeros((m_pad, n_pad));
        for val in a_pad.iter_mut() {
            *val = rng.gen_range(-5.0..5.0);
        }
        let mut b_pad = Array2::zeros((n_pad, p_pad));
        for val in b_pad.iter_mut() {
            *val = rng.gen_range(-5.0..5.0);
        }

        let c_fast_pad = mm.cp_matmul(&a_pad, &b_pad);
        let c_classical_pad = a_pad.dot(&b_pad);

        assert_eq!(c_fast_pad.dim(), (m_pad, p_pad));
        for i in 0..m_pad {
            for j in 0..p_pad {
                let diff = (c_fast_pad[[i, j]] - c_classical_pad[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch with padding for algorithm {} at ({}, {}): fast = {}, classical = {}",
                    algo_name, i, j, c_fast_pad[[i, j]], c_classical_pad[[i, j]]
                );
            }
        }
    }
}
