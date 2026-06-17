use ndarray::{Array1, Array2, Array3};

/// Natural column-major mapping (1-indexed): L(r, c; rows, cols) = r + (c - 1) * rows.
/// Equivalent to MATLAB's sub2ind for a 2D matrix of shape (rows, cols).
#[inline]
pub fn l_map(r: usize, c: usize, rows: usize, _cols: usize) -> usize {
    r + (c - 1) * rows
}

/// Natural column-major inverse mapping (1-indexed): L^-1(idx; rows, cols) = (row, col).
/// Equivalent to MATLAB's ind2sub for a 2D matrix of shape (rows, cols).
#[inline]
#[allow(dead_code)]
pub fn l_map_inv(idx: usize, rows: usize, _cols: usize) -> (usize, usize) {
    let r = (idx - 1) % rows + 1;
    let c = (idx - 1) / rows + 1;
    (r, c)
}

/// Inverse row-major mapping (1-indexed): L*(r, c; rows, cols) = (r - 1) * cols + c.
#[inline]
#[allow(dead_code)]
pub fn l_star_map(r: usize, c: usize, _rows: usize, cols: usize) -> usize {
    (r - 1) * cols + c
}

/// Inverse row-major inverse mapping (1-indexed): (L*)^-1(idx; rows, cols) = (row, col).
#[inline]
pub fn l_star_map_inv(idx: usize, _rows: usize, cols: usize) -> (usize, usize) {
    let r = (idx - 1) / cols + 1;
    let c = (idx - 1) % cols + 1;
    (r, c)
}

/// Returns the matrix multiplication tensor X representing <m, n, p> as defined in Exercise 3.
/// The shape of X will be (m*n, n*p, m*p).
pub fn matmul(m: usize, n: usize, p: usize) -> Array3<f64> {
    let mut x = Array3::zeros((m * n, n * p, m * p));

    for k in 1..=(m * p) {
        // (L*)^-1(k; m x p) = (k_r, k_c)
        let (k_r, k_c) = l_star_map_inv(k, m, p);

        for h in 1..=n {
            // i = L(k_r, h; m x n)
            let i = l_map(k_r, h, m, n);

            // j = L(h, k_c; n x p)
            let j = l_map(h, k_c, n, p);

            // Convert to 0-based indices for Array3 storage
            x[[i - 1, j - 1, k - 1]] = 1.0;
        }
    }

    x
}

/// Evaluates the mode product X x_1 vec_u^T x_2 vec_v^T.
/// The result is a 1D Array1 of size K = m * p.
pub fn evaluate_tensor_product(
    x: &Array3<f64>,
    vec_u: &Array1<f64>,
    vec_v: &Array1<f64>,
) -> Array1<f64> {
    let (shape_i, shape_j, shape_k) = x.dim();
    assert_eq!(
        vec_u.len(),
        shape_i,
        "vec_u length must match mode-1 dimension"
    );
    assert_eq!(
        vec_v.len(),
        shape_j,
        "vec_v length must match mode-2 dimension"
    );

    let mut z = Array1::zeros(shape_k);

    for k in 0..shape_k {
        let mut sum_k = 0.0;
        for j in 0..shape_j {
            for i in 0..shape_i {
                let x_val = x[[i, j, k]];
                if x_val != 0.0 {
                    sum_k += x_val * vec_u[i] * vec_v[j];
                }
            }
        }
        z[k] = sum_k;
    }

    z
}

/// Computes the standard matrix multiplication W = U * V and returns vec(W^T).
pub fn standard_matmul_vec_wt(u: &Array2<f64>, v: &Array2<f64>) -> Array1<f64> {
    // Perform standard matrix multiplication using ndarray's built-in dot product
    let w = u.dot(v);

    // Row-major vectorization of W is equivalent to column-major vectorization of W^T.
    Array1::from_iter(w.iter().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_matlab_mappings() {
        // sub2ind([2, 3], 1, 1) = 1
        assert_eq!(l_map(1, 1, 2, 3), 1);
        // sub2ind([2, 3], 2, 1) = 2
        assert_eq!(l_map(2, 1, 2, 3), 2);
        // sub2ind([2, 3], 1, 2) = 3
        assert_eq!(l_map(1, 2, 2, 3), 3);
        // sub2ind([2, 3], 2, 3) = 6
        assert_eq!(l_map(2, 3, 2, 3), 6);

        // [r, c] = ind2sub([2, 3], 3) -> (1, 2)
        assert_eq!(l_map_inv(3, 2, 3), (1, 2));
        // [r, c] = ind2sub([2, 3], 6) -> (2, 3)
        assert_eq!(l_map_inv(6, 2, 3), (2, 3));
    }

    #[test]
    fn test_example_2x2_slices() {
        let x = matmul(2, 2, 2);

        // Assert tensor shape is 4x4x4
        assert_eq!(x.dim(), (4, 4, 4));

        // Front slices from Exercise 3 PDF:
        // X1 = [1 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
        // X2 = [0 0 1 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
        // X3 = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 1 0 0]
        // X4 = [0 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 1]

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

        // Check each front slice (which varies i and j for a fixed k)
        for i in 0..4 {
            for j in 0..4 {
                // Recall index varies column-major in 2D slices as well,
                // but expected arrays are row-major from the visual display,
                // so we index expected arrays using [i * 4 + j].
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

        // Test various dimensions
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

            // Generate random U and V
            let vec_u: Vec<f64> = (0..(m * n)).map(|_| rng.gen_range(-10.0..10.0)).collect();
            let vec_v: Vec<f64> = (0..(n * p)).map(|_| rng.gen_range(-10.0..10.0)).collect();

            // Convert to ndarray matrices (transposed load to achieve column-major layout)
            let u_t = Array2::from_shape_vec((n, m), vec_u.clone()).unwrap();
            let u = u_t.t().to_owned();

            let v_t = Array2::from_shape_vec((p, n), vec_v.clone()).unwrap();
            let v = v_t.t().to_owned();

            // Vectorizations in column-major
            let nd_vec_u = Array1::from_vec(vec_u);
            let nd_vec_v = Array1::from_vec(vec_v);

            let res_tensor = evaluate_tensor_product(&x, &nd_vec_u, &nd_vec_v);
            let res_standard = standard_matmul_vec_wt(&u, &v);

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
}
