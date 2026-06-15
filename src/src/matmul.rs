/// Natural column-major mapping (1-indexed): L(r, c; rows, cols) = r + (c - 1) * rows.
#[inline]
pub fn l_map(r: usize, c: usize, rows: usize, _cols: usize) -> usize {
    r + (c - 1) * rows
}

/// Natural column-major inverse mapping (1-indexed): L^-1(idx; rows, cols) = (row, col).
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

/// A 3-dimensional tensor represented in column-major order.
/// Dimensions are (I, J, K), where:
/// - I = m * n (size of vec(U))
/// - J = n * p (size of vec(V))
/// - K = m * p (size of vec(W^T))
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor3D {
    pub shape: (usize, usize, usize),
    pub data: Vec<f64>,
}

impl Tensor3D {
    /// Create a new 3D tensor initialized with zeros of the given shape.
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Map a 3D index (i, j, k) to a 1D column-major flat index.
    /// In column-major order, the first dimension (i) varies fastest,
    /// then the second (j), and then the third (k).
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        let (shape_i, shape_j, _shape_k) = self.shape;
        i + j * shape_i + k * shape_i * shape_j
    }

    /// Retrieve the value at the 3D index (i, j, k).
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        let idx = self.index(i, j, k);
        self.data[idx]
    }

    /// Set the value at the 3D index (i, j, k).
    pub fn set(&mut self, i: usize, j: usize, k: usize, val: f64) {
        let idx = self.index(i, j, k);
        self.data[idx] = val;
    }
}

/// Returns the matrix multiplication tensor X representing <m, n, p> as defined in Exercise 3.
/// The shape of X will be (m*n, n*p, m*p).
pub fn matmul(m: usize, n: usize, p: usize) -> Tensor3D {
    let mut x = Tensor3D::new((m * n, n * p, m * p));

    // Loop k from 1 to m*p
    for k in 1..=(m * p) {
        // (L*)^-1(k; m x p) = (k_r, k_c)
        let (k_r, k_c) = l_star_map_inv(k, m, p);

        // Loop h from 1 to n
        for h in 1..=n {
            // i = L(k_r, h; m x n)
            let i = l_map(k_r, h, m, n);

            // j = L(h, k_c; n x p)
            let j = l_map(h, k_c, n, p);

            // Convert to 0-based indices for Tensor3D storage
            x.set(i - 1, j - 1, k - 1, 1.0);
        }
    }

    x
}

/// Evaluates the mode product X x_1 vec(U)^T x_2 vec(V)^T.
/// The result is a vector of size K = m * p.
pub fn evaluate_tensor_product(x: &Tensor3D, vec_u: &[f64], vec_v: &[f64]) -> Vec<f64> {
    let (shape_i, shape_j, shape_k) = x.shape;
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

    let mut z = vec![0.0; shape_k];

    for k in 0..shape_k {
        let mut sum_k = 0.0;
        for j in 0..shape_j {
            for i in 0..shape_i {
                let x_val = x.get(i, j, k);
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
pub fn standard_matmul_vec_wt(
    m: usize,
    n: usize,
    p: usize,
    vec_u: &[f64],
    vec_v: &[f64],
) -> Vec<f64> {
    let mut vec_wt = vec![0.0; m * p];

    for r in 1..=m {
        for c in 1..=p {
            let mut sum = 0.0;
            for h in 1..=n {
                // U_{r, h} located at index L(r, h; m, n) (1-indexed)
                let u_idx = l_map(r, h, m, n);
                let u_val = vec_u[u_idx - 1];

                // V_{h, c} located at index L(h, c; n, p) (1-indexed)
                let v_idx = l_map(h, c, n, p);
                let v_val = vec_v[v_idx - 1];

                sum += u_val * v_val;
            }
            // W^T_{c, r} = W_{r, c}, index under column-major ordering is L(c, r; p, m) (1-indexed)
            let wt_idx = l_map(c, r, p, m);
            vec_wt[wt_idx - 1] = sum;
        }
    }

    vec_wt
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
        assert_eq!(x.shape, (4, 4, 4));

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
                // Recall index(i, j, k) varies column-major in 2D slices as well,
                // but expected arrays are row-major from the visual display,
                // so we index expected arrays using [i * 4 + j].
                assert_eq!(
                    x.get(i, j, 0),
                    expected_x1[i * 4 + j],
                    "Mismatch at slice 0, index ({}, {})",
                    i,
                    j
                );
                assert_eq!(
                    x.get(i, j, 1),
                    expected_x2[i * 4 + j],
                    "Mismatch at slice 1, index ({}, {})",
                    i,
                    j
                );
                assert_eq!(
                    x.get(i, j, 2),
                    expected_x3[i * 4 + j],
                    "Mismatch at slice 2, index ({}, {})",
                    i,
                    j
                );
                assert_eq!(
                    x.get(i, j, 3),
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

            let res_tensor = evaluate_tensor_product(&x, &vec_u, &vec_v);
            let res_standard = standard_matmul_vec_wt(m, n, p, &vec_u, &vec_v);

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
