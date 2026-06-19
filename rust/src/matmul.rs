use crate::cp::CP;
use ndarray::{Array1, Array2, Array3};

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

/// Returns the matrix multiplication tensor X representing <m, n, p> as defined in report.
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

/// Evaluates the mode product X x_1 vec_a^T x_2 vec_b^T.
/// The result is a 1D Array1 of size K = m * p.
pub fn evaluate_tensor_product(
    x: &Array3<f64>,
    vec_a: &Array1<f64>,
    vec_b: &Array1<f64>,
) -> Array1<f64> {
    let (shape_i, shape_j, shape_k) = x.dim();
    assert_eq!(
        vec_a.len(),
        shape_i,
        "vec_a length must match mode-1 dimension"
    );
    assert_eq!(
        vec_b.len(),
        shape_j,
        "vec_b length must match mode-2 dimension"
    );

    let mut vec_c = Array1::zeros(shape_k);

    for k in 0..shape_k {
        let mut sum_k = 0.0;
        for j in 0..shape_j {
            for i in 0..shape_i {
                let x_val = x[[i, j, k]];
                if x_val != 0.0 {
                    sum_k += x_val * vec_a[i] * vec_b[j];
                }
            }
        }
        vec_c[k] = sum_k;
    }

    vec_c
}

/// Computes the standard matrix multiplication C = A * B and returns vec(C^T).
pub fn standard_matmul_vec_wt(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    // Perform standard matrix multiplication using ndarray's built-in dot product
    let c = a.dot(b);

    // Row-major vectorization of C is equivalent to column-major vectorization of C^T.
    Array1::from_iter(c.iter().cloned())
}

/// Computes C = A * B using the CP decomposition formula:
/// vec(C^T) = sum_{l=1}^r (u_l^T vec(A)) * (v_l^T vec(B)) * w_l
/// assuming row-major layout vectorization for A, B, C.
pub fn matmul_cp(
    a: &Array2<f64>,
    b: &Array2<f64>,
    u: &Array2<f64>,
    v: &Array2<f64>,
    w: &Array2<f64>,
) -> Array2<f64> {
    assert_eq!(a.dim(), (2, 2));
    assert_eq!(b.dim(), (2, 2));

    // Vectorize A and B in row-major order
    let vec_a = [a[[0, 0]], a[[0, 1]], a[[1, 0]], a[[1, 1]]];
    let vec_b = [b[[0, 0]], b[[0, 1]], b[[1, 0]], b[[1, 1]]];

    let mut c_vec = [0.0; 4];

    for l in 0..7 {
        let mut sum_u = 0.0;
        for i in 0..4 {
            sum_u += u[[i, l]] * vec_a[i];
        }
        let mut sum_v = 0.0;
        for i in 0..4 {
            sum_v += v[[i, l]] * vec_b[i];
        }
        let p_l = sum_u * sum_v;

        for i in 0..4 {
            c_vec[i] += p_l * w[[i, l]];
        }
    }

    Array2::from_shape_vec((2, 2), c_vec.to_vec()).unwrap()
}

/// Pads the matrices `a` and `b` to even dimensions if necessary.
///
/// If any of the dimensions (m, n, p) are odd, they are incremented by 1,
/// and the matrices are padded with zeros.
///
/// Returns:
/// - The (possibly padded) matrix A.
/// - The (possibly padded) matrix B.
/// - A boolean flag indicating whether padding was applied.
/// - The dimensions `(next_m, next_n, next_p)` of the padded matrices.
pub fn pad_matrices(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, bool, usize, usize, usize) {
    let (m, n) = a.dim();
    let (n_b, p) = b.dim();
    assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

    let mut next_m = m;
    let mut next_n = n;
    let mut next_p = p;
    let mut need_padding = false;

    if m % 2 != 0 {
        next_m += 1;
        need_padding = true;
    }
    if n % 2 != 0 {
        next_n += 1;
        need_padding = true;
    }
    if p % 2 != 0 {
        next_p += 1;
        need_padding = true;
    }

    if need_padding {
        let mut a_new = Array2::zeros((next_m, next_n));
        a_new.slice_mut(ndarray::s![..m, ..n]).assign(a);

        let mut b_new = Array2::zeros((next_n, next_p));
        b_new.slice_mut(ndarray::s![..n, ..p]).assign(b);

        (a_new, b_new, true, next_m, next_n, next_p)
    } else {
        (a.clone(), b.clone(), false, m, n, p)
    }
}

/// Helper to compute a single Strassen product M_l
#[expect(
    clippy::too_many_arguments,
    reason = "Internal helper for Strassen block multiplication recursion"
)]
fn compute_m_l(
    l: usize,
    m2: usize,
    n2: usize,
    p2: usize,
    a11: &Array2<f64>,
    a12: &Array2<f64>,
    a21: &Array2<f64>,
    a22: &Array2<f64>,
    b11: &Array2<f64>,
    b12: &Array2<f64>,
    b21: &Array2<f64>,
    b22: &Array2<f64>,
    u: &Array2<f64>,
    v: &Array2<f64>,
    multithreaded: bool,
) -> Array2<f64> {
    // Linear combination of A blocks
    let mut a_comb = Array2::zeros((m2, n2));
    if u[[0, l]] != 0.0 {
        a_comb = a_comb + a11 * u[[0, l]];
    }
    if u[[1, l]] != 0.0 {
        a_comb = a_comb + a12 * u[[1, l]];
    }
    if u[[2, l]] != 0.0 {
        a_comb = a_comb + a21 * u[[2, l]];
    }
    if u[[3, l]] != 0.0 {
        a_comb = a_comb + a22 * u[[3, l]];
    }

    // Linear combination of B blocks
    let mut b_comb = Array2::zeros((n2, p2));
    if v[[0, l]] != 0.0 {
        b_comb = b_comb + b11 * v[[0, l]];
    }
    if v[[1, l]] != 0.0 {
        b_comb = b_comb + b12 * v[[1, l]];
    }
    if v[[2, l]] != 0.0 {
        b_comb = b_comb + b21 * v[[2, l]];
    }
    if v[[3, l]] != 0.0 {
        b_comb = b_comb + b22 * v[[3, l]];
    }

    strassen_matmul_impl(&a_comb, &b_comb, multithreaded)
}

fn strassen_matmul_impl(a: &Array2<f64>, b: &Array2<f64>, multithreaded: bool) -> Array2<f64> {
    let (m, n) = a.dim();
    let (n_b, p) = b.dim();
    assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

    // Base Case 1: A and/or B are vectors
    if m == 1 || n == 1 || p == 1 || n <= 128 {
        return a.dot(b);
    }

    // Base Case 2: 2x2 matrices
    if m == 2 && n == 2 && p == 2 {
        let cp = CP::get_strassen();
        return matmul_cp(a, b, &cp.u, &cp.v, &cp.w);
    }

    // If any dimension is odd, pad to even
    let (a_padded, b_padded, need_padding, next_m, next_n, next_p) = pad_matrices(a, b);

    // Now partition the padded matrices into 4 equal blocks
    let m2 = next_m / 2;
    let n2 = next_n / 2;
    let p2 = next_p / 2;

    let a11 = a_padded.slice(ndarray::s![..m2, ..n2]).to_owned();
    let a12 = a_padded.slice(ndarray::s![..m2, n2..]).to_owned();
    let a21 = a_padded.slice(ndarray::s![m2.., ..n2]).to_owned();
    let a22 = a_padded.slice(ndarray::s![m2.., n2..]).to_owned();

    let b11 = b_padded.slice(ndarray::s![..n2, ..p2]).to_owned();
    let b12 = b_padded.slice(ndarray::s![..n2, p2..]).to_owned();
    let b21 = b_padded.slice(ndarray::s![n2.., ..p2]).to_owned();
    let b22 = b_padded.slice(ndarray::s![n2.., p2..]).to_owned();

    // Use Strassen's 7 multiplications using the parsed matrices U, V, W
    let cp = CP::get_strassen();

    // Compute the 7 products M_l
    // Parallelize with Rayon if block dimensions are large enough to offset scheduling overhead.
    const PARALLEL_CUTOFF: usize = 64;
    let m_products: Vec<Array2<f64>> =
        if multithreaded && m2 >= PARALLEL_CUTOFF && n2 >= PARALLEL_CUTOFF && p2 >= PARALLEL_CUTOFF
        {
            use rayon::prelude::*;
            (0..7)
                .into_par_iter()
                .map(|l| {
                    compute_m_l(
                        l, m2, n2, p2, &a11, &a12, &a21, &a22, &b11, &b12, &b21, &b22, &cp.u,
                        &cp.v, true,
                    )
                })
                .collect()
        } else {
            (0..7)
                .map(|l| {
                    compute_m_l(
                        l,
                        m2,
                        n2,
                        p2,
                        &a11,
                        &a12,
                        &a21,
                        &a22,
                        &b11,
                        &b12,
                        &b21,
                        &b22,
                        &cp.u,
                        &cp.v,
                        multithreaded,
                    )
                })
                .collect()
        };

    // Reconstruct C blocks from M_l using W
    let mut c11 = Array2::zeros((m2, p2));
    let mut c12 = Array2::zeros((m2, p2));
    let mut c21 = Array2::zeros((m2, p2));
    let mut c22 = Array2::zeros((m2, p2));

    for (l, m_prod) in m_products.iter().enumerate() {
        if cp.w[[0, l]] != 0.0 {
            c11 = c11 + m_prod * cp.w[[0, l]];
        }
        if cp.w[[1, l]] != 0.0 {
            c12 = c12 + m_prod * cp.w[[1, l]];
        }
        if cp.w[[2, l]] != 0.0 {
            c21 = c21 + m_prod * cp.w[[2, l]];
        }
        if cp.w[[3, l]] != 0.0 {
            c22 = c22 + m_prod * cp.w[[3, l]];
        }
    }

    // Combine the 4 blocks into a single matrix
    let mut c_padded = Array2::zeros((next_m, next_p));
    c_padded.slice_mut(ndarray::s![..m2, ..p2]).assign(&c11);
    c_padded.slice_mut(ndarray::s![..m2, p2..]).assign(&c12);
    c_padded.slice_mut(ndarray::s![m2.., ..p2]).assign(&c21);
    c_padded.slice_mut(ndarray::s![m2.., p2..]).assign(&c22);

    // Truncate to original size m x p
    if need_padding {
        c_padded.slice(ndarray::s![..m, ..p]).to_owned()
    } else {
        c_padded
    }
}

/// Computes C = A * B using Strassen's algorithm recursively (single-threaded).
pub fn strassen_matmul_single_thread(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    strassen_matmul_impl(a, b, false)
}

/// Computes C = A * B using Strassen's algorithm recursively.
/// - If m = 1, n = 1, or p = 1, uses classical matrix multiplication.
/// - If m = n = p = 2, uses matmul_cp with the Strassen CP decomposition matrices.
/// - Otherwise, partitions the matrices and calls itself recursively.
pub fn strassen_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    strassen_matmul_impl(a, b, true)
}
