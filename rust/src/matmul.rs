use crate::cp::CP;
use crate::l_map::{l_map, l_star_map_inv};
use ndarray::{Array1, Array2, Array3};

/// Matrix Multiplication operations and algorithms.
pub struct MatMul<'a> {
    cp: &'a CP,
}

impl Default for MatMul<'static> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> MatMul<'a> {
    /// Creates a new `MatMul` operator instance.
    pub fn new() -> MatMul<'static> {
        MatMul {
            cp: CP::get_strassen(),
        }
    }

    /// Creates a new `MatMul` operator instance with a custom CP decomposition.
    pub fn with_cp(cp: &'a CP) -> Self {
        Self { cp }
    }

    /// Returns the matrix multiplication tensor X representing <m, n, p> as defined in report.
    /// The shape of X will be (m*n, n*p, m*p).
    pub fn matmul(&self, m: usize, n: usize, p: usize) -> Array3<f64> {
        let mut x = Array3::zeros((m * n, n * p, m * p));

        for k in 1..=(m * p) {
            let (k_r, k_c) = l_star_map_inv(k, m, p);

            for h in 1..=n {
                let i = l_map(k_r, h, m, n);
                let j = l_map(h, k_c, n, p);

                x[[i - 1, j - 1, k - 1]] = 1.0;
            }
        }

        x
    }

    /// Evaluates the mode product X x_1 vec_a^T x_2 vec_b^T.
    /// The result is a 1D Array1 of size K = m * p.
    pub fn evaluate_tensor_product(
        &self,
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
    pub fn standard_matmul_vec_wt(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
        let c = a.dot(b);
        Array1::from_iter(c.iter().cloned())
    }

    /// Computes C = A * B using the CP decomposition formula:
    /// vec(C^T) = sum_{l=1}^r (u_l^T vec(A)) * (v_l^T vec(B)) * w_l
    /// assuming row-major layout vectorization for A, B, C.
    pub fn matmul_cp(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        assert_eq!(a.dim(), (self.cp.m, self.cp.n));
        assert_eq!(b.dim(), (self.cp.n, self.cp.p));

        let vec_a = Array1::from_iter(a.iter().cloned());
        let vec_b = Array1::from_iter(b.iter().cloned());

        // Resulting multiplication vector
        let mut c_vec = Array1::zeros(self.cp.m * self.cp.p);

        for l in 0..self.cp.rank {
            let s_l = self.cp.u.column(l).dot(&vec_a);
            let t_l = self.cp.v.column(l).dot(&vec_b);
            let m_l = s_l * t_l;

            c_vec.scaled_add(m_l, &self.cp.w.column(l));
        }

        c_vec.into_shape_with_order((self.cp.m, self.cp.p)).unwrap()
    }

    /// Pads the matrices `a` and `b` to multiples of the CP decomposition dimensions if necessary.
    pub fn pad_matrices(
        &self,
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

        if m % self.cp.m != 0 {
            next_m = m + (self.cp.m - m % self.cp.m);
            need_padding = true;
        }
        if n % self.cp.n != 0 {
            next_n = n + (self.cp.n - n % self.cp.n);
            need_padding = true;
        }
        if p % self.cp.p != 0 {
            next_p = p + (self.cp.p - p % self.cp.p);
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
    fn compute_m_l(
        &self,
        l: usize,
        a_blocks: &[Array2<f64>],
        b_blocks: &[Array2<f64>],
        multithreaded: bool,
    ) -> Array2<f64> {
        let a_comb = Self::combine_blocks(a_blocks, self.cp.u.column(l));
        let b_comb = Self::combine_blocks(b_blocks, self.cp.v.column(l));

        self.cp_matmul_impl(&a_comb, &b_comb, multithreaded)
    }

    /// Computes the dot product of a slice of matrix blocks
    /// weighted by a 1D vector of coefficients.
    ///
    /// This is an optimized in-place operation that uses `scaled_add` to avoid allocating
    /// temporary intermediate arrays, and skips operations for zero coefficients.
    fn combine_blocks(blocks: &[Array2<f64>], coeffs: ndarray::ArrayView1<f64>) -> Array2<f64> {
        let mut comb = Array2::zeros(blocks[0].dim());
        for (block, &coeff) in blocks.iter().zip(coeffs) {
            if coeff != 0.0 {
                comb.scaled_add(coeff, block);
            }
        }
        comb
    }

    /// Splits a matrix into grid blocks of specified block dimensions.
    fn split_into_blocks(
        matrix: &Array2<f64>,
        grid_rows: usize,
        grid_cols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Vec<Array2<f64>> {
        let mut blocks = Vec::with_capacity(grid_rows * grid_cols);
        for i in 0..grid_rows {
            for j in 0..grid_cols {
                let block = matrix
                    .slice(ndarray::s![
                        i * block_rows..(i + 1) * block_rows,
                        j * block_cols..(j + 1) * block_cols
                    ])
                    .to_owned();
                blocks.push(block);
            }
        }
        blocks
    }

    fn cp_matmul_impl(&self, a: &Array2<f64>, b: &Array2<f64>, multithreaded: bool) -> Array2<f64> {
        let (m, n) = a.dim();
        let (n_b, p) = b.dim();
        assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

        if m < self.cp.m || n < self.cp.n || p < self.cp.p || n <= 128 || m <= 128 || p <= 128 {
            return a.dot(b);
        }

        if m == self.cp.m && n == self.cp.n && p == self.cp.p {
            return self.matmul_cp(a, b);
        }

        let (a_padded, b_padded, need_padding, next_m, next_n, next_p) = self.pad_matrices(a, b);

        let m_block = next_m / self.cp.m;
        let n_block = next_n / self.cp.n;
        let p_block = next_p / self.cp.p;

        let a_blocks = Self::split_into_blocks(&a_padded, self.cp.m, self.cp.n, m_block, n_block);
        let b_blocks = Self::split_into_blocks(&b_padded, self.cp.n, self.cp.p, n_block, p_block);

        const PARALLEL_CUTOFF: usize = 64;
        let m_products: Vec<Array2<f64>> = if multithreaded
            && m_block >= PARALLEL_CUTOFF
            && n_block >= PARALLEL_CUTOFF
            && p_block >= PARALLEL_CUTOFF
        {
            use rayon::prelude::*;
            (0..self.cp.rank)
                .into_par_iter()
                .map(|l| self.compute_m_l(l, &a_blocks, &b_blocks, true))
                .collect()
        } else {
            (0..self.cp.rank)
                .map(|l| self.compute_m_l(l, &a_blocks, &b_blocks, multithreaded))
                .collect()
        };

        let mut c_blocks = vec![Array2::zeros((m_block, p_block)); self.cp.m * self.cp.p];
        for (l, m_prod) in m_products.iter().enumerate() {
            for (i, block) in c_blocks.iter_mut().enumerate() {
                let coeff = self.cp.w[[i, l]];
                if coeff != 0.0 {
                    block.scaled_add(coeff, m_prod);
                }
            }
        }

        let mut c_padded = Array2::zeros((next_m, next_p));
        for i in 0..self.cp.m {
            for j in 0..self.cp.p {
                let block_idx = i * self.cp.p + j;
                c_padded
                    .slice_mut(ndarray::s![
                        i * m_block..(i + 1) * m_block,
                        j * p_block..(j + 1) * p_block
                    ])
                    .assign(&c_blocks[block_idx]);
            }
        }

        if need_padding {
            c_padded.slice(ndarray::s![..m, ..p]).to_owned()
        } else {
            c_padded
        }
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively (single-threaded).
    pub fn cp_matmul_single_thread(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        self.cp_matmul_impl(a, b, false)
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively.
    pub fn cp_matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        self.cp_matmul_impl(a, b, true)
    }
}
