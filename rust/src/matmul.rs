use crate::cp::CP;
use crate::l_map::{l_map, l_star_map_inv};
use faer::{Col, Mat, MatRef};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct Tensor3 {
    pub data: Vec<f64>,
    pub shape: (usize, usize, usize),
}

impl Tensor3 {
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self {
            data: vec![0.0; shape.0 * shape.1 * shape.2],
            shape,
        }
    }

    pub fn dim(&self) -> (usize, usize, usize) {
        self.shape
    }
}

impl Index<[usize; 3]> for Tensor3 {
    type Output = f64;
    fn index(&self, index: [usize; 3]) -> &Self::Output {
        let [i, j, k] = index;
        &self.data[i * self.shape.1 * self.shape.2 + j * self.shape.2 + k]
    }
}

impl IndexMut<[usize; 3]> for Tensor3 {
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        let [i, j, k] = index;
        &mut self.data[i * self.shape.1 * self.shape.2 + j * self.shape.2 + k]
    }
}

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
    pub fn matmul(&self, m: usize, n: usize, p: usize) -> Tensor3 {
        let mut x = Tensor3::zeros((m * n, n * p, m * p));

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
    /// The result is a 1D Col of size K = m * p.
    pub fn evaluate_tensor_product(
        &self,
        x: &Tensor3,
        vec_a: &Col<f64>,
        vec_b: &Col<f64>,
    ) -> Col<f64> {
        let (shape_i, shape_j, shape_k) = x.dim();
        assert_eq!(
            vec_a.nrows(),
            shape_i,
            "vec_a length must match mode-1 dimension"
        );
        assert_eq!(
            vec_b.nrows(),
            shape_j,
            "vec_b length must match mode-2 dimension"
        );

        let mut vec_c = Col::<f64>::zeros(shape_k);

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

    /// Flatten a matrix in row-major order into a 1D column vector.
    fn flatten_row_major(matrix: &Mat<f64>) -> Col<f64> {
        let ncols = matrix.ncols();
        Col::from_fn(matrix.nrows() * ncols, |idx| {
            let r = idx / ncols;
            let c = idx % ncols;
            matrix[(r, c)]
        })
    }

    /// Computes the standard matrix multiplication C = A * B and returns vec(C^T).
    pub fn standard_matmul_vec_wt(&self, a: &Mat<f64>, b: &Mat<f64>) -> Col<f64> {
        Self::flatten_row_major(&(a * b))
    }

    /// Computes C = A * B using the CP decomposition formula:
    /// vec(C^T) = sum_{l=1}^r (u_l^T vec(A)) * (v_l^T vec(B)) * w_l
    /// assuming row-major layout vectorization for A, B, C.
    pub fn matmul_cp(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        assert_eq!((a.nrows(), a.ncols()), (self.cp.m, self.cp.n));
        assert_eq!((b.nrows(), b.ncols()), (self.cp.n, self.cp.p));

        let vec_a = Self::flatten_row_major(a);
        let vec_b = Self::flatten_row_major(b);

        // Resulting multiplication vector
        let mut c_vec = Col::<f64>::zeros(self.cp.m * self.cp.p);

        for l in 0..self.cp.rank {
            let u_col = self.cp.u.col(l);
            let mut s_l = 0.0;
            for i in 0..u_col.nrows() {
                s_l += u_col[i] * vec_a[i];
            }

            let v_col = self.cp.v.col(l);
            let mut t_l = 0.0;
            for i in 0..v_col.nrows() {
                t_l += v_col[i] * vec_b[i];
            }

            let m_l = s_l * t_l;

            let w_col = self.cp.w.col(l);
            for i in 0..c_vec.nrows() {
                c_vec[i] += m_l * w_col[i];
            }
        }

        Mat::from_fn(self.cp.m, self.cp.p, |r, c| c_vec[r * self.cp.p + c])
    }

    /// Core Strassen/CP recursive product step of the dynamic peeling algorithm.
    fn peel_core(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool, c: &mut Mat<f64>) {
        let core_m = a.nrows() - a.nrows() % self.cp.m;
        let core_n = a.ncols() - a.ncols() % self.cp.n;
        let core_p = b.ncols() - b.ncols() % self.cp.p;

        if core_m > 0 && core_n > 0 && core_p > 0 {
            let a_core = a.as_ref().get(0..core_m, 0..core_n);
            let b_core = b.as_ref().get(0..core_n, 0..core_p);
            let c_core = self.cp_matmul_impl(&a_core.to_owned(), &b_core.to_owned(), multithreaded);
            c.as_mut().get_mut(0..core_m, 0..core_p).copy_from(&c_core);
        }
    }

    /// Correction for the core product using the peeled column-row strip multiplication.
    fn correct_inner_dimension(&self, a: &Mat<f64>, b: &Mat<f64>, c: &mut Mat<f64>) {
        let core_m = a.nrows() - a.nrows() % self.cp.m;
        let core_n = a.ncols() - a.ncols() % self.cp.n;
        let core_p = b.ncols() - b.ncols() % self.cp.p;
        let extra_n = a.ncols() % self.cp.n;
        let n = a.ncols();

        if extra_n > 0 && core_m > 0 && core_p > 0 {
            let a_extra = a.as_ref().get(0..core_m, core_n..n);
            let b_extra = b.as_ref().get(core_n..n, 0..core_p);
            let c_extra = &a_extra.to_owned() * &b_extra.to_owned();
            let mut c_core_block = c.as_mut().get_mut(0..core_m, 0..core_p);
            for j in 0..core_p {
                for i in 0..core_m {
                    c_core_block[(i, j)] += c_extra[(i, j)];
                }
            }
        }
    }

    /// Computes the peeled far right columns of the matrix product.
    fn correct_right_columns(&self, a: &Mat<f64>, b: &Mat<f64>, c: &mut Mat<f64>) {
        let n = a.ncols();
        let p = b.ncols();
        let m = a.nrows();
        let extra_p = p % self.cp.p;
        let core_p = p - extra_p;

        if extra_p > 0 {
            let b_extra = b.as_ref().get(0..n, core_p..p);
            let c_extra = a * &b_extra.to_owned();
            c.as_mut().get_mut(0..m, core_p..p).copy_from(&c_extra);
        }
    }

    /// Computes the peeled bottom rows of the matrix product (excluding rightmost columns).
    fn correct_bottom_rows(&self, a: &Mat<f64>, b: &Mat<f64>, c: &mut Mat<f64>) {
        let n = a.ncols();
        let m = a.nrows();
        let extra_m = m % self.cp.m;
        let core_m = m - extra_m;
        let extra_p = b.ncols() % self.cp.p;
        let core_p = b.ncols() - extra_p;

        if extra_m > 0 && core_p > 0 {
            let a_extra = a.as_ref().get(core_m..m, 0..n);
            let b_extra = b.as_ref().get(0..n, 0..core_p);
            let c_extra = &a_extra.to_owned() * &b_extra.to_owned();
            c.as_mut().get_mut(core_m..m, 0..core_p).copy_from(&c_extra);
        }
    }

    /// Performs one step of dynamic peeling in the multiplication C = A * B.
    ///
    /// The input matrices `a` (M x N) and `b` (N x P) are split into a divisible core
    /// of dimensions `(m - extra_m) x (n - extra_n)` and `(n - extra_n) x (p - extra_p)`, respectively.
    /// The core multiplication is performed recursively using the CP fast matrix multiplication,
    /// and the peeled boundaries (extra rows/columns) are corrected using standard GEMM.
    fn dynamic_peeling(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());

        self.peel_core(a, b, multithreaded, &mut c);
        self.correct_inner_dimension(a, b, &mut c);
        self.correct_right_columns(a, b, &mut c);
        self.correct_bottom_rows(a, b, &mut c);

        c
    }

    /// Helper to compute a single Strassen product M_l
    fn compute_m_l(
        &self,
        l: usize,
        a_blocks: &[MatRef<'_, f64>],
        b_blocks: &[MatRef<'_, f64>],
        multithreaded: bool,
    ) -> Mat<f64> {
        let a_comb = Self::combine_blocks(a_blocks, self.cp.u.col(l));
        let b_comb = Self::combine_blocks(b_blocks, self.cp.v.col(l));

        self.cp_matmul_impl(&a_comb, &b_comb, multithreaded)
    }

    /// Computes the dot product of a slice of matrix blocks
    /// weighted by a 1D vector of coefficients.
    ///
    /// This is an optimized in-place operation that uses loops to avoid allocating
    /// temporary intermediate arrays, and skips operations for zero coefficients.
    fn combine_blocks(blocks: &[MatRef<'_, f64>], coeffs: faer::ColRef<'_, f64>) -> Mat<f64> {
        let mut comb = Mat::<f64>::zeros(blocks[0].nrows(), blocks[0].ncols());
        for (block, &coeff) in blocks.iter().zip(coeffs.iter()) {
            if coeff != 0.0 {
                for c in 0..comb.ncols() {
                    for r in 0..comb.nrows() {
                        comb[(r, c)] += coeff * block[(r, c)];
                    }
                }
            }
        }
        comb
    }

    /// Splits a matrix into grid blocks of specified block dimensions.
    fn split_into_blocks<'b>(
        matrix: &'b Mat<f64>,
        grid_rows: usize,
        grid_cols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Vec<MatRef<'b, f64>> {
        let mut blocks = Vec::with_capacity(grid_rows * grid_cols);
        for i in 0..grid_rows {
            for j in 0..grid_cols {
                let r_range = i * block_rows..(i + 1) * block_rows;
                let c_range = j * block_cols..(j + 1) * block_cols;
                let block = matrix.as_ref().get(r_range, c_range);
                blocks.push(block);
            }
        }
        blocks
    }

    fn cp_matmul_impl(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        let m = a.nrows();
        let n = a.ncols();
        let n_b = b.nrows();
        let p = b.ncols();
        assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

        if m < self.cp.m || n < self.cp.n || p < self.cp.p || n <= 128 || m <= 128 || p <= 128 {
            return a * b;
        }

        if m == self.cp.m && n == self.cp.n && p == self.cp.p {
            return self.matmul_cp(a, b);
        }

        let extra_m = m % self.cp.m;
        let extra_n = n % self.cp.n;
        let extra_p = p % self.cp.p;

        if extra_m > 0 || extra_n > 0 || extra_p > 0 {
            return self.dynamic_peeling(a, b, multithreaded);
        }

        let m_block = m / self.cp.m;
        let n_block = n / self.cp.n;
        let p_block = p / self.cp.p;

        let a_blocks = Self::split_into_blocks(a, self.cp.m, self.cp.n, m_block, n_block);
        let b_blocks = Self::split_into_blocks(b, self.cp.n, self.cp.p, n_block, p_block);

        const PARALLEL_CUTOFF: usize = 256;
        let m_products: Vec<Mat<f64>> = if multithreaded
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

        let mut c = Mat::<f64>::zeros(m, p);
        for (l, m_prod) in m_products.iter().enumerate() {
            for i in 0..self.cp.m {
                for j in 0..self.cp.p {
                    let coeff = self.cp.w[(i * self.cp.p + j, l)];
                    if coeff != 0.0 {
                        let mut block = c.as_mut().get_mut(
                            i * m_block..(i + 1) * m_block,
                            j * p_block..(j + 1) * p_block,
                        );
                        for c_idx in 0..p_block {
                            for r_idx in 0..m_block {
                                block[(r_idx, c_idx)] += coeff * m_prod[(r_idx, c_idx)];
                            }
                        }
                    }
                }
            }
        }

        c
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively (single-threaded).
    pub fn cp_matmul_single_thread(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        self.cp_matmul_impl(a, b, false)
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively.
    pub fn cp_matmul(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        self.cp_matmul_impl(a, b, true)
    }
}
