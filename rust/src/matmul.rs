use crate::cp::CP;
use crate::l_map::{l_map, l_star_map_inv};
use faer::{Mat, Col};
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

    /// Computes the standard matrix multiplication C = A * B and returns vec(C^T).
    pub fn standard_matmul_vec_wt(&self, a: &Mat<f64>, b: &Mat<f64>) -> Col<f64> {
        let c = a * b;
        Col::from_fn(c.nrows() * c.ncols(), |idx| {
            let r = idx / c.ncols();
            let col = idx % c.ncols();
            c[(r, col)]
        })
    }

    /// Computes C = A * B using the CP decomposition formula:
    /// vec(C^T) = sum_{l=1}^r (u_l^T vec(A)) * (v_l^T vec(B)) * w_l
    /// assuming row-major layout vectorization for A, B, C.
    pub fn matmul_cp(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        assert_eq!((a.nrows(), a.ncols()), (self.cp.m, self.cp.n));
        assert_eq!((b.nrows(), b.ncols()), (self.cp.n, self.cp.p));

        let vec_a = Col::from_fn(a.nrows() * a.ncols(), |idx| {
            let r = idx / a.ncols();
            let c = idx % a.ncols();
            a[(r, c)]
        });
        let vec_b = Col::from_fn(b.nrows() * b.ncols(), |idx| {
            let r = idx / b.ncols();
            let c = idx % b.ncols();
            b[(r, c)]
        });

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

        Mat::from_fn(self.cp.m, self.cp.p, |r, c| {
            c_vec[r * self.cp.p + c]
        })
    }

    /// Pads the matrices `a` and `b` to multiples of the CP decomposition dimensions if necessary.
    pub fn pad_matrices(
        &self,
        a: &Mat<f64>,
        b: &Mat<f64>,
    ) -> (Mat<f64>, Mat<f64>, bool, usize, usize, usize) {
        let m = a.nrows();
        let n = a.ncols();
        let n_b = b.nrows();
        let p = b.ncols();
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
            let mut a_new = Mat::<f64>::zeros(next_m, next_n);
            a_new.as_mut().get_mut(0..m, 0..n).copy_from(a);

            let mut b_new = Mat::<f64>::zeros(next_n, next_p);
            b_new.as_mut().get_mut(0..n, 0..p).copy_from(b);

            (a_new, b_new, true, next_m, next_n, next_p)
        } else {
            (a.clone(), b.clone(), false, m, n, p)
        }
    }

    /// Helper to compute a single Strassen product M_l
    fn compute_m_l(
        &self,
        l: usize,
        a_blocks: &[Mat<f64>],
        b_blocks: &[Mat<f64>],
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
    fn combine_blocks(blocks: &[Mat<f64>], coeffs: faer::ColRef<'_, f64>) -> Mat<f64> {
        let mut comb = Mat::<f64>::zeros(blocks[0].nrows(), blocks[0].ncols());
        for (block, &coeff) in blocks.iter().zip(coeffs.iter()) {
            if coeff != 0.0 {
                for r in 0..comb.nrows() {
                    for c in 0..comb.ncols() {
                        comb[(r, c)] += coeff * block[(r, c)];
                    }
                }
            }
        }
        comb
    }

    /// Splits a matrix into grid blocks of specified block dimensions.
    fn split_into_blocks(
        matrix: &Mat<f64>,
        grid_rows: usize,
        grid_cols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Vec<Mat<f64>> {
        let mut blocks = Vec::with_capacity(grid_rows * grid_cols);
        for i in 0..grid_rows {
            for j in 0..grid_cols {
                let r_range = i * block_rows..(i + 1) * block_rows;
                let c_range = j * block_cols..(j + 1) * block_cols;
                let block = matrix.as_ref().get(r_range, c_range).to_owned();
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

        let (a_padded, b_padded, need_padding, next_m, next_n, next_p) = self.pad_matrices(a, b);

        let m_block = next_m / self.cp.m;
        let n_block = next_n / self.cp.n;
        let p_block = next_p / self.cp.p;

        let a_blocks = Self::split_into_blocks(&a_padded, self.cp.m, self.cp.n, m_block, n_block);
        let b_blocks = Self::split_into_blocks(&b_padded, self.cp.n, self.cp.p, n_block, p_block);

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

        let mut c_blocks = vec![Mat::<f64>::zeros(m_block, p_block); self.cp.m * self.cp.p];
        for (l, m_prod) in m_products.iter().enumerate() {
            for (i, block) in c_blocks.iter_mut().enumerate() {
                let coeff = self.cp.w[(i, l)];
                if coeff != 0.0 {
                    for r in 0..block.nrows() {
                        for c in 0..block.ncols() {
                            block[(r, c)] += coeff * m_prod[(r, c)];
                        }
                    }
                }
            }
        }

        let mut c_padded = Mat::<f64>::zeros(next_m, next_p);
        for i in 0..self.cp.m {
            for j in 0..self.cp.p {
                let block_idx = i * self.cp.p + j;
                c_padded
                    .as_mut()
                    .get_mut(
                        i * m_block..(i + 1) * m_block,
                        j * p_block..(j + 1) * p_block
                    )
                    .copy_from(&c_blocks[block_idx]);
            }
        }

        if need_padding {
            c_padded.as_ref().get(0..m, 0..p).to_owned()
        } else {
            c_padded
        }
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively (single-threaded).
    pub fn cp_matmul_single_thread(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        self.cp_matmul_impl(a, b, false)
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively.
    pub fn cp_matmul(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        self.cp_matmul_impl(a, b, true)
    }

    /// Computes C = A * B using Intel MKL dgemm (FFI).
    pub fn mkl_matmul(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        let m = a.nrows();
        let k = a.ncols();
        let k_b = b.nrows();
        let n = b.ncols();
        assert_eq!(k, k_b, "Matrix dimensions must agree for multiplication");

        // Convert a and b to contiguous row-major layout
        let mut a_row_major = Vec::with_capacity(m * k);
        for r in 0..m {
            for c in 0..k {
                a_row_major.push(a[(r, c)]);
            }
        }

        let mut b_row_major = Vec::with_capacity(k * n);
        for r in 0..k {
            for c in 0..n {
                b_row_major.push(b[(r, c)]);
            }
        }

        let mut c_row_major = vec![0.0; m * n];

        unsafe {
            mkl_dgemm_wrapper(
                m as i32,
                n as i32,
                k as i32,
                a_row_major.as_ptr(),
                b_row_major.as_ptr(),
                c_row_major.as_mut_ptr(),
            );
        }

        // Convert c back to Mat (column-major layout)
        Mat::from_fn(m, n, |r, c| c_row_major[r * n + c])
    }
}

unsafe extern "C" {
    fn mkl_dgemm_wrapper(
        m: i32,
        n: i32,
        k: i32,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
    );
}
