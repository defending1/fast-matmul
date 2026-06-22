use crate::cp::CP;
use crate::dynamic_peeling::DynamicPeeling;
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

    /// Returns a reference to the underlying CP decomposition.
    pub(crate) fn cp(&self) -> &CP {
        self.cp
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

    /// Performs standard matrix multiplication using faer's low-level API with controlled parallelism.
    pub fn base_matmul(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        let mut c = Mat::zeros(a.nrows(), b.ncols());
        let par = if multithreaded {
            faer::get_global_parallelism()
        } else {
            faer::Parallelism::None
        };
        faer::linalg::matmul::matmul(c.as_mut(), a.as_ref(), b.as_ref(), None, 1.0, par);
        c
    }

    /// Performs one step of dynamic peeling in the multiplication C = A * B.
    ///
    /// Delegates to [`DynamicPeeling`], which handles the core CP product and
    /// the GEMM-based boundary corrections for odd or non-power-of-two dimensions.
    fn dynamic_peeling(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        DynamicPeeling::new(self, a, b, multithreaded).run()
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

    pub(crate) fn cp_matmul_impl(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        let m = a.nrows();
        let n = a.ncols();
        let n_b = b.nrows();
        let p = b.ncols();
        assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

        if m < self.cp.m || n < self.cp.n || p < self.cp.p || n <= 128 || m <= 128 || p <= 128 {
            return self.base_matmul(a, b, multithreaded);
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
