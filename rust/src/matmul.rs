use crate::cp::CP;
use crate::dynamic_peeling::DynamicPeeling;
use faer::{Col, Mat, MatRef};
use std::ops::{Index, IndexMut};

pub use crate::parallelism_mode::ParallelismMode;

/// A 3D tensor representation in row-major layout used in matrix multiplication.
#[derive(Clone, Debug)]
pub struct Tensor3 {
    /// Flat vector containing the elements of the tensor.
    pub data: Vec<f64>,
    /// The shape (dimensions) of the tensor as (depth, rows, cols).
    pub shape: (usize, usize, usize),
}

impl Tensor3 {
    /// Creates a new `Tensor3` of the specified shape with all elements initialized to zero.
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self {
            data: vec![0.0; shape.0 * shape.1 * shape.2],
            shape,
        }
    }

    /// Returns the shape dimensions (depth, rows, cols) of the 3D tensor.
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

    /// Computes classical matrix multiplication C = A * B using the underlying `faer` library,
    /// with optional multithreading.
    pub fn base_matmul(&self, a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) -> Mat<f64> {
        let mut c = Mat::zeros(a.nrows(), b.ncols());
        let par = if multithreaded && a.nrows() >= 384 && a.ncols() >= 384 && b.ncols() >= 384 {
            faer::get_global_parallelism()
        } else {
            faer::Par::Seq
        };
        faer::linalg::matmul::matmul(
            c.as_mut(),
            faer::Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            1.0,
            par,
        );
        c
    }

    /// Calculate the total number of recursion levels for the given input matrix shape.
    /// Performs one step of dynamic peeling in the multiplication C = A * B.
    ///
    /// Delegates to [`DynamicPeeling`], which handles the core CP product and
    /// the GEMM-based boundary corrections for odd or non-power-of-two dimensions.
    fn dynamic_peeling(&self, a: &Mat<f64>, b: &Mat<f64>, mode: ParallelismMode) -> Mat<f64> {
        DynamicPeeling::new(self, a, b, mode).run()
    }

    /// Helper to compute a single Strassen product M_l
    fn compute_m_l(
        &self,
        l: usize,
        a_blocks: &[MatRef<'_, f64>],
        b_blocks: &[MatRef<'_, f64>],
        mode: ParallelismMode,
    ) -> Mat<f64> {
        let a_comb = Self::combine_blocks(a_blocks, self.cp.u.col(l));
        let b_comb = Self::combine_blocks(b_blocks, self.cp.v.col(l));

        self.cp_matmul_impl(&a_comb, &b_comb, mode)
    }

    /// Computes the linear combination of matrix blocks for a single CP decomposition component.
    ///
    /// According to the CP-decomposition formulation in the report, for each rank component $l$,
    /// the matrix blocks are combined using the coefficients from matrices $U$ and $V$ of the
    /// decomposition:
    ///
    /// ```text
    ///   A_comb_l = sum_i ( U_{i, l} * A_i )
    ///   B_comb_l = sum_j ( V_{j, l} * B_j )
    /// ```
    ///
    /// This method computes these weighted sums. It takes a slice of matrix blocks (either $A_i$ or $B_j$)
    /// and accumulates their sum, scaling each block by the corresponding coefficient from
    /// the column `coeffs` (which represents the $l$-th column of matrix $U$ or $V$).
    fn combine_blocks(blocks: &[MatRef<'_, f64>], coeffs: faer::ColRef<'_, f64>) -> Mat<f64> {
        let mut comb = Mat::<f64>::zeros(blocks[0].nrows(), blocks[0].ncols());
        for (block, &coeff) in blocks.iter().zip(coeffs.iter()) {
            // To optimize performance, it skips blocks with a coefficient of exactly `0.0` and
            // accumulates directly into the output matrix `comb` without allocating temporary arrays.
            if coeff != 0.0 {
                comb += faer::Scale(coeff) * block;
            }
        }
        comb
    }

    /// Splits a matrix into grid blocks of specified block dimensions.
    ///
    /// # Visualizing a 2^N x 2^N matrix split into 4 blocks (grid_rows = 2, grid_cols = 2):
    ///
    /// ```text
    ///                     2^N columns
    ///         <--------------------------------->
    ///       +-------------------+-------------------+  ^
    ///       |                   |                   |  |
    ///       |      Block 0      |      Block 1      |  |
    ///       |      A_{0,0}      |      A_{0,1}      |  |
    ///       |                   |                   |  |
    ///       +-------------------+-------------------+  | 2^N rows
    ///       |                   |                   |  |
    ///       |      Block 2      |      Block 3      |  |
    ///       |      A_{1,0}      |      A_{1,1}      |  |
    ///       |                   |                   |  |
    ///       +-------------------+-------------------+  v
    ///       <------------------>
    ///           2^{N-1} cols
    /// ```
    ///
    /// The returned blocks are flattened in row-major order:
    /// `[A_{0,0}, A_{0,1}, A_{1,0}, A_{1,1}]`.
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

    /// Computes the intermediate matrix products M_l = A_blocks * B_blocks for each CP component
    /// and returns array [M_1,..., M_l].
    ///
    /// Depending on `multithreaded` and block dimensions, this may compute the products
    /// in parallel using Rayon or sequentially.
    fn compute_block_products(
        &self,
        a_blocks: &[MatRef<'_, f64>],
        b_blocks: &[MatRef<'_, f64>],
        m_block: usize,
        n_block: usize,
        p_block: usize,
        mode: ParallelismMode,
    ) -> Vec<Mat<f64>> {
        const PARALLEL_CUTOFF: usize = 256;

        match mode {
            ParallelismMode::Dfs => {
                // DFS: recursive steps are sequential
                (0..self.cp.rank)
                    .map(|l| self.compute_m_l(l, a_blocks, b_blocks, ParallelismMode::Dfs))
                    .collect()
            }
            ParallelismMode::Bfs => {
                // BFS: recursive steps are parallel
                use rayon::prelude::*;
                (0..self.cp.rank)
                    .into_par_iter()
                    .map(|l| self.compute_m_l(l, a_blocks, b_blocks, ParallelismMode::Bfs))
                    .collect()
            }
            ParallelismMode::Hybrid => {
                // Hybrid: BFS style at top levels, switch to DFS once blocks are small
                if m_block >= PARALLEL_CUTOFF
                    && n_block >= PARALLEL_CUTOFF
                    && p_block >= PARALLEL_CUTOFF
                {
                    use rayon::prelude::*;
                    (0..self.cp.rank)
                        .into_par_iter()
                        .map(|l| self.compute_m_l(l, a_blocks, b_blocks, ParallelismMode::Hybrid))
                        .collect()
                } else {
                    (0..self.cp.rank)
                        .map(|l| self.compute_m_l(l, a_blocks, b_blocks, ParallelismMode::Dfs))
                        .collect()
                }
            }
            ParallelismMode::Sequential => {
                // Sequential: all steps sequential
                (0..self.cp.rank)
                    .map(|l| self.compute_m_l(l, a_blocks, b_blocks, ParallelismMode::Sequential))
                    .collect()
            }
        }
    }

    /// Reconstructs the product matrix C from the computed CP block products.
    ///
    /// The matrix block products `M_l` are weighted by the coefficients in the
    /// W matrix of the CP decomposition and accumulated into the correct block positions
    /// of the final matrix C.
    ///
    /// # Example (Strassen's Algorithm, m = p = 2, rank = 7):
    ///
    /// The final matrix C is partitioned into 4 blocks of size `m_block` x `p_block`:
    /// - `C_{0,0}` (flat index 0)
    /// - `C_{0,1}` (flat index 1)
    /// - `C_{1,0}` (flat index 2)
    /// - `C_{1,1}` (flat index 3)
    ///
    /// For Strassen's algorithm, the output blocks are reconstructed using the 7 products $M_1 \dots M_7$:
    /// - `C_{0,0} = M_1 + M_4 - M_5 + M_7`
    /// - `C_{0,1} = M_3 + M_5`
    /// - `C_{1,0} = M_2 + M_4`
    /// - `C_{1,1} = M_1 - M_2 + M_3 + M_6`
    ///
    /// In this function:
    /// - For block `C_{0,1}` (flat index 1), the coefficients `self.cp.w[(1, l)]` are:
    ///   `1.0` for $l$ corresponding to $M_3$ and $M_5$, and `0.0` otherwise.
    /// - The loops iterate over each product index `l` (from 0 to 6), retrieve the coefficient,
    ///   and add the weighted product `coeff * M_l` to the slice of the target matrix `C`
    ///   corresponding to the `C_{i,j}` block.
    fn reconstruct_from_products(
        &self,
        m: usize,
        p: usize,
        m_block: usize,
        p_block: usize,
        m_products: &[Mat<f64>],
    ) -> Mat<f64> {
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
                        block += faer::Scale(coeff) * m_prod;
                    }
                }
            }
        }
        c
    }

    /// Internal implementation of the recursive matrix multiplication algorithm.
    ///
    /// Depending on dimensions and the parallelism mode, this may perform base matrix multiplication,
    /// standard CP-decomposition multiplication, dynamic peeling, or recursive splitting of blocks.
    pub(crate) fn cp_matmul_impl(
        &self,
        a: &Mat<f64>,
        b: &Mat<f64>,
        mode: ParallelismMode,
    ) -> Mat<f64> {
        let m = a.nrows();
        let n = a.ncols();
        let n_b = b.nrows();
        let p = b.ncols();
        assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

        if m < self.cp.m || n < self.cp.n || p < self.cp.p || n <= 128 || m <= 128 || p <= 128 {
            let leaf_multithreaded = matches!(mode, ParallelismMode::Dfs | ParallelismMode::Hybrid);
            return self.base_matmul(a, b, leaf_multithreaded);
        }

        if m == self.cp.m && n == self.cp.n && p == self.cp.p {
            return self.matmul_cp(a, b);
        }

        let extra_m = m % self.cp.m;
        let extra_n = n % self.cp.n;
        let extra_p = p % self.cp.p;

        if extra_m > 0 || extra_n > 0 || extra_p > 0 {
            return self.dynamic_peeling(a, b, mode);
        }

        let m_block = m / self.cp.m;
        let n_block = n / self.cp.n;
        let p_block = p / self.cp.p;

        let a_blocks = Self::split_into_blocks(a, self.cp.m, self.cp.n, m_block, n_block);
        let b_blocks = Self::split_into_blocks(b, self.cp.n, self.cp.p, n_block, p_block);

        let m_products =
            self.compute_block_products(&a_blocks, &b_blocks, m_block, n_block, p_block, mode);

        self.reconstruct_from_products(m, p, m_block, p_block, &m_products)
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively (single-threaded).
    pub fn cp_matmul_single_thread(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        self.cp_matmul_impl(a, b, ParallelismMode::Sequential)
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively with the specified parallel task switching mode.
    pub fn cp_matmul(&self, a: &Mat<f64>, b: &Mat<f64>, mode: ParallelismMode) -> Mat<f64> {
        self.cp_matmul_impl(a, b, mode)
    }
}
