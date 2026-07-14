use crate::cp::CP;
use crate::dynamic_peeling::DynamicPeeling;
use faer::{ColRef, Mat, MatRef, Scale};
use std::ops::{Index, IndexMut};

pub use crate::parallelism_mode::{BaseMatMul, ParallelismMode};

/// Configures Rayon's global thread pool to match standard concurrency environment variables
/// (RAYON_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS, or SLURM_CPUS_PER_TASK).
/// This prevents oversubscription on multi-socket high-core scientific cluster nodes.
pub fn init_rayon_threads() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| {
                std::env::var("OMP_NUM_THREADS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .or_else(|| {
                std::env::var("MKL_NUM_THREADS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .or_else(|| {
                std::env::var("SLURM_CPUS_PER_TASK")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            });
        if let Some(t) = threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build_global();
        }
    });
}

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
    ///
    /// # Arguments
    /// * `shape` - The shape (depth, rows, cols) of the new tensor.
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

/// Controls the recursion behavior of the CP decomposition matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursionLimit {
    /// Recurse up to a fixed number of levels, without limiting the size of the matrix.
    ///
    /// When the depth reaches `0`, the algorithm stops recursing and invokes the base matrix multiplication.
    Depth(usize),
    /// Recurse until any matrix dimension (m, n, or p) is less than or equal to the specified cutoff size.
    ///
    /// The algorithm recurses potentially to the last recursion child (no level limit), as long as
    /// the matrix dimensions are strictly greater than the cutoff size.
    Cutoff(usize),
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
        init_rayon_threads();
        MatMul {
            cp: CP::get_strassen(),
        }
    }

    /// Creates a new `MatMul` operator instance with a custom CP decomposition.
    ///
    /// # Arguments
    /// * `cp` - Reference to the custom CP decomposition.
    pub fn with_cp(cp: &'a CP) -> Self {
        init_rayon_threads();
        Self { cp }
    }

    /// Returns a reference to the underlying CP decomposition.
    pub(crate) fn cp(&self) -> &CP {
        self.cp
    }

    /// Computes the dot product of a column vector and a matrix flattened in row-major order,
    /// without performing any memory allocations.
    ///
    /// # Arguments
    /// * `vec` - A column vector.
    /// * `mat` - A matrix slice as `MatRef`.
    fn dot_product_flattened(vec: ColRef<'_, f64>, mat: MatRef<'_, f64>) -> f64 {
        assert_eq!(vec.nrows(), mat.nrows() * mat.ncols());
        let mut sum = 0.0;
        let mut i = 0;
        for r in 0..mat.nrows() {
            for c in 0..mat.ncols() {
                sum += vec[i] * mat[(r, c)];
                i += 1;
            }
        }
        sum
    }

    /// Computes C = A * B using the CP decomposition formula.
    /// Internal implementation using `MatRef` to avoid allocations.
    ///
    /// # Arguments
    /// * `dst` - The destination matrix slice as `MatMut`.
    /// * `a` - The left matrix operand as a `MatRef`.
    /// * `b` - The right matrix operand as a `MatRef`.
    fn matmul_cp_impl(
        &self,
        mut dst: faer::MatMut<'_, f64>,
        a: MatRef<'_, f64>,
        b: MatRef<'_, f64>,
    ) {
        assert_eq!((a.nrows(), a.ncols()), (self.cp.m, self.cp.n));
        assert_eq!((b.nrows(), b.ncols()), (self.cp.n, self.cp.p));

        dst.fill(0.0);

        for l in 0..self.cp.rank {
            let s_l = Self::dot_product_flattened(self.cp.u.col(l), a);
            let t_l = Self::dot_product_flattened(self.cp.v.col(l), b);

            let m_l = s_l * t_l;

            let w_col = self.cp.w.col(l);
            let p_dim = self.cp.p;
            for i in 0..dst.nrows() * dst.ncols() {
                let r = i / p_dim;
                let c = i % p_dim;
                dst[(r, c)] += m_l * w_col[i];
            }
        }
    }

    /// Computes C = A * B using the CP decomposition formula:
    /// vec(C^T) = sum_{l=1}^r (u_l^T vec(A)) * (v_l^T vec(B)) * w_l
    /// assuming row-major layout vectorization for A, B, C.
    ///
    /// # Arguments
    /// * `a` - The left matrix operand.
    /// * `b` - The right matrix operand.
    pub fn matmul_cp(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        let mut c = Mat::zeros(self.cp.m, self.cp.p);
        self.matmul_cp_impl(c.as_mut(), a.as_ref(), b.as_ref());
        c
    }

    /// Computes classical matrix multiplication C = A * B using `MatRef` inputs.
    /// Writes the result directly into `dst` according to the accumulation strategy `accum`.
    ///
    /// # Arguments
    /// * `dst` - The destination matrix slice as `MatMut`.
    /// * `accum` - The accumulation strategy (Add or Replace).
    /// * `a` - The left matrix operand as a `MatRef`.
    /// * `b` - The right matrix operand as a `MatRef`.
    /// * `multithreaded` - Whether to use multithreading.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    pub(crate) fn base_matmul_impl(
        &self,
        dst: faer::MatMut<'_, f64>,
        accum: faer::Accum,
        a: MatRef<'_, f64>,
        b: MatRef<'_, f64>,
        multithreaded: bool,
        base_choice: BaseMatMul,
    ) {
        init_rayon_threads();
        match base_choice {
            BaseMatMul::Faer => {
                let par =
                    if multithreaded && a.nrows() >= 256 && a.ncols() >= 256 && b.ncols() >= 256 {
                        faer::get_global_parallelism()
                    } else {
                        faer::Par::Seq
                    };
                faer::linalg::matmul::matmul(dst, accum, a, b, 1.0, par);
            }
            BaseMatMul::Dgemm => {
                crate::mkl::mkl_set_threads(if multithreaded { 0 } else { 1 });
                crate::mkl::mkl_matmul_impl(dst, accum, a, b);
            }
        }
    }

    /// Computes classical matrix multiplication C = A * B using the underlying library (faer or MKL/dgemm),
    /// with optional multithreading.
    ///
    /// # Arguments
    /// * `a` - The left matrix operand as a reference to `Mat`.
    /// * `b` - The right matrix operand as a reference to `Mat`.
    /// * `multithreaded` - Whether to use multithreading.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    pub fn base_matmul(
        &self,
        a: &Mat<f64>,
        b: &Mat<f64>,
        multithreaded: bool,
        base_choice: BaseMatMul,
    ) -> Mat<f64> {
        let mut c = Mat::zeros(a.nrows(), b.ncols());
        self.base_matmul_impl(
            c.as_mut(),
            faer::Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            multithreaded,
            base_choice,
        );
        c
    }

    /// Calculate the total number of recursion levels for the given input matrix shape.
    /// Performs one step of dynamic peeling in the multiplication C = A * B.
    ///
    /// Delegates to [`DynamicPeeling`], which handles the core CP product and
    /// the GEMM-based boundary corrections for odd or non-power-of-two dimensions.
    ///
    /// # Arguments
    /// * `a` - The left matrix operand as a `MatRef`.
    /// * `b` - The right matrix operand as a `MatRef`.
    /// * `mode` - The parallelism mode to use.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    fn dynamic_peeling(
        &self,
        mut dst: faer::MatMut<'_, f64>,
        a: MatRef<'_, f64>,
        b: MatRef<'_, f64>,
        mode: ParallelismMode,
        base_choice: BaseMatMul,
        recursion_limit: RecursionLimit,
    ) {
        let peeling = DynamicPeeling::new(self, a, b, mode, base_choice, recursion_limit);
        peeling.peel_core(dst.as_mut());
        peeling.correct_inner_dimension(dst.as_mut());
        peeling.correct_right_columns(dst.as_mut());
        peeling.correct_bottom_rows(dst.as_mut());
    }

    /// Helper to compute a single Strassen product M_l.
    ///
    /// # Arguments
    /// * `l` - The rank index.
    /// * `a_blocks` - The blocks of matrix A.
    /// * `b_blocks` - The blocks of matrix B.
    /// * `mode` - The parallelism mode.
    /// * `base_choice` - The backend choice.
    /// * `recursion_limit` - The recursion limit choice.
    fn compute_m_l(
        &self,
        l: usize,
        a_blocks: &[MatRef<'_, f64>],
        b_blocks: &[MatRef<'_, f64>],
        mode: ParallelismMode,
        base_choice: BaseMatMul,
        recursion_limit: RecursionLimit,
    ) -> Mat<f64> {
        let mut a_comb = Mat::<f64>::zeros(a_blocks[0].nrows(), a_blocks[0].ncols());
        let mut b_comb = Mat::<f64>::zeros(b_blocks[0].nrows(), b_blocks[0].ncols());
        // S_r in report
        Self::combine_blocks_into(a_comb.as_mut(), a_blocks, self.cp.u.col(l));
        // T_r in report
        Self::combine_blocks_into(b_comb.as_mut(), b_blocks, self.cp.v.col(l));

        let mut m_prod = Mat::<f64>::zeros(a_comb.nrows(), b_comb.ncols());
        self.cp_matmul_impl(
            m_prod.as_mut(),
            a_comb.as_ref(),
            b_comb.as_ref(),
            mode,
            base_choice,
            recursion_limit,
        );
        m_prod
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
    ///
    /// # Arguments
    /// * `dst` - The destination matrix slice where the combined blocks are written.
    /// * `blocks` - Slice of matrix block views.
    /// * `coeffs` - Coefficients vector.
    fn combine_blocks_into(
        mut dst: faer::MatMut<'_, f64>,
        blocks: &[MatRef<'_, f64>],
        coeffs: faer::ColRef<'_, f64>,
    ) {
        dst.fill(0.0);
        for (block, &coeff) in blocks.iter().zip(coeffs.iter()) {
            if coeff == 1.0 {
                dst += block;
            } else if coeff == -1.0 {
                dst -= block;
            } else if coeff != 0.0 {
                dst += Scale(coeff) * block;
            }
        }
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
    ///
    /// # Arguments
    /// * `matrix` - The matrix to split as a `MatRef`.
    /// * `grid_rows` - Number of rows in the block grid.
    /// * `grid_cols` - Number of columns in the block grid.
    /// * `block_rows` - Number of rows in each block.
    /// * `block_cols` - Number of columns in each block.
    fn vec_blocks(
        matrix: MatRef<'_, f64>,
        grid_rows: usize,
        grid_cols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Vec<MatRef<'_, f64>> {
        let mut blocks = Vec::with_capacity(grid_rows * grid_cols);
        for i in 0..grid_rows {
            for j in 0..grid_cols {
                let r_range = i * block_rows..(i + 1) * block_rows;
                let c_range = j * block_cols..(j + 1) * block_cols;
                let block = matrix.get(r_range, c_range);
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
    ///
    /// # Arguments
    /// * `a_blocks` - Slice of matrix block views for A.
    /// * `b_blocks` - Slice of matrix block views for B.
    /// * `m_block` - Number of rows in each block of A.
    /// * `n_block` - Number of columns in each block of A.
    /// * `p_block` - Number of columns in each block of B.
    /// * `mode` - The parallelism mode to use.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    #[allow(clippy::too_many_arguments)]
    fn compute_block_products(
        &self,
        a_blocks: &[MatRef<'_, f64>],
        b_blocks: &[MatRef<'_, f64>],
        _m_block: usize,
        _n_block: usize,
        _p_block: usize,
        mode: ParallelismMode,
        base_choice: BaseMatMul,
        recursion_limit: RecursionLimit,
    ) -> Vec<Mat<f64>> {
        match mode {
            ParallelismMode::Dfs => (0..self.cp.rank)
                .map(|l| {
                    self.compute_m_l(
                        l,
                        a_blocks,
                        b_blocks,
                        ParallelismMode::Dfs,
                        base_choice,
                        recursion_limit,
                    )
                })
                .collect(),
            ParallelismMode::Bfs => {
                use rayon::prelude::*;
                (0..self.cp.rank)
                    .into_par_iter()
                    .map(|l| {
                        self.compute_m_l(
                            l,
                            a_blocks,
                            b_blocks,
                            ParallelismMode::Bfs,
                            base_choice,
                            recursion_limit,
                        )
                    })
                    .collect()
            }
            ParallelismMode::Hybrid => {
                let level = match recursion_limit {
                    RecursionLimit::Depth(level) => level,
                    RecursionLimit::Cutoff(_) => {
                        panic!(
                            "Hybrid parallelism mode is only supported with RecursionLimit::Depth"
                        );
                    }
                };

                let r = self.cp.rank;
                let p_threads = rayon::current_num_threads().max(1);

                let r_pow_l = r.pow(level as u32);
                let k = r_pow_l - (r_pow_l % p_threads);
                let r_pow_l_minus_1 = r.pow(level.saturating_sub(1) as u32);

                let c = k.checked_div(r_pow_l_minus_1).unwrap_or(0);

                use rayon::prelude::*;
                (0..r)
                    .into_par_iter()
                    .map(|l| {
                        let child_mode = if l < c {
                            ParallelismMode::Bfs
                        } else {
                            ParallelismMode::Dfs
                        };
                        self.compute_m_l(
                            l,
                            a_blocks,
                            b_blocks,
                            child_mode,
                            base_choice,
                            recursion_limit,
                        )
                    })
                    .collect()
            }
            ParallelismMode::Sequential => (0..self.cp.rank)
                .map(|l| {
                    self.compute_m_l(
                        l,
                        a_blocks,
                        b_blocks,
                        ParallelismMode::Sequential,
                        base_choice,
                        recursion_limit,
                    )
                })
                .collect(),
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
    ///
    /// # Arguments
    /// * `m` - The number of rows in C.
    /// * `p` - The number of columns in C.
    /// * `m_block` - The row block size.
    /// * `p_block` - The column block size.
    /// * `m_products` - Slice of computed product blocks as `MatRef`.
    fn c_blocks_from_s_and_t(
        &self,
        mut c: faer::MatMut<'_, f64>,
        m_block: usize,
        p_block: usize,
        m_products: &[MatRef<'_, f64>],
    ) {
        c.fill(0.0);
        for i in 0..self.cp.m {
            for j in 0..self.cp.p {
                let mut block = c.as_mut().get_mut(
                    i * m_block..(i + 1) * m_block,
                    j * p_block..(j + 1) * p_block,
                );
                for (l, m_prod) in m_products.iter().enumerate() {
                    let coeff = self.cp.w[(i * self.cp.p + j, l)];
                    if coeff == 1.0 {
                        block += m_prod;
                    } else if coeff == -1.0 {
                        block -= m_prod;
                    } else if coeff != 0.0 {
                        block += Scale(coeff) * m_prod;
                    }
                }
            }
        }
    }

    /// Internal implementation of the recursive matrix multiplication algorithm.
    ///
    /// Depending on dimensions and the parallelism mode, this may perform base matrix multiplication,
    /// standard CP-decomposition multiplication, dynamic peeling, or recursive splitting of blocks.
    ///
    /// # Arguments
    /// * `dst` - The destination matrix slice as `MatMut`.
    /// * `a` - The left matrix operand as a `MatRef`.
    /// * `b` - The right matrix operand as a `MatRef`.
    /// * `mode` - The parallelism mode to use.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    /// * `recursion_limit` - The recursion limit choice (Depth or Cutoff).
    pub(crate) fn cp_matmul_impl(
        &self,
        dst: faer::MatMut<'_, f64>,
        a: MatRef<'_, f64>,
        b: MatRef<'_, f64>,
        mode: ParallelismMode,
        base_choice: BaseMatMul,
        recursion_limit: RecursionLimit,
    ) {
        if mode == ParallelismMode::Hybrid && matches!(recursion_limit, RecursionLimit::Cutoff(_)) {
            panic!("Hybrid parallelism mode is only supported with RecursionLimit::Depth");
        }

        let m = a.nrows();
        let n = a.ncols();
        let n_b = b.nrows();
        let p = b.ncols();
        assert_eq!(n, n_b, "Matrix dimensions must agree for multiplication");

        let stop_recursing = match recursion_limit {
            RecursionLimit::Depth(depth) => {
                depth == 0 || m < self.cp.m || n < self.cp.n || p < self.cp.p
            }
            RecursionLimit::Cutoff(cutoff) => {
                m < self.cp.m
                    || n < self.cp.n
                    || p < self.cp.p
                    || m <= cutoff
                    || n <= cutoff
                    || p <= cutoff
            }
        };

        if stop_recursing {
            let leaf_multithreaded = match mode {
                ParallelismMode::Dfs => true,
                ParallelismMode::Bfs => false,
                ParallelismMode::Sequential => false,
                ParallelismMode::Hybrid => {
                    // This can only occur if depth = 0 and mode is Hybrid
                    let p_threads = rayon::current_num_threads().max(1);
                    p_threads > 1
                }
            };
            self.base_matmul_impl(
                dst,
                faer::Accum::Replace,
                a,
                b,
                leaf_multithreaded,
                base_choice,
            );
            return;
        }

        if m == self.cp.m && n == self.cp.n && p == self.cp.p {
            self.matmul_cp_impl(dst, a, b);
            return;
        }

        let extra_m = m % self.cp.m;
        let extra_n = n % self.cp.n;
        let extra_p = p % self.cp.p;

        if extra_m > 0 || extra_n > 0 || extra_p > 0 {
            self.dynamic_peeling(dst, a, b, mode, base_choice, recursion_limit);
            return;
        }

        let m_block = m / self.cp.m;
        let n_block = n / self.cp.n;
        let p_block = p / self.cp.p;

        let a_blocks = Self::vec_blocks(a, self.cp.m, self.cp.n, m_block, n_block);
        let b_blocks = Self::vec_blocks(b, self.cp.n, self.cp.p, n_block, p_block);

        let next_limit = match recursion_limit {
            RecursionLimit::Depth(depth) => RecursionLimit::Depth(depth.saturating_sub(1)),
            RecursionLimit::Cutoff(cutoff) => RecursionLimit::Cutoff(cutoff),
        };

        let m_products = self.compute_block_products(
            &a_blocks,
            &b_blocks,
            m_block,
            n_block,
            p_block,
            mode,
            base_choice,
            next_limit,
        );

        let m_refs: Vec<MatRef<'_, f64>> = m_products.iter().map(|m| m.as_ref()).collect();
        self.c_blocks_from_s_and_t(dst, m_block, p_block, &m_refs);
    }

    /// Computes C = A * B using the CP decomposition algorithm recursively with the specified parallel task switching mode and base matrix multiplication choice.
    ///
    /// # Arguments
    /// * `a` - The left matrix operand as a reference to `Mat`.
    /// * `b` - The right matrix operand as a reference to `Mat`.
    /// * `mode` - The parallelism mode to use.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    /// * `recursion_limit` - The recursion limit choice (Depth or Cutoff).
    pub fn cp_matmul(
        &self,
        a: &Mat<f64>,
        b: &Mat<f64>,
        mode: ParallelismMode,
        base_choice: BaseMatMul,
        recursion_limit: RecursionLimit,
    ) -> Mat<f64> {
        let mut c = Mat::zeros(a.nrows(), b.ncols());
        self.cp_matmul_impl(
            c.as_mut(),
            a.as_ref(),
            b.as_ref(),
            mode,
            base_choice,
            recursion_limit,
        );
        c
    }
}
