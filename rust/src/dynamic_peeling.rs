use crate::matmul::{BaseMatMul, MatMul, ParallelismMode};
use faer::{Mat, MatRef};

/// Private helper to handle a single dynamic-peeling step of the multiplication C = A × B.
///
/// The operands `a` (m × n) and `b` (n × p) are split into a CP-divisible core and
/// up to three boundary regions (inner strip, right columns, bottom rows). The core is
/// computed via the fast CP algorithm; the boundary regions fall back to standard GEMM.
///
///
/// For example, in Strassen algorithm, input matrices `A` and `B` are partitioned as:
///
/// ```text
///            <--- n-1 --->  < 1 >                  <--- p-1 --->  < 1 >
///         +---------------+-------+  ^          +---------------+-------+  ^
///         |               |       |  |          |               |       |  |
///         |    A_{1,1}    |a_{1,2}|  | m-1      |    B_{1,1}    |b_{1,2}|  | n-1
///         |               |       |  |          |               |       |  |
///       A=|---------------+-------|  v        B=|---------------+-------|  v
///         |    a_{2,1}    |a_{2,2}|  ^ 1        |    b_{2,1}    |b_{2,2}|  ^ 1
///         +---------------+-------+  v          +---------------+-------+  v
/// ```
///
/// Where:
/// - `A_{1,1}` and `B_{1,1}` are the core matrices (divisible by the CP factors).
/// - `a_{1,2}` (column vector) and `a_{2,1}` (row vector) are peeled boundaries of `A`.
/// - `b_{1,2}` (column vector) and `b_{2,1}` (row vector) are peeled boundaries of `B`.
/// - `a_{2,2}` and `b_{2,2}` are scalar corner elements.
///
/// # Product Reconstruction
///
/// The product matrix `C = A × B` is computed block-by-block using the core CP product
/// and boundary GEMM corrections:
///
/// ```text
///                      <---------- p-1 ---------->  <------ 1 ------>
///                   +-----------------------------+-------------------+  ^
///                   |                             |                   |  |
///                   |  A_{1,1}B_{1,1} + a_{1,2}   |  A_{1,1}b_{1,2} + |  | m-1
///                   |           b_{2,1}           |   a_{1,2}b_{2,2}  |  |
///                 C=|-----------------------------+-------------------|  v
///                   |  a_{2,1}B_{1,1} + a_{2,2}   |  a_{2,1}b_{1,2} + |  ^ 1
///                   |           b_{2,1}           |   a_{2,2}b_{2,2}  |  |
///                   +-----------------------------+-------------------+  v
/// ```
///
/// Here, only `A_{1,1}B_{1,1}` is executed recursively using the fast CP algorithm (e.g. Strassen).
/// The remaining terms represent low-rank corrections computed using standard GEMM.
pub(crate) struct DynamicPeeling<'a, 'b> {
    matmul: &'a MatMul<'a>,
    a: MatRef<'b, f64>,
    b: MatRef<'b, f64>,
    mode: ParallelismMode,
    base_choice: BaseMatMul,
    pub(crate) core_m: usize,
    pub(crate) core_n: usize,
    pub(crate) core_p: usize,
    pub(crate) extra_m: usize,
    pub(crate) extra_n: usize,
    pub(crate) extra_p: usize,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) multithreaded: bool,
}

impl<'a, 'b> DynamicPeeling<'a, 'b> {
    /// Creates a new `DynamicPeeling` helper instance to multiply two matrices.
    ///
    /// # Arguments
    /// * `matmul` - Reference to the orchestrating `MatMul` instance.
    /// * `a` - The left matrix operand as a `MatRef`.
    /// * `b` - The right matrix operand as a `MatRef`.
    /// * `mode` - The parallelism mode to use.
    /// * `base_choice` - The backend choice (Faer or Dgemm).
    pub(crate) fn new(
        matmul: &'a MatMul<'a>,
        a: MatRef<'b, f64>,
        b: MatRef<'b, f64>,
        mode: ParallelismMode,
        base_choice: BaseMatMul,
    ) -> Self {
        let cp = matmul.cp();
        let m = a.nrows();
        let n = a.ncols();
        let p = b.ncols();

        let extra_m = m % cp.m;
        let extra_n = n % cp.n;
        let extra_p = p % cp.p;

        let core_m = m - extra_m;
        let core_n = n - extra_n;
        let core_p = p - extra_p;

        let multithreaded = matches!(mode, ParallelismMode::Dfs | ParallelismMode::Hybrid);

        Self {
            matmul,
            a,
            b,
            mode,
            base_choice,
            core_m,
            core_n,
            core_p,
            extra_m,
            extra_n,
            extra_p,
            m,
            n,
            p,
            multithreaded,
        }
    }

    /// Computes the CP recursive product for the largest CP-divisible core block.
    ///
    /// # Arguments
    /// * `c` - The mutable destination matrix `C` where results are written.
    pub(crate) fn peel_core(&self, c: &mut Mat<f64>) {
        if self.core_m > 0 && self.core_n > 0 && self.core_p > 0 {
            let a_core = self.a.get(0..self.core_m, 0..self.core_n);
            let b_core = self.b.get(0..self.core_n, 0..self.core_p);
            let c_core = self
                .matmul
                .cp_matmul_impl(a_core, b_core, self.mode, self.base_choice);
            c.as_mut().get_mut(0..self.core_m, 0..self.core_p).copy_from(&c_core);
        }
    }

    /// Adds the GEMM correction for the peeled inner-dimension (column-row) strip.
    ///
    /// # Arguments
    /// * `c` - The mutable destination matrix `C` where results are accumulated.
    pub(crate) fn correct_inner_dimension(&self, c: &mut Mat<f64>) {
        if self.extra_n > 0 && self.core_m > 0 && self.core_p > 0 {
            let a_extra = self.a.get(0..self.core_m, self.core_n..self.n);
            let b_extra = self.b.get(self.core_n..self.n, 0..self.core_p);
            let correction =
                self.matmul
                    .base_matmul_impl(a_extra, b_extra, self.multithreaded, self.base_choice);
            let mut c_block = c.as_mut().get_mut(0..self.core_m, 0..self.core_p);
            c_block += &correction;
        }
    }

    /// Fills the peeled far-right columns via standard GEMM.
    ///
    /// # Arguments
    /// * `c` - The mutable destination matrix `C` where results are copied.
    pub(crate) fn correct_right_columns(&self, c: &mut Mat<f64>) {
        if self.extra_p > 0 {
            let b_extra = self.b.get(0..self.n, self.core_p..self.p);
            let correction =
                self.matmul
                    .base_matmul_impl(self.a, b_extra, self.multithreaded, self.base_choice);
            c.as_mut().get_mut(0..self.m, self.core_p..self.p).copy_from(&correction);
        }
    }

    /// Fills the peeled bottom rows (excluding the rightmost columns) via standard GEMM.
    ///
    /// # Arguments
    /// * `c` - The mutable destination matrix `C` where results are copied.
    pub(crate) fn correct_bottom_rows(&self, c: &mut Mat<f64>) {
        if self.extra_m > 0 && self.core_p > 0 {
            let a_extra = self.a.get(self.core_m..self.m, 0..self.n);
            let b_extra = self.b.get(0..self.n, 0..self.core_p);
            let correction =
                self.matmul
                    .base_matmul_impl(a_extra, b_extra, self.multithreaded, self.base_choice);
            c.as_mut()
                .get_mut(self.core_m..self.m, 0..self.core_p)
                .copy_from(&correction);
        }
    }
}
