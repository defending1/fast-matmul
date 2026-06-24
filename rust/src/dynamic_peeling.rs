use crate::matmul::MatMul;
use faer::Mat;

/// Handles a single dynamic-peeling step of the multiplication C = A × B.
///
/// The operands `a` (m × n) and `b` (n × p) are split into a CP-divisible core and
/// up to three boundary regions (inner strip, right columns, bottom rows). The core is
/// computed via the fast CP algorithm; the boundary regions fall back to standard GEMM.
///
/// # Matrix Partitioning Diagram
///
/// Following the Dynamic Peeling formulation, input matrices `A` and `B` are partitioned as:
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
pub(crate) struct DynamicPeeling<'a> {
    matmul: &'a MatMul<'a>,
    a: &'a Mat<f64>,
    b: &'a Mat<f64>,
    multithreaded: bool,
}

impl<'a> DynamicPeeling<'a> {
    /// Creates a new `DynamicPeeling` helper instance to multiply two matrices.
    pub(crate) fn new(
        matmul: &'a MatMul<'a>,
        a: &'a Mat<f64>,
        b: &'a Mat<f64>,
        multithreaded: bool,
    ) -> Self {
        Self {
            matmul,
            a,
            b,
            multithreaded,
        }
    }

    /// Orchestrates the four peeling sub-steps and returns the completed product C.
    pub(crate) fn run(&self) -> Mat<f64> {
        let mut c = Mat::<f64>::zeros(self.a.nrows(), self.b.ncols());
        self.peel_core(&mut c);
        self.correct_inner_dimension(&mut c);
        self.correct_right_columns(&mut c);
        self.correct_bottom_rows(&mut c);
        c
    }

    /// Computes the CP recursive product for the largest CP-divisible core block.
    fn peel_core(&self, c: &mut Mat<f64>) {
        let cp = self.matmul.cp();
        let core_m = self.a.nrows() - self.a.nrows() % cp.m;
        let core_n = self.a.ncols() - self.a.ncols() % cp.n;
        let core_p = self.b.ncols() - self.b.ncols() % cp.p;

        if core_m > 0 && core_n > 0 && core_p > 0 {
            let a_core = self.a.as_ref().get(0..core_m, 0..core_n).to_owned();
            let b_core = self.b.as_ref().get(0..core_n, 0..core_p).to_owned();
            let c_core = self
                .matmul
                .cp_matmul_impl(&a_core, &b_core, self.multithreaded);
            c.as_mut().get_mut(0..core_m, 0..core_p).copy_from(&c_core);
        }
    }

    /// Adds the GEMM correction for the peeled inner-dimension (column-row) strip.
    fn correct_inner_dimension(&self, c: &mut Mat<f64>) {
        let cp = self.matmul.cp();
        let core_m = self.a.nrows() - self.a.nrows() % cp.m;
        let core_n = self.a.ncols() - self.a.ncols() % cp.n;
        let core_p = self.b.ncols() - self.b.ncols() % cp.p;
        let extra_n = self.a.ncols() % cp.n;
        let n = self.a.ncols();

        if extra_n > 0 && core_m > 0 && core_p > 0 {
            let a_extra = self.a.as_ref().get(0..core_m, core_n..n).to_owned();
            let b_extra = self.b.as_ref().get(core_n..n, 0..core_p).to_owned();
            let correction = self
                .matmul
                .base_matmul(&a_extra, &b_extra, self.multithreaded);
            let mut c_block = c.as_mut().get_mut(0..core_m, 0..core_p);
            for j in 0..core_p {
                for i in 0..core_m {
                    c_block[(i, j)] += correction[(i, j)];
                }
            }
        }
    }

    /// Fills the peeled far-right columns via standard GEMM.
    fn correct_right_columns(&self, c: &mut Mat<f64>) {
        let n = self.a.ncols();
        let p = self.b.ncols();
        let m = self.a.nrows();
        let extra_p = p % self.matmul.cp().p;
        let core_p = p - extra_p;

        if extra_p > 0 {
            let b_extra = self.b.as_ref().get(0..n, core_p..p).to_owned();
            let correction = self
                .matmul
                .base_matmul(self.a, &b_extra, self.multithreaded);
            c.as_mut().get_mut(0..m, core_p..p).copy_from(&correction);
        }
    }

    /// Fills the peeled bottom rows (excluding the rightmost columns) via standard GEMM.
    fn correct_bottom_rows(&self, c: &mut Mat<f64>) {
        let n = self.a.ncols();
        let m = self.a.nrows();
        let extra_m = m % self.matmul.cp().m;
        let core_m = m - extra_m;
        let extra_p = self.b.ncols() % self.matmul.cp().p;
        let core_p = self.b.ncols() - extra_p;

        if extra_m > 0 && core_p > 0 {
            let a_extra = self.a.as_ref().get(core_m..m, 0..n).to_owned();
            let b_extra = self.b.as_ref().get(0..n, 0..core_p).to_owned();
            let correction = self
                .matmul
                .base_matmul(&a_extra, &b_extra, self.multithreaded);
            c.as_mut()
                .get_mut(core_m..m, 0..core_p)
                .copy_from(&correction);
        }
    }
}
