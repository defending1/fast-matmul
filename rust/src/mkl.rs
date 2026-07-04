use faer::Mat;

use std::sync::OnceLock;

unsafe extern "C" {
    /// Wrapper for the Intel MKL dgemm function.
    fn mkl_dgemm_wrapper(
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
    /// Wrapper to set the number of threads for MKL.
    fn mkl_set_num_threads_wrapper(nt: i32);
    /// Wrapper to get the max thread count for MKL.
    fn mkl_get_max_threads_wrapper() -> i32;
}

/// A static variable to cache the maximum number of threads allowed by MKL.
static MAX_THREADS: OnceLock<i32> = OnceLock::new();

/// Set the number of threads used by MKL dynamically.
/// Setting to 1 runs sequentially; setting to 0 restores all possible threads (caching system default at startup).
///
/// # Arguments
/// * `num_threads` - The target number of threads. Setting <= 0 restores to the maximum available threads.
pub fn mkl_set_threads(num_threads: i32) {
    // Cache maximum threads at initialization before any adjustments are made
    let max = *MAX_THREADS.get_or_init(|| unsafe { mkl_get_max_threads_wrapper() });
    unsafe {
        if num_threads <= 0 {
            mkl_set_num_threads_wrapper(max);
        } else {
            mkl_set_num_threads_wrapper(num_threads);
        }
    }
}

/// Computes C = A * B using Intel MKL dgemm (FFI) with `MatRef` inputs.
/// Writes the result directly into `dst` according to the accumulation strategy `accum`.
///
/// # Arguments
/// * `dst` - The destination matrix slice as `MatMut`.
/// * `accum` - The accumulation strategy (Add or Replace).
/// * `a` - The left matrix operand as a `MatRef`.
/// * `b` - The right matrix operand as a `MatRef`.
pub(crate) fn mkl_matmul_impl(
    mut dst: faer::MatMut<'_, f64>,
    accum: faer::Accum,
    a: faer::MatRef<'_, f64>,
    b: faer::MatRef<'_, f64>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let k_b = b.nrows();
    let n = b.ncols();
    assert_eq!(k, k_b, "Matrix dimensions must agree for multiplication");

    // If any dimension is 0, nothing to do (if Replace, fill with zeros).
    if m == 0 || n == 0 || k == 0 {
        if matches!(accum, faer::Accum::Replace) {
            dst.fill(0.0);
        }
        return;
    }

    // Assert that the layout is standard contiguous column-major (row stride is 1)
    assert_eq!(a.row_stride(), 1, "Matrix A must be column-major");
    assert_eq!(b.row_stride(), 1, "Matrix B must be column-major");
    assert_eq!(dst.row_stride(), 1, "Matrix C must be column-major");

    let beta = match accum {
        faer::Accum::Replace => 0.0,
        faer::Accum::Add => 1.0,
    };

    unsafe {
        mkl_dgemm_wrapper(
            m.try_into().expect("m fits in i32"),
            n.try_into().expect("n fits in i32"),
            k.try_into().expect("k fits in i32"),
            1.0, // alpha
            a.as_ptr(),
            a.col_stride().try_into().expect("lda fits in i32"),
            b.as_ptr(),
            b.col_stride().try_into().expect("ldb fits in i32"),
            beta,
            dst.as_ptr_mut(),
            dst.col_stride().try_into().expect("ldc fits in i32"),
        );
    }
}

/// Computes C = A * B using Intel MKL dgemm (FFI).
/// Returns an empty or zero matrix early if any dimension is 0 to avoid invalid calls to MKL.
///
/// # Arguments
/// * `a` - The left matrix operand as a reference to `Mat`.
/// * `b` - The right matrix operand as a reference to `Mat`.
pub fn mkl_matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut c = Mat::zeros(a.nrows(), b.ncols());
    mkl_matmul_impl(c.as_mut(), faer::Accum::Replace, a.as_ref(), b.as_ref());
    c
}
