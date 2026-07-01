use faer::Mat;

use std::sync::OnceLock;

unsafe extern "C" {
    /// Wrapper for the Intel MKL dgemm function.
    fn mkl_dgemm_wrapper(
        m: i32,
        n: i32,
        k: i32,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
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
/// Returns an empty or zero matrix early if any dimension is 0 to avoid invalid calls to MKL.
///
/// # Arguments
/// * `a` - The left matrix operand as a `MatRef`.
/// * `b` - The right matrix operand as a `MatRef`.
pub(crate) fn mkl_matmul_impl(a: faer::MatRef<'_, f64>, b: faer::MatRef<'_, f64>) -> Mat<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let k_b = b.nrows();
    let n = b.ncols();
    assert_eq!(k, k_b, "Matrix dimensions must agree for multiplication");

    // If any dimension is 0, the result is a zero/empty matrix. Return early to avoid MKL internal checks failing.
    if m == 0 || n == 0 || k == 0 {
        return Mat::zeros(m, n);
    }

    // Assert that the layout is standard contiguous column-major (row stride is 1)
    assert_eq!(a.row_stride(), 1, "Matrix A must be column-major");
    assert_eq!(b.row_stride(), 1, "Matrix B must be column-major");

    let mut c = Mat::zeros(m, n);

    unsafe {
        mkl_dgemm_wrapper(
            m.try_into().expect("m fits in i32"),
            n.try_into().expect("n fits in i32"),
            k.try_into().expect("k fits in i32"),
            a.as_ptr(),
            a.col_stride().try_into().expect("lda fits in i32"),
            b.as_ptr(),
            b.col_stride().try_into().expect("ldb fits in i32"),
            c.as_ptr_mut(),
            c.col_stride().try_into().expect("ldc fits in i32"),
        );
    }

    c
}

/// Computes C = A * B using Intel MKL dgemm (FFI).
/// Returns an empty or zero matrix early if any dimension is 0 to avoid invalid calls to MKL.
///
/// # Arguments
/// * `a` - The left matrix operand as a reference to `Mat`.
/// * `b` - The right matrix operand as a reference to `Mat`.
pub fn mkl_matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    mkl_matmul_impl(a.as_ref(), b.as_ref())
}

