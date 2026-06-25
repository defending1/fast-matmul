use faer::Mat;

use std::sync::OnceLock;

unsafe extern "C" {
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
    fn mkl_set_num_threads_wrapper(nt: i32);
    fn mkl_get_max_threads_wrapper() -> i32;
}

static MAX_THREADS: OnceLock<i32> = OnceLock::new();

/// Set the number of threads used by MKL dynamically.
/// Setting to 1 runs sequentially; setting to 0 restores all possible threads (caching system default at startup).
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

/// Computes C = A * B using Intel MKL dgemm (FFI).
pub fn mkl_matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let k_b = b.nrows();
    let n = b.ncols();
    assert_eq!(k, k_b, "Matrix dimensions must agree for multiplication");

    // Assert that the layout is standard contiguous column-major (row stride is 1)
    assert_eq!(a.row_stride(), 1, "Matrix A must be column-major");
    assert_eq!(b.row_stride(), 1, "Matrix B must be column-major");

    let mut c = Mat::zeros(m, n);

    unsafe {
        mkl_dgemm_wrapper(
            m as i32,
            n as i32,
            k as i32,
            a.as_ptr(),
            a.col_stride() as i32,
            b.as_ptr(),
            b.col_stride() as i32,
            c.as_ptr_mut(),
            c.col_stride() as i32,
        );
    }

    c
}
