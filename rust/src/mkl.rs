use faer::Mat;

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

/// Computes C = A * B using Intel MKL dgemm (FFI).
pub fn mkl_matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
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
