use faer::Mat;
use fast_matmul::matmul::BaseMatMul;

/// Run the baseline sequential or parallel matrix multiplication,
/// selectively using `faer` or Intel MKL `dgemm` based on the specified choice.
pub fn base_matmul(
    a: &Mat<f64>,
    b: &Mat<f64>,
    multithreaded: bool,
    base_choice: BaseMatMul,
) -> Mat<f64> {
    match base_choice {
        BaseMatMul::Faer => {
            let mut c = Mat::zeros(a.nrows(), b.ncols());
            let par = if multithreaded {
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
        BaseMatMul::Dgemm => {
            // Adjust thread count for MKL dynamically based on concurrency requirements.
            // Single-threaded GEMM runs sequentially; multithreaded GEMM uses all available cores.
            fast_matmul::mkl::mkl_set_threads(if multithreaded { 0 } else { 1 });
            fast_matmul::mkl::mkl_matmul(a, b)
        }
    }
}
