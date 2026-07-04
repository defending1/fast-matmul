// MKL Dense GEMM FFI wrapper for matrix multiplication benchmarking.
#include <mkl.h>
#include <stdint.h>

_Static_assert(sizeof(MKL_INT) == sizeof(int32_t),
               "MKL must use LP64 linking (32-bit MKL_INT). Link with mkl_intel_lp64, not ilp64.");

void mkl_dgemm_wrapper(
    int32_t m,
    int32_t n,
    int32_t k,
    double alpha,
    const double *a,
    int32_t lda,
    const double *b,
    int32_t ldb,
    double beta,
    double *c,
    int32_t ldc
) {
    // Computes C = alpha * A * B + beta * C using column-major layout.
    // This allows direct FFI calls on faer::Mat layout without copies.
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha, a, lda,
                b, ldb,
                beta, c, ldc);
}

void mkl_set_num_threads_wrapper(int32_t nt) {
    mkl_set_num_threads(nt);
}

int32_t mkl_get_max_threads_wrapper(void) {
    return mkl_get_max_threads();
}
