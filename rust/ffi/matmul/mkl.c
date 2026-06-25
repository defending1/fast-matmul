// MKL Dense GEMM FFI wrapper for matrix multiplication benchmarking.
#include <mkl.h>
#include <stdint.h>

_Static_assert(sizeof(MKL_INT) == sizeof(int32_t),
               "MKL must use LP64 linking (32-bit MKL_INT). Link with mkl_intel_lp64, not ilp64.");

void mkl_dgemm_wrapper(
    int32_t m,
    int32_t n,
    int32_t k,
    const double *a,
    int32_t lda,
    const double *b,
    int32_t ldb,
    double *c,
    int32_t ldc
) {
    // Computes C = 1.0 * A * B + 0.0 * C using column-major layout.
    // This allows direct FFI calls on faer::Mat layout without copies.
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, a, lda,
                b, ldb,
                0.0, c, ldc);
}

void mkl_set_num_threads_wrapper(int32_t nt) {
    mkl_set_num_threads(nt);
}

int32_t mkl_get_max_threads_wrapper(void) {
    return mkl_get_max_threads();
}
