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
    const double *b,
    double *c
) {
    // Computes C = 1.0 * A * B + 0.0 * C
    // Since we use standard Row-Major layout for matrices:
    // A: m x k, leading dimension lda = k
    // B: k x n, leading dimension ldb = n
    // C: m x n, leading dimension ldc = n
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, a, k,
                b, n,
                0.0, c, n);
}

void mkl_set_num_threads_wrapper(int32_t nt) {
    mkl_set_num_threads(nt);
}

int32_t mkl_get_max_threads_wrapper(void) {
    return mkl_get_max_threads();
}
