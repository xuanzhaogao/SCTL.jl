// BLAS/LAPACK for SCTL from Julia's libblastrampoline (built with SCTLJL_BLAS_LBT).
//
// SCTL calls the LP64 Fortran symbols sgemm_, dgemm_, sgesvd_, dgesvd_ (sctl/blas.h, sctl/lapack.h). They are defined
// here (hidden: they only serve this library) and forward to the ILP64 entry points of libblastrampoline, i.e. to the
// BLAS that Julia's LinearAlgebra uses (OpenBLAS by default, or MKL etc. if the user loads one). Julia passes the
// function pointers at load time (sctljl_set_blas). Julia's OpenBLAS supports up to 512 threads calling it at once;
// an LP64 OpenBLAS32_jll supports only 32, which crashes PVFMM (it calls BLAS from every OpenMP thread) on larger
// nodes.
#include "sctljl.hpp"

#ifdef SCTLJL_BLAS_LBT

#include <cstdint>

namespace {
  using I = int64_t;
  // trailing hidden lengths of the character arguments (gfortran ABI; ignored by the C interfaces of OpenBLAS)
  using gemm_d_t = void (*)(const char*, const char*, const I*, const I*, const I*, const double*, const double*,
                            const I*, const double*, const I*, const double*, double*, const I*, size_t, size_t);
  using gemm_s_t = void (*)(const char*, const char*, const I*, const I*, const I*, const float*, const float*,
                            const I*, const float*, const I*, const float*, float*, const I*, size_t, size_t);
  using gesvd_d_t = void (*)(const char*, const char*, const I*, const I*, double*, const I*, double*, double*,
                             const I*, double*, const I*, double*, const I*, I*, size_t, size_t);
  using gesvd_s_t = void (*)(const char*, const char*, const I*, const I*, float*, const I*, float*, float*,
                             const I*, float*, const I*, float*, const I*, I*, size_t, size_t);
  gemm_d_t p_dgemm = nullptr;
  gemm_s_t p_sgemm = nullptr;
  gesvd_d_t p_dgesvd = nullptr;
  gesvd_s_t p_sgesvd = nullptr;
}

SCTLJL_EXPORT void sctljl_set_blas(void* dgemm, void* sgemm, void* dgesvd, void* sgesvd) {
  p_dgemm = reinterpret_cast<gemm_d_t>(dgemm);
  p_sgemm = reinterpret_cast<gemm_s_t>(sgemm);
  p_dgesvd = reinterpret_cast<gesvd_d_t>(dgesvd);
  p_sgesvd = reinterpret_cast<gesvd_s_t>(sgesvd);
}

extern "C" {

void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const double* ALPHA,
            const double* A, const int* LDA, const double* B, const int* LDB, const double* BETA, double* C,
            const int* LDC) {
  const I m = *M, n = *N, k = *K, lda = *LDA, ldb = *LDB, ldc = *LDC;
  p_dgemm(TRANSA, TRANSB, &m, &n, &k, ALPHA, A, &lda, B, &ldb, BETA, C, &ldc, 1, 1);
}

void sgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const float* ALPHA,
            const float* A, const int* LDA, const float* B, const int* LDB, const float* BETA, float* C,
            const int* LDC) {
  const I m = *M, n = *N, k = *K, lda = *LDA, ldb = *LDB, ldc = *LDC;
  p_sgemm(TRANSA, TRANSB, &m, &n, &k, ALPHA, A, &lda, B, &ldb, BETA, C, &ldc, 1, 1);
}

void dgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N, double* A, const int* LDA, double* S,
             double* U, const int* LDU, double* VT, const int* LDVT, double* WORK, const int* LWORK, int* INFO) {
  const I m = *M, n = *N, lda = *LDA, ldu = *LDU, ldvt = *LDVT, lwork = *LWORK;
  I info = 0;
  p_dgesvd(JOBU, JOBVT, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, WORK, &lwork, &info, 1, 1);
  *INFO = static_cast<int>(info);
}

void sgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N, float* A, const int* LDA, float* S,
             float* U, const int* LDU, float* VT, const int* LDVT, float* WORK, const int* LWORK, int* INFO) {
  const I m = *M, n = *N, lda = *LDA, ldu = *LDU, ldvt = *LDVT, lwork = *LWORK;
  I info = 0;
  p_sgesvd(JOBU, JOBVT, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, WORK, &lwork, &info, 1, 1);
  *INFO = static_cast<int>(info);
}

} // extern "C"

#endif // SCTLJL_BLAS_LBT
