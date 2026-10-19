// Core of the C API: error reporting, owned-vector handles, OpenMP thread control.
#include "sctljl.hpp"

namespace sctljl {
  static thread_local std::string last_error;
  void set_error(const char* msg) { last_error = msg; }
}

SCTLJL_EXPORT const char* sctljl_last_error() { return sctljl::last_error.c_str(); }

// ---- owned vector handles (outputs of size unknown in advance) ----

#define SCTLJL_VEC_API(SUFFIX, T)                                                                         \
  SCTLJL_EXPORT void* sctljl_vec_new_##SUFFIX() { return new sctl::Vector<T>(); }                          \
  SCTLJL_EXPORT void sctljl_vec_free_##SUFFIX(void* h) { delete static_cast<sctl::Vector<T>*>(h); }        \
  SCTLJL_EXPORT int64_t sctljl_vec_size_##SUFFIX(void* h) { return static_cast<sctl::Vector<T>*>(h)->Dim(); } \
  SCTLJL_EXPORT T* sctljl_vec_data_##SUFFIX(void* h) {                                                    \
    auto& v = *static_cast<sctl::Vector<T>*>(h);                                                           \
    return v.Dim() ? &v[0] : nullptr;                                                                      \
  }                                                                                                        \
  SCTLJL_EXPORT int sctljl_vec_resize_##SUFFIX(void* h, int64_t n) {                                      \
    SCTLJL_TRY                                                                                             \
    auto& v = *static_cast<sctl::Vector<T>*>(h);                                                           \
    if (v.Dim() != n) v.ReInit(n);                                                                         \
    SCTLJL_CATCH                                                                                           \
  }

SCTLJL_VEC_API(f64, double)
SCTLJL_VEC_API(i64, sctl::Long)

// ---- OpenMP ----

SCTLJL_EXPORT int sctljl_get_max_threads() { return omp_get_max_threads(); }
SCTLJL_EXPORT void sctljl_set_num_threads(int n) { omp_set_num_threads(n); }

// ---- build info ----

SCTLJL_EXPORT int sctljl_have(const char* feature) {
  const std::string f(feature);
#ifdef SCTL_HAVE_BLAS
  if (f == "BLAS") return 1;
#endif
#ifdef SCTL_HAVE_LAPACK
  if (f == "LAPACK") return 1;
#endif
#ifdef SCTL_HAVE_FFTW
  if (f == "FFTW") return 1;
#endif
#ifdef SCTL_HAVE_MPI
  if (f == "MPI") return 1;
#endif
#ifdef SCTL_HAVE_PVFMM
  if (f == "PVFMM") return 1;
#endif
  return 0;
}
