// Common header for the SCTL C-ABI shim used by SCTL.jl.
//
// Every exported function returns an `int` status: 0 on success, SCTLJL_ERR on a C++ exception (message in
// sctljl_last_error()), SCTLJL_ERR_CALLBACK when a Julia callback raised (the Julia side holds the exception).
//
// Inputs from Julia are wrapped as non-owning sctl::Vector views (see `view`). Outputs whose size is not known
// in advance are written to owned sctl::Vector objects allocated by Julia through sctljl_vec_new_* (handles).
#ifndef SCTLJL_HPP
#define SCTLJL_HPP

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>

#include <sctl/common.hpp>

namespace sctljl {
  struct Error : std::runtime_error { using std::runtime_error::runtime_error; };
  struct CallbackError : std::runtime_error { CallbackError() : std::runtime_error("Julia callback failed") {} };
  void set_error(const char* msg);
}

// SCTL's error macros call abort(), which would take the Julia process down; throw instead and let the C API
// convert it to an error code. This must come after sctl/common.hpp and before every other SCTL header.
#undef SCTL_ERROR
#define SCTL_ERROR(msg)                          \
  do {                                           \
    std::ostringstream sctljl_ss_;               \
    sctljl_ss_ << "SCTL Error: " << msg;         \
    throw ::sctljl::Error(sctljl_ss_.str());     \
  } while (0)

#undef SCTL_ASSERT
#define SCTL_ASSERT(cond)                                                                            \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      std::ostringstream sctljl_ss_;                                                                 \
      sctljl_ss_ << __FILE__ << ':' << __LINE__ << ": SCTL Assertion `" << #cond << "' failed";      \
      throw ::sctljl::Error(sctljl_ss_.str());                                                       \
    }                                                                                                \
  } while (0)

#include <sctl.hpp>

#define SCTLJL_EXPORT extern "C" __attribute__((visibility("default")))

#define SCTLJL_OK 0
#define SCTLJL_ERR 1
#define SCTLJL_ERR_CALLBACK 2

#define SCTLJL_TRY try {
#define SCTLJL_CATCH                                                          \
  }                                                                           \
  catch (const ::sctljl::CallbackError&) { return SCTLJL_ERR_CALLBACK; }      \
  catch (const std::exception& e) { ::sctljl::set_error(e.what()); return SCTLJL_ERR; } \
  catch (...) { ::sctljl::set_error("unknown C++ exception"); return SCTLJL_ERR; } \
  return SCTLJL_OK;

namespace sctljl {
  using sctl::Long;
  using sctl::Integer;

  using VecD = sctl::Vector<double>;
  using VecL = sctl::Vector<sctl::Long>;

  // Non-owning sctl::Vector view of a Julia array (read-only use).
  template <class T> inline sctl::Vector<T> view(const T* ptr, int64_t n) {
    return sctl::Vector<T>(n, (sctl::Iterator<T>)const_cast<T*>(ptr), false);
  }

  // Owned output vector behind a Julia handle.
  template <class T> inline sctl::Vector<T>& out(void* handle) { return *static_cast<sctl::Vector<T>*>(handle); }

  // Copy an sctl::Vector into a caller buffer of known size.
  template <class T> inline void copy_to(T* dst, const sctl::Vector<T>& v, int64_t n) {
    if (v.Dim() != n) throw Error("sctljl: output size mismatch (" + std::to_string(v.Dim()) + " != " + std::to_string(n) + ")");
    for (Long i = 0; i < n; i++) dst[i] = v[i];
  }

  // Discards everything written to std::cout while alive and `on` (SCTL logs to std::cout in some routines).
  struct NullBuf : std::streambuf {
    int overflow(int c) override { return traits_type::not_eof(c); }
  };
  struct CoutSilencer {
    NullBuf nb;
    std::streambuf* old = nullptr;
    explicit CoutSilencer(bool on) { if (on) old = std::cout.rdbuf(&nb); }
    ~CoutSilencer() { if (old) std::cout.rdbuf(old); }
    CoutSilencer(const CoutSilencer&) = delete;
    CoutSilencer& operator=(const CoutSilencer&) = delete;
  };

  inline sctl::Periodicity periodicity(int code) { return static_cast<sctl::Periodicity>(static_cast<uint8_t>(code)); }
}

#endif // SCTLJL_HPP
