// sctl::ParticleFMM<double,3> for one source group "src" and one target group "trg", with the FMM translation
// kernels of fmm_schemes.hpp. Built with PVFMM (SCTL_HAVE_PVFMM, no MPI) this is a fast multipole method for 40000 or
// more targets (or with periodicity); otherwise, and for fewer targets, a direct sum.
#include "sctljl.hpp"
#include "fmm_schemes.hpp"

using namespace sctljl;

namespace {

  struct FMMBase {
    virtual ~FMMBase() {}
    virtual void set_accuracy(Integer digits) = 0;
    virtual void set_periodicity(sctl::Periodicity p, double L) = 0;
    virtual void set_src(const VecD& X, const VecD& Xn) = 0;
    virtual void set_density(const VecD& F) = 0;
    virtual void set_trg(const VecD& X) = 0;
    virtual void eval(VecD& U, bool direct) const = 0;
  };

  template <class Ker> struct FMMImpl final : FMMBase {
    sctl::ParticleFMM<double, 3> fmm; // Comm::Self
    FMMImpl() { setup_fmm_kernels(fmm, Ker()); }
    void set_accuracy(Integer digits) override { fmm.SetAccuracy(digits); }
    void set_periodicity(sctl::Periodicity p, double L) override { fmm.SetPeriodicity(p, L); }
    void set_src(const VecD& X, const VecD& Xn) override { fmm.SetSrcCoord("src", X, Xn); }
    void set_density(const VecD& F) override { fmm.SetSrcDensity("src", F); }
    void set_trg(const VecD& X) override { fmm.SetTrgCoord("trg", X); }
    void eval(VecD& U, bool direct) const override {
      if (direct) fmm.EvalDirect(U, "trg");
      else fmm.Eval(U, "trg");
    }
  };

  // kernel ids as in capi_kernels.cpp
  FMMBase* new_fmm(int id) {
    switch (id) {
      case 0:  return new FMMImpl<sctl::Laplace3D_FxU>();
      case 1:  return new FMMImpl<sctl::Laplace3D_DxU>();
      case 2:  return new FMMImpl<sctl::Laplace3D_FxdU>();
      case 3:  return new FMMImpl<sctl::Laplace3D_Fxd2U>();
      case 4:  return new FMMImpl<sctl::Laplace3D_DxdU>();
      case 11: return new FMMImpl<sctl::Stokes3D_FxU>();
      case 12: return new FMMImpl<sctl::Stokes3D_DxU>();
      case 13: return new FMMImpl<sctl::Stokes3D_FxT>();
      default: return nullptr;
    }
  }

  FMMBase& fmm(void* h) { return *static_cast<FMMBase*>(h); }

} // namespace

SCTLJL_EXPORT int sctljl_fmm_new(void** out, int kernel_id) {
  SCTLJL_TRY
  *out = new_fmm(kernel_id);
  if (!*out) throw Error("sctljl: kernel " + std::to_string(kernel_id) + " has no FMM support");
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_fmm_free(void* h) { delete static_cast<FMMBase*>(h); }

SCTLJL_EXPORT int sctljl_fmm_set_accuracy(void* h, int64_t digits) {
  SCTLJL_TRY
  fmm(h).set_accuracy((Integer)digits);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_fmm_set_periodicity(void* h, int code, double L) {
  SCTLJL_TRY
  fmm(h).set_periodicity(periodicity(code), L);
  SCTLJL_CATCH
}

// X: 3*ns coordinates; Xn: 3*ns normals, or nn = 0 for kernels without normals
SCTLJL_EXPORT int sctljl_fmm_set_sources(void* h, const double* X, int64_t ns, const double* Xn, int64_t nn) {
  SCTLJL_TRY
  fmm(h).set_src(view(X, 3 * ns), nn ? view(Xn, 3 * nn) : VecD());
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_fmm_set_density(void* h, const double* F, int64_t n) {
  SCTLJL_TRY
  fmm(h).set_density(view(F, n));
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_fmm_set_targets(void* h, const double* X, int64_t nt) {
  SCTLJL_TRY
  fmm(h).set_trg(view(X, 3 * nt));
  SCTLJL_CATCH
}

// U: n = trg_dim * nt values
SCTLJL_EXPORT int sctljl_fmm_eval(void* h, double* U, int64_t n, int direct) {
  SCTLJL_TRY
  VecD U_;
  fmm(h).eval(U_, direct != 0);
  copy_to(U, U_, n);
  SCTLJL_CATCH
}
