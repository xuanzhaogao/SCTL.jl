// Kernel functions (sctl/kernel_functions.hpp): dimensions, names, kernel matrices and direct N-body sums.
//
// Kernels are selected by an integer id (the order of `with_kernel` below, mirrored in src/kernels.jl). The
// context (wavenumber of the Helmholtz kernels) is passed by pointer and only used for the duration of the call.
//
// Layouts (all AoS):
//   Xs, Xt, Xn: {x1,y1,z1, x2,...}; F: Ns*SrcDim {f_1[0..SrcDim), f_2...}; U: Nt*TrgDim.
//   M (KernelMatrix): row-major (Ns*SrcDim) x (Nt*TrgDim), M[s*SrcDim+i][t*TrgDim+j]; U = F^T M.
#include "sctljl.hpp"

using namespace sctljl;

namespace {

  // Calls f(Kernel()) for kernel id `id`; throws on an unknown id.
  template <class F> void with_kernel(int id, F&& f) {
    switch (id) {
      case 0:  return f(sctl::Laplace3D_FxU());
      case 1:  return f(sctl::Laplace3D_DxU());
      case 2:  return f(sctl::Laplace3D_FxdU());
      case 3:  return f(sctl::Laplace3D_Fxd2U());
      case 4:  return f(sctl::Laplace3D_DxdU());
      case 5:  return f(sctl::BiotSavart3D_FxU());
      case 6:  return f(sctl::BiotSavart3D_FxdU());
      case 7:  return f(sctl::Helmholtz3D_FxU());
      case 8:  return f(sctl::Helmholtz3D_DxU());
      case 9:  return f(sctl::Helmholtz3D_FxdU());
      case 10: return f(sctl::HelmholtzDiff3D_FxdU());
      case 11: return f(sctl::Stokes3D_FxU());
      case 12: return f(sctl::Stokes3D_DxU());
      case 13: return f(sctl::Stokes3D_FxT());
      case 14: return f(sctl::Stokes3D_FSxU());
      case 15: return f(sctl::Stokes3D_FxUP());
      default: throw Error("sctljl: unknown kernel id " + std::to_string(id));
    }
  }

  template <class Ker> void check_sizes(int64_t ns, int64_t nn, int64_t nt) {
    if (ns < 0 || nt < 0) throw Error("sctljl: negative number of points");
    if (Ker::NormalDim() && nn != ns) throw Error("sctljl: kernel needs one normal per source point");
  }

}

// dims = {CoordDim, NormalDim, SrcDim, TrgDim}
SCTLJL_EXPORT int sctljl_kernel_dims(int id, int64_t* dims) {
  SCTLJL_TRY
  with_kernel(id, [dims](const auto& ker) {
    using Ker = std::decay_t<decltype(ker)>;
    dims[0] = Ker::CoordDim();
    dims[1] = Ker::NormalDim();
    dims[2] = Ker::SrcDim();
    dims[3] = Ker::TrgDim();
  });
  SCTLJL_CATCH
}

// *name points to a static string (valid for the lifetime of the library).
SCTLJL_EXPORT int sctljl_kernel_name(int id, const char** name) {
  SCTLJL_TRY
  with_kernel(id, [name](const auto& ker) { *name = ker.Name().c_str(); });
  SCTLJL_CATCH
}

// M: caller buffer of (ns*SrcDim)*(nt*TrgDim) doubles, row-major (Ns*SrcDim) x (Nt*TrgDim). Xn has nn points
// (nn = ns when the kernel takes normals; ignored otherwise). ctx: pointer to the context (wavenumber) or NULL.
SCTLJL_EXPORT int sctljl_kernel_matrix(int id, const double* ctx, double* M, const double* Xs, int64_t ns,
                                       const double* Xn, int64_t nn, const double* Xt, int64_t nt, int openmp) {
  SCTLJL_TRY
  with_kernel(id, [=](auto ker) {
    using Ker = decltype(ker);
    check_sizes<Ker>(ns, nn, nt);
    constexpr int64_t D = Ker::CoordDim();
    ker.SetCtxPtr(const_cast<double*>(ctx));
    sctl::Matrix<double> M_(ns * Ker::SrcDim(), nt * Ker::TrgDim(), (sctl::Iterator<double>)M, false);
    const VecD Xs_ = view(Xs, ns * D), Xt_ = view(Xt, nt * D);
    const VecD Xn_ = (Ker::NormalDim() ? view(Xn, nn * Ker::NormalDim()) : VecD());
    if (openmp) ker.template KernelMatrix<double, true>(M_, Xt_, Xs_, Xn_);
    else ker.template KernelMatrix<double, false>(M_, Xt_, Xs_, Xn_);
  });
  SCTLJL_CATCH
}

// U (nt*TrgDim, caller buffer) += potential of the densities F (ns*SrcDim) at Xt.
SCTLJL_EXPORT int sctljl_kernel_eval(int id, const double* ctx, double* U, const double* Xs, int64_t ns,
                                     const double* Xn, int64_t nn, const double* F, const double* Xt, int64_t nt, int openmp) {
  SCTLJL_TRY
  with_kernel(id, [=](auto ker) {
    using Ker = decltype(ker);
    check_sizes<Ker>(ns, nn, nt);
    constexpr int64_t D = Ker::CoordDim();
    ker.SetCtxPtr(const_cast<double*>(ctx));
    VecD U_(nt * Ker::TrgDim(), (sctl::Iterator<double>)U, false);
    const VecD Xs_ = view(Xs, ns * D), Xt_ = view(Xt, nt * D), F_ = view(F, ns * Ker::SrcDim());
    const VecD Xn_ = (Ker::NormalDim() ? view(Xn, nn * Ker::NormalDim()) : VecD());
    if (openmp) ker.template Eval<double, true>(U_, Xt_, Xs_, Xn_, F_);
    else ker.template Eval<double, false>(U_, Xt_, Xs_, Xn_, F_);
  });
  SCTLJL_CATCH
}
