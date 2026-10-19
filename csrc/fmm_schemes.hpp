// FMM translation kernels for each source-to-target (S2T) kernel, shared by ParticleFMM (capi_fmm.cpp) and
// BoundaryIntegralOp (capi_bie*.cpp). Include after sctljl.hpp.
//
// PVFMM caches its precomputed operators in a file named after all kernels of an FMM, so a kernel must always get the
// same translation kernels; the table below is the only place they are chosen. Translations act on equivalent
// densities: the Laplace single layer for all Laplace kernels; for Stokes, the Stokeslet plus source (Stokes3D_FSxU)
// for multipole expansions and the Stokeslet for local ones, as in sctl::ParticleFMM::test. S2L = S2M and M2L = M2M.
#ifndef SCTLJL_FMM_SCHEMES_HPP
#define SCTLJL_FMM_SCHEMES_HPP

namespace sctljl {

  template <class Ker> struct FMMScheme { static constexpr bool supported = false; };

#define SCTLJL_FMM_SCHEME(KER, S2M, M2M, L2L, M2T, L2T) \
  template <> struct FMMScheme<sctl::KER> {             \
    static constexpr bool supported = true;              \
    using s2m = sctl::S2M;                               \
    using m2m = sctl::M2M;                               \
    using l2l = sctl::L2L;                               \
    using m2t = sctl::M2T;                               \
    using l2t = sctl::L2T;                               \
  };

  //                KER                 S2M                M2M, M2L           L2L                M2T                L2T
  SCTLJL_FMM_SCHEME(Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU)
  SCTLJL_FMM_SCHEME(Laplace3D_DxU,    Laplace3D_DxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU)
  SCTLJL_FMM_SCHEME(Laplace3D_FxdU,   Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxdU,   Laplace3D_FxdU)
  SCTLJL_FMM_SCHEME(Laplace3D_DxdU,   Laplace3D_DxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxdU,   Laplace3D_FxdU)
  SCTLJL_FMM_SCHEME(Laplace3D_Fxd2U,  Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_FxU,    Laplace3D_Fxd2U,  Laplace3D_Fxd2U)
  SCTLJL_FMM_SCHEME(Stokes3D_FxU,     Stokes3D_FxU,     Stokes3D_FSxU,    Stokes3D_FxU,     Stokes3D_FSxU,    Stokes3D_FxU)
  SCTLJL_FMM_SCHEME(Stokes3D_DxU,     Stokes3D_DxU,     Stokes3D_FSxU,    Stokes3D_FxU,     Stokes3D_FSxU,    Stokes3D_FxU)
  // traction: no kernel maps Stokes3D_FSxU's 4 equivalent-density components to traction, so all translations use
  // the Stokeslet (less accurate per multipole order)
  SCTLJL_FMM_SCHEME(Stokes3D_FxT,     Stokes3D_FxU,     Stokes3D_FxU,     Stokes3D_FxU,     Stokes3D_FxT,     Stokes3D_FxT)

#undef SCTLJL_FMM_SCHEME

  // Kernels that read a context pointer (the wavenumber). Others must not get one: ParticleFMM treats kernels with a
  // context pointer as not evaluable by PVFMM and falls back to direct sums.
  template <class Ker> constexpr bool kernel_uses_ctx() {
    return std::is_same<Ker, sctl::Helmholtz3D_FxU>::value || std::is_same<Ker, sctl::Helmholtz3D_DxU>::value ||
           std::is_same<Ker, sctl::Helmholtz3D_FxdU>::value || std::is_same<Ker, sctl::HelmholtzDiff3D_FxdU>::value;
  }

  // Configure fmm (no sources/targets yet) for S2T kernel `ker` with source "src" and target "trg".
  template <class Ker, class FMM> void setup_fmm_kernels(FMM& fmm, const Ker& ker) {
    using S = FMMScheme<Ker>;
    static_assert(S::supported, "no FMM translation kernels for this kernel");
    const typename S::s2m s2m;
    const typename S::m2m m2m;
    const typename S::l2l l2l;
    const typename S::m2t m2t;
    const typename S::l2t l2t;
    fmm.SetKernels(m2m, m2m, l2l);
    fmm.AddSrc("src", s2m, s2m);
    fmm.AddTrg("trg", m2t, l2t);
    fmm.SetKernelS2T("src", "trg", ker);
  }

} // namespace sctljl

#endif // SCTLJL_FMM_SCHEMES_HPP
