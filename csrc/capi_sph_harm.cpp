// Spherical harmonics on the unit sphere (sctl::SphericalHarmonics<double>).
//
// Arrangement codes: 0 = SHCArrange::ALL, 1 = ROW_MAJOR, 2 = COL_MAJOR_NONZERO.
// All outputs are caller-allocated buffers whose size Julia computes in advance (checked by copy_to).
// SCTL guards its sizes with plain `assert` (compiled out under NDEBUG) and indexes fixed-size caches of
// SCTL_SHMAXDEG entries, so the size checks below are repeated here to turn would-be UB into an exception.
#include "sctljl.hpp"

using namespace sctljl;
using SH = sctl::SphericalHarmonics<double>;

namespace {
  sctl::SHCArrange arrangement(int code) {
    switch (code) {
      case 0: return sctl::SHCArrange::ALL;
      case 1: return sctl::SHCArrange::ROW_MAJOR;
      case 2: return sctl::SHCArrange::COL_MAJOR_NONZERO;
    }
    throw Error("sctljl: invalid SHCArrange code " + std::to_string(code));
  }

  int64_t ncoeff(int64_t p, int code) {  // real numbers per scalar field
    switch (code) {
      case 0: return 2 * (p + 1) * (p + 1);
      case 1: return (p + 1) * (p + 2);
      case 2: return (p + 1) * (p + 1);
    }
    throw Error("sctljl: invalid SHCArrange code " + std::to_string(code));
  }

  // p_extra: SCTL works internally at order p + p_extra (vector routines use p + 1).
  void check_order(int64_t p, int64_t p_extra = 0) {
    if (p < 0 || p + p_extra >= SCTL_SHMAXDEG) throw Error("sctljl: spherical harmonic order out of range");
  }
  void check_grid(int64_t Nt, int64_t Np) {
    if (Nt < 1 || Nt > SCTL_SHMAXDEG || Np < 1 || Np >= SCTL_SHMAXDEG) throw Error("sctljl: grid size out of range");
  }
  void check_multiple(int64_t n, int64_t block, const char* what) {
    if (block <= 0 || n <= 0 || n % block != 0) throw Error(std::string("sctljl: invalid length of ") + what);
  }
}

// ---- scalar spherical harmonics ----

SCTLJL_EXPORT int sctljl_sph_grid2shc(double* S, int64_t nS, const double* X, int64_t nX, int64_t Nt, int64_t Np, int64_t p, int arr) {
  SCTLJL_TRY
  check_order(p); check_grid(Nt, Np); check_multiple(nX, Nt * Np, "grid values");
  VecD S_(nS);
  S_ = 0;  // SHCArrange::ALL leaves the (unused) m > n entries unwritten
  SH::Grid2SHC(view(X, nX), Nt, Np, p, S_, arrangement(arr));
  copy_to(S, S_, nS);
  SCTLJL_CATCH
}

// X_theta / X_phi may be NULL.
SCTLJL_EXPORT int sctljl_sph_shc2grid(double* X, double* X_theta, double* X_phi, int64_t nX,
                                      const double* S, int64_t nS, int arr, int64_t p, int64_t Nt, int64_t Np) {
  SCTLJL_TRY
  check_order(p); check_grid(Nt, Np); check_multiple(nS, ncoeff(p, arr), "coefficients");
  VecD X_, Xt_, Xp_;  // SHC2Grid requires X != NULL whenever X_phi is requested
  SH::SHC2Grid(view(S, nS), arrangement(arr), p, Nt, Np, &X_, X_theta ? &Xt_ : nullptr, X_phi ? &Xp_ : nullptr);
  copy_to(X, X_, nX);
  if (X_theta) copy_to(X_theta, Xt_, nX);
  if (X_phi) copy_to(X_phi, Xp_, nX);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_sph_shceval(double* X, int64_t nX, const double* S, int64_t nS, int arr, int64_t p,
                                     const double* theta_phi, int64_t ntp) {
  SCTLJL_TRY
  check_order(p); check_multiple(nS, ncoeff(p, arr), "coefficients"); check_multiple(ntp, 2, "theta_phi");
  VecD X_;
  SH::SHCEval(view(S, nS), arrangement(arr), p, view(theta_phi, ntp), X_);
  copy_to(X, X_, nX);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_sph_shc2pole(double* P, int64_t nP, const double* S, int64_t nS, int arr, int64_t p) {
  SCTLJL_TRY
  check_order(p); check_multiple(nS, ncoeff(p, arr), "coefficients");
  VecD P_;
  SH::SHC2Pole(view(S, nS), arrangement(arr), p, P_);
  copy_to(P, P_, nP);
  SCTLJL_CATCH
}

// ---- vector spherical harmonics ----

SCTLJL_EXPORT int sctljl_sph_grid2vecshc(double* S, int64_t nS, const double* X, int64_t nX, int64_t Nt, int64_t Np, int64_t p, int arr) {
  SCTLJL_TRY
  check_order(p, 1); check_grid(Nt, Np); check_multiple(nX, 3 * Nt * Np, "grid values");
  VecD S_(nS);
  S_ = 0;  // SHCArrange::ALL leaves the (unused) m > n entries unwritten
  SH::Grid2VecSHC(view(X, nX), Nt, Np, p, S_, arrangement(arr));
  copy_to(S, S_, nS);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_sph_vecshc2grid(double* X, int64_t nX, const double* S, int64_t nS, int arr, int64_t p, int64_t Nt, int64_t Np) {
  SCTLJL_TRY
  check_order(p, 1); check_grid(Nt, Np); check_multiple(nS, 3 * ncoeff(p, arr), "coefficients");
  VecD X_;
  SH::VecSHC2Grid(view(S, nS), arrangement(arr), p, Nt, Np, X_);
  copy_to(X, X_, nX);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_sph_vecshceval(double* X, int64_t nX, const double* S, int64_t nS, int arr, int64_t p,
                                        const double* theta_phi, int64_t ntp) {
  SCTLJL_TRY
  check_order(p, 1); check_multiple(nS, 3 * ncoeff(p, arr), "coefficients"); check_multiple(ntp, 2, "theta_phi");
  VecD X_;
  SH::VecSHCEval(view(S, nS), arrangement(arr), p, view(theta_phi, ntp), X_);
  copy_to(X, X_, nX);
  SCTLJL_CATCH
}

// ---- layer potentials on the unit sphere ----

// kind: 0 = single layer, 1 = double layer
SCTLJL_EXPORT int sctljl_sph_laplace_eval(double* U, int64_t nU, const double* S, int64_t nS, int arr, int64_t p,
                                          const double* coord, int64_t ncoord, int interior, int kind) {
  SCTLJL_TRY
  check_order(p, 1); check_multiple(nS, ncoeff(p, arr), "coefficients"); check_multiple(ncoord, 3, "coordinates");
  VecD U_;
  if (kind == 0) SH::LaplaceEvalSL(view(S, nS), arrangement(arr), p, view(coord, ncoord), interior != 0, U_);
  else if (kind == 1) SH::LaplaceEvalDL(view(S, nS), arrangement(arr), p, view(coord, ncoord), interior != 0, U_);
  else throw Error("sctljl: invalid Laplace kernel kind");
  copy_to(U, U_, nU);
  SCTLJL_CATCH
}

// kind: 0 = single layer, 1 = double layer, 2 = traction of the single layer at targets with normals `norm`,
// 3 = traction of the single layer on the sphere itself (normal = radial direction; `norm` ignored).
SCTLJL_EXPORT int sctljl_sph_stokes_eval(double* U, int64_t nU, const double* S, int64_t nS, int arr, int64_t p,
                                         const double* coord, int64_t ncoord, const double* norm, int interior, int kind) {
  SCTLJL_TRY
  check_order(p, 2); check_multiple(nS, 3 * ncoeff(p, arr), "coefficients"); check_multiple(ncoord, 3, "coordinates");
  const auto S_ = view(S, nS);
  const auto X_ = view(coord, ncoord);
  const auto a = arrangement(arr);
  VecD U_;
  switch (kind) {
    case 0: SH::StokesEvalSL(S_, a, p, X_, interior != 0, U_); break;
    case 1: SH::StokesEvalDL(S_, a, p, X_, interior != 0, U_); break;
    case 2:
      if (!norm) throw Error("sctljl: StokesEvalKL needs normals");
      SH::StokesEvalKL(S_, a, p, X_, view(norm, ncoord), interior != 0, U_);
      break;
    case 3: SH::StokesEvalKSelf(S_, a, p, X_, interior != 0, U_); break;
    default: throw Error("sctljl: invalid Stokes kernel kind");
  }
  copy_to(U, U_, nU);
  SCTLJL_CATCH
}

// ---- Gauss-Legendre rule in cos(theta) and cache ----

// p1 + 1 nodes / weights on [-1, 1].
SCTLJL_EXPORT int sctljl_sph_legendre_nodes(double* x, int64_t p1) {
  SCTLJL_TRY
  check_order(p1);
  copy_to(x, SH::LegendreNodes(p1), p1 + 1);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_sph_legendre_weights(double* w, int64_t p1) {
  SCTLJL_TRY
  check_order(p1);
  copy_to(w, SH::LegendreWeights(p1), p1 + 1);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_sph_clear() {
  SCTLJL_TRY
  SH::Clear();
  SCTLJL_CATCH
}
