// SDC (spectral deferred correction) ODE solver.
//
// Threading: SDC invokes the RHS and monitor callbacks only from the calling thread (never inside an OpenMP
// region), so Julia callbacks are safe here.
#include "sctljl.hpp"

#include <iostream>
#include <streambuf>

using namespace sctljl;

using SDCD = sctl::SDC<double>;

// RHS callback: write du/dt at u into dudt (both of length n); correction_idx / substep_idx are the SDC indices.
// Return 0, or nonzero if the Julia callback threw.
typedef int (*sctljl_sdc_fn)(void* ctx, double* dudt, const double* u, int64_t n, int64_t correction_idx, int64_t substep_idx);

// Monitor callback after each accepted adaptive step; return 0, or nonzero if the Julia callback threw.
typedef int (*sctljl_sdc_monitor_fn)(void* ctx, double t, double dt, const double* u, int64_t n);

namespace {
  SDCD::Fn0 make_rhs(sctljl_sdc_fn fn, void* ctx) {
    return [fn, ctx](VecD* dudt, const VecD& u, const Integer correction_idx, const Integer substep_idx) {
      const Long N = u.Dim();
      if (dudt->Dim() != N) dudt->ReInit(N);
      if (fn(ctx, N ? &(*dudt)[0] : nullptr, N ? &u[0] : nullptr, N, correction_idx, substep_idx) != 0) throw CallbackError();
    };
  }
}

SCTLJL_EXPORT int sctljl_sdc_new(void** out, int64_t order) {
  SCTLJL_TRY
  *out = new SDCD((Integer)order);
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_sdc_free(void* sdc) { delete static_cast<SDCD*>(sdc); }

SCTLJL_EXPORT int sctljl_sdc_order(void* sdc, int64_t* order) {
  SCTLJL_TRY
  *order = static_cast<SDCD*>(sdc)->Order();
  SCTLJL_CATCH
}

// One SDC step u = u0 + int_0^dt F(u). u_substep (may be NULL): row-major (order x n), row i = solution at substep i.
SCTLJL_EXPORT int sctljl_sdc_step(void* sdc, double* u, const double* u0, int64_t n, double dt, sctljl_sdc_fn fn, void* ctx,
                                  int64_t picard_iters, double picard_tol, double* error_interp, double* error_picard,
                                  int64_t* iter_count, double* u_substep) {
  SCTLJL_TRY
  const SDCD& S = *static_cast<SDCD*>(sdc);
  const SDCD::Fn0 F = make_rhs(fn, ctx);
  VecD u_;
  sctl::Matrix<double> Msub;
  Integer iters = 0;
  S(&u_, dt, view(u0, n), F, (Integer)picard_iters, picard_tol, error_interp, error_picard, &iters, u_substep ? &Msub : nullptr);
  copy_to(u, u_, n);
  *iter_count = iters;
  if (u_substep) {
    const Long order = S.Order();
    if (Msub.Dim(0) != order || Msub.Dim(1) != n) throw Error("sctljl: unexpected substep matrix size");
    for (Long i = 0; i < order * n; i++) u_substep[i] = Msub[0][i];
  }
  SCTLJL_CATCH
}

// Adaptive solve on [0, T]. monitor may be NULL. Returns the final time in *t_final and the error estimate in *error.
SCTLJL_EXPORT int sctljl_sdc_solve(void* sdc, double* u, const double* u0, int64_t n, double dt, double T, double tol,
                                   sctljl_sdc_fn fn, sctljl_sdc_monitor_fn monitor, void* ctx, int continue_with_errors,
                                   int verbose, double* t_final, double* error) {
  SCTLJL_TRY
  const SDCD& S = *static_cast<SDCD*>(sdc);
  const SDCD::Fn0 F = make_rhs(fn, ctx);
  const SDCD::MonitorFn mon = [monitor, ctx](double t, double dt, const VecD& u) {
    const Long N = u.Dim();
    if (monitor(ctx, t, dt, N ? &u[0] : nullptr, N) != 0) throw CallbackError();
  };
  VecD u_;
  double err = 0;
  {
    const CoutSilencer silence(verbose == 0);
    *t_final = S.AdaptiveSolve(&u_, dt, T, view(u0, n), F, tol, monitor ? &mon : nullptr, continue_with_errors != 0, &err);
  }
  copy_to(u, u_, n);
  *error = err;
  SCTLJL_CATCH
}
