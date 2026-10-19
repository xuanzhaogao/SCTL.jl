// GMRES linear solver and KrylovPrecond (Krylov-subspace recycling preconditioner).
//
// Threading: GMRES invokes the operator callback only from the calling thread (never inside an OpenMP region),
// so a Julia callback is safe here.
#include "sctljl.hpp"

#include <cstdio>

using namespace sctljl;

using KrylovPrecondD = sctl::KrylovPrecond<double>;

// ---- KrylovPrecond ----

SCTLJL_EXPORT int sctljl_krylov_new(void** out) {
  SCTLJL_TRY
  *out = new KrylovPrecondD();
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_krylov_free(void* P) { delete static_cast<KrylovPrecondD*>(P); }

SCTLJL_EXPORT int sctljl_krylov_size(void* P, int64_t* size, int64_t* rank) {
  SCTLJL_TRY
  const auto& P_ = *static_cast<KrylovPrecondD*>(P);
  *size = P_.Size();
  *rank = P_.Rank();
  SCTLJL_CATCH
}

// Apply the preconditioner in place to x[0..n-1] (identity when n != Size()).
SCTLJL_EXPORT int sctljl_krylov_apply(void* P, double* x, int64_t n) {
  SCTLJL_TRY
  VecD x_(n, (sctl::Iterator<double>)x, false);
  static_cast<KrylovPrecondD*>(P)->Apply(x_, sctl::Comm::Self());
  SCTLJL_CATCH
}

// ---- GMRES ----

// Linear operator callback: write A*x into Ax (both of length n); return 0, or nonzero if the Julia callback threw.
typedef int (*sctljl_linop_fn)(void* ctx, double* Ax, const double* x, int64_t n);

// Solve A x = b. On entry x[0..n-1] is the initial guess if has_x0 != 0 (otherwise ignored); on exit the solution.
// gs: 0 = modified Gram-Schmidt, 1 = classical Gram-Schmidt. precond: a KrylovPrecond (updated) or NULL.
SCTLJL_EXPORT int sctljl_gmres(double* x, int has_x0, sctljl_linop_fn fn, void* ctx, const double* b, int64_t n,
                               double tol, int64_t max_iter, int use_abs_tol, int verbose, int gs, int64_t num_reorth,
                               void* precond, int64_t* iters) {
  SCTLJL_TRY
  using GMRESD = sctl::GMRES<double>;
  // ParallelOp contract: Ax is preallocated to x.Dim() (possibly a fixed-size view); only ReInit on mismatch.
  const GMRESD::ParallelOp A = [fn, ctx](VecD* Ax, const VecD& x) {
    const Long N = x.Dim();
    if (Ax->Dim() != N) Ax->ReInit(N);
    if (fn(ctx, N ? &(*Ax)[0] : nullptr, N ? &x[0] : nullptr, N) != 0) throw CallbackError();
  };
  const GMRESD solver(sctl::Comm::Self(), verbose != 0, gs == 1 ? GMRESD::GramSchmidt::CGS : GMRESD::GramSchmidt::MGS,
                      (Integer)num_reorth);
  VecD x_;
  if (has_x0) {
    x_.ReInit(n);
    for (Long i = 0; i < n; i++) x_[i] = x[i];
  }
  Long solve_iter = 0;
  solver(&x_, A, view(b, n), tol, (Integer)max_iter, use_abs_tol != 0, &solve_iter, static_cast<KrylovPrecondD*>(precond));
  if (verbose) fflush(stdout);
  copy_to(x, x_, n);
  *iters = solve_iter;
  SCTLJL_CATCH
}
