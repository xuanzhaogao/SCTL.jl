// Quadrature rules (ChebQuadRule, LegQuadRule, InterpQuadRule) and LagrangeInterp.
#include "sctljl.hpp"

using namespace sctljl;

// ---- Fejér (Chebyshev) and Gauss-Legendre rules on [0,1] ----

SCTLJL_EXPORT int sctljl_chebquad(double* nds, double* wts, int64_t n) {
  SCTLJL_TRY
  VecD x, w;
  sctl::ChebQuadRule<double>::ComputeNdsWts(&x, &w, n);
  copy_to(nds, x, n);
  copy_to(wts, w, n);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_legquad(double* nds, double* wts, int64_t n) {
  SCTLJL_TRY
  VecD x, w;
  sctl::LegQuadRule<double>::ComputeNdsWts(&x, &w, n);
  copy_to(nds, x, n);
  copy_to(wts, w, n);
  SCTLJL_CATCH
}

// Legendre polynomial of `degree` and its derivative at x in [-1,1]; P or dP may be NULL.
SCTLJL_EXPORT int sctljl_legpoly(double* P, double* dP, const double* x, int64_t n, int64_t degree) {
  SCTLJL_TRY
  VecD P_, dP_;
  sctl::LegQuadRule<double>::LegPoly<double>(P ? &P_ : nullptr, dP ? &dP_ : nullptr, view(x, n), degree);
  if (P) copy_to(P, P_, n);
  if (dP) copy_to(dP, dP_, n);
  SCTLJL_CATCH
}

// ---- InterpQuadRule ----

// M: row-major (nnodes x nfuncs), M[i][j] = j-th integrand at nds[i].
SCTLJL_EXPORT int sctljl_interpquad_build(void* quad_nds, void* quad_wts, double* cond,
                                          const double* M, const double* nds, const double* wts, int64_t nnodes, int64_t nfuncs,
                                          double eps, int64_t order, double nds_a, double nds_b, int use_svd, int verbose) {
  SCTLJL_TRY
  const CoutSilencer silence(verbose == 0);
  const sctl::Matrix<double> M_(nnodes, nfuncs, (sctl::Iterator<double>)const_cast<double*>(M), false);
  *cond = sctl::InterpQuadRule<double>::Build(out<double>(quad_nds), out<double>(quad_wts), M_, view(nds, nnodes), view(wts, nnodes),
                                              eps, order, nds_a, nds_b, use_svd != 0);
  SCTLJL_CATCH
}

// Several rules at once; rule k is written to quad_nds[k], quad_wts[k] (arrays of handles of length nrules).
SCTLJL_EXPORT int sctljl_interpquad_build_multi(void** quad_nds, void** quad_wts, double* cond, int64_t nrules,
                                                const double* M, const double* nds, const double* wts, int64_t nnodes, int64_t nfuncs,
                                                const double* eps_vec, int64_t n_eps, const int64_t* order_vec, int64_t n_order,
                                                double nds_a, double nds_b, int use_svd, int verbose) {
  SCTLJL_TRY
  const CoutSilencer silence(verbose == 0);
  const sctl::Matrix<double> M_(nnodes, nfuncs, (sctl::Iterator<double>)const_cast<double*>(M), false);
  sctl::Vector<VecD> qn, qw;
  const VecD cond_ = sctl::InterpQuadRule<double>::Build(qn, qw, M_, view(nds, nnodes), view(wts, nnodes),
                                                         view(eps_vec, n_eps), view<Long>(order_vec, n_order), nds_a, nds_b, use_svd != 0);
  if (cond_.Dim() != nrules || qn.Dim() != nrules || qw.Dim() != nrules) throw Error("sctljl: unexpected number of quadrature rules");
  for (Long k = 0; k < nrules; k++) {
    out<double>(quad_nds[k]) = qn[k];
    out<double>(quad_wts[k]) = qw[k];
    cond[k] = cond_[k];
  }
  SCTLJL_CATCH
}

// Integrand callback: evaluate all integrands at x[0..n-1], resize `M` (an f64 handle) to n*nfuncs and fill it
// row-major (M[i*nfuncs+j] = j-th integrand at x[i]); return nfuncs, or -1 on error.
typedef int64_t (*sctljl_integrand_fn)(void* ctx, const double* x, int64_t n, void* M);

SCTLJL_EXPORT int sctljl_interpquad_build_fn(void* quad_nds, void* quad_wts, double* cond, sctljl_integrand_fn fn, void* ctx,
                                             double a, double b, double eps, int64_t order, double nds_a, double nds_b, int use_svd, int verbose) {
  SCTLJL_TRY
  const CoutSilencer silence(verbose == 0);
  const auto integrands = [fn, ctx](const VecD& x) {
    VecD buf;
    const int64_t nfuncs = fn(ctx, x.Dim() ? &x[0] : nullptr, x.Dim(), &buf);
    if (nfuncs < 0) throw CallbackError();
    if (buf.Dim() != x.Dim() * nfuncs) throw Error("sctljl: integrand callback returned wrong size");
    sctl::Matrix<double> M(x.Dim(), nfuncs);
    for (Long i = 0; i < buf.Dim(); i++) M[0][i] = buf[i];
    return M;
  };
  *cond = sctl::InterpQuadRule<double>::Build(out<double>(quad_nds), out<double>(quad_wts), integrands, a, b, eps, order, nds_a, nds_b, use_svd != 0);
  SCTLJL_CATCH
}

// ---- LagrangeInterp ----

// wts: row-major (ns x nt)
SCTLJL_EXPORT int sctljl_lagrange_interp(double* wts, const double* src, int64_t ns, const double* trg, int64_t nt) {
  SCTLJL_TRY
  VecD w;
  sctl::LagrangeInterp<double>::Interpolate(w, view(src, ns), view(trg, nt));
  copy_to(wts, w, ns * nt);
  SCTLJL_CATCH
}

// f: dof blocks of n values (f = [f1(x1..xn), f2(x1..xn), ...]); df has the same layout.
SCTLJL_EXPORT int sctljl_lagrange_derivative(double* df, const double* f, int64_t nf, const double* nds, int64_t n) {
  SCTLJL_TRY
  VecD d;
  sctl::LagrangeInterp<double>::Derivative(d, view(f, nf), view(nds, n));
  copy_to(df, d, nf);
  SCTLJL_CATCH
}
