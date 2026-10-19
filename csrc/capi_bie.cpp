// Boundary integral operators: QuadElemList (sctl/experimental/quad_element.*) and BoundaryIntegralOp.
//
// quad_element.cpp is #included (through capi_bie.hpp) in every capi_bie*.cpp; everything in it is templates or
// inline functions, so that is ODR-safe. The BoundaryIntegralOp instantiations are split by kernel family across
// capi_bie.cpp (Laplace FxU, DxU), capi_bie_laplace.cpp, capi_bie_stokes.cpp and capi_bie_helmholtz.cpp.
//
// QuadElemList<double>: the constructor takes the node coordinates AoS, element by element; within an element node
// p = i*order + j is at the parameter point (u_i, v_j) with u, v = ParamNodes(order) (Gauss-Legendre on [0,1]).
// Internally SCTL stores, per element, x of all nodes, then y, then z; that layout is not exposed.
//
// BoundaryIntegralOp<double,Kernel>: one C++ class template per kernel, behind a virtual interface and selected by
// the kernel id of capi_kernels.cpp. The op object owns the kernel context (wavenumber), to which the kernel copies
// inside the op point. Without PVFMM the far field is a direct O(N^2) sum (ParticleFMM::EvalDirect).
#include "sctljl.hpp"
#include "capi_bie.hpp"

using namespace sctljl;

namespace sctljl {
namespace bie {

  BIOpBase* new_op_laplace_sl(int id, double ctx, bool trg_dot_prod) {
    switch (id) {
      case 0:  return new BIOp<sctl::Laplace3D_FxU>(ctx, trg_dot_prod);
      case 1:  return new BIOp<sctl::Laplace3D_DxU>(ctx, trg_dot_prod);
      default: return nullptr;
    }
  }

} // namespace bie
} // namespace sctljl

using namespace sctljl::bie;

namespace {

  BIOpBase* new_op(int id, double ctx, bool trg_dot_prod) {
    for (auto f : {new_op_laplace_sl, new_op_laplace_grad, new_op_stokes, new_op_helmholtz}) {
      if (BIOpBase* op = f(id, ctx, trg_dot_prod)) return op;
    }
    throw Error("sctljl: kernel id " + std::to_string(id) + " is not supported by BoundaryIntegralOp");
  }

  QEL& qel(void* h) { return *static_cast<QEL*>(h); }
  BIOpBase& bop(void* h) { return *static_cast<BIOpBase*>(h); }

}

// ---- QuadElemList ----

// coord: AoS node coordinates, n doubles (a multiple of 3*order^2).
SCTLJL_EXPORT int sctljl_quadelem_new(void** h, int64_t order, const double* coord, int64_t n) {
  SCTLJL_TRY
  *h = nullptr;
  if (order <= 0) throw Error("sctljl: order must be positive");
  if (n % (3 * order * order)) throw Error("sctljl: coordinate length is not a multiple of 3*order^2");
  *h = new QEL((sctl::Integer)order, view(coord, n));
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_quadelem_free(void* h) { delete static_cast<QEL*>(h); }

// out = {Size(), Order()}
SCTLJL_EXPORT int sctljl_quadelem_info(void* h, int64_t* out) {
  SCTLJL_TRY
  out[0] = qel(h).Size();
  out[1] = qel(h).Order();
  SCTLJL_CATCH
}

// X, Xn: caller buffers of 3*Nnodes doubles (either may be NULL).
SCTLJL_EXPORT int sctljl_quadelem_node_coord(void* h, double* X, double* Xn, int64_t nnodes) {
  SCTLJL_TRY
  VecD X_, Xn_;
  qel(h).GetNodeCoord(X ? &X_ : nullptr, Xn ? &Xn_ : nullptr, nullptr);
  if (X) copy_to(X, X_, 3 * nnodes);
  if (Xn) copy_to(Xn, Xn_, 3 * nnodes);
  SCTLJL_CATCH
}

// Far-field (= node) quadrature weights and far-field cut-off distances for tolerance tol (Nnodes each; may be NULL).
SCTLJL_EXPORT int sctljl_quadelem_far_nodes(void* h, double* wts, double* dist_far, int64_t nnodes, double tol) {
  SCTLJL_TRY
  VecD X_, Xn_, w_, d_;
  VecL cnt;
  qel(h).GetFarFieldNodes(X_, Xn_, w_, d_, cnt, tol);
  if (wts) copy_to(wts, w_, nnodes);
  if (dist_far) copy_to(dist_far, d_, nnodes);
  SCTLJL_CATCH
}

// Geometry of element elem (0-based) at the nu x nv parameter grid; output point a*nv+b at (u[a], v[b]).
// X, Xn, dXdu, dXdv: 3*nu*nv doubles, Xa: nu*nv doubles; each may be NULL.
SCTLJL_EXPORT int sctljl_quadelem_geom(void* h, double* X, double* Xn, double* Xa, double* dXdu, double* dXdv,
                                       const double* u, int64_t nu, const double* v, int64_t nv, int64_t elem) {
  SCTLJL_TRY
  if (elem < 0 || elem >= qel(h).Size()) throw Error("sctljl: element index out of range");
  const int64_t N = nu * nv;
  VecD X_, Xn_, Xa_, du_, dv_;
  qel(h).GetGeom(X ? &X_ : nullptr, Xn ? &Xn_ : nullptr, Xa ? &Xa_ : nullptr, dXdu ? &du_ : nullptr, dXdv ? &dv_ : nullptr,
                 view(u, nu), view(v, nv), elem);
  if (X) copy_to(X, X_, 3 * N);
  if (Xn) copy_to(Xn, Xn_, 3 * N);
  if (Xa) copy_to(Xa, Xa_, N);
  if (dXdu) copy_to(dXdu, du_, 3 * N);
  if (dXdv) copy_to(dXdv, dv_, 3 * N);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_quadelem_param_nodes(double* nds, int64_t order) {
  SCTLJL_TRY
  if (order <= 0) throw Error("sctljl: order must be positive");
  copy_to(nds, QEL::ParamNodes((sctl::Integer)order), order);
  SCTLJL_CATCH
}

// scheme: 0 = TensorProduct, 1 = Duffy, 2 = Hedgehog
SCTLJL_EXPORT int sctljl_quadelem_set_scheme(void* h, int scheme) {
  SCTLJL_TRY
  if (scheme < 0 || scheme > 2) throw Error("sctljl: invalid quadrature scheme");
  qel(h).SetQuadScheme(static_cast<QEL::QuadScheme>(scheme));
  SCTLJL_CATCH
}

// Writes <fname>000000.vtu; F (nF doubles, AoS at the nodes) may be empty.
SCTLJL_EXPORT int sctljl_quadelem_write_vtk(void* h, const char* fname, const double* F, int64_t nF) {
  SCTLJL_TRY
  qel(h).WriteVTK(fname, nF ? view(F, nF) : VecD());
  SCTLJL_CATCH
}

// ---- BoundaryIntegralOp ----

SCTLJL_EXPORT int sctljl_bie_new(void** h, int kernel_id, double ctx, int trg_dot_prod) {
  SCTLJL_TRY
  *h = nullptr;
  *h = new_op(kernel_id, ctx, trg_dot_prod != 0);
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_bie_free(void* h) { delete static_cast<BIOpBase*>(h); }

// Copies the element list (later changes to it, e.g. its quadrature scheme, do not affect the op).
SCTLJL_EXPORT int sctljl_bie_add_elem_list(void* h, void* elist, const char* name) {
  SCTLJL_TRY
  bop(h).add(qel(elist), name);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_delete_elem_list(void* h, const char* name) {
  SCTLJL_TRY
  bop(h).del(name);
  SCTLJL_CATCH
}

// n = 0: targets are the surface nodes (the default).
SCTLJL_EXPORT int sctljl_bie_set_target_coord(void* h, const double* X, int64_t n) {
  SCTLJL_TRY
  bop(h).set_trg_coord(n ? view(X, n) : VecD());
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_set_target_normal(void* h, const double* Xn, int64_t n) {
  SCTLJL_TRY
  bop(h).set_trg_normal(n ? view(Xn, n) : VecD());
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_set_accuracy(void* h, double tol) {
  SCTLJL_TRY
  if (!(tol > 0)) throw Error("sctljl: accuracy must be positive");
  bop(h).set_accuracy(tol);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_get_accuracy(void* h, double* tol) {
  SCTLJL_TRY
  *tol = bop(h).accuracy();
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_dim(void* h, int k, int64_t* d) {
  SCTLJL_TRY
  if (k != 0 && k != 1) throw Error("sctljl: dim index must be 0 (input) or 1 (output)");
  *d = bop(h).dim(k);
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_setup(void* h) {
  SCTLJL_TRY
  bop(h).setup();
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_bie_clear_setup(void* h) {
  SCTLJL_TRY
  bop(h).clear_setup();
  SCTLJL_CATCH
}

// U: caller buffer of nU = Dim(1) doubles (overwritten); F: nF = Dim(0) doubles.
SCTLJL_EXPORT int sctljl_bie_compute_potential(void* h, double* U, int64_t nU, const double* F, int64_t nF) {
  SCTLJL_TRY
  const BIOpBase& op = bop(h);
  if (op.dim(0) != nF) throw Error("sctljl: density length does not match Dim(0)");
  if (op.dim(1) != nU) throw Error("sctljl: potential length does not match Dim(1)");
  VecD U_(nU, (sctl::Iterator<double>)U, false);
  op.compute(U_, view(F, nF));
  if (U_.Dim() != nU) throw Error("sctljl: unexpected potential size");
  SCTLJL_CATCH
}

// In place: U (n doubles, a multiple of the number of surface nodes) *= sqrt(element area), or its inverse.
SCTLJL_EXPORT int sctljl_bie_sqrt_scaling(void* h, double* U, int64_t n, int inverse) {
  SCTLJL_TRY
  VecD U_(n, (sctl::Iterator<double>)U, false);
  if (inverse) bop(h).inv_sqrt_scaling(U_);
  else bop(h).sqrt_scaling(U_);
  SCTLJL_CATCH
}
