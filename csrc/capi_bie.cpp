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

  QuadBase* new_quad_laplace_sl(int id, double ctx) {
    switch (id) {
      case 0:  return new Quad<sctl::Laplace3D_FxU>(ctx);
      case 1:  return new Quad<sctl::Laplace3D_DxU>(ctx);
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

// ---- near-field pieces ----
//
// The steps of BoundaryIntegralOp's near-field setup, separately: the near list (which targets are near which
// element) and the accurate element-to-target matrices of a QuadElemList's quadrature. Julia assembles the near
// correction from them (src/bie_pieces.jl), and can replace any step.

namespace {
  QuadBase* new_quad(int id, double ctx) {
    for (auto f : {new_quad_laplace_sl, new_quad_laplace_grad, new_quad_stokes, new_quad_helmholtz}) {
      if (QuadBase* q = f(id, ctx)) return q;
    }
    throw Error("sctljl: kernel id " + std::to_string(id) + " is not supported by the element quadratures");
  }
  QuadBase& quad(void* h) { return *static_cast<QuadBase*>(h); }
}

// Near list of SCTL's BuildNearList (the near test of BoundaryIntegralOp): target t is near element e if it lies
// within rad[i] of a node i of e. Sources: ns nodes Xs (AoS), element by element, elem_cnt[e] nodes in element e.
// Outputs: ptr (nelem+1, caller buffer) and hidx (owned i64 vector): the targets near element e are
// idx[ptr[e]:ptr[e+1]-1], 0-based and sorted.
SCTLJL_EXPORT int sctljl_near_list(const double* Xs, const double* rad, int64_t ns, const int64_t* elem_cnt, int64_t nelem,
                                   const double* Xt, int64_t nt, int64_t* ptr, void* hidx) {
  SCTLJL_TRY
  VecL& idx = *static_cast<VecL*>(hidx);
  VecL cnt(nelem), dsp(nelem);
  int64_t total = 0;
  for (int64_t e = 0; e < nelem; e++) {
    if (elem_cnt[e] < 0) throw Error("sctljl: negative node count");
    cnt[e] = elem_cnt[e];
    dsp[e] = total;
    total += elem_cnt[e];
  }
  if (total != ns) throw Error("sctljl: the element node counts do not add up to the number of nodes");
  for (int64_t e = 0; e <= nelem; e++) ptr[e] = 0;
  idx.ReInit(0);
  if (ns == 0 || nt == 0) return SCTLJL_OK;
  // BuildNearList carries the target normals to the near pairs; the target index goes in that slot
  VecD payload(3 * nt);
  for (int64_t i = 0; i < nt; i++) {
    payload[3*i] = (double)i;
    payload[3*i+1] = payload[3*i+2] = 0;
  }
  VecD Xtrg_near, Xn_trg_near;
  VecL near_elem_cnt, near_elem_dsp, near_trg_cnt, near_trg_dsp;
  sctl::SortScatter<sctl::Long> near_sort_scatter;
  sctl::BuildNearList<double, 3>(Xtrg_near, Xn_trg_near, near_elem_cnt, near_elem_dsp, near_sort_scatter, near_trg_cnt,
                                 near_trg_dsp, view(Xt, 3 * nt), payload, view(Xs, 3 * ns), view(rad, ns), cnt, dsp,
                                 sctl::Comm::Self());
  const sctl::Long npair = Xn_trg_near.Dim() / 3;
  idx.ReInit(npair);
  for (sctl::Long k = 0; k < npair; k++) idx[k] = (sctl::Long)Xn_trg_near[3*k];
  for (int64_t e = 0; e < nelem; e++) ptr[e+1] = ptr[e] + near_elem_cnt[e];
  if (ptr[nelem] != npair) throw Error("sctljl: inconsistent near list");
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_quad_new(void** h, int kernel_id, double ctx) {
  SCTLJL_TRY
  *h = nullptr;
  *h = new_quad(kernel_id, ctx);
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_quad_free(void* h) { delete static_cast<QuadBase*>(h); }

// Accurate matrix from element e (0-based) to nt targets Xt (normals Xn, nn = nt or 0): SCTL's row-major
// (nnode*SrcDim) x (nt*TrgDim_out), nM doubles.
SCTLJL_EXPORT int sctljl_quad_near(void* q, void* el, int64_t e, const double* Xt, int64_t nt, const double* Xn, int64_t nn,
                                   double tol, double* M, int64_t nM) {
  SCTLJL_TRY
  if (e < 0 || e >= qel(el).Size()) throw Error("sctljl: element index out of range");
  sctl::Matrix<double> M_;
  quad(q).near(M_, qel(el), e, view(Xt, 3 * nt), nn ? view(Xn, 3 * nn) : VecD(), tol);
  if ((int64_t)(M_.Dim(0) * M_.Dim(1)) != nM) throw Error("sctljl: near matrix has an unexpected size");
  if (nM) std::copy(M_.begin(), M_.begin() + nM, M);
  SCTLJL_CATCH
}

// Self-interaction matrices of all elements, concatenated: element e row-major (nnode*SrcDim) x (nnode*TrgDim_out).
SCTLJL_EXPORT int sctljl_quad_self(void* q, void* el, double tol, int trg_dot_prod, double* M, int64_t nM) {
  SCTLJL_TRY
  sctl::Vector<sctl::Matrix<double>> M_lst;
  quad(q).self(M_lst, qel(el), tol, trg_dot_prod != 0);
  int64_t off = 0;
  for (sctl::Long e = 0; e < M_lst.Dim(); e++) {
    const int64_t n = M_lst[e].Dim(0) * M_lst[e].Dim(1);
    if (off + n > nM) throw Error("sctljl: self-interaction matrices have an unexpected size");
    if (n) std::copy(M_lst[e].begin(), M_lst[e].begin() + n, M + off);
    off += n;
  }
  if (off != nM) throw Error("sctljl: self-interaction matrices have an unexpected size");
  SCTLJL_CATCH
}

// Accurate blocks for a near list (ptr, idx as from sctljl_near_list): element e to its near targets, in parallel.
// Block e is row-major (nnode*SrcDim) x (ntrg_e*TrgDim_out), the blocks consecutive in element order (nout doubles).
// A target that coincides with a node of the element gets that node's self-interaction row, as in
// BoundaryIntegralOp (with the element's own normal for trg_normal_dot_prod).
SCTLJL_EXPORT int sctljl_quad_near_blocks(void* q, void* el, const int64_t* ptr, const int64_t* idx, const double* Xt,
                                          int64_t nt, const double* Xn, int64_t nn, double tol, double* out, int64_t nout) {
  SCTLJL_TRY
  const QEL& L = qel(el);
  const QuadBase& Q = quad(q);
  const sctl::Long nelem = L.Size();
  const sctl::Long nnode = (sctl::Long)L.Order() * L.Order();
  const bool dot = (nn > 0);
  if (dot && nn != nt) throw Error("sctljl: need one target normal per target");
  const sctl::Long dof = dot ? Q.trg_dim() / 3 : Q.trg_dim();
  const sctl::Long N0 = nnode * Q.src_dim();
  VecL off(nelem + 1);
  off[0] = 0;
  for (sctl::Long e = 0; e < nelem; e++) off[e+1] = off[e] + N0 * (ptr[e+1] - ptr[e]) * dof;
  if (off[nelem] != nout) throw Error("sctljl: output buffer has the wrong size");
  for (sctl::Long k = 0; k < ptr[nelem]; k++) if (idx[k] < 0 || idx[k] >= nt) throw Error("sctljl: target index out of range");

  VecD Xsurf;
  L.GetNodeCoord(&Xsurf, nullptr, nullptr);
  // node of element e that target t coincides with, or -1
  const auto on_node = [&](sctl::Long e, sctl::Long t) {
    for (sctl::Long i = 0; i < nnode; i++) {
      double r2 = 0;
      for (int k = 0; k < 3; k++) {
        const double d = Xt[3*t+k] - Xsurf[(e*nnode+i)*3+k];
        r2 += d * d;
      }
      if (r2 == 0) return i;
    }
    return (sctl::Long)-1;
  };
  bool any_on_node = false;
  #pragma omp parallel for schedule(dynamic) reduction(||: any_on_node)
  for (sctl::Long e = 0; e < nelem; e++)
    for (sctl::Long k = ptr[e]; k < ptr[e+1]; k++) any_on_node = any_on_node || (on_node(e, idx[k]) >= 0);
  sctl::Vector<sctl::Matrix<double>> K_self;
  if (any_on_node) Q.self(K_self, L, tol, dot);

  std::string err;
  #pragma omp parallel for schedule(dynamic)
  for (sctl::Long e = 0; e < nelem; e++) {
    const sctl::Long ntrg = ptr[e+1] - ptr[e];
    if (!ntrg) continue;
    try {
      sctl::Matrix<double> B(N0, ntrg * dof, out + off[e], false); // row-major view of block e
      sctl::Vector<sctl::Long> off_node(ntrg);
      VecD Xt_e, Xn_e;
      for (sctl::Long k = 0; k < ntrg; k++) {
        const sctl::Long t = idx[ptr[e] + k];
        off_node[k] = on_node(e, t);
        if (off_node[k] < 0) {
          for (int d = 0; d < 3; d++) Xt_e.PushBack(Xt[3*t+d]);
          if (dot) for (int d = 0; d < 3; d++) Xn_e.PushBack(Xn[3*t+d]);
        }
      }
      sctl::Matrix<double> M;
      if (Xt_e.Dim()) Q.near(M, L, e, Xt_e, Xn_e, tol);
      for (sctl::Long k = 0, j = 0; k < ntrg; k++) {
        for (sctl::Long l = 0; l < N0; l++) {
          for (sctl::Long k1 = 0; k1 < dof; k1++) {
            B[l][k*dof+k1] = (off_node[k] >= 0) ? K_self[e][l][off_node[k]*dof+k1] : M[l][j*dof+k1];
          }
        }
        if (off_node[k] < 0) j++;
      }
    } catch (const std::exception& ex) {
      #pragma omp critical
      if (err.empty()) err = ex.what();
    }
  }
  if (!err.empty()) throw Error(err);
  SCTLJL_CATCH
}
