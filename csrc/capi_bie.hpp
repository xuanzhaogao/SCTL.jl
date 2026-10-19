// Private header shared by the capi_bie*.cpp translation units (BoundaryIntegralOp instantiations are split
// across them to keep compile time and memory per TU moderate). Include after sctljl.hpp.
#ifndef SCTLJL_CAPI_BIE_HPP
#define SCTLJL_CAPI_BIE_HPP

#include "sctljl.hpp"
#include "fmm_schemes.hpp"

// All templates / inline functions, so including it in several translation units is fine.
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

namespace sctljl {
namespace bie {

  using QEL = sctl::QuadElemList<double>;

  struct BIOpBase {
    virtual ~BIOpBase() {}
    virtual void add(const QEL& elist, const std::string& name) = 0;
    virtual void del(const std::string& name) = 0;
    virtual void set_trg_coord(const VecD& X) = 0;
    virtual void set_trg_normal(const VecD& Xn) = 0;
    virtual void set_accuracy(double tol) = 0;
    virtual double accuracy() const = 0;
    virtual Long dim(int k) const = 0;
    virtual void setup() const = 0;
    virtual void clear_setup() const = 0;
    virtual void compute(VecD& U, const VecD& F) const = 0;
    virtual void sqrt_scaling(VecD& U) const = 0;
    virtual void inv_sqrt_scaling(VecD& U) const = 0;
  };

  template <class Ker> struct BIOp final : BIOpBase {
    double ctx; // must be initialized before `op`, which copies a kernel pointing to it
    sctl::BoundaryIntegralOp<double, Ker> op;

    static Ker make_kernel(double* ctx) {
      Ker ker;
      if constexpr (kernel_uses_ctx<Ker>()) ker.SetCtxPtr(ctx); // others: none, or the FMM is not used
      return ker;
    }
    BIOp(double ctx_, bool trg_dot_prod) : ctx(ctx_), op(make_kernel(&ctx), trg_dot_prod) {
      if constexpr (FMMScheme<Ker>::supported) { // far field by PVFMM (when built with it) for 40000+ targets
        using S = FMMScheme<Ker>;
        const Ker ker = make_kernel(&ctx);
        op.SetFMMKer(typename S::s2m(), typename S::s2m(), ker, typename S::m2m(), typename S::m2m(), typename S::m2t(),
                     typename S::l2l(), typename S::l2t());
      }
    }
    BIOp(const BIOp&) = delete;

    void add(const QEL& elist, const std::string& name) override { op.AddElemList(elist, name); } // copies elist
    void del(const std::string& name) override { op.DeleteElemList(name); }
    void set_trg_coord(const VecD& X) override { op.SetTargetCoord(X); }
    void set_trg_normal(const VecD& Xn) override { op.SetTargetNormal(Xn); }
    void set_accuracy(double tol) override { op.SetAccuracy(tol); }
    double accuracy() const override { return op.GetAccuracy(); }
    Long dim(int k) const override { return op.Dim(k); }
    void setup() const override { op.Setup(); }
    void clear_setup() const override { op.ClearSetup(); }
    void compute(VecD& U, const VecD& F) const override { op.ComputePotential(U, F); }
    // SqrtScaling/InvSqrtScaling read the node counts without setting them up; Dim() sets them up.
    void sqrt_scaling(VecD& U) const override { op.Dim(0); op.SqrtScaling(U); }
    void inv_sqrt_scaling(VecD& U) const override { op.Dim(0); op.InvSqrtScaling(U); }
  };

  // The singular and near-singular quadratures of a QuadElemList for one kernel, without an operator: the pieces of
  // BoundaryIntegralOp's near-field setup (capi_bie.cpp, "near-field pieces").
  struct QuadBase {
    virtual ~QuadBase() {}
    virtual int src_dim() const = 0;
    virtual int trg_dim() const = 0;
    // M: (nnode*SrcDim) x (Ntrg*TrgDim_out) from element e to the targets Xt (with normals Xn for trg_dot_prod)
    virtual void near(sctl::Matrix<double>& M, const QEL& el, Long e, const VecD& Xt, const VecD& Xn, double tol) const = 0;
    // M_lst[e]: (nnode*SrcDim) x (nnode*TrgDim_out) from element e to its own nodes
    virtual void self(sctl::Vector<sctl::Matrix<double>>& M_lst, const QEL& el, double tol, bool trg_dot_prod) const = 0;
  };

  template <class Ker> struct Quad final : QuadBase {
    double ctx; // before `ker`, which points to it
    Ker ker;
    explicit Quad(double ctx_) : ctx(ctx_), ker(BIOp<Ker>::make_kernel(&ctx)) {}
    Quad(const Quad&) = delete;
    int src_dim() const override { return Ker::SrcDim(); }
    int trg_dim() const override { return Ker::TrgDim(); }
    void near(sctl::Matrix<double>& M, const QEL& el, Long e, const VecD& Xt, const VecD& Xn, double tol) const override {
      QEL::NearInterac<Ker>(M, Xt, Xn, ker, tol, e, &el);
    }
    void self(sctl::Vector<sctl::Matrix<double>>& M_lst, const QEL& el, double tol, bool trg_dot_prod) const override {
      if (M_lst.Dim() != el.Size()) M_lst.ReInit(el.Size()); // SCTL sizes each matrix, not the list
      QEL::SelfInterac<Ker>(M_lst, ker, tol, trg_dot_prod, &el);
    }
  };

  // Factories, one per translation unit: return a new op (quadrature) for kernel id `id` (ids as in
  // capi_kernels.cpp), or nullptr if that translation unit does not instantiate this kernel.
  BIOpBase* new_op_laplace_sl(int id, double ctx, bool trg_dot_prod);   // capi_bie.cpp
  BIOpBase* new_op_laplace_grad(int id, double ctx, bool trg_dot_prod); // capi_bie_laplace.cpp
  BIOpBase* new_op_stokes(int id, double ctx, bool trg_dot_prod);       // capi_bie_stokes.cpp
  BIOpBase* new_op_helmholtz(int id, double ctx, bool trg_dot_prod);    // capi_bie_helmholtz.cpp
  QuadBase* new_quad_laplace_sl(int id, double ctx);
  QuadBase* new_quad_laplace_grad(int id, double ctx);
  QuadBase* new_quad_stokes(int id, double ctx);
  QuadBase* new_quad_helmholtz(int id, double ctx);

} // namespace bie
} // namespace sctljl

#endif // SCTLJL_CAPI_BIE_HPP
