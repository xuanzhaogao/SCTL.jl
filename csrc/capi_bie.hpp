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

  // Factories, one per translation unit: return a new op for kernel id `id` (ids as in capi_kernels.cpp), or
  // nullptr if that translation unit does not instantiate this kernel.
  BIOpBase* new_op_laplace_sl(int id, double ctx, bool trg_dot_prod);   // capi_bie.cpp
  BIOpBase* new_op_laplace_grad(int id, double ctx, bool trg_dot_prod); // capi_bie_laplace.cpp
  BIOpBase* new_op_stokes(int id, double ctx, bool trg_dot_prod);       // capi_bie_stokes.cpp
  BIOpBase* new_op_helmholtz(int id, double ctx, bool trg_dot_prod);    // capi_bie_helmholtz.cpp

} // namespace bie
} // namespace sctljl

#endif // SCTLJL_CAPI_BIE_HPP
