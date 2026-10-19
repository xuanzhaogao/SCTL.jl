// BoundaryIntegralOp instantiations (helmholtz kernels); see capi_bie.cpp.
#include "sctljl.hpp"
#include "capi_bie.hpp"

namespace sctljl {
namespace bie {

  BIOpBase* new_op_helmholtz(int id, double ctx, bool trg_dot_prod) {
    switch (id) {
      case 7:  return new BIOp<sctl::Helmholtz3D_FxU>(ctx, trg_dot_prod);
      case 8:  return new BIOp<sctl::Helmholtz3D_DxU>(ctx, trg_dot_prod);
      case 9:  return new BIOp<sctl::Helmholtz3D_FxdU>(ctx, trg_dot_prod);
      default:  return nullptr;
    }
  }

  QuadBase* new_quad_helmholtz(int id, double ctx) {
    switch (id) {
      case 7:  return new Quad<sctl::Helmholtz3D_FxU>(ctx);
      case 8:  return new Quad<sctl::Helmholtz3D_DxU>(ctx);
      case 9:  return new Quad<sctl::Helmholtz3D_FxdU>(ctx);
      default:  return nullptr;
    }
  }

} // namespace bie
} // namespace sctljl
