// BoundaryIntegralOp instantiations (laplace kernels); see capi_bie.cpp.
#include "sctljl.hpp"
#include "capi_bie.hpp"

namespace sctljl {
namespace bie {

  BIOpBase* new_op_laplace_grad(int id, double ctx, bool trg_dot_prod) {
    switch (id) {
      case 2:  return new BIOp<sctl::Laplace3D_FxdU>(ctx, trg_dot_prod);
      case 4:  return new BIOp<sctl::Laplace3D_DxdU>(ctx, trg_dot_prod);
      default:  return nullptr;
    }
  }

} // namespace bie
} // namespace sctljl
