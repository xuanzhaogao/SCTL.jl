// BoundaryIntegralOp instantiations (stokes kernels); see capi_bie.cpp.
#include "sctljl.hpp"
#include "capi_bie.hpp"

namespace sctljl {
namespace bie {

  BIOpBase* new_op_stokes(int id, double ctx, bool trg_dot_prod) {
    switch (id) {
      case 11: return new BIOp<sctl::Stokes3D_FxU>(ctx, trg_dot_prod);
      case 12: return new BIOp<sctl::Stokes3D_DxU>(ctx, trg_dot_prod);
      case 13: return new BIOp<sctl::Stokes3D_FxT>(ctx, trg_dot_prod);
      default:  return nullptr;
    }
  }

} // namespace bie
} // namespace sctljl
