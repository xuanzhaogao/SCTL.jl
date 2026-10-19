// Point trees: sctl::PtTree<double, DIM> for DIM = 1, 2, 3 (MPI-free, Comm::Self()).
//
// Every function takes the tree handle and its dimension `dim`; the handle points to a PtTree<double, dim>.
// Coordinates are AoS (x1, y1, z1, x2, ...) in [0,1)^dim. Node indices returned here are 0-based, -1 = none.
#include "sctljl.hpp"

using namespace sctljl;

namespace {
  template <class F> void with_tree(void* h, int dim, F&& f) {
    if (!h) throw Error("sctljl: null tree handle");
    switch (dim) {
      case 1: f(*static_cast<sctl::PtTree<double, 1>*>(h)); break;
      case 2: f(*static_cast<sctl::PtTree<double, 2>*>(h)); break;
      case 3: f(*static_cast<sctl::PtTree<double, 3>*>(h)); break;
      default: throw Error("sctljl: tree dimension must be 1, 2 or 3");
    }
  }
  template <class T> constexpr sctl::Integer tree_dim() { return std::decay_t<T>::Dim(); }
}

SCTLJL_EXPORT int sctljl_pttree_new(void** out, int dim) {
  SCTLJL_TRY
  *out = nullptr;
  switch (dim) {
    case 1: *out = new sctl::PtTree<double, 1>(); break;
    case 2: *out = new sctl::PtTree<double, 2>(); break;
    case 3: *out = new sctl::PtTree<double, 3>(); break;
    default: throw Error("sctljl: tree dimension must be 1, 2 or 3");
  }
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_pttree_free(void* h, int dim) {
  switch (dim) {
    case 1: delete static_cast<sctl::PtTree<double, 1>*>(h); break;
    case 2: delete static_cast<sctl::PtTree<double, 2>*>(h); break;
    case 3: delete static_cast<sctl::PtTree<double, 3>*>(h); break;
    default: break;
  }
}

// coord: n points (n*dim doubles). periodicity: sctl::Periodicity bitmask (bit k = axis k periodic).
SCTLJL_EXPORT int sctljl_pttree_update_refinement(void* h, int dim, const double* coord, int64_t n, int64_t M, int balance21,
                                                  int per, int halo_size) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) {
    t.UpdateRefinement(view(coord, n * dim), M, balance21 != 0, sctljl::periodicity(per), halo_size);
  });
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_pttree_add_particles(void* h, int dim, const char* name, const double* coord, int64_t n) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.AddParticles(name, view(coord, n * dim)); });
  SCTLJL_CATCH
}

// data: len doubles (dof per particle of the group, in the caller's particle order).
SCTLJL_EXPORT int sctljl_pttree_add_particle_data(void* h, int dim, const char* data_name, const char* particle_name,
                                                  const double* data, int64_t len) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.AddParticleData(data_name, particle_name, view(data, len)); });
  SCTLJL_CATCH
}

// Allocate `dof` unwritten values per particle (in tree order; fill with sctljl_pttree_set_data).
SCTLJL_EXPORT int sctljl_pttree_alloc_particle_data(void* h, int dim, const char* data_name, const char* particle_name, int64_t dof) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.AddParticleData(data_name, particle_name, (Long)dof); });
  SCTLJL_CATCH
}

// Particle data scattered back to the caller's original particle order (an f64 handle).
SCTLJL_EXPORT int sctljl_pttree_get_particle_data(void* h, int dim, const char* data_name, void* data) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.GetParticleData(out<double>(data), data_name); });
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_pttree_delete_particle_data(void* h, int dim, const char* data_name) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.DeleteParticleData(data_name); });
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_pttree_num_nodes(void* h, int dim, int64_t* n) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { *n = t.GetNodeMID().Dim(); });
  SCTLJL_CATCH
}

// Per node i (n = number of nodes, buffers caller-allocated):
//   coord[i*dim + k]   lower corner of the box, depth[i]   (box side 2^-depth),
//   leaf[i], ghost[i]  (0/1), parent[i],
//   child[i*2^dim + c] (c = sum_k b_k 2^k, b_k = upper half along axis k),
//   nbr[i*3^dim + j]   (j = sum_k (o_k + 1) 3^k, o_k in {-1, 0, 1} = offset along axis k; same depth),
// all indices 0-based, -1 = none.
SCTLJL_EXPORT int sctljl_pttree_node_info(void* h, int dim, int64_t n, double* coord, int64_t* depth, uint8_t* leaf, uint8_t* ghost,
                                          int64_t* parent, int64_t* child, int64_t* nbr) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) {
    constexpr sctl::Integer DIM = tree_dim<decltype(t)>();
    constexpr Long NCHILD = 1 << DIM;
    constexpr Long NNBR = sctl::pow<DIM, Long>(3);
    const auto& mid = t.GetNodeMID();
    const auto& attr = t.GetNodeAttr();
    const auto& lst = t.GetNodeLists();
    if (mid.Dim() != n || attr.Dim() != n || lst.Dim() != n) throw Error("sctljl: tree node count mismatch");
    for (Long i = 0; i < n; i++) {
      double c[DIM];
      mid[i].Coord(c);
      for (sctl::Integer k = 0; k < DIM; k++) coord[i * DIM + k] = c[k];
      depth[i] = mid[i].Depth();
      leaf[i] = attr[i].Leaf;
      ghost[i] = attr[i].Ghost;
      parent[i] = lst[i].parent;
      for (Long j = 0; j < NCHILD; j++) child[i * NCHILD + j] = lst[i].child[j];
      for (Long j = 0; j < NNBR; j++) nbr[i * NNBR + j] = lst[i].nbr[j];
    }
  });
  SCTLJL_CATCH
}

// [begin, end): 0-based owned node range.
SCTLJL_EXPORT int sctljl_pttree_owned_range(void* h, int dim, int64_t* begin, int64_t* end) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) {
    Long b = 0, e = 0;
    t.GetOwnedRange(b, e);
    *begin = b;
    *end = e;
  });
  SCTLJL_CATCH
}

// Node data `name` (stored as doubles; concatenated in node order) and its per-node item counts.
SCTLJL_EXPORT int sctljl_pttree_get_data(void* h, int dim, const char* name, void* data, void* cnt) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) {
    VecD d;
    VecL c;
    t.GetData(d, c, name);
    out<double>(data) = d;  // deep copies of the views
    out<Long>(cnt) = c;
  });
  SCTLJL_CATCH
}

// Overwrite the values of node data `name` in place (same layout and length as sctljl_pttree_get_data).
SCTLJL_EXPORT int sctljl_pttree_set_data(void* h, int dim, const char* name, const double* data, int64_t len) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) {
    VecD d;
    VecL c;
    t.GetData(d, c, name);
    if (d.Dim() != len) throw Error("sctljl: node data size mismatch (" + std::to_string(len) + " != " + std::to_string(d.Dim()) + ")");
    for (Long i = 0; i < len; i++) d[i] = data[i];
  });
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_pttree_write_tree_vtk(void* h, int dim, const char* fname, int show_ghost) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.WriteTreeVTK(fname, show_ghost != 0); });
  SCTLJL_CATCH
}

SCTLJL_EXPORT int sctljl_pttree_write_particle_vtk(void* h, int dim, const char* fname, const char* data_name, int show_ghost) {
  SCTLJL_TRY
  with_tree(h, dim, [&](auto& t) { t.WriteParticleVTK(fname, data_name, show_ghost != 0); });
  SCTLJL_CATCH
}

// 1 if every x[i] lies in [0, 1) (NaN fails), else 0. Parallel: the input check is O(N) and runs on every
// refinement and particle insertion.
SCTLJL_EXPORT int sctljl_unit_box_ok(const double* x, int64_t n) {
  int ok = 1;
  #pragma omp parallel for reduction(&& : ok) if (n > 100000)
  for (int64_t i = 0; i < n; i++) ok = ok && (x[i] >= 0 && x[i] < 1);
  return ok;
}
