# SCTL.jl

Julia wrapper for [SCTL](https://github.com/dmalhotra/SCTL), the Scientific Computing Template Library
(header-only C++). This is the MPI-free build and works in `Float64` only.

It works through a thin C-ABI shim (`csrc/`) compiled against the SCTL headers into `libsctljl.so`, which Julia calls
with `ccall`. SCTL errors and failed assertions do not abort the process. They are raised as `SCTL.SCTLError`, and
exceptions thrown inside Julia callbacks (GMRES operators, ODE right-hand sides, integrands) are re-thrown unchanged.

## Installation

Requirements: a C++17 compiler with OpenMP (GCC ≥ 9) and `make`. SCTL itself comes as a git submodule
(`deps/SCTL`, pinned to the tested commit of https://github.com/dmalhotra/SCTL):

```bash
git clone --recursive git@github.com:xuanzhaogao/SCTL.jl.git   # or, in an existing clone: git submodule update --init
```
```julia
using Pkg; Pkg.develop(path="/path/to/SCTL.jl"); Pkg.build("SCTL")
```

To build against another SCTL checkout, set `ENV["SCTL_ROOT"]` before `Pkg.build`.

The build takes a few minutes, since the boundary-integral operators are template-heavy (about 2 min with 8 cores and
around 1.7 GB of RAM per compiler process). Build options are environment variables read by `deps/build.jl`:

| variable | default | purpose |
|:-|:-|:-|
| `SCTL_ROOT` | `deps/SCTL` (submodule), then `../SCTL` | SCTL checkout |
| `CXX` | `g++` | compiler |
| `SCTL_JL_MARCH` | `-march=native` | e.g. `-march=x86-64-v3` for a library shared by machines with different CPUs |
| `SCTL_JL_CXXFLAGS` | | extra flags, e.g. `-DSCTL_HAVE_FFTW` |
| `SCTL_JL_LDFLAGS` | | extra link flags, e.g. `-lfftw3` |

By default no optional dependencies are used. SCTL falls back to its own FFT, GEMM and SVD. `SCTL.has_feature(:FFTW)`
reports what was compiled in. `SCTL_JL_LIB=/path/libsctljl.so` points the package at another build; it is read when
the package is precompiled, so changing it later needs a recompile.

Tests: `julia --project=. test/runtests.jl`. This is about 1700 tests and takes about 40 s.

## What is wrapped

| area | Julia API | SCTL |
|:-|:-|:-|
| quadrature | `chebquad`, `legquad`, `legendre_poly`, `interp_quad_rule`, `interp_quad_rules` | `ChebQuadRule`, `LegQuadRule`, `InterpQuadRule` |
| interpolation | `lagrange_interp_matrix`, `lagrange_interp`, `lagrange_derivative` | `LagrangeInterp` |
| linear solver | `gmres`, `KrylovPrecond`, `apply!` | `GMRES`, `KrylovPrecond` |
| ODE solver | `SDC`, `sdc_step`, `sdc_solve`, `sdc_nodes`, `order` | `SDC` (spectral deferred correction) |
| kernels | `Laplace3D_FxU`, `Stokes3D_DxU`, `Helmholtz3D_FxU(k)`, … (16 kernels), `kernel_matrix`, `kernel_eval[!]`, `src_dim`, `trg_dim`, … | `kernel_functions.hpp`, `GenericKernel` |
| boundary integrals | `QuadElemList`, `BoundaryIntegralOp`, `add_elem_list!`, `set_target_coord!`, `compute_potential[!]`, … | `BoundaryIntegralOp`, `QuadElemList` (experimental) |
| spherical harmonics | `grid2shc`, `shc2grid`, `shceval`, `shc2pole`, vector variants, `laplace_eval_sl/dl`, `stokes_eval_sl/dl/kl/kself`, `SHCArrange` | `SphericalHarmonics` |
| trees | `PtTree{DIM}` (DIM = 1, 2, 3), `update_refinement!`, `add_particles!`, `add_particle_data!`, `get_particle_data`, `node_info`, `get_data`, `write_tree_vtk`, … | `Tree`, `PtTree`, `Morton` |
| FFT | `FFTPlan(:r2c / :c2c / :c2c_inv / :c2r, dims; howmany)`, `execute[!]` | `FFT` |
| OpenMP | `SCTL.get_num_threads`, `SCTL.set_num_threads` | |

Not wrapped: MPI (`Comm`), `ParticleFMM`/PVFMM, the containers (`Vector`, `Matrix`, `Tensor`, `Vec`, memory
managers; use Julia arrays), `VTUData`, `Profile`, the legacy `ChebBasis`, the GPU tree.

Every exported function has a docstring (`?gmres`) that gives the exact array layouts. The conventions are:

- Points are `DIM × N` matrices, and per-point data (densities, potentials) is `dof × N`.
- SCTL's row-major matrices are presented with the shape that makes `y = M * x` work in Julia, e.g.
  `lagrange_interp_matrix(src, trg)` is `length(trg) × length(src)`.
- Tree, element and node indices are 1-based, with `0` meaning "none".
- Complex values (Helmholtz, FFT) are interleaved `(re, im)`. The FFT takes and returns `ComplexF64` arrays, is
  unnormalized, and uses `exp(-2πi…)` for the forward transform.

## Example

```julia
using SCTL, LinearAlgebra

x, w = legquad(10)                                    # Gauss-Legendre on [0, 1]
sol = gmres((y, v) -> mul!(y, A, v), b; tol=1e-12)    # matrix-free GMRES -> (x=..., iters=...)
u, t, err = sdc_solve(SDC(8), (du, u) -> (du .= -u), [1.0], 0.1, 1.0, 1e-10)
U = kernel_eval(Laplace3D_FxU(), Xs, F, Xt)            # direct N-body sum, F is 1×Ns, U is 1×Nt

elems = QuadElemList((e, u, v) -> surface_point(e, u, v), 8, nelem)
op = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=1e-8)
add_elem_list!(op, elems)
set_target_coord!(op, targets)                        # default targets: the surface nodes
U = compute_potential(op, density)
```

See [`examples/quickstart.jl`](examples/quickstart.jl) for a runnable tour of every module.

## Performance

`bench/` compares every module against the same SCTL calls made from C++ on the same inputs, built with the same
flags (`bench/run.sh`; `bench/cluster_bench.slurm` runs it on an exclusive Rusty node). Results from 2026-10-18 on
genoa (AMD EPYC 9474F, 96 cores) and icelake (Xeon Platinum 8362, 64 cores), as the time ratio Julia / C++:

| | ratio |
|:-|:-|
| compute-bound calls on 1 thread (N-body sums, kernel matrices, boundary integrals, spherical harmonics, FFT, trees, GMRES on 200k unknowns) | 0.93–1.07 |
| the same on 16–96 threads | 1.0 within the timing noise of ~1 ms threaded calls (single runs ±10–40%) |
| `grid2shc` on 16–96 threads, median of 400 calls, wrapper vs plain C++ loop on the same node | 0.98–1.08 |

The fixed costs are:
- **Per call:** about 0.2–0.3 µs, e.g. `legquad(16)` takes 0.8 µs vs 0.6 µs.
- **Per Julia callback invocation (GMRES operator, SDC right-hand side):** about 40–60 ns. A complete SDC step on a
  4-element state takes 3.6 µs vs 1.3 µs; on a 10⁵-element state it costs 1.1–1.2× native.
- **Fresh outputs:** functions that allocate a large output (e.g. `kernel_matrix`) pay for fresh memory on every call.
  Use the in-place form (`kernel_matrix!`, `kernel_eval!`, `compute_potential!`, `execute!`) in loops; with a reused
  buffer `kernel_matrix!` matches C++ at every thread count.

Callback arguments are reused buffers for vectors of up to 512 entries and zero-copy views of SCTL's memory for longer
ones.

## Notes and limitations

- **Threads.** SCTL uses OpenMP; control it with `OMP_NUM_THREADS` or `SCTL.set_num_threads`. On load the package
  sets `OMP_WAIT_POLICY=PASSIVE` (unless you set it already), so idle OpenMP threads don't spin against Julia's own.
  Julia callbacks are only ever invoked on the calling thread.
- **No FMM.** Without PVFMM, `BoundaryIntegralOp`'s far field is a direct O(N_nodes · N_targets) sum, and
  periodicity is unavailable.
- **Supported kernels.** `BoundaryIntegralOp` supports Laplace FxU/DxU/FxdU/DxdU, Helmholtz FxU/DxU/FxdU and Stokes
  FxU/DxU/FxT (`bie_supported_kernels()`). `QuadElemList` orders must be 4, 8, 12, 16 or 20 for use in an operator.
  `AddElemList` copies the list, so set the quadrature scheme before adding it.
- **Spherical-harmonic memory.** The first spherical-harmonic call allocates about 200 MB of lookup tables inside
  SCTL; `sph_clear_cache()` frees them. Don't call it while another thread uses spherical harmonics.
- **Low-order SDC.** Adaptive solves with order < 6 can fail to reach tight tolerances; this is SCTL's step
  heuristic. Use a higher order or `continue_with_errors=true`.
- **Assertion coverage.** A few SCTL checks run inside OpenMP parallel regions; an exception there would terminate the
  process. The Julia side validates arguments up front so those checks don't fire.
- **Workaround for an SCTL bug.** SCTL leaves the unused coefficients of the `SHC_ALL` layout uninitialized; the shim
  zeroes them.
