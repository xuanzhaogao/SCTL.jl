# SCTL.jl

SCTL.jl is a Julia interface to [SCTL](https://github.com/dmalhotra/SCTL), the *Scientific Computing Template
Library* by Dhairya Malhotra. SCTL is a header-only C++ library with high-order numerical building blocks:

- quadrature rules
- spectral and ODE solvers
- spherical harmonics
- layer potentials and boundary integral operators
- adaptive particle trees
- FFTs

All of it is vectorized with SIMD and parallelized with OpenMP.

The wrapper exposes these components with Julia arrays in and Julia arrays out. Compute-heavy calls run at the same
speed as calling SCTL from C++ (see [Performance](@ref)). This is the MPI-free build and works in `Float64`.

## Installation

You need a C++17 compiler with OpenMP (GCC ≥ 9) and `make`. SCTL itself comes as a git submodule of the package.

```bash
git clone --recursive https://github.com/xuanzhaogao/SCTL.jl.git
```

```julia
using Pkg
Pkg.develop(path = "SCTL.jl")
Pkg.build("SCTL")      # compiles the C++ shim into SCTL.jl/deps/usr/lib/libsctljl.so (a few minutes)
```

These environment variables are read by `Pkg.build`:

| variable | default | purpose |
|:--|:--|:--|
| `SCTL_ROOT` | `deps/SCTL` (submodule), then `../SCTL` | the SCTL checkout to compile against |
| `CXX` | `g++` | C++ compiler |
| `SCTL_JL_MARCH` | `-march=native` | target CPU, e.g. `-march=x86-64-v3` when one build serves different machines |
| `SCTL_JL_PVFMM` | `1` | `0` builds without the FMM (the `deps/pvfmm` submodule); `PVFMM_ROOT` picks another checkout |
| `SCTL_JL_BLAS` | `1` | `0` builds without BLAS/LAPACK (from the `OpenBLAS32_jll` package) |
| `SCTL_JL_CXXFLAGS` | | extra flags, e.g. `-DSCTL_HAVE_FFTW` |
| `SCTL_JL_LDFLAGS` | | extra link flags, e.g. `-lfftw3` |

Run the tests with `julia --project=. test/runtests.jl` (about 1700 tests, under a minute).

## A first example

```@example index
using SCTL, LinearAlgebra

x, w = legquad(10)                       # 10-point Gauss-Legendre rule on [0, 1]
dot(w, exp.(x)) - (exp(1) - 1)           # quadrature error
```

```@example index
# Laplace single-layer potential of 1000 random unit charges, by direct summation
Xs = rand(3, 1000)                       # sources: 3 × N
Xt = [2.0 0 0; 0 2.0 0]'                 # targets: 3 × 2
U = kernel_eval(Laplace3D_FxU(), Xs, ones(1, 1000), Xt)
U ≈ [sum(1 / (4π * norm(Xt[:, t] - Xs[:, s])) for s in 1:1000) for t in 1:2]'
```

## What is wrapped

| SCTL component | Julia API | manual page |
|:--|:--|:--|
| `ChebQuadRule`, `LegQuadRule`, `InterpQuadRule`, `LagrangeInterp` | [`legquad`](@ref), [`interp_quad_rule`](@ref), [`lagrange_interp_matrix`](@ref), … | [Quadrature and interpolation](@ref) |
| `GMRES`, `KrylovPrecond` | [`gmres`](@ref), [`KrylovPrecond`](@ref) | [GMRES](@ref) |
| `SDC` | [`SDC`](@ref), [`sdc_step`](@ref), [`sdc_solve`](@ref) | [SDC ODE solver](@ref) |
| `kernel_functions.hpp`, `GenericKernel` | [`Laplace3D_FxU`](@ref), …, [`kernel_eval`](@ref), [`kernel_matrix`](@ref) | [Kernel functions](@ref) |
| `ParticleFMM` (PVFMM, without MPI) | [`ParticleFMM`](@ref), [`fmm_eval`](@ref) | [Fast multipole method](@ref) |
| `BoundaryIntegralOp`, `QuadElemList` | [`BoundaryIntegralOp`](@ref), [`QuadElemList`](@ref) | [Boundary integral operators](@ref) |
| `SphericalHarmonics` | [`grid2shc`](@ref), [`shc2grid`](@ref), [`shceval`](@ref), … | [Spherical harmonics](@ref) |
| `Tree`, `PtTree`, `Morton` | [`PtTree`](@ref), [`update_refinement!`](@ref), [`node_info`](@ref), … | [Point trees](@ref) |
| `FFT` | [`FFTPlan`](@ref), [`execute`](@ref) | [FFT](@ref) |

These are not wrapped:
- MPI (`Comm`). PVFMM is compiled in without MPI and runs in the Julia process.
- SCTL's containers (`Vector`, `Matrix`, `Tensor`, `Vec`) and memory managers; Julia arrays take their place.
- `VTUData`, `Profile`, the legacy `ChebBasis`, and the experimental GPU tree.

Read [Conventions](@ref) first: it explains the array layouts, errors, threads and callbacks that every module shares.

```@docs
SCTL
```
