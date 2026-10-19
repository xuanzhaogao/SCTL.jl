# Conventions

These conventions apply to every module of SCTL.jl.

```@setup conv
using SCTL, LinearAlgebra
```

## Array layouts

SCTL stores data in flat C++ arrays in row-major order. SCTL.jl passes Julia arrays to SCTL without copying them,
so each function documents the Julia shape whose column-major memory is exactly SCTL's layout:

| data | SCTL (C++) | Julia |
|:--|:--|:--|
| points in ``d`` dimensions | `{x₁, y₁, z₁, x₂, y₂, z₂, …}` (array of structs) | `d × N` matrix (a flat vector of length `d*N` also works) |
| per-point data with `dof` components | `{f₁¹, …, f₁ᵈᵒᶠ, f₂¹, …}` | `dof × N` matrix |
| row-major matrix ``M`` of size ``m × n`` | `M[i][j]` at `i*n + j` | the same memory read as an ``n × m`` matrix, i.e. ``M^T`` |
| complex numbers | interleaved `(re, im)` | `ComplexF64` arrays (FFT), or `2k` real components (Helmholtz kernels) |

When SCTL returns a matrix, SCTL.jl gives it the shape that makes `y = M * x` correct in Julia. For example,
[`lagrange_interp_matrix(src, trg)`](@ref lagrange_interp_matrix) is `length(trg) × length(src)`, and
[`kernel_matrix`](@ref) is `(trg_dim*Nt) × (src_dim*Ns)`. Each docstring states the exact shape.

Indices returned to Julia, such as tree node and element indices, are **1-based**, and `0` means "none". Inputs of
any real element type are accepted; non-`Float64` arrays are converted, which makes a copy.

```@example conv
X = rand(3, 5)                 # five points in 3D, one per column
vec(X)[1:3] == X[:, 1]         # memory order = SCTL's {x1, y1, z1, x2, ...}
```

## Errors

Arguments are checked on the Julia side first, and violations throw the usual `ArgumentError`, `DimensionMismatch`
or `BoundsError`. A failed check inside SCTL, an `SCTL_ASSERT` or `SCTL_ERROR`, would normally abort the process.
SCTL.jl compiles SCTL so that these throw a C++ exception instead, which surfaces in Julia as [`SCTL.SCTLError`](@ref).
The Julia session survives:

```@example conv
try
    sdc_solve(SDC(4), (du, u) -> (du .= -u), [1.0], 0.1, 1.0, 1e-14)   # tolerance out of reach for order 4
catch e
    e
end
```

An exception thrown inside a Julia callback (a GMRES operator, an ODE right-hand side, an integrand) stops the SCTL
routine and is rethrown with its original type:

```@example conv
try
    gmres((y, x) -> throw(DomainError(x[1], "bad input")), ones(3))
catch e
    typeof(e)
end
```

A few SCTL checks run inside OpenMP parallel regions, where an exception cannot propagate and would terminate the
process. The Julia-side argument checks exist so that these checks never fire.

## Objects and memory

Stateful SCTL objects are Julia `mutable struct`s that own a C++ object:
- [`SDC`](@ref) and [`KrylovPrecond`](@ref)
- [`FFTPlan`](@ref)
- [`PtTree`](@ref)
- [`QuadElemList`](@ref) and [`BoundaryIntegralOp`](@ref)

A finalizer frees the C++ object, and `close(obj)` frees it immediately; this is useful for large trees and operators.
Using an object after `close` throws an `ArgumentError` instead of crashing. While SCTL works on an object, the
object is kept alive (GC-rooted), even if your code holds no other reference to it.

## Threads

SCTL parallelizes with OpenMP:

- Set the number of threads with the environment variable `OMP_NUM_THREADS` before starting Julia, or at run time
  with [`SCTL.set_num_threads`](@ref). [`SCTL.get_num_threads`](@ref) reports the current value.
- SCTL calls BLAS (Julia's own, see below) inside its OpenMP loops, where a multi-threaded BLAS would oversubscribe
  the cores. So while an SCTL routine that can use BLAS runs (the FMM, boundary integral operators, spherical
  harmonic transforms, quadrature construction), Julia's BLAS is limited to one thread; the previous count
  (`BLAS.get_num_threads()`) is restored when it returns. Other Julia tasks calling BLAS at that time also get one
  thread.
- When it loads, SCTL.jl sets `OMP_WAIT_POLICY=PASSIVE` unless you have set it yourself. Otherwise idle OpenMP threads
  spin after each parallel region and compete with Julia's own threads. Benchmarks that call SCTL in a tight loop
  may run faster with `OMP_WAIT_POLICY=ACTIVE`.
- Julia's threads (`julia -t`) and OpenMP threads are separate pools. If both are busy at once, the machine is
  oversubscribed.

Julia callbacks are always invoked on the thread that called the SCTL routine, never from OpenMP worker threads.
Calling independent SCTL routines from several Julia tasks at the same time is supported. One exception: a callback
that yields to the scheduler (I/O, `sleep`, channels) while another task on the same thread enters SCTL can upset
SCTL's per-thread scratch memory, so keep callbacks free of task switches.

## Callbacks

[`gmres`](@ref), [`sdc_step`](@ref), [`sdc_solve`](@ref) and [`interp_quad_rule`](@ref) take Julia functions:

- **Fast dispatch.** Each callback goes through a trampoline specialized on your function's type, so it costs
  about 50 ns plus the work you do.
- **Argument vectors** are valid only during the call. Vectors of up to 512 entries are reused Julia buffers.
  Longer ones are views of SCTL's own memory, which means no copies. In both cases:
  - write only to the output argument (do not modify the input);
  - do not keep references to either argument after returning.
- **SCTL's console output** (per-step logs, rule summaries) is suppressed unless you pass `verbose = true`.

## Optional dependencies

By default the library includes:
- **PVFMM**, compiled from its headers without MPI. It powers [`ParticleFMM`](@ref) and the far field of
  [`BoundaryIntegralOp`](@ref).
- **BLAS and LAPACK**, Julia's own (`LinearAlgebra.BLAS`: OpenBLAS, or MKL etc. if you load one) through
  libblastrampoline. SCTL's 32-bit-integer calls are forwarded to the 64-bit interface, so no other BLAS library is
  needed. (An earlier version used `OpenBLAS32_jll`, which is built for at most 32 threads and crashed the FMM on
  larger nodes.) `SCTL.has_feature(:BLAS_LBT)` reports this configuration. BLAS matters mostly for the one-time FMM
  precomputation: without it, Stokes operators take minutes instead of seconds.

FFTW is not included by default; SCTL uses its own FFT. MPI is never used. [`SCTL.has_feature`](@ref) reports what the
build includes:

```@example conv
(; PVFMM = SCTL.has_feature(:PVFMM), BLAS = SCTL.has_feature(:BLAS), LAPACK = SCTL.has_feature(:LAPACK),
   BLAS_LBT = SCTL.has_feature(:BLAS_LBT), FFTW = SCTL.has_feature(:FFTW), MPI = SCTL.has_feature(:MPI))
```

The FMM stores precomputed operators on disk, in [`SCTL.fmm_cache_dir`](@ref); see
[Fast multipole method](@ref).

## Reference

```@docs
SCTL.SCTLError
SCTL.get_num_threads
SCTL.set_num_threads
SCTL.has_feature
```
