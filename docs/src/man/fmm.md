# Fast multipole method

[`ParticleFMM`](@ref) evaluates the same particle sums as [`kernel_eval`](@ref),

```math
U(x_t) = \sum_{s=1}^{N_s} K(x_t - x_s)\, F_s, \qquad t = 1, \dots, N_t,
```

in ``O(N_s + N_t)`` work instead of ``O(N_s N_t)``. It wraps SCTL's `ParticleFMM<double,3>`, whose fast path is
[PVFMM](https://github.com/dmalhotra/pvfmm), a kernel-independent FMM by Malhotra and Biros. SCTL.jl compiles PVFMM's
headers into its library **without MPI**: the FMM runs in the Julia process, parallelized with OpenMP, and you need no
MPI installation, no MPI.jl and no `mpirun`.

The same machinery gives [`BoundaryIntegralOp`](@ref) an FMM far field (see
[Boundary integral operators](@ref)).

```@setup fmm
using SCTL, LinearAlgebra, Printf
```

## Supported kernels

```@example fmm
fmm_supported_kernels()
```

For these kernels, SCTL.jl picks the FMM translation kernels that act on the equivalent densities:
- **Laplace kernels:** the Laplace single layer for every translation.
- **Stokes velocity kernels:** the Stokeslet plus a source term (`Stokes3D_FSxU`) for multipole expansions and the
  Stokeslet for local expansions.
- **`Stokes3D_FxT` (traction):** the Stokeslet for every translation.

The other kernels have no FMM path and give an `ArgumentError`; use [`kernel_eval`](@ref) for those. The Helmholtz
kernels, for example, need their wavenumber, which PVFMM does not pass to kernels.

## Tutorial

### One-shot evaluation

[`fmm_eval`](@ref) takes the same arguments as [`kernel_eval`](@ref). Here are 10⁵ charges and 10⁵ targets:

```@example fmm
Xs = rand(3, 100_000)
Xt = rand(3, 100_000)
F = randn(1, 100_000)

U = fmm_eval(Laplace3D_FxU(), Xs, F, Xt; digits = 8)

# check 100 targets against the direct sum
sample = 1:1000:100_000
Uref = kernel_eval(Laplace3D_FxU(), Xs, F, Xt[:, sample])
maximum(abs, U[:, sample] .- Uref) / maximum(abs, Uref)
```

The first evaluation for a kernel and accuracy also precomputes PVFMM's translation operators and saves them in
[`SCTL.fmm_cache_dir`](@ref). Later evaluations, including those in new Julia sessions, load them from there.

### Reusing points

A [`ParticleFMM`](@ref) object keeps its octree: the tree is rebuilt only when the points change. Solving an integral
equation with [`gmres`](@ref) applies the operator to many densities on fixed points, which is exactly that fast case.

```@example fmm
fmm = ParticleFMM(Stokes3D_FxU(); digits = 6)
set_sources!(fmm, Xs)
set_targets!(fmm, Xt)
set_density!(fmm, randn(3, 100_000))
t_first = @elapsed fmm_evaluate(fmm)        # builds the tree (and, the first time ever, the operators)
set_density!(fmm, randn(3, 100_000))
t_next = @elapsed U = fmm_evaluate(fmm)     # same points: tree and operators reused
@printf("first evaluation %.2f s, next %.2f s; U is %d × %d\n", t_first, t_next, size(U)...)
```

[`fmm_evaluate!`](@ref) writes into an existing array, and [`fmm_direct`](@ref) evaluates the same sum directly,
which is useful for checking accuracy on small problems. Free the object with `close(fmm)`, or leave it to the
garbage collector.

### Double layers and gradients

Kernels with source normals take them in [`set_sources!`](@ref):

```@example fmm
Xn = randn(3, 100_000)
U = fmm_eval(Laplace3D_DxU(), Xs, Xn, F, Xt; digits = 8)
Uref = kernel_eval(Laplace3D_DxU(), Xs, Xn, F, Xt[:, sample])
maximum(abs, U[:, sample] .- Uref) / maximum(abs, Uref)
```

### Periodic boundary conditions

With `periodic = true`, every axis is periodic; pass one `Bool` per axis for some of them. The sum then runs over
all periodic images of the sources. Points must lie in ``[0, \mathrm{period})`` along the periodic axes.

Periodic sums of the Laplace potential converge only for densities with zero net charge. Their gradient (the field)
is well defined, and it is invariant under shifting every point (modulo the period):

```@example fmm
F0 = F .- sum(F) / length(F)                     # zero net charge
E1 = fmm_eval(Laplace3D_FxdU(), Xs, F0, Xt; digits = 8, periodic = true)
shift(X) = mod.(X .+ [0.3, 0.6, 0.1], 1.0)
E2 = fmm_eval(Laplace3D_FxdU(), shift(Xs), F0, shift(Xt); digits = 8, periodic = true)
maximum(abs, E2 .- E1) / maximum(abs, E1)
```

## When the FMM is used

| situation | evaluation |
|:--|:--|
| fewer than 40000 targets, not periodic | direct sum (SCTL's choice: faster at this size) |
| 40000 or more targets, or periodic | PVFMM |
| library built without PVFMM ([`SCTL.has_feature`](@ref)`(:PVFMM) == false`) | always a direct sum; periodicity unavailable |

## Accuracy and cost

`digits` sets the multipole order of PVFMM, ``m = 2\lfloor 0.55\,\mathrm{digits} \rfloor``. These relative errors
were measured with 6×10⁴ uniformly random points:

| kernel | `digits = 6` | `digits = 8` | `digits = 10` | `digits = 12` |
|:--|:--|:--|:--|:--|
| `Laplace3D_FxU`, `Laplace3D_DxU`, `Laplace3D_FxdU` | ≈ 2e-6 | ≈ 7e-8 | | ≈ 6e-11 |
| `Stokes3D_FxU`, `Stokes3D_DxU` | | ≈ 2e-6 | ≈ 7e-8 | |

Stokes kernels reach about two digits less than requested, and `Stokes3D_FxT` less again. Ask for more digits if you
need them.

The run time grows linearly with the number of points. On one 96-core AMD EPYC 9474F node, with 10⁶ sources and 10⁶
targets, Laplace at 6 digits takes:

| threads | 1 | 16 | 96 |
|:--|:--|:--|:--|
| evaluation, same points [s] | 4.25 | 0.29 | 0.078 |
| first evaluation, building the tree [s] | 4.90 | 0.66 | 0.62 |

A direct sum takes about 10 s on 96 threads, so the FMM is roughly 125× faster here.

**Precomputation is a one-time cost** per translation kernel and `digits`. The operators depend only on the M2L
kernel, so all Laplace kernels share one file per `digits` (`Precomp_Laplace3D-FxU_m<order>.data`), as do
`Stokes3D_FxU` and `Stokes3D_DxU` (`Stokes3D-FSxU`), while `Stokes3D_FxT` has its own (`Stokes3D-FxU`).
Computing them takes seconds for Laplace (≈ 20 s at 12 digits) and
minutes for Stokes (≈ 1.5 min at 8 digits and ≈ 4 min at 10 digits on 16 threads). The operators are saved to
[`SCTL.fmm_cache_dir`](@ref) and loaded from there afterwards.

## Pitfalls

- **Precompute before parallel jobs.** Several processes computing the same operator file at once may read each
  other's partial files. When running many independent jobs (e.g. with disBatch), run one small evaluation per
  kernel and `digits` first, so later jobs only read the cache.
- **The cache directory must exist.** It is `ENV["PVFMM_DIR"]` if set before `using SCTL`, otherwise a scratch space
  in the Julia depot, which SCTL.jl creates. PVFMM terminates the process if the directory is missing, so do not
  delete the directory while Julia is running.
- **Watch the dynamic range.** The accuracy is relative to the largest potential. Targets very close to sources get
  near-field (direct) treatment, but potentials that vary over many orders of magnitude lose relative accuracy in
  the smaller values.
- **Threads.** The FMM uses OpenMP (`OMP_NUM_THREADS`, [`SCTL.set_num_threads`](@ref)). It scales well up to tens of
  threads; building the tree stops scaling at around 32 threads.

## Reference

```@docs
ParticleFMM
fmm_supported_kernels
set_sources!
set_density!
set_targets!
fmm_evaluate
fmm_evaluate!
fmm_direct
fmm_eval
SCTL.fmm_cache_dir
close(::ParticleFMM)
```
