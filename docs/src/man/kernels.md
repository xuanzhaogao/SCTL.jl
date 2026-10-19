# Kernel functions

SCTL's kernel functions (`sctl/kernel_functions.hpp`) are the Green's functions of the 3D Laplace, Helmholtz and
Stokes equations and their derivatives. Each one is a C++ type alias of the form `sctl::GenericKernel<uKernel>`,
for example `sctl::Laplace3D_FxU`. SCTL.jl exposes each alias as a Julia type of the same name, a subtype of
[`AbstractKernel`](@ref). A kernel ``K(r)``, with ``r = x_t - x_s`` the vector from a source to a target, maps a
density ``F_s`` of `src_dim` components at source ``x_s`` to a potential of `trg_dim` components at each target
``x_t``:

```math
U(x_t) = \sum_{s=1}^{N_s} K(x_t - x_s)\, F_s .
```

Double-layer kernels also depend on the unit normal ``n_s`` at each source (`normal_dim = 3`). Two operations are
provided:

- [`kernel_eval`](@ref) / [`kernel_eval!`](@ref) compute this direct ``O(N_s N_t)`` sum (SCTL's
  `GenericKernel::Eval`), vectorized and optionally OpenMP-parallel. The matrix is never formed.
- [`kernel_matrix`](@ref) / [`kernel_matrix!`](@ref) build the dense matrix of the sum (SCTL's `KernelMatrix`).
  Use it when the same geometry is applied to many densities, or when you need the matrix itself, e.g. for a
  quadrature correction or a dense solve.

The same kernel types select the operator of a [`BoundaryIntegralOp`](@ref) (see
[Boundary integral operators](@ref)).

## The kernels

All formulas include SCTL's scale factor (`uKerScaleFactor`). They use ``\rho = |r|`` and the unit viscosity for
Stokes. "N" means the kernel takes the source normal.

| Type | N | `src_dim` | `trg_dim` | Kernel |
|:-----|:-:|:--:|:--:|:-------|
| `Laplace3D_FxU()` | | 1 | 1 | ``\frac{1}{4\pi\rho}`` (single layer) |
| `Laplace3D_DxU()` | ✓ | 1 | 1 | ``\frac{r\cdot n_s}{4\pi\rho^3}`` (double layer, ``\partial_{n_s}`` of the single layer) |
| `Laplace3D_FxdU()` | | 1 | 3 | ``-\frac{r}{4\pi\rho^3}`` (gradient of the single layer at the target) |
| `Laplace3D_Fxd2U()` | | 1 | 9 | ``\frac{1}{4\pi}\big(\frac{3 r r^T}{\rho^5} - \frac{I}{\rho^3}\big)`` (Hessian, row-major) |
| `Laplace3D_DxdU()` | ✓ | 1 | 3 | ``\frac{1}{4\pi}\big(\frac{n_s}{\rho^3} - \frac{3 (r\cdot n_s) r}{\rho^5}\big)`` (gradient of the double layer) |
| `BiotSavart3D_FxU()` | | 3 | 3 | ``\frac{f\times r}{4\pi\rho^3}`` |
| `BiotSavart3D_FxdU()` | | 3 | 9 | gradient of the Biot-Savart field |
| `Helmholtz3D_FxU(k)` | | 2 | 2 | ``\frac{e^{ik\rho}}{4\pi\rho}`` (complex) |
| `Helmholtz3D_DxU(k)` | ✓ | 2 | 2 | ``(1 - ik\rho)\, e^{ik\rho}\frac{r\cdot n_s}{4\pi\rho^3}`` (complex) |
| `Helmholtz3D_FxdU(k)` | | 2 | 6 | ``(ik\rho - 1)\, e^{ik\rho}\frac{r}{4\pi\rho^3}`` (complex gradient) |
| `HelmholtzDiff3D_FxdU(k)` | | 2 | 6 | ``\nabla \frac{e^{ik\rho} - 1}{4\pi\rho}``, no cancellation as ``k\rho \to 0`` |
| `Stokes3D_FxU()` | | 3 | 3 | ``\frac{1}{8\pi}\big(\frac{I}{\rho} + \frac{r r^T}{\rho^3}\big)`` (Stokeslet) |
| `Stokes3D_DxU()` | ✓ | 3 | 3 | ``\frac{3 (r\cdot n_s)\, r r^T}{4\pi\rho^5}`` (stresslet / double layer) |
| `Stokes3D_FxT()` | | 3 | 9 | ``-\frac{3\, r_i r_j r_k}{4\pi\rho^5}`` (stress of the Stokeslet, row-major ``3\times 3``) |
| `Stokes3D_FSxU()` | | 4 | 3 | Stokeslet on ``f`` plus ``\frac{s\, r}{8\pi\rho^3}`` for a source strength ``s`` |
| `Stokes3D_FxUP()` | | 3 | 4 | velocity and half the Stokeslet pressure, ``p = \frac{f\cdot r}{8\pi\rho^3}`` |

The Helmholtz kernels take the wavenumber `k` as a field. In C++ this is the context pointer `ctx_ptr`, which
SCTL.jl sets for you. `Stokes3D_FSxU` and `Stokes3D_FxUP` exist for SCTL's FMM translations, and their scalings
follow that use, as noted in the table. [`coord_dim`](@ref), [`normal_dim`](@ref), [`src_dim`](@ref),
[`trg_dim`](@ref) and [`kernel_name`](@ref) query a kernel:

```@example kernels
using SCTL, LinearAlgebra

for ker in (Laplace3D_DxU(), Helmholtz3D_FxdU(2.0), Stokes3D_FxT())
    println(rpad(kernel_name(ker), 18), " normal_dim=", normal_dim(ker),
            "  src_dim=", src_dim(ker), "  trg_dim=", trg_dim(ker))
end
```

## Tutorial

### Direct sums

Sources and targets are `3 × N` matrices. Densities are `src_dim × Ns` matrices and potentials `trg_dim × Nt`
matrices. The Laplace single layer of random charges, checked against the formula:

```@example kernels
Xs = randn(3, 200)                     # sources
Xt = randn(3, 50) .+ [4.0, 0, 0]       # targets
q  = randn(1, 200)                     # charges: src_dim × Ns
U  = kernel_eval(Laplace3D_FxU(), Xs, q, Xt)          # trg_dim × Nt = 1 × 50
Uref = [sum(q[s] / (4π * norm(Xt[:, t] - Xs[:, s])) for s in 1:200) for t in 1:50]
size(U), maximum(abs, vec(U) - Uref) / maximum(abs, Uref)
```

Double-layer kernels take the source normals as the third argument, also `3 × Ns`:

```@example kernels
Ns = randn(3, 200); Ns ./= sqrt.(sum(abs2, Ns; dims=1))   # unit normals
Ud = kernel_eval(Laplace3D_DxU(), Xs, Ns, q, Xt)
Udref = [sum(q[s] * dot(Xt[:, t] - Xs[:, s], Ns[:, s]) / (4π * norm(Xt[:, t] - Xs[:, s])^3) for s in 1:200)
         for t in 1:50]
maximum(abs, vec(Ud) - Udref) / maximum(abs, Udref)
```

A gradient kernel returns one 3-vector per target. Here the gradient of the single layer is compared with central
differences:

```@example kernels
G = kernel_eval(Laplace3D_FxdU(), Xs, q, Xt)          # 3 × Nt
h = 1e-5
FD = vcat([(kernel_eval(Laplace3D_FxU(), Xs, q, Xt .+ h .* e) -
            kernel_eval(Laplace3D_FxU(), Xs, q, Xt .- h .* e)) / 2h for e in eachcol(Matrix(1.0I, 3, 3))]...)
maximum(abs, G - FD) / maximum(abs, G)
```

### Kernel matrices

`kernel_matrix(ker, Xs, Xn, Xt)` returns the matrix `K` with `vec(U) == K * vec(F)`. A Stokeslet example:

```@example kernels
ker = Stokes3D_FxU()
f = randn(3, 200)                                     # forces: 3 × Ns
K = kernel_matrix(ker, Xs, Xt)                        # (3*50) × (3*200)
u = kernel_eval(ker, Xs, f, Xt)                       # 3 × 50
size(K), norm(K * vec(f) - vec(u)) / norm(u)
```

The ``3\times 3`` block that couples source `s` to target `t` is `K[3t-2:3t, 3s-2:3s]`:

```@example kernels
t, s = 7, 11
r = Xt[:, t] - Xs[:, s]; ρ = norm(r)
norm(K[3t-2:3t, 3s-2:3s] - (I / ρ + r * r' / ρ^3) / 8π)
```

To build matrices repeatedly for geometries of the same size, preallocate once and use
[`kernel_matrix!`](@ref). It overwrites the whole matrix:

```@example kernels
Kbuf = similar(K)
for _ in 1:3
    Xt .+= 0.1 .* randn(3, 50)                         # moved targets
    kernel_matrix!(Kbuf, ker, Xs, Xt)
end
norm(Kbuf * vec(f) - vec(kernel_eval(ker, Xs, f, Xt))) / norm(Kbuf * vec(f))
```

Likewise, [`kernel_eval!`](@ref) writes into a given array. With `accumulate=true` it adds to the existing
contents, which is SCTL's native behavior. This sums the potentials of two source sets:

```@example kernels
Xs2, q2 = randn(3, 30) .- [4.0, 0, 0], randn(1, 30)
Ubuf = zeros(1, 50)
kernel_eval!(Ubuf, Laplace3D_FxU(), Xs, q, Xt)                        # overwrite
kernel_eval!(Ubuf, Laplace3D_FxU(), Xs2, q2, Xt; accumulate=true)     # add
norm(Ubuf - kernel_eval(Laplace3D_FxU(), hcat(Xs, Xs2), hcat(q, q2), Xt))
```

### Helmholtz: complex values as real pairs

SCTL evaluates in real arithmetic and stores every complex number as a `(re, im)` pair of consecutive entries. A
scalar complex density is therefore a `2 × Ns` real matrix, which is exactly the memory layout of a
`Vector{ComplexF64}`. `reinterpret` converts in both directions without copying:

```@example kernels
k = 3.0
c = randn(ComplexF64, 200)                                       # complex charges
Uh = kernel_eval(Helmholtz3D_FxU(k), Xs, reinterpret(Float64, c), Xt)    # 2 × Nt
uc = vec(reinterpret(ComplexF64, vec(Uh)))                       # back to complex
uref = [sum(exp(im * k * norm(Xt[:, t] - Xs[:, s])) / (4π * norm(Xt[:, t] - Xs[:, s])) * c[s] for s in 1:200)
        for t in 1:50]
maximum(abs, uc - uref) / maximum(abs, uref)
```

In the kernel matrix, each complex entry ``g`` becomes the real ``2\times 2`` block
``\begin{pmatrix}\Re g & -\Im g\\ \Im g & \Re g\end{pmatrix}``, which acts as multiplication by ``g``:

```@example kernels
Kh = kernel_matrix(Helmholtz3D_FxU(k), Xs, Xt)
ρ = norm(Xt[:, 1] - Xs[:, 1]); g = exp(im * k * ρ) / (4π * ρ)
Kh[1:2, 1:2] ≈ [real(g) -imag(g); imag(g) real(g)]
```

`Helmholtz3D_FxdU` returns `(re ∂x, im ∂x, re ∂y, im ∂y, re ∂z, im ∂z)`, that is, `6 × Nt`.

### Coincident points

Every kernel is set to zero at ``r = 0``: SCTL masks ``1/\rho`` where ``\rho = 0``. A source therefore does not
interact with a target at the same position, so passing the same points as sources and targets gives the sum over
all other points:

```@example kernels
Kself = kernel_matrix(Laplace3D_FxU(), Xs, Xs)
all(iszero, diag(Kself)), all(isfinite, Kself)
```

Singular integrals over surfaces need special quadratures, which a [`BoundaryIntegralOp`](@ref) provides.

## Data layouts

All arrays are column-major Julia arrays whose memory is SCTL's array-of-structs order.

| Quantity | Julia shape | Memory order (SCTL) |
|:---------|:------------|:--------------------|
| source / target coordinates `Xs`, `Xt` | `3 × Ns`, `3 × Nt` (or a vector of length `3N`) | `x1 y1 z1 x2 y2 z2 …` |
| source normals `Xn` | `3 × Ns`; `nothing` or empty if `normal_dim == 0` | as coordinates |
| densities `F` | `src_dim × Ns` (or a vector of length `src_dim*Ns`) | all components of source 1, then source 2, … |
| potentials `U` | `trg_dim × Nt` | all components of target 1, then target 2, … |
| complex values (Helmholtz) | an extra leading factor 2: `(re, im)` | `re im` pairs; `reinterpret(Float64, ::Vector{ComplexF64})` |
| `Laplace3D_Fxd2U`, `Stokes3D_FxT`, `BiotSavart3D_FxdU` output | `9 × Nt` | row-major ``3\times 3`` per target |
| kernel matrix `K` | `(trg_dim*Nt) × (src_dim*Ns)` | SCTL's row-major `(Ns*SrcDim) × (Nt*TrgDim)` matrix ``M``, i.e. ``K = M^T`` |

The kernel matrix entry `K[(t-1)*trg_dim + j, (s-1)*src_dim + i]` is the response of component `j` at target `t`
to component `i` of the density at source `s`. Because SCTL's row-major ``M`` and the Julia column-major ``K``
occupy the same memory, no transpose is ever made.

## Pitfalls and limitations

- **Cost.** `kernel_eval` is a direct ``O(N_s N_t)`` sum. There is no FMM in this build. For large point sets,
  form the matrix only if `trg_dim*Nt × src_dim*Ns` doubles fit in memory.
- **Threads.** With `openmp=true` (the default), `kernel_eval` parallelizes over targets and `kernel_matrix` over
  sources, using the threads set by `SCTL.set_num_threads`. Pass `openmp=false` when you already parallelize at a
  higher level, e.g. calling from several Julia threads.
- **Accuracy.** Evaluation uses SCTL's vectorized approximate `rsqrt`/`sincos` at full double precision, so
  results agree with naive formulas to a relative ``\sim 10^{-14}``, not to the last bit.
- **Normals are not normalized** by SCTL. Pass unit normals to double-layer kernels.
- **`kernel_eval!` accumulates only on request.** SCTL's `Eval` adds to its output. The Julia wrapper zeroes `U`
  first unless `accumulate=true`.
- **Scale conventions.** The `Stokes3D_FxUP` pressure is half the Stokeslet pressure, and the `Stokes3D_FSxU`
  source term is half the velocity of a point source of unit flux. Both are SCTL conventions for FMM translations;
  rescale them if you use these kernels for physics.

## API reference

```@docs
AbstractKernel
Laplace3D_FxU
Laplace3D_DxU
Laplace3D_FxdU
Laplace3D_Fxd2U
Laplace3D_DxdU
BiotSavart3D_FxU
BiotSavart3D_FxdU
Helmholtz3D_FxU
Helmholtz3D_DxU
Helmholtz3D_FxdU
HelmholtzDiff3D_FxdU
Stokes3D_FxU
Stokes3D_DxU
Stokes3D_FxT
Stokes3D_FSxU
Stokes3D_FxUP
coord_dim
normal_dim
src_dim
trg_dim
kernel_name
kernel_matrix
kernel_matrix!
kernel_eval
kernel_eval!
```
