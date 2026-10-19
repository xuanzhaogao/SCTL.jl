# Kernel functions (sctl/kernel_functions.hpp): kernel matrices and direct N-body sums.

export AbstractKernel
export Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU, Laplace3D_Fxd2U, Laplace3D_DxdU
export BiotSavart3D_FxU, BiotSavart3D_FxdU
export Helmholtz3D_FxU, Helmholtz3D_DxU, Helmholtz3D_FxdU, HelmholtzDiff3D_FxdU
export Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT, Stokes3D_FSxU, Stokes3D_FxUP
export coord_dim, normal_dim, src_dim, trg_dim, kernel_name
export kernel_matrix, kernel_matrix!, kernel_eval, kernel_eval!

"""
    AbstractKernel

Supertype of the SCTL kernel functions (`sctl::GenericKernel` aliases of `kernel_functions.hpp`). Each kernel
``K(r)``, ``r = x_t - x_s``, maps a density of [`src_dim`](@ref) components at a source point to a potential of
[`trg_dim`](@ref) components at a target point; kernels with [`normal_dim`](@ref) `== 3` also take the source normal
(double-layer kernels).

Data layouts used by [`kernel_matrix`](@ref) and [`kernel_eval`](@ref) (all column-major Julia arrays, i.e. SCTL's
array-of-structs order):

- coordinates and normals: `3 × N` matrices (or vectors of length `3N`);
- densities: `src_dim(ker) × Ns`; potentials: `trg_dim(ker) × Nt`;
- complex-valued kernels (Helmholtz) store each complex number as a `(re, im)` pair of consecutive components, so a
  scalar Helmholtz density is `2 × Ns`; `reinterpret(Float64, z)` of a `Vector{ComplexF64}` gives this layout.

The kernels include their scale factors: e.g. `Laplace3D_FxU` is ``1/(4π|r|)``. Self-interactions (``r = 0``) are
skipped: every kernel evaluates to zero at coincident points.
"""
abstract type AbstractKernel end

"""
    Laplace3D_FxU()

Laplace single-layer kernel ``1/(4π|r|)`` (`src_dim = trg_dim = 1`).
"""
struct Laplace3D_FxU <: AbstractKernel end

"""
    Laplace3D_DxU()

Laplace double-layer kernel ``(r·n_s)/(4π|r|^3)``, ``r = x_t - x_s`` (the normal derivative of the single layer at
the source; `src_dim = trg_dim = 1`, normals required).
"""
struct Laplace3D_DxU <: AbstractKernel end

"""
    Laplace3D_FxdU()

Gradient of the Laplace single layer at the target, ``-r/(4π|r|^3)`` (`src_dim = 1`, `trg_dim = 3`).
"""
struct Laplace3D_FxdU <: AbstractKernel end

"""
    Laplace3D_Fxd2U()

Hessian of the Laplace single layer at the target, ``(3 r r^T/|r|^5 - I/|r|^3)/(4π)`` (`src_dim = 1`,
`trg_dim = 9`, row-major 3×3 per target, symmetric).
"""
struct Laplace3D_Fxd2U <: AbstractKernel end

"""
    Laplace3D_DxdU()

Gradient (at the target) of the Laplace double layer, ``(n_s/|r|^3 - 3 (r·n_s) r/|r|^5)/(4π)`` (`src_dim = 1`,
`trg_dim = 3`, normals required).
"""
struct Laplace3D_DxdU <: AbstractKernel end

"""
    BiotSavart3D_FxU()

Biot-Savart kernel: the field ``f × r/(4π|r|^3)`` of a vector density `f` (`src_dim = trg_dim = 3`).
"""
struct BiotSavart3D_FxU <: AbstractKernel end

"""
    BiotSavart3D_FxdU()

Gradient (at the target) of the Biot-Savart field (`src_dim = 3`, `trg_dim = 9`).
"""
struct BiotSavart3D_FxdU <: AbstractKernel end

"""
    Helmholtz3D_FxU(k)

Helmholtz single-layer kernel ``e^{ik|r|}/(4π|r|)`` with wavenumber `k` (complex, `src_dim = trg_dim = 2` as
`(re, im)` pairs).
"""
struct Helmholtz3D_FxU <: AbstractKernel
    wavenumber::Float64
end

"""
    Helmholtz3D_DxU(k)

Helmholtz double-layer kernel, the normal derivative at the source of ``e^{ik|r|}/(4π|r|)``:
``(1 - ik|r|) e^{ik|r|} (r·n_s)/(4π|r|^3)`` (complex, `src_dim = trg_dim = 2`, normals required). Reduces to
[`Laplace3D_DxU`](@ref) as `k → 0`.
"""
struct Helmholtz3D_DxU <: AbstractKernel
    wavenumber::Float64
end

"""
    Helmholtz3D_FxdU(k)

Gradient at the target of the Helmholtz single layer, ``(ik|r| - 1) e^{ik|r|} r/(4π|r|^3)`` (complex,
`src_dim = 2`, `trg_dim = 6`: `(re, im)` of each of the 3 components).
"""
struct Helmholtz3D_FxdU <: AbstractKernel
    wavenumber::Float64
end

"""
    HelmholtzDiff3D_FxdU(k)

Gradient at the target of the difference of the Helmholtz and Laplace single layers,
``∇(e^{ik|r|} - 1)/(4π|r|)``, evaluated without cancellation for small `k|r|` (layout as
[`Helmholtz3D_FxdU`](@ref)).
"""
struct HelmholtzDiff3D_FxdU <: AbstractKernel
    wavenumber::Float64
end

"""
    Stokes3D_FxU()

Stokes single-layer kernel (Stokeslet, unit viscosity) ``(I/|r| + r r^T/|r|^3)/(8π)`` (`src_dim = trg_dim = 3`).
"""
struct Stokes3D_FxU <: AbstractKernel end

"""
    Stokes3D_DxU()

Stokes double-layer kernel (stresslet) ``3 (r·n_s) r r^T/(4π|r|^5)`` (`src_dim = trg_dim = 3`, normals required).
"""
struct Stokes3D_DxU <: AbstractKernel end

"""
    Stokes3D_FxT()

Stress tensor of the Stokeslet, ``-3 r_i r_j r_k/(4π|r|^5)``: density `f` (`src_dim = 3`) to the 3×3 stress
(`trg_dim = 9`, row-major); dotted with the target normal it gives the traction.
"""
struct Stokes3D_FxT <: AbstractKernel end

"""
    Stokes3D_FSxU()

Stokeslet plus source/sink: density `(f, s)` (`src_dim = 4`) to velocity (`trg_dim = 3`),
``(I/|r| + r r^T/|r|^3) f/(8π) + s r/(8π|r|^3)``. Used for FMM translations.
"""
struct Stokes3D_FSxU <: AbstractKernel end

"""
    Stokes3D_FxUP()

Stokeslet velocity and (half) pressure: density `f` (`src_dim = 3`) to `(u, p)` (`trg_dim = 4`) with
`p = (f·r)/(8π|r|^3)` (half the Stokeslet pressure; a convention of SCTL's FMM translations).
"""
struct Stokes3D_FxUP <: AbstractKernel end

# kernel id (as in csrc/capi_kernels.cpp) and dimensions (CoordDim, NormalDim, SrcDim, TrgDim)
_kernel_info(::Laplace3D_FxU) = (0, (3, 0, 1, 1))
_kernel_info(::Laplace3D_DxU) = (1, (3, 3, 1, 1))
_kernel_info(::Laplace3D_FxdU) = (2, (3, 0, 1, 3))
_kernel_info(::Laplace3D_Fxd2U) = (3, (3, 0, 1, 9))
_kernel_info(::Laplace3D_DxdU) = (4, (3, 3, 1, 3))
_kernel_info(::BiotSavart3D_FxU) = (5, (3, 0, 3, 3))
_kernel_info(::BiotSavart3D_FxdU) = (6, (3, 0, 3, 9))
_kernel_info(::Helmholtz3D_FxU) = (7, (3, 0, 2, 2))
_kernel_info(::Helmholtz3D_DxU) = (8, (3, 3, 2, 2))
_kernel_info(::Helmholtz3D_FxdU) = (9, (3, 0, 2, 6))
_kernel_info(::HelmholtzDiff3D_FxdU) = (10, (3, 0, 2, 6))
_kernel_info(::Stokes3D_FxU) = (11, (3, 0, 3, 3))
_kernel_info(::Stokes3D_DxU) = (12, (3, 3, 3, 3))
_kernel_info(::Stokes3D_FxT) = (13, (3, 0, 3, 9))
_kernel_info(::Stokes3D_FSxU) = (14, (3, 0, 4, 3))
_kernel_info(::Stokes3D_FxUP) = (15, (3, 0, 3, 4))

_kernel_id(ker::AbstractKernel) = _kernel_info(ker)[1]
# context passed to the kernel (the wavenumber of the Helmholtz kernels)
_kernel_ctx(ker::AbstractKernel) = hasfield(typeof(ker), :wavenumber) ? Float64(getfield(ker, :wavenumber)) : 0.0

"""
    coord_dim(ker)

Dimension of the coordinate space (3 for all SCTL kernels).
"""
coord_dim(ker::AbstractKernel) = _kernel_info(ker)[2][1]

"""
    normal_dim(ker)

Number of components of the source normal the kernel needs: 3 for double-layer kernels, 0 otherwise.
"""
normal_dim(ker::AbstractKernel) = _kernel_info(ker)[2][2]

"""
    src_dim(ker)

Number of (real) components of the density at each source point.
"""
src_dim(ker::AbstractKernel) = _kernel_info(ker)[2][3]

"""
    trg_dim(ker)

Number of (real) components of the potential at each target point.
"""
trg_dim(ker::AbstractKernel) = _kernel_info(ker)[2][4]

# dimensions as reported by the C++ library (used by the tests to check the table above)
function _kernel_dims_c(ker::AbstractKernel)
    d = zeros(Int64, 4)
    _check(ccall((:sctljl_kernel_dims, libsctljl), Cint, (Cint, Ptr{Int64}), _kernel_id(ker), d))
    return Tuple(d)
end

"""
    kernel_name(ker) -> String

SCTL's name of the kernel, e.g. `"Laplace3D-FxU"`.
"""
function kernel_name(ker::AbstractKernel)
    p = Ref{Cstring}()
    _check(ccall((:sctljl_kernel_name, libsctljl), Cint, (Cint, Ref{Cstring}), _kernel_id(ker), p))
    return unsafe_string(p[])
end

# Points as a contiguous Float64 array with 3 rows; returns (array, number of points).
function _points(X, what::AbstractString)
    X_ = _f64(X)
    if X_ isa AbstractVector
        length(X_) % 3 == 0 || throw(DimensionMismatch("$what: length must be a multiple of 3"))
        return X_, length(X_) ÷ 3
    end
    ndims(X_) == 2 && size(X_, 1) == 3 || throw(DimensionMismatch("$what must be a 3×N matrix (or a vector of length 3N)"))
    return X_, size(X_, 2)
end

function _normals(ker::AbstractKernel, Xn, ns::Integer)
    if normal_dim(ker) == 0
        (Xn === nothing || isempty(Xn)) || throw(ArgumentError("$(kernel_name(ker)) takes no normals; pass an empty array or nothing"))
        return Float64[], 0
    end
    Xn === nothing && throw(ArgumentError("$(kernel_name(ker)) needs source normals"))
    Xn_, nn = _points(Xn, "normals")
    nn == ns || throw(DimensionMismatch("need one normal per source point ($ns), got $nn"))
    return Xn_, nn
end

"""
    kernel_matrix(ker, Xs, Xn, Xt; openmp=true) -> K
    kernel_matrix(ker, Xs, Xt; openmp=true)

Kernel matrix from the sources `Xs` (with normals `Xn`, `3 × Ns`; empty or `nothing` for kernels without normals) to
the targets `Xt` (`3 × Nt`). Returns `K` of size `(trg_dim*Nt, src_dim*Ns)` with
`K[(t-1)*trg_dim + j, (s-1)*src_dim + i]` = component `j` of the potential at target `t` due to component `i` of the
density at source `s`, so that `vec(U) = K * vec(F)` for densities `F` (`src_dim × Ns`) and potentials `U`
(`trg_dim × Nt`).

(SCTL fills a row-major `(Ns*SrcDim) × (Nt*TrgDim)` matrix ``M``; `K` is the same memory read column-major, i.e.
``K = M^T``.) `openmp` parallelizes over the sources with OpenMP.
"""
function kernel_matrix(ker::AbstractKernel, Xs, Xn, Xt; openmp::Bool=true)
    ns, nt = _points(Xs, "source coordinates")[2], _points(Xt, "target coordinates")[2]
    K = Matrix{Float64}(undef, trg_dim(ker) * nt, src_dim(ker) * ns)  # SCTL zeroes it
    return kernel_matrix!(K, ker, Xs, Xn, Xt; openmp=openmp)
end
kernel_matrix(ker::AbstractKernel, Xs, Xt; openmp::Bool=true) = kernel_matrix(ker, Xs, nothing, Xt; openmp=openmp)

"""
    kernel_matrix!(K, ker, Xs, Xn, Xt; openmp=true) -> K
    kernel_matrix!(K, ker, Xs, Xt; openmp=true)

In-place [`kernel_matrix`](@ref): overwrite `K`, a `Matrix{Float64}` of size `(trg_dim*Nt, src_dim*Ns)`. Reusing `K`
across calls avoids allocating (and page-faulting) a new matrix each time.
"""
function kernel_matrix!(K::Matrix{Float64}, ker::AbstractKernel, Xs, Xn, Xt; openmp::Bool=true)
    Xs_, ns = _points(Xs, "source coordinates")
    Xt_, nt = _points(Xt, "target coordinates")
    Xn_, nn = _normals(ker, Xn, ns)
    size(K) == (trg_dim(ker) * nt, src_dim(ker) * ns) ||
        throw(DimensionMismatch("K must be $(trg_dim(ker) * nt)×$(src_dim(ker) * ns), got $(size(K))"))
    (ns == 0 || nt == 0) && return fill!(K, 0)
    ctx = Ref(_kernel_ctx(ker))
    _check(ccall((:sctljl_kernel_matrix, libsctljl), Cint,
                 (Cint, Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint),
                 _kernel_id(ker), ctx, K, Xs_, ns, Xn_, nn, Xt_, nt, openmp))
    return K
end
kernel_matrix!(K::Matrix{Float64}, ker::AbstractKernel, Xs, Xt; openmp::Bool=true) =
    kernel_matrix!(K, ker, Xs, nothing, Xt; openmp=openmp)

"""
    kernel_eval!(U, ker, Xs, Xn, F, Xt; openmp=true, accumulate=false) -> U
    kernel_eval!(U, ker, Xs, F, Xt; openmp=true, accumulate=false) -> U

In-place [`kernel_eval`](@ref): write (or with `accumulate=true`, add) the potential at the targets into `U`, a
`Float64` array with `trg_dim(ker)*Nt` elements (e.g. `trg_dim × Nt`).
"""
function kernel_eval!(U::Array{Float64}, ker::AbstractKernel, Xs, Xn, F, Xt; openmp::Bool=true, accumulate::Bool=false)
    Xs_, ns = _points(Xs, "source coordinates")
    Xt_, nt = _points(Xt, "target coordinates")
    Xn_, nn = _normals(ker, Xn, ns)
    F_ = _f64(F)
    length(F_) == src_dim(ker) * ns ||
        throw(DimensionMismatch("densities must have src_dim × Ns = $(src_dim(ker)) × $ns elements, got $(length(F_))"))
    length(U) == trg_dim(ker) * nt ||
        throw(DimensionMismatch("potentials must have trg_dim × Nt = $(trg_dim(ker)) × $nt elements, got $(length(U))"))
    accumulate || fill!(U, 0)
    (ns == 0 || nt == 0) && return U
    ctx = Ref(_kernel_ctx(ker))
    _check(ccall((:sctljl_kernel_eval, libsctljl), Cint,
                 (Cint, Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Int64, Cint),
                 _kernel_id(ker), ctx, U, Xs_, ns, Xn_, nn, F_, Xt_, nt, openmp))
    return U
end
kernel_eval!(U::Array{Float64}, ker::AbstractKernel, Xs, F, Xt; kw...) = kernel_eval!(U, ker, Xs, nothing, F, Xt; kw...)

"""
    kernel_eval(ker, Xs, Xn, F, Xt; openmp=true) -> U
    kernel_eval(ker, Xs, F, Xt; openmp=true)

Direct N-body sum ``U(x_t) = \\sum_s K(x_t - x_s) F_s`` (SCTL's `GenericKernel::Eval`, vectorized and, with
`openmp=true`, OpenMP-parallel over the targets). `Xs` (`3 × Ns`) are the sources with normals `Xn` (`3 × Ns`, empty
or `nothing` for kernels without normals), `F` the densities (`src_dim × Ns`, or a vector of that length) and `Xt`
(`3 × Nt`) the targets. Returns the potentials as a `trg_dim × Nt` matrix. Coincident source/target pairs contribute
nothing.
"""
function kernel_eval(ker::AbstractKernel, Xs, Xn, F, Xt; openmp::Bool=true)
    _, nt = _points(Xt, "target coordinates")
    return kernel_eval!(Matrix{Float64}(undef, trg_dim(ker), nt), ker, Xs, Xn, F, Xt; openmp=openmp)
end
kernel_eval(ker::AbstractKernel, Xs, F, Xt; openmp::Bool=true) = kernel_eval(ker, Xs, nothing, F, Xt; openmp=openmp)
