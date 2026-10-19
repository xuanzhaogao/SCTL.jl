# sctl::ParticleFMM: fast multipole method (PVFMM, compiled in without MPI) for particle sums of the kernel functions.

export ParticleFMM, set_sources!, set_density!, set_targets!, fmm_evaluate, fmm_evaluate!, fmm_direct, fmm_eval
export fmm_supported_kernels

# ---- precomputed-operator cache ----

const _UUID = Base.UUID("ac9f61ce-527e-4020-a439-f70cd5e361ae")

"""
    SCTL.fmm_cache_dir()

Directory where PVFMM stores the operators it precomputes for each kernel and accuracy (files `Precomp_*.data`).
Computing them takes from seconds (Laplace) to minutes (Stokes, high accuracy) the first time; later sessions load
them. The directory is `ENV["PVFMM_DIR"]` if that is set when SCTL is loaded, otherwise a scratch space in the
first Julia depot. Delete its files to force recomputation.
"""
fmm_cache_dir() = get(ENV, "PVFMM_DIR", "")

function _init_pvfmm_cache()
    dir = get(ENV, "PVFMM_DIR", "")
    if isempty(dir)
        dir = joinpath(first(DEPOT_PATH), "scratchspaces", string(_UUID), "pvfmm")
        ENV["PVFMM_DIR"] = dir
    end
    # PVFMM exits the process if PVFMM_DIR is not a directory
    try
        mkpath(dir)
    catch e
        @warn "SCTL: cannot create the PVFMM cache directory $dir; FMM evaluations will fail" exception = e
    end
    return nothing
end

# ---- kernels ----

const _FMM_KERNELS = (Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU, Laplace3D_Fxd2U, Laplace3D_DxdU,
                      Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT)

"""
    fmm_supported_kernels()

Kernel types that [`ParticleFMM`](@ref) supports. For the others (Helmholtz, Biot-Savart, `Stokes3D_FSxU`,
`Stokes3D_FxUP`), PVFMM cannot be used; use [`kernel_eval`](@ref).
"""
fmm_supported_kernels() = collect(_FMM_KERNELS)

_fmm_supported(ker::AbstractKernel) = any(T -> ker isa T, _FMM_KERNELS)

# ---- ParticleFMM ----

"""
    ParticleFMM(kernel; digits=10, periodic=false, period=1.0)

Fast multipole evaluation of the sum
``U(x_t) = \\sum_s K(x_t - x_s) F_s`` (SCTL's `ParticleFMM<double,3>` with PVFMM) for a kernel from
[`fmm_supported_kernels`](@ref): `Laplace3D_FxU`, `Laplace3D_DxU`, `Laplace3D_FxdU`, `Laplace3D_Fxd2U`,
`Laplace3D_DxdU`, `Stokes3D_FxU`, `Stokes3D_DxU` and `Stokes3D_FxT`. The result is the same as
[`kernel_eval`](@ref), to accuracy `digits`, in O(N) work instead of O(Ns·Nt).

Set the points and density with [`set_sources!`](@ref), [`set_density!`](@ref) and [`set_targets!`](@ref), then
call [`fmm_evaluate`](@ref) (or [`fmm_evaluate!`](@ref)). The tree is rebuilt only when points change, so
evaluating new densities on fixed points (e.g. inside [`gmres`](@ref)) is the fast case. [`fmm_eval`](@ref) does
all of this in one call.

- `digits`: requested accuracy (relative to the largest potential). Measured: Laplace kernels reach it; Stokes
  kernels about 2 digits less (`digits = 10` gives ≈ 1e-7); `Stokes3D_FxT` less again.
- `periodic`: `true` (all axes), or one `Bool` per axis; the periodic box is ``[0, \\mathrm{period})`` along each
  periodic axis.

The FMM is used for 40000 or more targets, or with periodicity; smaller problems are summed directly (SCTL's
choice). It needs the library built with PVFMM ([`SCTL.has_feature`](@ref)`(:PVFMM)`); otherwise every evaluation
is a direct sum, and periodicity is unavailable. The first evaluation for a kernel and `digits` precomputes
translation operators, cached on disk ([`SCTL.fmm_cache_dir`](@ref)).

The C++ object is freed by a finalizer, or explicitly with `close(fmm)`.
"""
mutable struct ParticleFMM{K<:AbstractKernel}
    ptr::Ptr{Cvoid}
    kernel::K
    digits::Int
    ns::Int          # number of sources (-1: not set)
    nt::Int          # number of targets (-1: not set)
    density::Bool    # density set for the current sources
    periodic::NTuple{3,Bool}
    period::Float64
    function ParticleFMM(ker::K; digits::Integer = 10, periodic = false, period::Real = 1.0) where {K<:AbstractKernel}
        _fmm_supported(ker) ||
            throw(ArgumentError("$(kernel_name(ker)) has no FMM support; use kernel_eval (see fmm_supported_kernels())"))
        1 <= digits <= 15 || throw(ArgumentError("digits must be in 1:15, got $digits"))
        per = _periodic3(periodic)
        any(per) && !has_feature(:PVFMM) &&
            throw(ArgumentError("periodicity needs the library built with PVFMM (SCTL.has_feature(:PVFMM) is false)"))
        any(per) && !(period > 0) && throw(ArgumentError("period must be positive"))
        p = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_fmm_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Cint), p, _kernel_id(ker)))
        f = new{K}(p[], ker, Int(digits), -1, -1, false, per, Float64(period))
        finalizer(_free!, f)
        _check(ccall((:sctljl_fmm_set_accuracy, libsctljl), Cint, (Ptr{Cvoid}, Int64), f, digits))
        if any(per)
            code = sum(per[k] << (k - 1) for k in 1:3)
            _check(ccall((:sctljl_fmm_set_periodicity, libsctljl), Cint, (Ptr{Cvoid}, Cint, Float64), f, code, period))
        end
        return f
    end
end

function _periodic3(periodic)
    periodic isa Bool && return (periodic, periodic, periodic)
    length(periodic) == 3 && all(p -> p isa Bool, periodic) ||
        throw(ArgumentError("periodic must be a Bool or three Bools (one per axis)"))
    return Tuple(periodic)
end

function _free!(f::ParticleFMM)
    if f.ptr != C_NULL
        ccall((:sctljl_fmm_free, libsctljl), Cvoid, (Ptr{Cvoid},), f.ptr)
        f.ptr = C_NULL
    end
    return nothing
end

"""
    close(fmm::ParticleFMM)

Free the C++ object now (otherwise done by the finalizer). The object cannot be used afterwards.
"""
Base.close(f::ParticleFMM) = _free!(f)

function _ptr(f::ParticleFMM)
    f.ptr == C_NULL && throw(ArgumentError("ParticleFMM has been closed"))
    return f.ptr
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, f::ParticleFMM) = _ptr(f)

Base.show(io::IO, f::ParticleFMM) =
    print(io, "ParticleFMM(", kernel_name(f.kernel), "; digits=", f.digits,
          any(f.periodic) ? ", periodic=$(f.periodic), period=$(f.period)" : "",
          f.ns >= 0 ? ", $(f.ns) sources" : "", f.nt >= 0 ? ", $(f.nt) targets" : "",
          f.ptr == C_NULL ? ", closed)" : ")")

function _check_periodic_box(f::ParticleFMM, X, what)
    any(f.periodic) || return
    for k in 1:3
        f.periodic[k] || continue
        for i in k:3:length(X)
            0 <= X[i] < f.period ||
                throw(ArgumentError("$what must lie in [0, period) along periodic axis $k"))
        end
    end
end

"""
    set_sources!(fmm, Xs[, Xn]) -> fmm

Set the source points `Xs` (`3 × Ns`) and, for kernels with normals (`Laplace3D_DxU`, `Laplace3D_DxdU`,
`Stokes3D_DxU`), the source normals `Xn` (`3 × Ns`). Invalidates the density.
"""
function set_sources!(f::ParticleFMM, Xs, Xn = nothing)
    X_, ns = _points(Xs, "source coordinates")
    N_, nn = _normals(f.kernel, Xn, ns)
    _check_periodic_box(f, X_, "source coordinates")
    _check(ccall((:sctljl_fmm_set_sources, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
                 f, X_, ns, N_, nn))
    f.ns = ns
    f.density = false
    return f
end

"""
    set_density!(fmm, F) -> fmm

Set the source density `F` (`src_dim × Ns`, or a vector of length `src_dim*Ns`) for the current sources.
"""
function set_density!(f::ParticleFMM, F)
    f.ns >= 0 || throw(ArgumentError("set the sources (set_sources!) before the density"))
    F_ = _f64(F)
    length(F_) == src_dim(f.kernel) * f.ns ||
        throw(DimensionMismatch("density must have src_dim*Ns = $(src_dim(f.kernel) * f.ns) values, got $(length(F_))"))
    _check(ccall((:sctljl_fmm_set_density, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64), f, F_, length(F_)))
    f.density = true
    return f
end

"""
    set_targets!(fmm, Xt) -> fmm

Set the target points `Xt` (`3 × Nt`).
"""
function set_targets!(f::ParticleFMM, Xt)
    X_, nt = _points(Xt, "target coordinates")
    _check_periodic_box(f, X_, "target coordinates")
    _check(ccall((:sctljl_fmm_set_targets, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64), f, X_, nt))
    f.nt = nt
    return f
end

function _fmm_eval!(U, f::ParticleFMM, direct::Bool)
    (f.ns >= 0 && f.nt >= 0 && f.density) ||
        throw(ArgumentError("set the sources, the density and the targets before evaluating"))
    length(U) == trg_dim(f.kernel) * f.nt ||
        throw(DimensionMismatch("output must have trg_dim*Nt = $(trg_dim(f.kernel) * f.nt) values, got $(length(U))"))
    direct && any(f.periodic) && throw(ArgumentError("direct evaluation is not periodic"))
    f.nt == 0 && return U
    _check(ccall((:sctljl_fmm_eval, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64, Cint), f, U, length(U), direct))
    return U
end

"""
    fmm_evaluate!(U, fmm) -> U

Evaluate the potential at the targets into `U` (a `Float64` array with `trg_dim*Nt` elements, e.g. `trg_dim × Nt`).
"""
fmm_evaluate!(U::Array{Float64}, f::ParticleFMM) = _fmm_eval!(U, f, false)

"""
    fmm_evaluate(fmm) -> U

Evaluate the potential at the targets: a `trg_dim × Nt` matrix (layout as for [`kernel_eval`](@ref)).
"""
fmm_evaluate(f::ParticleFMM) = fmm_evaluate!(Matrix{Float64}(undef, trg_dim(f.kernel), max(f.nt, 0)), f)

"""
    fmm_direct(fmm) -> U

The same sum by direct evaluation (SCTL's `EvalDirect`), e.g. to check accuracy on small problems.
"""
fmm_direct(f::ParticleFMM) = _fmm_eval!(Matrix{Float64}(undef, trg_dim(f.kernel), max(f.nt, 0)), f, true)

"""
    fmm_eval(ker, Xs, Xn, F, Xt; digits=10, periodic=false, period=1.0) -> U
    fmm_eval(ker, Xs, F, Xt; digits=10, ...)

One-shot [`ParticleFMM`](@ref) evaluation, with the same arguments and result as [`kernel_eval`](@ref).
"""
function fmm_eval(ker::AbstractKernel, Xs, Xn, F, Xt; kw...)
    f = ParticleFMM(ker; kw...)
    try
        set_sources!(f, Xs, Xn)
        set_density!(f, F)
        set_targets!(f, Xt)
        return fmm_evaluate(f)
    finally
        close(f)
    end
end
fmm_eval(ker::AbstractKernel, Xs, F, Xt; kw...) = fmm_eval(ker, Xs, nothing, F, Xt; kw...)
