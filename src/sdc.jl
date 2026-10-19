# Spectral deferred correction (SDC) ODE solver.

export SDC, order, sdc_nodes, sdc_step, sdc_solve

"""
    SDC(order)

A spectral deferred correction solver (`sctl::SDC<double>`) of the given `order ≥ 2` for autonomous ODEs
``u' = F(u)``. Each step uses `order` substep nodes on the step interval (see [`sdc_nodes`](@ref)) and, by default,
`order` Picard (correction) iterations.

Use [`sdc_step`](@ref) for single steps and [`sdc_solve`](@ref) for adaptive time stepping. The C++ object is freed
by a finalizer, or explicitly with `close(sdc)`.
"""
mutable struct SDC
    ptr::Ptr{Cvoid}
    order::Int
    function SDC(order::Integer)
        order >= 2 || throw(ArgumentError("SDC order must be ≥ 2, got $order"))
        p = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_sdc_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Int64), p, order))
        o = Ref{Int64}(0)
        _check(ccall((:sctljl_sdc_order, libsctljl), Cint, (Ptr{Cvoid}, Ref{Int64}), p[], o))
        S = new(p[], Int(o[]))
        finalizer(close, S)
        return S
    end
end

"""
    close(sdc::SDC)

Free the underlying C++ object now (also done by the finalizer). Further use of `sdc` throws.
"""
function Base.close(S::SDC)
    if S.ptr != C_NULL
        ccall((:sctljl_sdc_free, libsctljl), Cvoid, (Ptr{Cvoid},), S.ptr)
        S.ptr = C_NULL
    end
    return nothing
end

function _sdc_ptr(S::SDC)
    S.ptr == C_NULL && throw(ArgumentError("SDC object has been freed"))
    return S.ptr
end

# Passing the object itself to ccall (instead of its raw pointer) keeps it rooted, so the finalizer cannot free the
# C++ object during the call.
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::SDC) = _sdc_ptr(x)

Base.show(io::IO, S::SDC) = print(io, "SDC(", S.order, S.ptr == C_NULL ? ", <freed>)" : ")")

"""
    order(sdc::SDC) -> Int

Order of the SDC scheme (also its number of substep nodes).
"""
order(S::SDC) = S.order

"""
    sdc_nodes(sdc::SDC) -> Vector{Float64}

The substep nodes of one SDC step, as fractions of the step size in `[0, 1]`: the `order` Chebyshev points of the
second kind `τ_i = (1 - cos(π i / (order - 1))) / 2`, `i = 0, …, order-1` (so `τ_1 = 0`, `τ_end = 1`).
Column `i` of `sdc_step(...; substeps=true).substeps` is the solution at time `dt * τ_i`.
"""
sdc_nodes(S::SDC) = [(1 - cospi(i / (S.order - 1))) / 2 for i in 0:S.order-1]

# Callback context: the RHS (two- or four-argument form), the monitor, and the state length.
struct _SDCFns{F,M,I}
    f!::F
    monitor::M
    n::Int
    bufs::CallbackBufs
end
_SDCFns(f!, monitor, n, indices::Bool) = _SDCFns{typeof(f!),typeof(monitor),indices}(f!, monitor, n, CallbackBufs(n))

_call_rhs(fns::_SDCFns{F,M,false}, dudt, u, ci, si) where {F,M} = fns.f!(dudt, u)
_call_rhs(fns::_SDCFns{F,M,true}, dudt, u, ci, si) where {F,M} = fns.f!(dudt, u, Int(ci), Int(si))

# The trampolines are specialized on the concrete state type (the C side passes the state object as its void*
# context), so calling the user's functions is statically dispatched.
function _sdc_rhs_tramp(st::CallbackState, dudt::Ptr{Float64}, u::Ptr{Float64}, n::Int64, ci::Int64, si::Int64)::Cint
    ok = _guard(st) do
        fns = st.f
        n == fns.n || throw(DimensionMismatch("SDC right-hand side called with a vector of length $n"))
        _with_cbufs((du_, u_) -> _call_rhs(fns, du_, u_, ci, si), fns.bufs, dudt, u, n)
    end
    return ok ? Cint(0) : Cint(1)
end

function _sdc_monitor_tramp(st::CallbackState, t::Float64, dt::Float64, u::Ptr{Float64}, n::Int64)::Cint
    ok = _guard(st) do
        st.f.monitor(t, dt, copy(unsafe_wrap(Array, u, n)))
    end
    return ok ? Cint(0) : Cint(1)
end

_sdc_rhs_cfun(::S) where {S<:CallbackState} =
    @cfunction(_sdc_rhs_tramp, Cint, (Ref{S}, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Int64))
_sdc_monitor_cfun(::S) where {S<:CallbackState} =
    @cfunction(_sdc_monitor_tramp, Cint, (Ref{S}, Float64, Float64, Ptr{Float64}, Int64))

function _check_u0(u0)
    isempty(u0) && throw(ArgumentError("u0 must not be empty"))
    return Vector{Float64}(u0)
end

"""
    sdc_step(sdc, f!, u0, dt; picard_iters=-1, picard_tol=0, substeps=false, indices=false)
        -> (u=..., error_interp=..., error_picard=..., iters=..., substeps=...)

One SDC step for ``u' = F(u)``: computes ``u(dt) = u_0 + \\int_0^{dt} F(u(s))\\,ds``.

`f!(dudt, u)` overwrites `dudt` with ``F(u)``. With `indices=true` it is called as
`f!(dudt, u, correction_idx, substep_idx)` instead (the 0-based Picard iteration and substep indices, as in SCTL).
`dudt` and `u` are valid only during the call (reused buffers for short states, zero-copy views of SCTL's memory for
long ones): do not modify `u` and do not keep references to either. Exceptions thrown by
`f!` abort the step and are rethrown unchanged.

Keywords: `picard_iters` is the maximum number of Picard iterations (`-1` = `order(sdc)`); the iteration stops early
once the Picard update is below `picard_tol` (or stagnates).

Returns a `NamedTuple`: the solution `u` at `dt`; `error_interp`, an estimate of the truncation error of the
solution interpolant; `error_picard`, the size of the last Picard update (times `dt`); `iters`, the number of Picard
iterations performed; and, if `substeps=true`, `substeps::Matrix{Float64}` of size `length(u0) × order(sdc)` whose
column `i` is the solution at time `dt * sdc_nodes(sdc)[i]` (first column `u0`, last column `u`), else `nothing`.
"""
function sdc_step(S::SDC, f!, u0::AbstractVector{<:Real}, dt::Real; picard_iters::Integer=-1, picard_tol::Real=0,
                  substeps::Bool=false, indices::Bool=false)
    (picard_iters == -1 || picard_iters >= 1) ||
        throw(ArgumentError("picard_iters must be -1 (default: order) or ≥ 1, got $picard_iters"))
    u0_ = _check_u0(u0)
    n = length(u0_)
    u = Vector{Float64}(undef, n)
    Usub = substeps ? Matrix{Float64}(undef, n, S.order) : nothing  # column-major n × order == row-major order × n
    ei, ep, it = Ref(0.0), Ref(0.0), Ref{Int64}(0)
    st = CallbackState(_SDCFns(f!, nothing, n, indices))
    cfn = _sdc_rhs_cfun(st)
    GC.@preserve S st begin
        status = ccall((:sctljl_sdc_step, libsctljl), Cint,
                       (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int64, Float64,
                        Ref{Float64}, Ref{Float64}, Ref{Int64}, Ptr{Float64}),
                       S, u, u0_, n, dt, cfn, pointer_from_objref(st), picard_iters, picard_tol,
                       ei, ep, it, substeps ? Usub : C_NULL)
    end
    _check(status, st)
    return (u=u, error_interp=ei[], error_picard=ep[], iters=Int(it[]), substeps=Usub)
end

"""
    sdc_solve(sdc, f!, u0, dt, T, tol; monitor=nothing, continue_with_errors=false, indices=false, verbose=false)
        -> (u=..., t=..., err=...)

Solve ``u' = F(u)``, ``u(0) = u_0`` adaptively on `[0, T]` to the tolerance `tol`, starting with step size `dt`.
`f!` is as in [`sdc_step`](@ref). If given, `monitor(t, dt, u)` is called after each accepted step with the new time
`t`, the step size `dt` just taken and (a copy of) the solution at `t`; exceptions it throws abort the solve and are
rethrown unchanged.

Returns the solution `u` at the final time, the final time `t` reached (equal to `T` on success) and `err`, the
accumulated error estimate. If the tolerance cannot be met, an `SCTLError` is thrown unless
`continue_with_errors=true`, in which case the best solution found is returned (with `t < T` and/or `err > tol`;
SCTL then also prints a warning to stderr). `verbose=true` lets SCTL print its per-step log to stdout.
"""
function sdc_solve(S::SDC, f!, u0::AbstractVector{<:Real}, dt::Real, T::Real, tol::Real; monitor=nothing,
                   continue_with_errors::Bool=false, indices::Bool=false, verbose::Bool=false)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    T > 0 || throw(ArgumentError("T must be positive"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    u0_ = _check_u0(u0)
    n = length(u0_)
    u = Vector{Float64}(undef, n)
    tf, err = Ref(0.0), Ref(0.0)
    st = CallbackState(_SDCFns(f!, monitor, n, indices))
    cfn = _sdc_rhs_cfun(st)
    mfn = monitor === nothing ? C_NULL :
          _sdc_monitor_cfun(st)
    GC.@preserve S st begin
        status = ccall((:sctljl_sdc_solve, libsctljl), Cint,
                       (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Int64, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
                        Ptr{Cvoid}, Cint, Cint, Ref{Float64}, Ref{Float64}),
                       S, u, u0_, n, dt, T, tol, cfn, mfn, pointer_from_objref(st), continue_with_errors,
                       verbose, tf, err)
    end
    _check(status, st)
    return (u=u, t=tf[], err=err[])
end
