# GMRES linear solver and the Krylov-subspace recycling preconditioner (KrylovPrecond).

export gmres, KrylovPrecond, apply!

"""
    KrylovPrecond()

A Krylov-subspace preconditioner (`sctl::KrylovPrecond<double>`) that is built up by [`gmres`](@ref) solves:
pass it as `gmres(A, b; precond=P)` and each solve appends a low-rank correction ``P ← P (I + U Q^T)`` built from
its Krylov subspace, so later solves with the same operator and a nearby right-hand side need fewer iterations.

A fresh preconditioner is the identity. If it is used with a system of a different size, it is reset.

- `size(P)` is `(n, n)` where `n` is the system size it was built for (`0` while empty),
- `rank(P)` is the cumulative dimension of the stored Krylov subspaces,
- `apply!(P, x)` / `P * x` apply it to a vector.

The C++ object is freed by a finalizer, or explicitly with `close(P)`.
"""
mutable struct KrylovPrecond
    ptr::Ptr{Cvoid}
    function KrylovPrecond()
        p = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_krylov_new, libsctljl), Cint, (Ref{Ptr{Cvoid}},), p))
        P = new(p[])
        finalizer(close, P)
        return P
    end
end

"""
    close(P::KrylovPrecond)

Free the underlying C++ object now (also done by the finalizer). Further use of `P` throws.
"""
function Base.close(P::KrylovPrecond)
    if P.ptr != C_NULL
        ccall((:sctljl_krylov_free, libsctljl), Cvoid, (Ptr{Cvoid},), P.ptr)
        P.ptr = C_NULL
    end
    return nothing
end

function _krylov_ptr(P::KrylovPrecond)
    P.ptr == C_NULL && throw(ArgumentError("KrylovPrecond has been freed"))
    return P.ptr
end

# Passing the object itself to ccall (instead of its raw pointer) keeps it rooted, so the finalizer cannot free the
# C++ object during the call.
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::KrylovPrecond) = _krylov_ptr(x)

function _size_rank(P::KrylovPrecond)
    n, r = Ref{Int64}(0), Ref{Int64}(0)
    _check(ccall((:sctljl_krylov_size, libsctljl), Cint, (Ptr{Cvoid}, Ref{Int64}, Ref{Int64}), P, n, r))
    return Int(n[]), Int(r[])
end

Base.size(P::KrylovPrecond) = (n = _size_rank(P)[1]; (n, n))
Base.size(P::KrylovPrecond, d::Integer) = d < 1 ? throw(ArgumentError("dimension must be ≥ 1")) : d <= 2 ? size(P)[1] : 1

"""
    rank(P::KrylovPrecond)

Cumulative dimension of the Krylov subspaces stored in `P` (`0` for a fresh preconditioner).
"""
LinearAlgebra.rank(P::KrylovPrecond) = _size_rank(P)[2]

function Base.show(io::IO, P::KrylovPrecond)
    if P.ptr == C_NULL
        print(io, "KrylovPrecond(<freed>)")
    else
        n, r = _size_rank(P)
        print(io, "KrylovPrecond(size=", n, ", rank=", r, ")")
    end
end

"""
    apply!(P::KrylovPrecond, x::AbstractVector{Float64}) -> x

Apply the preconditioner to `x` in place. An empty preconditioner (`size(P) == (0, 0)`) is the identity;
otherwise `length(x)` must equal `size(P, 1)`.
"""
function apply!(P::KrylovPrecond, x::AbstractVector{Float64})
    n, _ = _size_rank(P)
    n == 0 && return x
    length(x) == n || throw(DimensionMismatch("vector has length $(length(x)), preconditioner has size $n"))
    x_ = x isa Vector{Float64} ? x : Vector{Float64}(x)
    GC.@preserve P _check(ccall((:sctljl_krylov_apply, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64),
                                P, x_, n))
    x_ !== x && copyto!(x, x_)
    return x
end

Base.:*(P::KrylovPrecond, x::AbstractVector{<:Real}) = apply!(P, Vector{Float64}(x))

# The operator callback: A!(y, x) on reused buffers (small n) or zero-copy views of SCTL's vectors (see _with_cbufs).
struct _LinOp{F}
    A!::F
    n::Int
    bufs::CallbackBufs
end
_LinOp(A!, n) = _LinOp(A!, n, CallbackBufs(n))

function (op::_LinOp)(Ax::Ptr{Float64}, x::Ptr{Float64}, n::Int64)
    n == op.n || throw(DimensionMismatch("GMRES operator called with a vector of length $n"))
    _with_cbufs(op.A!, op.bufs, Ax, x, n)
end

# Specialized on the concrete state type (the C side passes the state object as its void* context), so calling the
# user's operator is statically dispatched.
function _linop_tramp(st::CallbackState, Ax::Ptr{Float64}, x::Ptr{Float64}, n::Int64)::Cint
    ok = _guard(st) do
        st.f(Ax, x, n)
    end
    return ok ? Cint(0) : Cint(1)
end

_linop_cfun(::S) where {S<:CallbackState} = @cfunction(_linop_tramp, Cint, (Ref{S}, Ptr{Float64}, Ptr{Float64}, Int64))

_as_op(A::AbstractMatrix) = (y, x) -> mul!(y, A, x)
_as_op(A) = A

"""
    gmres(A, b; tol=1e-10, maxiter=-1, abstol=false, orthogonalization=:mgs, reorth=0,
          precond=nothing, x0=nothing, verbose=false) -> (x=..., iters=...)

Solve the square linear system `A x = b` with SCTL's (unrestarted, right-preconditioned) GMRES. Returns a
`NamedTuple` with the solution `x::Vector{Float64}` and the number of iterations `iters` (so
`x, iters = gmres(A, b)` also works).

`A` is either an `AbstractMatrix` (applied with `mul!`) or a function `A!(y, x)` that overwrites `y` with
`A * x` in place. `y` and `x` are valid only during the call (reused buffers for short vectors, zero-copy views of
SCTL's memory for long ones): do not modify `x` and do not keep references to either.
Exceptions thrown by `A!` abort the solve and are rethrown unchanged.

Keywords:
- `tol`: stop when the residual norm is `≤ tol * norm(b)` (or `≤ tol` if `abstol=true`).
- `maxiter`: maximum number of iterations; `-1` means `length(b)` (no restarts are performed).
- `orthogonalization`: Gram-Schmidt scheme of the Arnoldi step, `:mgs` (modified, default) or `:cgs` (classical).
- `reorth`: number of extra re-orthogonalization passes (`1` is recommended with `:cgs`).
- `precond`: a [`KrylovPrecond`](@ref) used as right preconditioner; it is updated with this solve's Krylov
  subspace, so it accelerates later solves with the same operator.
- `x0`: initial guess (default: zero).
- `verbose`: print the residual norm at each iteration (from C, to the process' stdout).
"""
function gmres(A, b::AbstractVector{<:Real}; tol::Real=1e-10, maxiter::Integer=-1, abstol::Bool=false,
               orthogonalization::Symbol=:mgs, reorth::Integer=0, precond::Union{Nothing,KrylovPrecond}=nothing,
               x0::Union{Nothing,AbstractVector{<:Real}}=nothing, verbose::Bool=false)
    n = length(b)
    if A isa AbstractMatrix
        size(A) == (n, n) || throw(DimensionMismatch("A has size $(size(A)), expected ($n, $n) to match b"))
    end
    x0 === nothing || length(x0) == n ||
        throw(DimensionMismatch("x0 has length $(length(x0)), expected $n"))
    orthogonalization in (:mgs, :cgs) ||
        throw(ArgumentError("orthogonalization must be :mgs or :cgs, got :$orthogonalization"))
    reorth >= 0 || throw(ArgumentError("reorth must be non-negative"))
    tol >= 0 || throw(ArgumentError("tol must be non-negative"))
    maxiter >= -1 || throw(ArgumentError("maxiter must be -1 (no limit) or non-negative"))
    x = x0 === nothing ? zeros(n) : Vector{Float64}(x0)
    n == 0 && return (x=x, iters=0)
    b_ = _f64(b)
    st = CallbackState(_LinOp(_as_op(A), n))
    cfn = _linop_cfun(st)
    pptr = precond === nothing ? C_NULL : _krylov_ptr(precond)
    iters = Ref{Int64}(0)
    GC.@preserve st precond begin
        status = ccall((:sctljl_gmres, libsctljl), Cint,
                       (Ptr{Float64}, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Int64, Float64, Int64, Cint, Cint,
                        Cint, Int64, Ptr{Cvoid}, Ref{Int64}),
                       x, x0 !== nothing, cfn, pointer_from_objref(st), b_, n, tol, maxiter, abstol, verbose,
                       orthogonalization === :cgs ? 1 : 0, reorth, pptr, iters)
    end
    _check(status, st)
    return (x=x, iters=Int(iters[]))
end
