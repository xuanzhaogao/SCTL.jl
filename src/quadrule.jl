# Quadrature rules (ChebQuadRule, LegQuadRule, InterpQuadRule) and Lagrange interpolation.

export chebquad, legquad, legendre_poly
export interp_quad_rule, interp_quad_rules
export lagrange_interp_matrix, lagrange_interp, lagrange_derivative

"""
    chebquad(n) -> (x, w)

Fejér's first quadrature rule with `n` nodes on `[0, 1]`: the nodes are the Chebyshev points of the first kind
(an open rule) and the weights integrate the Chebyshev interpolant exactly.
"""
function chebquad(n::Integer)
    n > 0 || throw(ArgumentError("n must be positive"))
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    _check(ccall((:sctljl_chebquad, libsctljl), Cint, (Ptr{Float64}, Ptr{Float64}, Int64), x, w, n))
    return x, w
end

"""
    legquad(n) -> (x, w)

Gauss-Legendre quadrature rule with `n` nodes on `[0, 1]`.
"""
function legquad(n::Integer)
    n > 0 || throw(ArgumentError("n must be positive"))
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    _check(ccall((:sctljl_legquad, libsctljl), Cint, (Ptr{Float64}, Ptr{Float64}, Int64), x, w, n))
    return x, w
end

"""
    legendre_poly(x, degree) -> (P, dP)

Legendre polynomial ``P_\\mathrm{degree}`` and its derivative at the points `x` in `[-1, 1]`.
"""
function legendre_poly(x::AbstractVector{<:Real}, degree::Integer)
    degree >= 0 || throw(ArgumentError("degree must be non-negative"))
    x_ = _f64(x)
    n = length(x_)
    P = Vector{Float64}(undef, n)
    dP = Vector{Float64}(undef, n)
    n == 0 && return P, dP
    _check(ccall((:sctljl_legpoly, libsctljl), Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Int64),
                 P, dP, x_, n, degree))
    return P, dP
end

function _check_discretization(M, nodes, weights)
    length(nodes) == length(weights) || throw(DimensionMismatch("nodes and weights must have the same length"))
    size(M, 1) == length(nodes) || throw(DimensionMismatch("M must have one row per discretization node"))
    size(M, 2) > 0 || throw(ArgumentError("M must have at least one column (integrand)"))
end

"""
    interp_quad_rule(M, nodes, weights; eps=1e-16, order=0, interval=(0, 0), use_svd=true) -> (x, w, cond)

Generalized Chebyshev quadrature (InterpQuadRule): build a rule that integrates the functions sampled in the columns
of `M` (`M[i, j]` = `j`-th integrand at `nodes[i]`), where `nodes`, `weights` is a quadrature that resolves them.

`eps` is the truncation tolerance and `order` (if > 0) caps the number of nodes. When `interval = (a, b)` with
`a != b`, the nodes are picked from `[a, b]`. Returns the nodes, the weights and the condition number of the
interpolation matrix. `verbose=true` lets SCTL print a summary of each rule. See DOI:10.1137/080737046.
"""
function interp_quad_rule(M::AbstractMatrix{<:Real}, nodes::AbstractVector{<:Real}, weights::AbstractVector{<:Real};
                          eps::Real=1e-16, order::Integer=0, interval=(0.0, 0.0), use_svd::Bool=true, verbose::Bool=false)
    _check_discretization(M, nodes, weights)
    Mt = permutedims(_f64(M))  # row-major (nnodes x nfuncs)
    nds, wts = _f64(nodes), _f64(weights)
    cond = Ref(0.0)
    _withvecs(Float64, Float64) do hx, hw
        _check(ccall((:sctljl_interpquad_build, libsctljl), Cint,
                     (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Int64,
                      Float64, Int64, Float64, Float64, Cint, Cint),
                     hx, hw, cond, Mt, nds, wts, size(M, 1), size(M, 2),
                     eps, order, interval[1], interval[2], use_svd, verbose))
        return collect(hx), collect(hw), cond[]
    end
end

"""
    interp_quad_rules(M, nodes, weights; eps=Float64[], order=Int[], interval=(0, 0), use_svd=true)

Build several generalized Chebyshev quadrature rules at once, one per entry of `eps` and/or `order` (if both are
given they must have the same length and each rule uses the smaller node count). Returns a vector of `(x, w, cond)`.
"""
function interp_quad_rules(M::AbstractMatrix{<:Real}, nodes::AbstractVector{<:Real}, weights::AbstractVector{<:Real};
                           eps::AbstractVector{<:Real}=Float64[], order::AbstractVector{<:Integer}=Int[],
                           interval=(0.0, 0.0), use_svd::Bool=true, verbose::Bool=false)
    _check_discretization(M, nodes, weights)
    nrules = max(length(eps), length(order))
    nrules > 0 || throw(ArgumentError("give at least one tolerance in `eps` or one node count in `order`"))
    (isempty(eps) || isempty(order) || length(eps) == length(order)) ||
        throw(DimensionMismatch("eps and order must have the same length"))
    Mt = permutedims(_f64(M))
    nds, wts = _f64(nodes), _f64(weights)
    eps_, order_ = _f64(eps), _i64(order)
    hx = [VecHandle{Float64}() for _ in 1:nrules]
    hw = [VecHandle{Float64}() for _ in 1:nrules]
    try
        cond = Vector{Float64}(undef, nrules)
        px, pw = [h.ptr for h in hx], [h.ptr for h in hw]
        _check(ccall((:sctljl_interpquad_build_multi, libsctljl), Cint,
                     (Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                      Int64, Int64, Ptr{Float64}, Int64, Ptr{Int64}, Int64, Float64, Float64, Cint, Cint),
                     px, pw, cond, nrules, Mt, nds, wts, size(M, 1), size(M, 2),
                     eps_, length(eps_), order_, length(order_), interval[1], interval[2], use_svd, verbose))
        return [(collect(hx[k]), collect(hw[k]), cond[k]) for k in 1:nrules]
    finally
        foreach(free!, hx)
        foreach(free!, hw)
    end
end

# Specialized on the concrete state type (the C side passes the state object as its void* context).
function _integrand_tramp(st::CallbackState, x::Ptr{Float64}, n::Int64, out::Ptr{Cvoid})::Int64
    nfuncs = Ref{Int64}(-1)
    _guard(st) do
        M = st.f(copy(unsafe_wrap(Array, x, n)))
        size(M, 1) == n || throw(DimensionMismatch("integrand function must return a matrix with length(x) rows"))
        copyto!(VecHandle{Float64}(out), vec(permutedims(M)))
        nfuncs[] = size(M, 2)
    end
    return nfuncs[]
end

_integrand_cfun(::S) where {S<:CallbackState} = @cfunction(_integrand_tramp, Int64, (Ref{S}, Ptr{Float64}, Int64, Ptr{Cvoid}))

"""
    interp_quad_rule(f, a, b; eps=1e-16, order=0, interval=(0, 0), use_svd=true) -> (x, w, cond)

Generalized Chebyshev quadrature for the integrands `f` on `[a, b]`: `f(x::Vector{Float64})` returns a matrix with
`M[i, j]` = `j`-th integrand at `x[i]`. The integrands are first discretized by an adaptive composite
Gauss-Legendre rule to tolerance `eps`.
"""
function interp_quad_rule(f, a::Real, b::Real; eps::Real=1e-16, order::Integer=0, interval=(0.0, 0.0), use_svd::Bool=true, verbose::Bool=false)
    st = CallbackState(f)
    cfn = _integrand_cfun(st)
    cond = Ref(0.0)
    _withvecs(Float64, Float64) do hx, hw
        GC.@preserve st begin
            status = ccall((:sctljl_interpquad_build_fn, libsctljl), Cint,
                           (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Float64}, Ptr{Cvoid}, Ptr{Cvoid},
                            Float64, Float64, Float64, Int64, Float64, Float64, Cint, Cint),
                           hx, hw, cond, cfn, pointer_from_objref(st),
                           a, b, eps, order, interval[1], interval[2], use_svd, verbose)
        end
        _check(status, st)
        return collect(hx), collect(hw), cond[]
    end
end

"""
    lagrange_interp_matrix(src, trg) -> W

Lagrange interpolation matrix from nodes `src` to points `trg`: `W * f(src) ≈ f(trg)`
(`W` has size `length(trg) × length(src)`).
"""
function lagrange_interp_matrix(src::AbstractVector{<:Real}, trg::AbstractVector{<:Real})
    s, t = _f64(src), _f64(trg)
    ns, nt = length(s), length(t)
    ns > 0 || throw(ArgumentError("need at least one source node"))
    W = Matrix{Float64}(undef, nt, ns)  # column-major nt x ns == SCTL's row-major ns x nt
    nt == 0 && return W
    _check(ccall((:sctljl_lagrange_interp, libsctljl), Cint, (Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
                 W, s, ns, t, nt))
    return W
end

"""
    lagrange_interp(src, f, trg)

Interpolate values `f` at nodes `src` (a vector, or a matrix with one column per function) to the points `trg`.
"""
lagrange_interp(src::AbstractVector{<:Real}, f::AbstractVecOrMat, trg::AbstractVector{<:Real}) =
    lagrange_interp_matrix(src, trg) * f

"""
    lagrange_derivative(f, nodes)

Derivative, at `nodes`, of the polynomial interpolating `f` through `nodes`. `f` is a vector of length
`length(nodes)` or a matrix with one column per function; the result has the same shape.
"""
function lagrange_derivative(f::AbstractVecOrMat{<:Real}, nodes::AbstractVector{<:Real})
    size(f, 1) == length(nodes) || throw(DimensionMismatch("f must have one row per node"))
    f_, x = _f64(f), _f64(nodes)
    df = similar(f_)
    isempty(f_) && return df
    _check(ccall((:sctljl_lagrange_derivative, libsctljl), Cint, (Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
                 df, f_, length(f_), x, length(x)))
    return df
end
