# Near-field pieces of a boundary integral operator, separately: far-field nodes and near radii, the near list, the
# element quadratures (accurate element-to-target matrices) and the assembly of the sparse near correction. Each step
# returns plain Julia arrays, so any of them can be inspected or replaced (e.g. by a quadrature written in Julia).

export far_field_nodes, NearList, near_list, near_targets, npairs
export ElemQuadrature, near_matrix, self_matrices, near_blocks, node_rule_block, node_rule_blocks
export assemble_near_correction, near_correction

"""
    far_field_nodes(elist; tol = 1e-10) -> (; X, normal, weights, radius)

The far-field (smooth node) quadrature of `elist` and the near radius of each node, as
[`BoundaryIntegralOp`](@ref) uses them at accuracy `tol`: node positions `X` and unit normals `normal`
(`3 × nnodes`), weights (area element times tensor Gauss-Legendre weights) and `radius`. Beyond `radius[i]` of
every node `i` of an element the node rule is accurate to `tol`; closer targets are near the element (see
[`near_list`](@ref)). The radius is the distance to the Bernstein ellipse of `ρ = (64 / (15 tol))^(1/(2 order))` in
parameter space, scaled by the tangent length.
"""
function far_field_nodes(el::QuadElemList; tol::Real = 1e-10)
    tol > 0 || throw(ArgumentError("tol must be positive"))
    X, N = node_coord(el)
    n = nnodes(el)
    w = Vector{Float64}(undef, n)
    r = Vector{Float64}(undef, n)
    n == 0 || _check(ccall((:sctljl_quadelem_far_nodes, libsctljl), Cint,
                           (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Int64, Float64), el, w, r, n, tol))
    return (; X = X, normal = N, weights = w, radius = r)
end

# ---------------------------------------------------------------------------------------------------------------
# near list
# ---------------------------------------------------------------------------------------------------------------

"""
    NearList

Which targets are near which element: the targets near element `e` are [`near_targets`](@ref)`(nl, e)` (sorted,
1-based), stored compressed as `nl.trg[nl.ptr[e]:nl.ptr[e+1]-1]`. `nl.ntrg` is the number of targets. From
[`near_list`](@ref), or construct it yourself (`NearList(ptr, trg, ntrg)`) for your own criterion.
"""
struct NearList
    ptr::Vector{Int}
    trg::Vector{Int}
    ntrg::Int
    function NearList(ptr::Vector{Int}, trg::Vector{Int}, ntrg::Integer)
        (!isempty(ptr) && ptr[1] == 1 && ptr[end] == length(trg) + 1 && issorted(ptr)) ||
            throw(ArgumentError("ptr must start at 1, be non-decreasing and end at length(trg) + 1"))
        all(t -> 1 <= t <= ntrg, trg) || throw(ArgumentError("target indices must be in 1:$ntrg"))
        return new(ptr, trg, Int(ntrg))
    end
end

nelem(nl::NearList) = length(nl.ptr) - 1
"""
    near_targets(nl::NearList, e) -> view

The (sorted, 1-based) indices of the targets near element `e`.
"""
near_targets(nl::NearList, e::Integer) = view(nl.trg, nl.ptr[e]:nl.ptr[e+1]-1)
"""
    npairs(nl::NearList)

The number of (element, near target) pairs.
"""
npairs(nl::NearList) = length(nl.trg)
Base.show(io::IO, nl::NearList) =
    print(io, "NearList(", nelem(nl), " elements, ", nl.ntrg, " targets, ", npairs(nl), " near pairs)")

"""
    near_list(Xs, radius, nodes_per_elem, Xt) -> NearList
    near_list(elist, Xt = nothing; tol = 1e-10) -> NearList

SCTL's near test (that of [`BoundaryIntegralOp`](@ref), an octree search): target `t` is near element `e` if it
lies within `radius[i]` of some node `i` of `e`.

The first form works for any discretization: source nodes `Xs` (`3 × Ns`, element by element), one radius per node,
and `nodes_per_elem` (an integer, or one count per element); targets `Xt` (`3 × Nt`). The second takes the nodes and
radii of [`far_field_nodes`](@ref)`(elist; tol)`, with the surface nodes as default targets.
"""
function near_list(Xs, radius::AbstractVector{<:Real}, nodes_per_elem, Xt)
    Xs_, ns = _points(Xs, "source nodes")
    Xt_, nt = _points(Xt, "targets")
    length(radius) == ns || throw(DimensionMismatch("need one radius per source node ($ns), got $(length(radius))"))
    cnt = nodes_per_elem isa Integer ?
        (nodes_per_elem > 0 && ns % nodes_per_elem == 0 ? fill(Int64(nodes_per_elem), ns ÷ nodes_per_elem) :
         throw(DimensionMismatch("$ns nodes are not a whole number of elements of $nodes_per_elem nodes"))) :
        _i64(collect(nodes_per_elem))
    sum(cnt; init = 0) == ns || throw(DimensionMismatch("the node counts add up to $(sum(cnt; init = 0)), not $ns"))
    ne = length(cnt)
    ptr = Vector{Int64}(undef, ne + 1)
    r = _f64(collect(radius))
    trg = _withvecs(Int64) do hidx
        _check(ccall((:sctljl_near_list, libsctljl), Cint,
                     (Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Int64}, Int64, Ptr{Float64}, Int64, Ptr{Int64}, Ptr{Cvoid}),
                     Xs_, r, ns, cnt, ne, Xt_, nt, ptr, hidx))
        collect(hidx)
    end
    return NearList(Vector{Int}(ptr .+ 1), Vector{Int}(trg .+ 1), nt)
end

function near_list(el::QuadElemList, Xt = nothing; tol::Real = 1e-10)
    ff = far_field_nodes(el; tol = tol)
    return near_list(ff.X, ff.radius, order(el)^2, Xt === nothing ? ff.X : Xt)
end

# ---------------------------------------------------------------------------------------------------------------
# element quadratures
# ---------------------------------------------------------------------------------------------------------------

"""
    ElemQuadrature(kernel)

The singular and near-singular quadratures of [`QuadElemList`](@ref)s for `kernel` (one of
[`bie_supported_kernels()`](@ref)), without an operator: [`near_matrix`](@ref), [`near_blocks`](@ref) and
[`self_matrices`](@ref) return the accurate element-to-target matrices that [`BoundaryIntegralOp`](@ref) builds in
its setup. The scheme is the element list's ([`set_quad_scheme!`](@ref)).

All matrices map nodal densities of one element to potentials at targets, laid out like [`kernel_matrix`](@ref):
`(dof*Nt) × (src_dim*nnode)`, row `(t-1)*dof + j`, column `(i-1)*src_dim + k` for node `i` of the element, with
`dof = trg_dim(kernel)` (or `trg_dim/3` when target normals are given: the potential dotted with the normal, as in
`trg_normal_dot_prod`). They include the quadrature weights.
"""
mutable struct ElemQuadrature{K<:AbstractKernel}
    ptr::Ptr{Cvoid}
    kernel::K
    function ElemQuadrature(ker::K) where {K<:AbstractKernel}
        K in _BIE_KERNELS ||
            throw(ArgumentError("element quadratures are not available for $(kernel_name(ker)); supported: $(join(nameof.(_BIE_KERNELS), ", "))"))
        h = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_quad_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Cint, Float64), h, _kernel_id(ker), _kernel_ctx(ker)))
        q = new{K}(h[], ker)
        finalizer(_free!, q)
        return q
    end
end

function _free!(q::ElemQuadrature)
    if q.ptr != C_NULL
        ccall((:sctljl_quad_free, libsctljl), Cvoid, (Ptr{Cvoid},), q.ptr)
        q.ptr = C_NULL
    end
    return nothing
end
Base.close(q::ElemQuadrature) = _free!(q)
function Base.unsafe_convert(::Type{Ptr{Cvoid}}, q::ElemQuadrature)
    q.ptr == C_NULL && throw(ArgumentError("ElemQuadrature has been closed"))
    return q.ptr
end
Base.show(io::IO, q::ElemQuadrature) = print(io, "ElemQuadrature(", kernel_name(q.kernel), ")")

function _trg_dof(ker, dot::Bool)
    dot && trg_dim(ker) % 3 != 0 &&
        throw(ArgumentError("target normals need a kernel whose trg_dim is a multiple of 3 ($(kernel_name(ker)) has $(trg_dim(ker)))"))
    return dot ? trg_dim(ker) ÷ 3 : trg_dim(ker)
end

function _trg_normals(normals, nt)
    normals === nothing && return Float64[], 0
    N_, nn = _points(normals, "target normals")
    nn == nt || throw(DimensionMismatch("need one normal per target ($nt), got $nn"))
    return N_, nn
end

"""
    near_matrix(q::ElemQuadrature, elist, e, Xt; normals = nothing, tol = 1e-10) -> Matrix

The accurate matrix from element `e` of `elist` to the targets `Xt` (`3 × Nt`), by `elist`'s near-singular rule
(split at each target's closest point, refined toward it). For targets that coincide with a node of the element use
[`near_blocks`](@ref) or [`self_matrices`](@ref). Layout: see [`ElemQuadrature`](@ref).
"""
function near_matrix(q::ElemQuadrature, el::QuadElemList, e::Integer, Xt; normals = nothing, tol::Real = 1e-10)
    1 <= e <= nelem(el) || throw(BoundsError(1:nelem(el), e))
    Xt_, nt = _points(Xt, "targets")
    N_, nn = _trg_normals(normals, nt)
    dof = _trg_dof(q.kernel, nn > 0)
    M = Matrix{Float64}(undef, dof * nt, src_dim(q.kernel) * order(el)^2)
    @serial_blas _check(ccall((:sctljl_quad_near, libsctljl), Cint,
                              (Ptr{Cvoid}, Ptr{Cvoid}, Int64, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Float64, Ptr{Float64}, Int64),
                              q, el, e - 1, Xt_, nt, N_, nn, tol, M, length(M)))
    return M
end

"""
    self_matrices(q::ElemQuadrature, elist; tol = 1e-10, trg_normal_dot_prod = false) -> Array{Float64,3}

The singular (self-interaction) matrices of all elements: `S[:, :, e]` maps the nodal densities of element `e` to
the potential at its own nodes (the principal value for the Duffy and tensor-product schemes), `(dof*nnode) ×
(src_dim*nnode)`. With `trg_normal_dot_prod` the potential is dotted with the element normal at each node.
"""
function self_matrices(q::ElemQuadrature, el::QuadElemList; tol::Real = 1e-10, trg_normal_dot_prod::Bool = false)
    dof = _trg_dof(q.kernel, trg_normal_dot_prod)
    nn = order(el)^2
    S = Array{Float64,3}(undef, dof * nn, src_dim(q.kernel) * nn, nelem(el))
    @serial_blas _check(ccall((:sctljl_quad_self, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Cint, Ptr{Float64}, Int64),
                              q, el, tol, trg_normal_dot_prod, S, length(S)))
    return S
end

"""
    near_blocks(q::ElemQuadrature, elist, nl::NearList, Xt; normals = nothing, tol = 1e-10) -> Vector{Matrix{Float64}}

The accurate matrices of all near pairs, as [`BoundaryIntegralOp`](@ref) builds them (in parallel): block `e` maps
element `e`'s nodal densities to its near targets `near_targets(nl, e)` (in that order). A target that coincides
with a node of the element gets the self-interaction row of that node (with the element's normal for the dot
product). Layout: see [`ElemQuadrature`](@ref).
"""
function near_blocks(q::ElemQuadrature, el::QuadElemList, nl::NearList, Xt; normals = nothing, tol::Real = 1e-10)
    nelem(nl) == nelem(el) || throw(DimensionMismatch("near list has $(nelem(nl)) elements, the element list $(nelem(el))"))
    Xt_, nt = _points(Xt, "targets")
    nt == nl.ntrg || throw(DimensionMismatch("near list has $(nl.ntrg) targets, got $nt"))
    N_, nn = _trg_normals(normals, nt)
    dof = _trg_dof(q.kernel, nn > 0)
    ncol = src_dim(q.kernel) * order(el)^2
    buf = Vector{Float64}(undef, ncol * dof * npairs(nl))
    ptr0 = Int64.(nl.ptr .- 1)
    idx0 = Int64.(nl.trg .- 1)
    @serial_blas _check(ccall((:sctljl_quad_near_blocks, libsctljl), Cint,
                              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64,
                               Float64, Ptr{Float64}, Int64),
                              q, el, ptr0, idx0, Xt_, nt, N_, nn, tol, buf, length(buf)))
    blocks = Vector{Matrix{Float64}}(undef, nelem(nl))
    off = 0
    for e in 1:nelem(nl)
        m = dof * (nl.ptr[e+1] - nl.ptr[e])
        blocks[e] = reshape(buf[off+1:off+m*ncol], m, ncol)
        off += m * ncol
    end
    return blocks
end

# ---------------------------------------------------------------------------------------------------------------
# node rule and assembly
# ---------------------------------------------------------------------------------------------------------------

"""
    node_rule_block(kernel, Xs, Xn, w, Xt; normals = nothing) -> Matrix

The plain node rule from source nodes `Xs` (normals `Xn` for kernels with source normals, else `nothing`; weights
`w`) to the targets `Xt`: [`kernel_matrix`](@ref) with each source column scaled by its weight, dotted with the
target `normals` if given. This is what the far field (FMM) contributes for a pair, and what a near correction
subtracts. Layout as [`ElemQuadrature`](@ref).
"""
function node_rule_block(ker::AbstractKernel, Xs, Xn, w::AbstractVector{<:Real}, Xt; normals = nothing)
    K = kernel_matrix(ker, Xs, normal_dim(ker) > 0 ? Xn : nothing, Xt)
    sd = src_dim(ker)
    size(K, 2) == sd * length(w) || throw(DimensionMismatch("need one weight per source node"))
    for s in eachindex(w), k in 1:sd
        @views K[:, (s-1)*sd+k] .*= w[s]
    end
    normals === nothing && return K
    N_, nt = _trg_normals(normals, size(K, 1) ÷ trg_dim(ker))
    dof = _trg_dof(ker, true)
    Nm = reshape(N_, 3, nt)
    D = zeros(dof * nt, size(K, 2))
    for t in 1:nt, a in 1:dof, l in 1:3
        @views D[(t-1)*dof+a, :] .+= Nm[l, t] .* K[((t-1)*dof+a-1)*3+l, :]
    end
    return D
end

"""
    node_rule_blocks(kernel, elist, nl::NearList, Xt; normals = nothing, tol = 1e-10) -> Vector{Matrix{Float64}}

[`node_rule_block`](@ref) for every element of `elist` and its near targets, with the far-field nodes of
[`far_field_nodes`](@ref): the part of the far field that a near correction replaces.
"""
function node_rule_blocks(ker::AbstractKernel, el::QuadElemList, nl::NearList, Xt; normals = nothing, tol::Real = 1e-10)
    ff = far_field_nodes(el; tol = tol)
    Xt_ = reshape(_points(Xt, "targets")[1], 3, :)
    N_ = normals === nothing ? nothing : reshape(_trg_normals(normals, size(Xt_, 2))[1], 3, :)
    nn = order(el)^2
    blocks = Vector{Matrix{Float64}}(undef, nelem(nl))
    Threads.@threads for e in 1:nelem(nl)
        rg = (e-1)*nn+1:e*nn
        ts = near_targets(nl, e)
        blocks[e] = node_rule_block(ker, ff.X[:, rg], ff.normal[:, rg], ff.weights[rg], Xt_[:, ts];
                                    normals = N_ === nothing ? nothing : N_[:, ts])
    end
    return blocks
end

"""
    assemble_near_correction(nl::NearList, blocks, nodes_per_elem; src_dim = 1, dof = 1, nsrc = nothing,
                             subtract = nothing) -> SparseMatrixCSC

Scatter per-element blocks (block `e`: `(dof * length(near_targets(nl, e))) × (src_dim * nodes of e)`, layout as
[`ElemQuadrature`](@ref)) into the sparse `(dof*Nt) × (src_dim*Nsrc)` matrix acting on all nodal densities, element
`e`'s nodes being consecutive. With `subtract` (blocks of the same shapes, e.g. [`node_rule_blocks`](@ref)) the
entries are `blocks[e] - subtract[e]`: the near correction, to be added to the node rule. The blocks can come from
any quadrature, e.g. one written in Julia.
"""
function assemble_near_correction(nl::NearList, blocks::AbstractVector{<:AbstractMatrix}, nodes_per_elem;
                                  src_dim::Integer = 1, dof::Integer = 1, nsrc = nothing, subtract = nothing)
    ne = nelem(nl)
    length(blocks) == ne || throw(DimensionMismatch("need one block per element ($ne)"))
    cnt = nodes_per_elem isa Integer ? fill(Int(nodes_per_elem), ne) : collect(Int, nodes_per_elem)
    length(cnt) == ne || throw(DimensionMismatch("need one node count per element ($ne)"))
    off = cumsum([0; cnt])
    ns = nsrc === nothing ? off[end] : Int(nsrc)
    nz = sum(length, blocks; init = 0)
    I = Vector{Int}(undef, nz); J = Vector{Int}(undef, nz); V = Vector{Float64}(undef, nz)
    k = 0
    for e in 1:ne
        ts = near_targets(nl, e)
        B = blocks[e]
        size(B) == (dof * length(ts), src_dim * cnt[e]) ||
            throw(DimensionMismatch("block $e is $(size(B)), expected $((dof * length(ts), src_dim * cnt[e]))"))
        S = subtract === nothing ? nothing : subtract[e]
        S === nothing || size(S) == size(B) || throw(DimensionMismatch("subtracted block $e has the wrong size"))
        for c in axes(B, 2), r in axes(B, 1)
            k += 1
            t = ts[(r-1) ÷ dof + 1]
            I[k] = (t-1)*dof + (r-1) % dof + 1
            J[k] = off[e]*src_dim + c
            V[k] = S === nothing ? B[r, c] : B[r, c] - S[r, c]
        end
    end
    return sparse(I, J, V, dof * nl.ntrg, src_dim * ns)
end

"""
    near_correction(kernel, elist; targets = nothing, normals = nothing, trg_normal_dot_prod = false, tol = 1e-10)
        -> (C, nl)

SCTL's near correction for `elist`, assembled from the pieces ([`near_list`](@ref), [`near_blocks`](@ref),
[`node_rule_blocks`](@ref), [`assemble_near_correction`](@ref)): the sparse matrix `C` such that

    U = (node rule over all nodes) * F + C * F

is [`BoundaryIntegralOp`](@ref)'s potential at the targets, for nodal densities `F`. Targets default to the surface
nodes; `trg_normal_dot_prod = true` there dots the potential with the surface normals, or give `normals` for other
targets. Returns the near list too, so the pieces can be recomputed or replaced selectively.
"""
function near_correction(ker::AbstractKernel, el::QuadElemList; targets = nothing, normals = nothing,
                         trg_normal_dot_prod::Bool = false, tol::Real = 1e-10)
    ff = far_field_nodes(el; tol = tol)
    Xt = targets === nothing ? ff.X : targets
    Nt = normals !== nothing ? normals :
         (trg_normal_dot_prod ? (targets === nothing ? ff.normal :
                                 throw(ArgumentError("give the normals of the targets"))) : nothing)
    nl = near_list(ff.X, ff.radius, order(el)^2, Xt)
    q = ElemQuadrature(ker)
    acc = near_blocks(q, el, nl, Xt; normals = Nt, tol = tol)
    close(q)
    plain = node_rule_blocks(ker, el, nl, Xt; normals = Nt, tol = tol)
    C = assemble_near_correction(nl, acc, order(el)^2; src_dim = src_dim(ker),
                                 dof = _trg_dof(ker, Nt !== nothing), subtract = plain)
    return C, nl
end
