# Boundary integral operators: quadrilateral surface elements (QuadElemList) and BoundaryIntegralOp.

export QuadElemList, nelem, nnodes, order, param_nodes, node_coord, node_weights, elem_geom
export set_quad_scheme!, quad_scheme, write_vtk
export BoundaryIntegralOp, bie_supported_kernels, add_elem_list!, delete_elem_list!, elem_list_names
export set_target_coord!, set_target_normal!, set_accuracy!, accuracy, dim, setup!, clear_setup!
export compute_potential, compute_potential!, sqrt_scaling!, inv_sqrt_scaling!

# ---------------------------------------------------------------------------------------------------------------
# QuadElemList
# ---------------------------------------------------------------------------------------------------------------

const _QUAD_SCHEMES = (:tensor_product, :duffy, :hedgehog)

"""
    QuadElemList(order, coord)
    QuadElemList(f, order, nelem)

A list of quadrilateral surface elements in 3D (SCTL's experimental `QuadElemList<double>`). Each element is the
tensor-product polynomial map from the parameter square ``[0,1]^2`` that interpolates its `order × order` nodes,
which lie at the parameter points `(u_i, v_j)`, `u = v =` [`param_nodes(order)`](@ref) (Gauss-Legendre nodes on
`[0, 1]`). The unit normal is along ``∂X/∂u × ∂X/∂v``.

`coord` holds the node coordinates in SCTL's order (element by element, node `(i, j)` at index `(i-1)*order + j`
within its element, i.e. `v` varies fastest), as any of

- an array of size `(3, order, order, nelem)` with `coord[:, j, i, e]` = node at `(u_i, v_j)` of element `e`
  (**note** the `j, i` order: the second index runs over `v`);
- a `3 × (order^2 * nelem)` matrix (node `(i, j)` of element `e` in column `(e-1)*order^2 + (i-1)*order + j`);
- a vector of length `3 * order^2 * nelem` with the same content.

(SCTL copies these array-of-structs coordinates into its own per-element `x…, y…, z…` storage.) The second form
calls `f(e, u, v)` (with `e` in `1:nelem`, `u, v` in `[0, 1]`) for every node and expects a point (3 numbers), which
avoids getting the layout wrong.

For use in a [`BoundaryIntegralOp`](@ref), `order` must be one of 4, 8, 12, 16, 20. Node data (densities,
potentials, [`node_coord`](@ref), [`node_weights`](@ref)) follow the same node order. The C++ object is freed by a
finalizer, or explicitly with `close(elist)`.
"""
mutable struct QuadElemList
    ptr::Ptr{Cvoid}
    order::Int
    nelem::Int
    scheme::Symbol
    function QuadElemList(order::Integer, coord::AbstractArray{<:Real})
        order > 0 || throw(ArgumentError("order must be positive"))
        c = _f64(coord)
        npe = 3 * order^2
        if ndims(c) == 4
            size(c)[1:3] == (3, order, order) ||
                throw(DimensionMismatch("coord must have size (3, order, order, nelem) = (3, $order, $order, nelem), got $(size(c))"))
        elseif ndims(c) == 2
            size(c, 1) == 3 || throw(DimensionMismatch("coord must be a 3×N matrix"))
        elseif ndims(c) != 1
            throw(DimensionMismatch("coord must be a vector, a 3×N matrix or a (3, order, order, nelem) array"))
        end
        length(c) % npe == 0 ||
            throw(DimensionMismatch("number of coordinates ($(length(c))) is not a multiple of 3*order^2 = $npe"))
        h = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_quadelem_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Int64, Ptr{Float64}, Int64),
                     h, order, c, length(c)))
        el = new(h[], order, length(c) ÷ npe, :duffy)
        finalizer(_free!, el)
        return el
    end
end

function QuadElemList(f, order::Integer, nelem::Integer)
    order > 0 || throw(ArgumentError("order must be positive"))
    nelem >= 0 || throw(ArgumentError("nelem must be non-negative"))
    t = param_nodes(order)
    X = Array{Float64}(undef, 3, order, order, nelem)
    for e in 1:nelem, i in 1:order, j in 1:order
        p = f(e, t[i], t[j])
        length(p) == 3 || throw(DimensionMismatch("f(e, u, v) must return 3 coordinates"))
        X[:, j, i, e] .= p
    end
    return QuadElemList(order, X)
end

function _free!(el::QuadElemList)
    if el.ptr != C_NULL
        ccall((:sctljl_quadelem_free, libsctljl), Cvoid, (Ptr{Cvoid},), el.ptr)
        el.ptr = C_NULL
    end
    return nothing
end

"""
    close(elist::QuadElemList)

Free the C++ element list now (otherwise done by the finalizer). Operators it was added to keep their own copies.
"""
Base.close(el::QuadElemList) = _free!(el)

function _ptr(el::QuadElemList)
    el.ptr == C_NULL && throw(ArgumentError("QuadElemList has been closed"))
    return el.ptr
end

# Passing the object itself to ccall (instead of its raw pointer) keeps it rooted, so the finalizer cannot free the
# C++ object during the call.
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::QuadElemList) = _ptr(x)

Base.show(io::IO, el::QuadElemList) =
    print(io, "QuadElemList(order=", el.order, ", nelem=", el.nelem, ", scheme=:", el.scheme, el.ptr == C_NULL ? ", closed)" : ")")

function _qel_info(el::QuadElemList)
    info = zeros(Int64, 2)
    _check(ccall((:sctljl_quadelem_info, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Int64}), el, info))
    return info
end

"""
    nelem(elist::QuadElemList) -> Int

Number of elements (SCTL's `Size()`).
"""
nelem(el::QuadElemList) = Int(_qel_info(el)[1])

"""
    order(elist::QuadElemList) -> Int

Number of nodes of each element in each parameter direction (SCTL's `Order()`).
"""
order(el::QuadElemList) = Int(_qel_info(el)[2])

"""
    nnodes(elist::QuadElemList) -> Int

Total number of nodes, `order^2 * nelem`.
"""
nnodes(el::QuadElemList) = el.order^2 * el.nelem

"""
    param_nodes(order) -> Vector{Float64}

Parameter values (in each of `u`, `v`) of the element nodes: the `order` Gauss-Legendre nodes on `[0, 1]`.
"""
function param_nodes(n::Integer)
    n > 0 || throw(ArgumentError("order must be positive"))
    x = Vector{Float64}(undef, n)
    _check(ccall((:sctljl_quadelem_param_nodes, libsctljl), Cint, (Ptr{Float64}, Int64), x, n))
    return x
end

"""
    node_coord(elist) -> (X, N)

Node positions `X` and unit normals `N` (each `3 × nnodes(elist)`, in node order; see [`QuadElemList`](@ref)).
"""
function node_coord(el::QuadElemList)
    n = nnodes(el)
    X = Matrix{Float64}(undef, 3, n)
    N = Matrix{Float64}(undef, 3, n)
    _ptr(el)  # throws if closed
    n == 0 && return X, N
    _check(ccall((:sctljl_quadelem_node_coord, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Int64),
                 el, X, N, n))
    return X, N
end

"""
    node_weights(elist) -> Vector{Float64}

Surface quadrature weights at the nodes (area element times the tensor-product Gauss-Legendre weights), so that
`sum(node_weights(elist) .* f)` approximates the surface integral of `f` (`sum` alone gives the area). These are the
far-field quadrature weights of the [`BoundaryIntegralOp`](@ref).
"""
function node_weights(el::QuadElemList)
    n = nnodes(el)
    w = Vector{Float64}(undef, n)
    _ptr(el)  # throws if closed
    n == 0 && return w
    _check(ccall((:sctljl_quadelem_far_nodes, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Int64, Float64),
                 el, w, C_NULL, n, 1e-10))
    return w
end

"""
    elem_geom(elist, e, u, v) -> (X, normal, area, dXdu, dXdv)

Geometry of element `e` (1-based) on the tensor-product grid of parameters `u` (length `nu`) and `v` (length `nv`),
values in `[0, 1]`: positions `X`, unit normals, the area element ``|∂X/∂u × ∂X/∂v|`` (a vector) and the tangents,
each `3 × (nu*nv)` (the area `nu*nv`) with point `(a, b)` = `(u[a], v[b])` in column `(a-1)*nv + b` (`v` fastest, as
for the nodes; `reshape(X, 3, nv, nu)` gives `X[:, b, a]`).
"""
function elem_geom(el::QuadElemList, e::Integer, u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    1 <= e <= el.nelem || throw(BoundsError(el, e))
    u_, v_ = _f64(u), _f64(v)
    N = length(u_) * length(v_)
    X, Xn, du, dv = (Matrix{Float64}(undef, 3, N) for _ in 1:4)
    Xa = Vector{Float64}(undef, N)
    _ptr(el)  # throws if closed
    if N > 0
        _check(ccall((:sctljl_quadelem_geom, libsctljl), Cint,
                     (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                      Ptr{Float64}, Int64, Ptr{Float64}, Int64, Int64),
                     el, X, Xn, Xa, du, dv, u_, length(u_), v_, length(v_), e - 1))
    end
    return (X=X, normal=Xn, area=Xa, dXdu=du, dXdv=dv)
end

"""
    set_quad_scheme!(elist, scheme) -> elist

Quadrature scheme for the singular (target on a node of the element) and near-singular interactions:
`:duffy` (the default), `:tensor_product` or `:hedgehog`. At on-surface targets, `:duffy` and `:tensor_product`
give the principal value and `:hedgehog` the limit from the side the normal points to (for a double-layer kernel
they differ by the jump, half the density). `:duffy` and `:tensor_product` are only correct for kernels that grow
at most like ``1/r`` along the surface (single and double layers, gradients dotted with the target normal); use
`:hedgehog` for steeper ones (SCTL prints a warning otherwise).

Operators copy the element list in [`add_elem_list!`](@ref): set the scheme before adding it.
"""
function set_quad_scheme!(el::QuadElemList, scheme::Symbol)
    k = findfirst(==(scheme), _QUAD_SCHEMES)
    k === nothing && throw(ArgumentError("scheme must be one of $(_QUAD_SCHEMES), got :$scheme"))
    _check(ccall((:sctljl_quadelem_set_scheme, libsctljl), Cint, (Ptr{Cvoid}, Cint), el, k - 1))
    el.scheme = scheme
    return el
end

"""
    quad_scheme(elist) -> Symbol

The quadrature scheme set by [`set_quad_scheme!`](@ref).
"""
quad_scheme(el::QuadElemList) = el.scheme

"""
    write_vtk(elist, fname[, F]) -> String

Write the elements (each as a grid of quadrilateral cells through its nodes and edges) to the VTK unstructured-grid
file `fname * "000000.vtu"` (SCTL appends the 6-digit process rank), optionally with node data `F` (any number of
values per node, `dof × nnodes`). Returns the file name.
"""
function write_vtk(el::QuadElemList, fname::AbstractString, F=nothing)
    F_ = F === nothing ? Float64[] : _f64(F)
    n = nnodes(el)
    (isempty(F_) || (n > 0 && length(F_) % n == 0)) ||
        throw(DimensionMismatch("node data must have a multiple of nnodes = $n values"))
    _check(ccall((:sctljl_quadelem_write_vtk, libsctljl), Cint, (Ptr{Cvoid}, Cstring, Ptr{Float64}, Int64),
                 el, fname, F_, length(F_)))
    out = string(fname, "000000.vtu")
    isfile(out) || throw(SCTLError("could not write $out"))
    return out
end

# ---------------------------------------------------------------------------------------------------------------
# BoundaryIntegralOp
# ---------------------------------------------------------------------------------------------------------------

const _BIE_KERNELS = (Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU, Laplace3D_DxdU,
                      Helmholtz3D_FxU, Helmholtz3D_DxU, Helmholtz3D_FxdU,
                      Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT)

"""
    bie_supported_kernels() -> Tuple

The kernel types for which a [`BoundaryIntegralOp`](@ref) is compiled.
"""
bie_supported_kernels() = _BIE_KERNELS

const _BIE_ORDERS = (4, 8, 12, 16, 20)

"""
    BoundaryIntegralOp(kernel; trg_normal_dot_prod=false, accuracy=1e-10)

Boundary integral operator ``U(x) = \\int_Γ K(x - y) F(y) dS_y`` of an [`AbstractKernel`](@ref) over the element
lists added with [`add_elem_list!`](@ref) (SCTL's `BoundaryIntegralOp<double,Kernel>`), with singular and
near-singular quadratures from the element lists (see [`set_quad_scheme!`](@ref)) and the smooth node quadrature
elsewhere. Supported kernels: [`bie_supported_kernels()`](@ref) (`Laplace3D_FxU`, `Laplace3D_DxU`, `Laplace3D_FxdU`,
`Laplace3D_DxdU`, `Helmholtz3D_FxU`, `Helmholtz3D_DxU`, `Helmholtz3D_FxdU`, `Stokes3D_FxU`, `Stokes3D_DxU`,
`Stokes3D_FxT`).

With `trg_normal_dot_prod=true` the potential (`trg_dim` a multiple of 3) is dotted with the target normals, e.g.
`Laplace3D_FxdU` gives the adjoint double layer ``∂S/∂n_x``; the normals are the surface normals for the default
(on-surface) targets, otherwise they must be given with [`set_target_normal!`](@ref).

Layout: the input density has `src_dim(kernel)` values per surface node (nodes of all element lists, the lists in
lexicographic order of their names); the output has `trg_dim(kernel)` (or `trg_dim/3`) values per target. Targets
are the surface nodes unless set with [`set_target_coord!`](@ref).

This build has no PVFMM: the far field is a direct ``O(N_{nodes} N_{targets})`` sum (OpenMP-parallel), so it is
meant for moderate sizes. Periodicity needs PVFMM and is not available. The C++ object is freed by a finalizer, or
explicitly with `close(op)`.
"""
mutable struct BoundaryIntegralOp{K<:AbstractKernel}
    ptr::Ptr{Cvoid}
    kernel::K
    trg_normal_dot_prod::Bool
    elem_lists::Dict{String,QuadElemList} # the Julia lists added (SCTL holds copies)
    ntrg::Int                            # number of user targets (-1: surface nodes)
    nnormal::Int                         # number of user target normals (-1: none)
    function BoundaryIntegralOp(ker::K; trg_normal_dot_prod::Bool=false, accuracy::Real=1e-10) where {K<:AbstractKernel}
        K in _BIE_KERNELS ||
            throw(ArgumentError("BoundaryIntegralOp is not available for $(kernel_name(ker)); supported: $(join(nameof.(_BIE_KERNELS), ", "))"))
        (!trg_normal_dot_prod || trg_dim(ker) % 3 == 0) ||
            throw(ArgumentError("trg_normal_dot_prod needs a kernel whose trg_dim is a multiple of 3 ($(kernel_name(ker)) has $(trg_dim(ker)))"))
        h = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_bie_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Cint, Float64, Cint),
                     h, _kernel_id(ker), _kernel_ctx(ker), trg_normal_dot_prod))
        op = new{K}(h[], ker, trg_normal_dot_prod, Dict{String,QuadElemList}(), -1, -1)
        finalizer(_free!, op)
        set_accuracy!(op, accuracy)
        return op
    end
end

function _free!(op::BoundaryIntegralOp)
    if op.ptr != C_NULL
        ccall((:sctljl_bie_free, libsctljl), Cvoid, (Ptr{Cvoid},), op.ptr)
        op.ptr = C_NULL
    end
    return nothing
end

"""
    close(op::BoundaryIntegralOp)

Free the C++ operator now (otherwise done by the finalizer). The operator cannot be used afterwards.
"""
Base.close(op::BoundaryIntegralOp) = _free!(op)

function _ptr(op::BoundaryIntegralOp)
    op.ptr == C_NULL && throw(ArgumentError("BoundaryIntegralOp has been closed"))
    return op.ptr
end

# Passing the object itself to ccall (instead of its raw pointer) keeps it rooted, so the finalizer cannot free the
# C++ object during the call.
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::BoundaryIntegralOp) = _ptr(x)

Base.show(io::IO, op::BoundaryIntegralOp) =
    print(io, "BoundaryIntegralOp(", kernel_name(op.kernel), op.trg_normal_dot_prod ? ", trg_normal_dot_prod" : "",
          "; elem_lists=", elem_list_names(op), op.ptr == C_NULL ? ", closed)" : ")")

"""
    add_elem_list!(op, elist[, name="elements"]) -> op

Add a copy of the element list `elist` (a [`QuadElemList`](@ref) of order 4, 8, 12, 16 or 20) under `name`,
replacing a list of the same name. Later changes to `elist` (e.g. [`set_quad_scheme!`](@ref)) do not affect `op`;
add it again to update. Clears the setup.
"""
function add_elem_list!(op::BoundaryIntegralOp, el::QuadElemList, name::AbstractString="elements")
    el.order in _BIE_ORDERS ||
        throw(ArgumentError("BoundaryIntegralOp needs elements of order 4, 8, 12, 16 or 20, got $(el.order)"))
    _check(ccall((:sctljl_bie_add_elem_list, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), op, el, name))
    op.elem_lists[String(name)] = el
    return op
end

"""
    delete_elem_list!(op, name="elements") -> op

Remove the element list `name` (no-op if there is none). Clears the setup.
"""
function delete_elem_list!(op::BoundaryIntegralOp, name::AbstractString="elements")
    _check(ccall((:sctljl_bie_delete_elem_list, libsctljl), Cint, (Ptr{Cvoid}, Cstring), op, name))
    delete!(op.elem_lists, String(name))
    return op
end

"""
    elem_list_names(op) -> Vector{String}

Names of the element lists of `op`, in the order their nodes appear in the density vector (sorted).
"""
elem_list_names(op::BoundaryIntegralOp) = sort!(collect(keys(op.elem_lists)))

"""
    set_target_coord!(op, X) -> op

Set the target points (`3 × Nt`, or a vector of length `3Nt`); `nothing` or an empty array restores the default,
the surface nodes. Targets may lie on, near or far from the surface; a target that coincides with a surface node
uses that element's singular quadrature.
"""
function set_target_coord!(op::BoundaryIntegralOp, X)
    X_, n = X === nothing ? (Float64[], 0) : _points(X, "target coordinates")
    _check(ccall((:sctljl_bie_set_target_coord, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64), op, X_, length(X_)))
    op.ntrg = n == 0 ? -1 : n
    return op
end

"""
    set_target_normal!(op, N) -> op

Set the target normals (`3 × Nt`, same number as the targets) for an operator with `trg_normal_dot_prod=true`;
`nothing` or an empty array restores the default (the surface normals, only valid with the default targets).
"""
function set_target_normal!(op::BoundaryIntegralOp, N)
    N_, n = N === nothing ? (Float64[], 0) : _points(N, "target normals")
    _check(ccall((:sctljl_bie_set_target_normal, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64), op, N_, length(N_)))
    op.nnormal = n == 0 ? -1 : n
    return op
end

"""
    set_accuracy!(op, tol) -> op

Quadrature accuracy tolerance (default `1e-10`); clears the setup.
"""
function set_accuracy!(op::BoundaryIntegralOp, tol::Real)
    tol > 0 || throw(ArgumentError("tolerance must be positive"))
    _check(ccall((:sctljl_bie_set_accuracy, libsctljl), Cint, (Ptr{Cvoid}, Float64), op, tol))
    return op
end

"""
    accuracy(op) -> Float64

The quadrature accuracy tolerance.
"""
function accuracy(op::BoundaryIntegralOp)
    t = Ref(0.0)
    _check(ccall((:sctljl_bie_get_accuracy, libsctljl), Cint, (Ptr{Cvoid}, Ref{Float64}), op, t))
    return t[]
end

"""
    dim(op::BoundaryIntegralOp, i) -> Int

Length of the input density (`i = 1`: `src_dim × total nodes`) or of the output potential (`i = 2`:
`trg_dim × targets`, or `trg_dim/3 × targets` with `trg_normal_dot_prod`). SCTL's `Dim(i-1)`.
"""
function dim(op::BoundaryIntegralOp, i::Integer)
    i in (1, 2) || throw(ArgumentError("i must be 1 (input) or 2 (output)"))
    d = Ref{Int64}(0)
    _check(ccall((:sctljl_bie_dim, libsctljl), Cint, (Ptr{Cvoid}, Cint, Ref{Int64}), op, i - 1, d))
    return Int(d[])
end

# values per target in the output
_out_dof(op::BoundaryIntegralOp) = op.trg_normal_dot_prod ? trg_dim(op.kernel) ÷ 3 : trg_dim(op.kernel)

# SCTL does not check that the target normals match the targets (it would read out of bounds): check here.
function _check_targets(op::BoundaryIntegralOp)
    if op.trg_normal_dot_prod
        if op.ntrg >= 0
            op.nnormal == op.ntrg ||
                throw(ArgumentError("trg_normal_dot_prod with custom targets needs one target normal per target (set_target_normal!)"))
        elseif op.nnormal >= 0
            n = sum(nnodes, values(op.elem_lists); init=0)
            op.nnormal == n ||
                throw(ArgumentError("target normals ($(op.nnormal)) do not match the default targets (surface nodes, $n)"))
        end
    end
end

"""
    setup!(op) -> op

Precompute the far-field quadrature, the near-target lists and the singular/near-singular correction matrices
(otherwise done by the first [`compute_potential`](@ref)).
"""
function setup!(op::BoundaryIntegralOp)
    _check_targets(op)
    _check(ccall((:sctljl_bie_setup, libsctljl), Cint, (Ptr{Cvoid},), op))
    return op
end

"""
    clear_setup!(op) -> op

Discard the precomputed data (recomputed on the next evaluation).
"""
function clear_setup!(op::BoundaryIntegralOp)
    _check(ccall((:sctljl_bie_clear_setup, libsctljl), Cint, (Ptr{Cvoid},), op))
    return op
end

"""
    compute_potential!(U, op, F) -> U

In-place [`compute_potential`](@ref): `U` is a `Float64` array with `dim(op, 2)` elements (overwritten).
"""
function compute_potential!(U::Array{Float64}, op::BoundaryIntegralOp, F)
    _check_targets(op)
    F_ = _f64(F)
    nin, nout = dim(op, 1), dim(op, 2)
    length(F_) == nin || throw(DimensionMismatch("density must have dim(op, 1) = $nin values, got $(length(F_))"))
    length(U) == nout || throw(DimensionMismatch("potential must have dim(op, 2) = $nout values, got $(length(U))"))
    nout == 0 && return U
    _check(ccall((:sctljl_bie_compute_potential, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
                 op, U, nout, F_, nin))
    return U
end

"""
    compute_potential(op, F) -> U

Apply the operator to the density `F` (`dim(op, 1)` values: a vector, or a `src_dim × nnodes` matrix). Returns the
potential at the targets: a vector of `dim(op, 2)` values if `F` is a vector, else a `dof × Nt` matrix (`dof` =
`trg_dim`, or `trg_dim/3` with `trg_normal_dot_prod`).
"""
function compute_potential(op::BoundaryIntegralOp, F::AbstractArray)
    nout = dim(op, 2)
    dof = _out_dof(op)
    U = F isa AbstractVector ? Vector{Float64}(undef, nout) : Matrix{Float64}(undef, dof, nout ÷ dof)
    return compute_potential!(U, op, F)
end

function _scaling!(op::BoundaryIntegralOp, U::Array{Float64}, inverse::Bool)
    n = sum(nnodes, values(op.elem_lists); init=0)
    (n > 0 ? length(U) % n == 0 : isempty(U)) ||
        throw(DimensionMismatch("U must have a whole number of values per surface node ($n nodes), got $(length(U))"))
    isempty(U) && return U
    _check(ccall((:sctljl_bie_sqrt_scaling, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64, Cint),
                 op, U, length(U), inverse))
    return U
end

"""
    sqrt_scaling!(op, U) -> U

Multiply the node data `U` (any number of values per surface node, in node order) by the square root of the area of
each node's element, in place. With [`inv_sqrt_scaling!`](@ref) this symmetrizes the scaling of the operator, e.g.
for iterative solvers.
"""
sqrt_scaling!(op::BoundaryIntegralOp, U::Array{Float64}) = _scaling!(op, U, false)

"""
    inv_sqrt_scaling!(op, U) -> U

Divide the node data `U` by the square root of the area of each node's element, in place (inverse of
[`sqrt_scaling!`](@ref)).
"""
inv_sqrt_scaling!(op::BoundaryIntegralOp, U::Array{Float64}) = _scaling!(op, U, true)
