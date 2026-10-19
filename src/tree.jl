# Adaptive point trees (sctl::PtTree<double, DIM>): quadtrees/octrees refined so that each leaf holds at most
# `max_pts` points, with named particle groups and particle data that follow the refinement.

export PtTree, update_refinement!, add_particles!, add_particle_data!, get_particle_data, delete_particle_data!
export num_nodes, node_info, owned_range, get_data, set_data!, write_tree_vtk, write_particle_vtk

"""
    PtTree{DIM}()
    PtTree(DIM)

An adaptive `DIM`-dimensional point tree (`sctl::PtTree<double, DIM>`, `DIM` ∈ 1, 2, 3; a binary tree, quadtree or
octree) over the unit box `[0, 1)^DIM`, MPI-free.

Typical use:

```julia
tree = PtTree{3}()
add_particles!(tree, "pt", X)               # X: 3×N coordinates in [0,1)^3
add_particle_data!(tree, "f", "pt", f)      # f: dof×N (or a length-N vector), in the order of X
update_refinement!(tree, X; max_pts=100)    # refine; groups and data are redistributed over the new leaves
nodes = node_info(tree)                     # boxes, flags, parent/children/neighbors
data, cnt = get_data(tree, "f")             # data in node order, cnt[i] items in node i
get_particle_data(tree, "f") == f           # back in the original order
```

The tree is a linear tree: its nodes are stored in Morton (depth-first) order, node 1 being the root, and every
non-leaf node has all `2^DIM` children. Particles are counted in the leaves only.
"""
mutable struct PtTree{DIM}
    ptr::Ptr{Cvoid}
    groups::Dict{String,Int}                 # particle group => number of particles
    data::Dict{String,Tuple{String,Int}}     # data set => (particle group, dof); a group's coordinates are the set named after it
    function PtTree{DIM}() where {DIM}
        DIM isa Integer && 1 <= DIM <= 3 || throw(ArgumentError("PtTree dimension must be 1, 2 or 3"))
        out = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_pttree_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Cint), out, DIM))
        t = new{DIM}(out[], Dict{String,Int}(), Dict{String,Tuple{String,Int}}())
        finalizer(_free!, t)
        return t
    end
end
PtTree(dim::Integer) = PtTree{Int(dim)}()

function _free!(t::PtTree{DIM}) where {DIM}
    if t.ptr != C_NULL
        ccall((:sctljl_pttree_free, libsctljl), Cvoid, (Ptr{Cvoid}, Cint), t.ptr, DIM)
        t.ptr = C_NULL
    end
    return nothing
end

"""
    close(tree::PtTree)

Release the tree's C++ object now (otherwise done by the finalizer). The tree cannot be used afterwards.
"""
Base.close(t::PtTree) = _free!(t)

function _ptr(t::PtTree)
    t.ptr == C_NULL && throw(ArgumentError("PtTree has been closed"))
    return t.ptr
end

# Passing the object itself to ccall (instead of its raw pointer) keeps it rooted, so the finalizer cannot free the
# C++ object during the call.
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PtTree) = _ptr(x)

Base.show(io::IO, t::PtTree{DIM}) where {DIM} =
    t.ptr == C_NULL ? print(io, "PtTree{$DIM}(closed)") :
    print(io, "PtTree{$DIM}(", num_nodes(t), " nodes, groups: ", join(sort!(collect(keys(t.groups))), ", "), ")")

# DIM×N coordinates (matrix, or flat AoS vector) as a flat Float64 array, checked to lie in [0,1)^DIM.
function _coords(::PtTree{DIM}, coord::AbstractArray{<:Real}) where {DIM}
    if coord isa AbstractMatrix
        size(coord, 1) == DIM || throw(DimensionMismatch("coordinates must be a $DIM×N matrix, got $(size(coord))"))
    else
        coord isa AbstractVector || throw(DimensionMismatch("coordinates must be a $DIM×N matrix or a flat vector"))
        length(coord) % DIM == 0 || throw(DimensionMismatch("flat coordinate vector length must be a multiple of $DIM"))
    end
    c = _f64(coord)
    ccall((:sctljl_unit_box_ok, libsctljl), Cint, (Ptr{Float64}, Int64), c, length(c)) == 1 || throw(ArgumentError("coordinates must lie in the unit box [0, 1)^$DIM"))
    return c, length(c) ÷ DIM
end

function _periodicity(::PtTree{DIM}, periodic) where {DIM}
    if periodic isa Bool
        return periodic ? (1 << DIM) - 1 : 0
    end
    length(periodic) == DIM || throw(DimensionMismatch("periodic must be a Bool or $DIM Bools (one per axis)"))
    mask = 0
    for (k, p) in enumerate(periodic)
        p isa Bool || throw(ArgumentError("periodic must be a Bool or $DIM Bools (one per axis)"))
        p && (mask |= 1 << (k - 1))
    end
    return mask
end

"""
    update_refinement!(tree, coord; max_pts=1, balance21=false, periodic=false, halo_size=-1) -> tree

Rebuild the tree refinement from the points `coord` (a `DIM×N` matrix or flat `(x1, y1, z1, x2, ...)` vector in
`[0, 1)^DIM`) so that every leaf holds at most `max_pts` of them (except where more than `max_pts` points coincide
down to the maximum depth, 20). Particle groups and their data are
redistributed over the new nodes.

- `balance21`: make the tree level-restricted (2:1 balanced): leaves that touch (including at corners) differ by
  at most one level.
- `periodic`: `false` (none), `true` (all axes) or one `Bool` per axis, e.g. `(true, false, true)`; affects the
  neighbor lists (they wrap around) and the 2:1 balancing.
- `halo_size`: with `k >= 0`, `2^k` neighboring boxes are included in the halo (ghost) region of other MPI ranks;
  irrelevant in this MPI-free build (there are no ghost nodes).
"""
function update_refinement!(t::PtTree{DIM}, coord::AbstractArray{<:Real}; max_pts::Integer=1, balance21::Bool=false,
                            periodic=false, halo_size::Integer=-1) where {DIM}
    max_pts >= 1 || throw(ArgumentError("max_pts must be positive"))
    halo_size >= -1 || throw(ArgumentError("halo_size must be >= -1"))
    c, n = _coords(t, coord)
    n >= 1 || throw(ArgumentError("need at least one point"))
    mask = _periodicity(t, periodic)
    _check(ccall((:sctljl_pttree_update_refinement, libsctljl), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Float64}, Int64, Int64, Cint, Cint, Cint),
                 t, DIM, c, n, max_pts, balance21, mask, halo_size))
    return t
end

_name(s::AbstractString) = (n = String(s); isempty(n) && throw(ArgumentError("name must not be empty")); n)

"""
    add_particles!(tree, name, coord) -> tree

Add a group of particles named `name` with coordinates `coord` (`DIM×N` matrix or flat vector in `[0, 1)^DIM`).
The coordinates are stored as the particle data set `name` (`dof = DIM`). Adding a group under an existing group's
name replaces it; the data sets attached to it are then emptied (dof 0) and must be added again. The points do
not refine the tree: call [`update_refinement!`](@ref) for that.
"""
function add_particles!(t::PtTree{DIM}, name::AbstractString, coord::AbstractArray{<:Real}) where {DIM}
    nm = _name(name)
    haskey(t.data, nm) && !haskey(t.groups, nm) &&
        throw(ArgumentError("\"$nm\" is the name of a particle data set, not of a particle group"))
    c, n = _coords(t, coord)
    _check(ccall((:sctljl_pttree_add_particles, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Ptr{Float64}, Int64),
                 t, DIM, nm, c, n))
    for (k, (g, _)) in t.data  # mirror SCTL: the group's other data sets are emptied
        g == nm && k != nm && (t.data[k] = (g, 0))
    end
    t.groups[nm] = n
    t.data[nm] = (nm, DIM)
    return t
end

function _check_data_name(t::PtTree, data_name::String, particle_name::String)
    haskey(t.groups, particle_name) || throw(ArgumentError("no particle group named \"$particle_name\""))
    haskey(t.groups, data_name) &&
        throw(ArgumentError("\"$data_name\" is a particle group (its coordinates); use add_particles! to change them"))
end

"""
    add_particle_data!(tree, data_name, particle_name, data) -> tree
    add_particle_data!(tree, data_name, particle_name, dof::Integer) -> tree

Attach data to the particle group `particle_name`: `data` is a `dof×N` matrix (or a length-`N` vector, `dof = 1`)
in the order of the group's coordinates as passed to [`add_particles!`](@ref). An existing data set of the name is
replaced. The data follow the particles through [`update_refinement!`](@ref).

With an integer `dof`, allocate `dof` *uninitialized* values per particle instead, to be filled in node order with
[`set_data!`](@ref).
"""
function add_particle_data!(t::PtTree{DIM}, data_name::AbstractString, particle_name::AbstractString,
                            data::AbstractVecOrMat{<:Real}) where {DIM}
    dn, pn = _name(data_name), _name(particle_name)
    _check_data_name(t, dn, pn)
    N = t.groups[pn]
    dof = data isa AbstractVector ? 1 : size(data, 1)
    size(data, ndims(data)) == N ||
        throw(DimensionMismatch("data must have one column per particle of \"$pn\" ($N), got size $(size(data))"))
    N == 0 && data isa AbstractMatrix && (dof = size(data, 1))
    d = _f64(data)
    _check(ccall((:sctljl_pttree_add_particle_data, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Cstring, Ptr{Float64}, Int64),
                 t, DIM, dn, pn, d, length(d)))
    t.data[dn] = (pn, dof)
    return t
end

function add_particle_data!(t::PtTree{DIM}, data_name::AbstractString, particle_name::AbstractString, dof::Integer) where {DIM}
    dn, pn = _name(data_name), _name(particle_name)
    _check_data_name(t, dn, pn)
    dof >= 0 || throw(ArgumentError("dof must be non-negative"))
    _check(ccall((:sctljl_pttree_alloc_particle_data, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Cstring, Int64),
                 t, DIM, dn, pn, dof))
    t.data[dn] = (pn, Int(dof))
    return t
end

function _dataset(t::PtTree, name::AbstractString)
    nm = String(name)
    haskey(t.data, nm) || throw(ArgumentError("no particle data set named \"$nm\""))
    return nm, t.data[nm]...
end

"""
    get_particle_data(tree, data_name) -> Matrix{Float64}

The particle data set `data_name` (or a group's coordinates, under the group's name) as a `dof×N` matrix, scattered
back to the original order of the group's particles (as passed to [`add_particles!`](@ref)), whatever the
refinement in between. An emptied set gives a `0×N` matrix.
"""
function get_particle_data(t::PtTree{DIM}, data_name::AbstractString) where {DIM}
    nm, g, dof = _dataset(t, data_name)
    N = t.groups[g]
    _withvecs(Float64) do h
        _check(ccall((:sctljl_pttree_get_particle_data, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Ptr{Cvoid}),
                     t, DIM, nm, h))
        v = collect(h)
        length(v) == dof * N || throw(SCTLError("unexpected particle data size $(length(v)) for dof $dof, N $N"))
        return reshape(v, dof, N)
    end
end

"""
    delete_particle_data!(tree, data_name) -> tree

Delete a particle data set. Deleting a particle group (by its name) deletes the group's coordinates and every
data set attached to it.
"""
function delete_particle_data!(t::PtTree{DIM}, data_name::AbstractString) where {DIM}
    nm, g, _ = _dataset(t, data_name)
    _check(ccall((:sctljl_pttree_delete_particle_data, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring), t, DIM, nm))
    if nm == g
        filter!(kv -> kv.second[1] != g, t.data)
        delete!(t.groups, g)
    else
        delete!(t.data, nm)
    end
    return t
end

"""
    num_nodes(tree) -> Int

Number of tree nodes (leaves and interior nodes).
"""
function num_nodes(t::PtTree{DIM}) where {DIM}
    n = Ref{Int64}(0)
    _check(ccall((:sctljl_pttree_num_nodes, libsctljl), Cint, (Ptr{Cvoid}, Cint, Ref{Int64}), t, DIM, n))
    return Int(n[])
end

"""
    node_info(tree) -> NamedTuple

All nodes of the tree, in Morton order (node 1 is the root), as a named tuple of arrays over the `n` nodes:

- `coord::Matrix{Float64}` (`DIM×n`): lower corner of node `i`'s box, `coord[:, i]` (from `Morton::Coord`);
- `depth::Vector{Int}`: depth (root 0); the box is `coord[:, i] .+ [0, 2.0^-depth[i])^DIM`;
- `leaf::Vector{Bool}`, `ghost::Vector{Bool}`: node attributes (no ghosts without MPI);
- `parent::Vector{Int}`: parent index (`0` for the root);
- `children::Matrix{Int}` (`2^DIM × n`): `children[c, i]` with `c = 1 + Σ_k b_k 2^(k-1)`, `b_k = 1` for the upper
  half along axis `k`, i.e. `reshape(children[:, i], ntuple(_ -> 2, DIM))[b_1+1, …, b_DIM+1]`; `0` for a leaf;
- `neighbors::Matrix{Int}` (`3^DIM × n`): same-depth neighbors, `neighbors[j, i]` with
  `j = 1 + Σ_k (o_k + 1) 3^(k-1)` for the offset `o_k ∈ (-1, 0, 1)` (in box sizes) along axis `k`, i.e.
  `reshape(neighbors[:, i], ntuple(_ -> 3, DIM))[o_1+2, …, o_DIM+2]`; the center entry is `i` itself; `0` when that
  box is not in the tree or lies outside the (non-periodic) domain. Periodic axes wrap around.

All indices are 1-based Julia indices with `0` meaning *none*.
"""
function node_info(t::PtTree{DIM}) where {DIM}
    n = num_nodes(t)
    coord = Matrix{Float64}(undef, DIM, n)
    depth = Vector{Int64}(undef, n)
    leaf = Vector{UInt8}(undef, n)
    ghost = Vector{UInt8}(undef, n)
    parent = Vector{Int64}(undef, n)
    children = Matrix{Int64}(undef, 1 << DIM, n)
    neighbors = Matrix{Int64}(undef, 3^DIM, n)
    _check(ccall((:sctljl_pttree_node_info, libsctljl), Cint,
                 (Ptr{Cvoid}, Cint, Int64, Ptr{Float64}, Ptr{Int64}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}),
                 t, DIM, n, coord, depth, leaf, ghost, parent, children, neighbors))
    return (coord=coord, depth=depth, leaf=leaf .!= 0, ghost=ghost .!= 0,
            parent=parent .+ 1, children=children .+ 1, neighbors=neighbors .+ 1)
end

"""
    owned_range(tree) -> UnitRange{Int}

The (1-based) indices of the nodes owned by this process; the others are ghosts. Without MPI this is
`1:num_nodes(tree)`.
"""
function owned_range(t::PtTree{DIM}) where {DIM}
    b, e = Ref{Int64}(0), Ref{Int64}(0)
    _check(ccall((:sctljl_pttree_owned_range, libsctljl), Cint, (Ptr{Cvoid}, Cint, Ref{Int64}, Ref{Int64}),
                 t, DIM, b, e))
    return Int(b[]) + 1:Int(e[])
end

"""
    get_data(tree, name) -> (data, cnt)

The tree's node data for the particle data set `name` (a group's coordinates under the group's name), in node
order: `cnt[i]` is the number of particles in node `i` (nonzero for leaves only, `sum(cnt) == N`) and `data` is a
`dof×sum(cnt)` matrix whose columns `offs[i]+1 : offs[i]+cnt[i]`, with `offs = cumsum([0; cnt[1:end-1]])`, belong
to node `i`. (This is the tree's internal ordering; [`get_particle_data`](@ref) returns the original order.)
"""
function get_data(t::PtTree{DIM}, name::AbstractString) where {DIM}
    nm, _, dof = _dataset(t, name)
    _withvecs(Float64, Int64) do hd, hc
        _check(ccall((:sctljl_pttree_get_data, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Ptr{Cvoid}, Ptr{Cvoid}),
                     t, DIM, nm, hd, hc))
        d, c = collect(hd), collect(hc)
        total = sum(c; init=0)
        length(d) == dof * total || throw(SCTLError("unexpected node data size $(length(d)) for dof $dof"))
        return reshape(d, dof, total), c
    end
end

"""
    set_data!(tree, name, data) -> tree

Overwrite the values of the particle data set `name` in node order (the layout of [`get_data`](@ref): a
`dof×sum(cnt)` matrix, or a vector of the same length). Used to fill a set allocated with
`add_particle_data!(tree, name, group, dof)`.
"""
function set_data!(t::PtTree{DIM}, name::AbstractString, data::AbstractArray{<:Real}) where {DIM}
    nm, g, dof = _dataset(t, name)
    length(data) == dof * t.groups[g] ||
        throw(DimensionMismatch("data must have $(dof * t.groups[g]) values (dof $dof × $(t.groups[g]) particles), got $(length(data))"))
    data isa AbstractMatrix && size(data, 1) != dof && throw(DimensionMismatch("data must have $dof rows"))
    d = _f64(data)
    _check(ccall((:sctljl_pttree_set_data, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Ptr{Float64}, Int64),
                 t, DIM, nm, d, length(d)))
    return t
end

function _vtk_files(fname::AbstractString)
    files = [fname * "000000.vtu", fname * ".pvtu"]
    all(isfile, files) || error("could not write the VTK files $(files)")
    return files
end

"""
    write_tree_vtk(tree, fname; show_ghost=false) -> Vector{String}

Write the leaf boxes as a VTK unstructured grid: `fname * "000000.vtu"` (this rank's piece) and `fname * ".pvtu"`
(the parallel index). Returns the two paths.
"""
function write_tree_vtk(t::PtTree{DIM}, fname::AbstractString; show_ghost::Bool=false) where {DIM}
    _check(ccall((:sctljl_pttree_write_tree_vtk, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Cint),
                 t, DIM, fname, show_ghost))
    return _vtk_files(fname)
end

"""
    write_particle_vtk(tree, fname, data_name; show_ghost=false) -> Vector{String}

Write the particles of the group that `data_name` belongs to, with the values of `data_name` attached (none when
`data_name` is the group itself), as VTK files `fname * "000000.vtu"` and `fname * ".pvtu"`. Returns the paths.
"""
function write_particle_vtk(t::PtTree{DIM}, fname::AbstractString, data_name::AbstractString; show_ghost::Bool=false) where {DIM}
    nm, _, _ = _dataset(t, data_name)
    _check(ccall((:sctljl_pttree_write_particle_vtk, libsctljl), Cint, (Ptr{Cvoid}, Cint, Cstring, Cstring, Cint),
                 t, DIM, fname, nm, show_ghost))
    return _vtk_files(fname)
end
