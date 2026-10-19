# Point trees

[`PtTree`](@ref) wraps SCTL's `sctl::PtTree<double, DIM>`, an adaptive binary tree (`DIM = 1`), quadtree
(`DIM = 2`) or octree (`DIM = 3`) on the unit box ``[0, 1)^{\mathrm{DIM}}``. You give it a set of points and a
maximum number of points per box `max_pts`. It subdivides every box holding more than `max_pts` points into
``2^{\mathrm{DIM}}`` children of half the size, and repeats until no box needs splitting. Two optional constraints
are available: 2:1 balance (adjacent leaves differ by at most one level) and periodicity along any subset of the
axes. These trees are the spatial data structure behind fast multipole methods, near-field quadrature selection,
neighbor search and adaptive discretizations.

A tree node at depth ``\ell`` with lower corner ``c`` is the box ``c + [0, 2^{-\ell})^{\mathrm{DIM}}``. SCTL
identifies it by a Morton key (`sctl::Morton<DIM>`): the interleaved bits of the integer coordinates
``\lfloor 2^{20} c \rfloor`` together with the depth. The maximum depth is `MAX_DEPTH = 20`. The nodes form a
*linear tree*: one flat array in Morton (Z-order, depth-first preorder) order. Every interior node has all
``2^{\mathrm{DIM}}`` children, and particles live in the leaves. For each node, the base class `sctl::Tree<DIM>`
stores its Morton key (`GetNodeMID`), its leaf and ghost flags (`GetNodeAttr`), and its parent, children and
*colleagues* (`GetNodeLists`). Colleagues are the same-depth boxes that touch the node, including itself.
`PtTree` adds named **particle groups** (coordinates) and **particle data** (values attached to a group). Both
are redistributed automatically each time the tree is refined.

SCTL.jl uses `sctl::Comm::Self()`; there is no MPI. Every node is owned by the single process, and the ghost and
halo machinery has no effect.

## Tutorial

### Refining a clustered point set

```@example tree
using SCTL, LinearAlgebra
using Random; Random.seed!(1)   # reproducible output

N = 20_000
X = clamp.(0.5 .+ 0.5 .* (2 .* rand(2, N) .- 1) .^ 3, 0, prevfloat(1.0))   # 2 × N, dense near the center
tree = PtTree{2}()
add_particles!(tree, "pt", X)               # group "pt": coordinates stored as data set "pt" (dof 2)
update_refinement!(tree, X; max_pts=40)     # refine around the points
ni = node_info(tree)
leaves = findall(ni.leaf)
println(num_nodes(tree), " nodes, ", length(leaves), " leaves")
println("leaves per depth: ", [d => count(==(d), ni.depth[leaves]) for d in extrema(ni.depth[leaves])[1]:maximum(ni.depth)])
```

[`get_data`](@ref) returns a data set in the tree's node order, together with the number of particles in each
node. Only leaves hold particles:

```@example tree
pts, cnt = get_data(tree, "pt")                 # pts: 2 × N in node order; cnt: one count per node
offs = cumsum([0; cnt[1:end-1]])                # first column of node i is offs[i] + 1
s = 2.0 .^ -ni.depth                            # box side lengths
inside = all(i -> all(j -> all(ni.coord[:, i] .<= pts[:, j] .< ni.coord[:, i] .+ s[i]), offs[i]+1:offs[i]+cnt[i]), 1:num_nodes(tree))
(sum(cnt) == N, all(cnt[.!ni.leaf] .== 0), maximum(cnt), inside)
```

### The node arrays on a small example

Six points in 2D with `max_pts = 1`:

```@example tree
P = [0.1 0.6 0.1 0.6 0.8 0.9;
     0.1 0.1 0.6 0.6 0.8 0.9]
small = PtTree{2}()
add_particles!(small, "pt", P)
update_refinement!(small, P; max_pts=1)
sn = node_info(small)
for i in 1:num_nodes(small)
    println(lpad(i, 2), ": corner ", sn.coord[:, i], "  depth ", sn.depth[i], sn.leaf[i] ? "  leaf" : "      ",
            "  parent ", sn.parent[i])
end
```

The root is node 1. The upper-right quadrant (node 5) holds three points and is split twice more. The column
`sn.children[:, i]` lists the ``2^{\mathrm{DIM}}`` children. Reshaped to `(2, 2)`, entry `[b1+1, b2+1]` is the
child in the lower (`b = 0`) or upper (`b = 1`) half along x and y:

```@example tree
reshape(sn.children[:, 1], 2, 2)   # [lower-left upper-left; lower-right upper-right] in (x, y) terms
```

The column `sn.neighbors[:, i]` lists the ``3^{\mathrm{DIM}}`` same-depth neighbors. Reshaped to `(3, 3)`, entry
`[o1+2, o2+2]` is the box offset by `(o1, o2)` box sizes. The center `[2, 2]` is the node itself, and `0` means
the box does not exist, either because it lies outside the domain or because that region is covered by a coarser
leaf:

```@example tree
reshape(sn.neighbors[:, 5], 3, 3)   # node 5 = upper-right quadrant at depth 1
```

### Near-neighbor interactions through the neighbor lists

A common task is to sum a compactly supported interaction over all pairs closer than a cutoff ``r``:

```math
u_i = \sum_{j \ne i,\ |x_i - x_j| < r} q_j \Big(1 - \frac{|x_i - x_j|}{r}\Big).
```

If ``r`` is at most the smallest leaf side, every partner of a point in leaf ``B`` lies in a leaf that touches
``B``. All leaf boundaries lie on the grid of the smallest leaves, so a non-touching leaf is at least one smallest
side away. For a leaf ``B``, the touching leaves are found by two walks:

- **Same size or coarser:** the leaves among the colleagues of ``B`` and of its ancestors.
- **Finer:** descend from the non-leaf colleagues of ``B`` into the children that touch ``B``.

```@example tree
function touching_leaves(ni, i)
    DIM = size(ni.coord, 1)
    s = 2.0 .^ -ni.depth
    touches(a, b) = all(d -> ni.coord[d, a] <= ni.coord[d, b] + s[b] && ni.coord[d, b] <= ni.coord[d, a] + s[a], 1:DIM)
    out = Int[]
    a = i
    while a != 0                                   # colleagues of i and of its ancestors
        for c in ni.neighbors[:, a]
            c > 0 && ni.leaf[c] && touches(c, i) && !(c in out) && push!(out, c)
        end
        a = ni.parent[a]
    end
    stack = [c for c in ni.neighbors[:, i] if c > 0 && !ni.leaf[c]]
    while !isempty(stack)                          # finer leaves next to i
        c = pop!(stack)
        for ch in ni.children[:, c]
            touches(ch, i) || continue
            ni.leaf[ch] ? push!(out, ch) : push!(stack, ch)
        end
    end
    return out
end

M = 3000
Y = rand(2, M)
q = randn(M)
t2 = PtTree{2}()
add_particles!(t2, "src", Y)
add_particle_data!(t2, "q", "src", q)
add_particle_data!(t2, "id", "src", collect(1.0:M))   # original index of each point
update_refinement!(t2, Y; max_pts=20)
n2 = node_info(t2)
r = minimum(2.0 .^ -n2.depth[n2.leaf])                # cutoff = smallest leaf side

y, c2 = get_data(t2, "src")
qn, _ = get_data(t2, "q")
id, _ = get_data(t2, "id")
o2 = cumsum([0; c2[1:end-1]])
u = zeros(M)                                          # result, in the original order
for b in findall(n2.leaf), j in touching_leaves(n2, b)
    for it in o2[b]+1:o2[b]+c2[b], js in o2[j]+1:o2[j]+c2[j]
        ρ = hypot(y[1, it] - y[1, js], y[2, it] - y[2, js])
        0 < ρ < r && (u[Int(id[it])] += qn[js] * (1 - ρ / r))
    end
end

dist(i, j) = hypot(Y[1, i] - Y[1, j], Y[2, i] - Y[2, j])
u_direct = [sum((ρ = dist(i, j); 0 < ρ < r ? q[j] * (1 - ρ / r) : 0.0) for j in 1:M) for i in 1:M]
(r, maximum(abs, u - u_direct))
```

The tree version visits only nearby pairs and agrees with the ``O(M^2)`` direct sum to rounding.

### Particle data round trip

Particle data follow their particles through every refinement. [`get_particle_data`](@ref) always returns them in
the order in which the group's coordinates were passed to [`add_particles!`](@ref):

```@example tree
f = vcat(sin.(2π .* X[1:1, :]), X[2:2, :] .^ 2)       # 2 × N data
add_particle_data!(tree, "f", "pt", f)
update_refinement!(tree, X; max_pts=10)               # refine again: data are redistributed
(get_particle_data(tree, "f") == f, get_particle_data(tree, "pt") == X, num_nodes(tree))
```

A data set can also be allocated without values and filled in node order with [`set_data!`](@ref). The following
stores ``x + y`` for every particle:

```@example tree
add_particle_data!(tree, "sum", "pt", 1)              # 1 uninitialized value per particle
pts, _ = get_data(tree, "pt")
set_data!(tree, "sum", sum(pts; dims=1))              # node order
maximum(abs, get_particle_data(tree, "sum") .- sum(X; dims=1))
```

### 2:1 balance and periodicity

With `balance21 = true`, leaves that touch, including at a corner, differ by at most one level. The tree gains
nodes in exchange:

```@example tree
function max_jump(ni)
    lv = findall(ni.leaf)
    worst = 0
    for i in lv, j in touching_leaves(ni, i)
        worst = max(worst, abs(ni.depth[i] - ni.depth[j]))
    end
    worst
end
Z = X[:, 1:2000]
tb = PtTree{2}()
add_particles!(tb, "pt", Z)
update_refinement!(tb, Z; max_pts=1)
a = (num_nodes(tb), max_jump(node_info(tb)))
update_refinement!(tb, Z; max_pts=1, balance21=true)
b = (num_nodes(tb), max_jump(node_info(tb)))
println("unbalanced: ", a, "   balanced: ", b, "   (nodes, max level jump)")
```

`periodic` makes the neighbor lists wrap around (and 2:1 balancing applies across the periodic boundaries). It
is `true` for all axes, or one `Bool` per axis. In a tree that is periodic in x only, the depth-1 box at the
origin has its left neighbor at x = 0.5, but no neighbor below it:

```@example tree
tp = PtTree{2}()
add_particles!(tp, "pt", P)
update_refinement!(tp, P; max_pts=1, periodic=(true, false))
np_ = node_info(tp)
i = findfirst(i -> np_.depth[i] == 1 && np_.coord[:, i] == [0, 0], 1:num_nodes(tp))
nb = reshape(np_.neighbors[:, i], 3, 3)
(left = np_.coord[:, nb[1, 2]], below = nb[2, 1])
```

### VTK output

[`write_tree_vtk`](@ref) writes the leaf boxes and [`write_particle_vtk`](@ref) writes the particles with a data set
attached. ParaView reads the files:

```@example tree
dir = mktempdir()
files = [write_tree_vtk(tree, joinpath(dir, "tree")); write_particle_vtk(tree, joinpath(dir, "pts"), "f")]
basename.(files)
```

## Data layouts

**Inputs**

| argument | Julia shape | SCTL C++ |
|:---|:---|:---|
| coordinates (`add_particles!`, `update_refinement!`) | `DIM × N` matrix, or flat vector `(x1, y1, z1, x2, …)` | `Vector<double>` AoS, `coord[i*DIM + k]` |
| particle data (`add_particle_data!`) | `dof × N` matrix (or length-`N` vector, `dof = 1`), in the group's original order | `Vector<double>` with `data[i*dof + k]` |
| `periodic` | `false`, `true`, or `DIM` Bools | `sctl::Periodicity` bitmask (bit `k-1` = axis `k`) |

**`node_info(tree)`**: `n = num_nodes(tree)`, with nodes in Morton order and node 1 the root. Every index is
1-based, and `0` means *none*. SCTL itself uses 0-based indices and `-1`.

| field | Julia shape | meaning |
|:---|:---|:---|
| `coord` | `DIM × n` | lower corner of node `i` (`Morton::Coord`) |
| `depth` | `n` | depth; box side `2.0^-depth[i]` |
| `leaf`, `ghost` | `n` `Bool`s | `NodeAttr::Leaf`, `NodeAttr::Ghost` (no ghosts without MPI) |
| `parent` | `n` | parent index; `0` for the root |
| `children` | `2^DIM × n` | `children[c, i]`, `c = 1 + Σ_k b_k 2^(k-1)`, `b_k = 1` for the upper half along axis `k`; all `0` for a leaf |
| `neighbors` | `3^DIM × n` | same-depth colleagues, `neighbors[j, i]`, `j = 1 + Σ_k (o_k + 1) 3^(k-1)`, `o_k ∈ {-1, 0, 1}` |

Equivalently, `reshape(children[:, i], ntuple(_ -> 2, DIM))[b_1+1, …, b_DIM+1]` and
`reshape(neighbors[:, i], ntuple(_ -> 3, DIM))[o_1+2, …, o_DIM+2]`. The center neighbor entry is `i` itself. A
neighbor is `0` outside a non-periodic domain, and also where the tree has no node of that depth there, which
happens when the region is covered by a coarser leaf.

**`get_data(tree, name)`** returns `(data, cnt)`:

| value | Julia shape | meaning |
|:---|:---|:---|
| `cnt` | `n` `Int`s | particles in node `i`; nonzero only for leaves, `sum(cnt) == N` |
| `data` | `dof × sum(cnt)` | node `i` owns columns `offs[i]+1 : offs[i]+cnt[i]`, `offs = cumsum([0; cnt[1:end-1]])` |

The coordinates of a group are the data set named after the group, with `dof = DIM`. Within a node, the particles
are sorted by Morton key. [`get_particle_data`](@ref) returns the `dof × N` matrix in the original particle order
instead. [`set_data!`](@ref) takes the node-order layout of `get_data`.

**Owned range.** [`owned_range`](@ref) is `GetOwnedRange` as a 1-based `UnitRange`; without MPI it is
`1:num_nodes(tree)`.

## Pitfalls and limitations

- **The unit box.** Coordinates must lie in ``[0, 1)^{\mathrm{DIM}}``. A value of exactly `1.0` is rejected, and
  so are negative values and `NaN`, with `ArgumentError`. SCTL itself would silently clamp them. Rescale your data
  first, for example `(x .- lo) ./ (L * (1 + eps()))`.
- **Resolution.** Points are binned on the ``2^{-20}`` grid (`MAX_DEPTH = 20`). Points that share a finest-level
  cell cannot be separated, so a depth-20 leaf may hold more than `max_pts` points.
- **Indices change on refinement.** Each [`update_refinement!`](@ref) call rebuilds the node arrays. Call
  [`node_info`](@ref) and [`get_data`](@ref) again afterwards. The refinement depends only on the `coord` argument
  of `update_refinement!`, not on the particle groups. You can refine on one point set and store others.
- **Neighbor lists are same-depth only.** Leaves of a different size next to a node are reached through the
  parent and child links, as in the tutorial. `balance21 = true` limits the size difference to one level.
- **Periodicity** affects only the neighbor lists and the balancing. Coordinates are not wrapped, and the tree
  does not store whether it is periodic.
- **Data-set names.** Every name is either a particle group or a data set attached to one; a name cannot be both.
  Calling [`add_particles!`](@ref) again with an existing group's name replaces the group, and its data sets become
  empty (`dof = 0`) until you add them again. Deleting a group deletes its data sets.
  `add_particle_data!(tree, name, group, dof)` allocates **uninitialized** values.
- **Copies.** [`get_data`](@ref) and [`get_particle_data`](@ref) return copies. Changing them does not change the
  tree; use [`set_data!`](@ref) for that.
- **Threads and memory.** Tree construction runs OpenMP-parallel inside SCTL. Node arrays take
  ``O(n\,3^{\mathrm{DIM}})`` integers, so a 3D tree uses 27 neighbor entries per node. A `PtTree` is freed by its
  finalizer, or immediately with `close(tree)`.
- **MPI.** The tree uses `Comm::Self()`. `halo_size` and `show_ghost` are accepted for parity with SCTL but have no
  effect in this build.

## API reference

```@docs
PtTree
update_refinement!
add_particles!
add_particle_data!
get_particle_data
delete_particle_data!
num_nodes
node_info
owned_range
get_data
set_data!
write_tree_vtk
write_particle_vtk
close(::PtTree)
```
