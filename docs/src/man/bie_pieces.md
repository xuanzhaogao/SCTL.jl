# Near-field pieces

A [`BoundaryIntegralOp`](@ref) adds two things for nodal densities `F`:
- **The far field:** the smooth node quadrature over all nodes, evaluated by the FMM.
- **A sparse near correction `C`:** for each element and each target close to it, the accurate quadrature
  minus the node rule.

```math
U = K_{\mathrm{nodes}} F + C F, \qquad
C_{e,t} = (\text{accurate rule of element } e \text{ at target } t) - (\text{node rule of } e \text{ at } t).
```

The functions on this page run the steps that build `C` one at a time and return plain Julia arrays. You can
inspect or replace any step, for example by computing the accurate rule with a quadrature of your own, while
keeping SCTL's classification and its far field:

| step | function | result |
|:--|:--|:--|
| far-field nodes and near radii | [`far_field_nodes`](@ref) | positions, normals, weights, radius per node |
| which targets are near which element | [`near_list`](@ref) | a [`NearList`](@ref) (compressed, per element) |
| accurate element-to-target matrices | [`near_blocks`](@ref), [`near_matrix`](@ref), [`self_matrices`](@ref) of an [`ElemQuadrature`](@ref) | one dense block per element |
| node rule of the same pairs | [`node_rule_blocks`](@ref), [`node_rule_block`](@ref) | blocks of the same shapes |
| assembly | [`assemble_near_correction`](@ref) | `SparseMatrixCSC` |
| all of the above | [`near_correction`](@ref) | `(C, near list)` |

The near test of [`near_list`](@ref) works for any discretization: you give it nodes, one radius per node and the
number of nodes per element. So it also classifies panels that are not [`QuadElemList`](@ref)s.

```@setup pieces
using SCTL, LinearAlgebra, SparseArrays, Printf
```

## The steps

The surface of the cube ``[-1,1]^3``, with 2 × 2 flat elements of order 8 per face (normals outward):

```@example pieces
el = QuadElemList(8, 24) do e, u, v
    f, r = divrem(e - 1, 4)                 # face f, element (i, j) on it
    i, j = divrem(r, 2)
    a, b = i + u - 1, j + v - 1             # face coordinates in [-1, 1]
    f == 0 ? [a, b, 1.0] : f == 1 ? [b, a, -1.0] : f == 2 ? [1.0, a, b] :
    f == 3 ? [-1.0, b, a] : f == 4 ? [b, 1.0, a] : [a, -1.0, b]
end
ff = far_field_nodes(el; tol = 1e-10)
extrema(ff.radius)
```

A target is near an element if it lies within `radius` of one of the element's nodes. Here the targets are the
surface nodes themselves:

```@example pieces
nl = near_list(el; tol = 1e-10)
```

```@example pieces
ts = near_targets(nl, 1)  # targets near element 1: its own 64 nodes, nodes of its neighbours on the same face and across the edges
(length(ts), nnodes(el))
```

The accurate rows come from SCTL's quadrature for the element list, here the double layer:
- A target on a node of the element gets that node's row of the self-interaction matrix (Duffy rule,
  principal value).
- Other targets get the near-singular rule: the element is split at the target's closest point and refined
  dyadically toward it.

Each block maps the element's nodal densities to its near targets, in the layout of [`kernel_matrix`](@ref):

```@example pieces
ker = Laplace3D_DxU()
q = ElemQuadrature(ker)
acc = near_blocks(q, el, nl, ff.X; tol = 1e-10)
size(acc[1])
```

Subtracting the node rule of the same pairs and scattering the blocks gives the correction. Together with the node
rule over all nodes (the far field), it is the boundary integral operator:

```@example pieces
plain = node_rule_blocks(ker, el, nl, ff.X; tol = 1e-10)
C = assemble_near_correction(nl, acc, order(el)^2; subtract = plain)

F = randn(nnodes(el))
K = node_rule_block(ker, ff.X, ff.normal, ff.weights, ff.X)     # node rule, all nodes to all targets
op = BoundaryIntegralOp(ker; accuracy = 1e-10)
add_elem_list!(op, el)
norm(K * F + C * F - compute_potential(op, F)) / norm(compute_potential(op, F))
```

For a constant density, the double layer is ``-1/2`` on the surface (principal value):

```@example pieces
extrema(K * ones(nnodes(el)) + C * ones(nnodes(el)))
```

[`near_correction`](@ref) runs all of these steps in one call.

## Replacing a step: a quadrature written in Julia

Suppose you want your own rule for the targets that are near an element without lying on it. Here it is an
upsampled tensor Gauss-Legendre rule:
- evaluate the element's geometry at `m × m` points with [`elem_geom`](@ref);
- interpolate the nodal density to those points with [`lagrange_interp_matrix`](@ref);
- apply the kernel matrix.

```@example pieces
function upsampled_block(ker, el, e, Xt; m = 32)
    u, w = legquad(m)
    g = elem_geom(el, e, u, u)                        # point (a, b) in column (a-1)*m + b
    W = vec(w * w') .* g.area                         # quadrature weights
    L = lagrange_interp_matrix(param_nodes(order(el)), u)
    P = kron(L, L)                                    # nodal values -> values at the m × m points
    return kernel_matrix(ker, g.X, g.normal, Xt) * (W .* P)
end
nothing # hide
```

Its rows converge to SCTL's as `m` grows. The targets of element 1 on the neighbouring faces include nodes next to
the shared edges, which are nearly singular for this rule:

```@example pieces
off = findall(>(order(el)^2), ts)                     # targets near element 1 but not on it
for m in (16, 32, 64, 128)
    B = upsampled_block(ker, el, 1, ff.X[:, ts[off]]; m = m)
    @printf("m = %3d: max difference from SCTL's rows %.1e\n", m, maximum(abs, B - acc[1][off, :]))
end
```

To use it, replace those rows of the blocks and assemble as before. The classification, the self-interaction rows
and the far field stay SCTL's:

```@example pieces
mine = map(1:nelem(el)) do e
    B = copy(acc[e])
    ts = near_targets(nl, e)
    off = findall(t -> !(t in (e-1)*64+1:e*64), ts)
    B[off, :] = upsampled_block(ker, el, e, ff.X[:, ts[off]]; m = 128)
    B
end
C2 = assemble_near_correction(nl, mine, order(el)^2; subtract = plain)
norm((C2 - C) * F) / norm(K * F + C * F)
```

## Classifying other panels

[`near_list`](@ref) only needs node positions and radii, so it works for panels from another code. The radius can
come from your own accuracy criterion, or from SCTL's rule as in [`far_field_nodes`](@ref):
- In parameter space, the rule takes the distance from each Gauss-Legendre node to the Bernstein ellipse with
  `ρ = (64 / (15 tol))^(1/(2 order))`.
- It scales that by the tangent length.

```@example pieces
r = fill(0.25, nnodes(el))                            # e.g. a fixed radius
targets = [0.5 0.0; 0.5 0.0; 1.1 3.0]                 # one target above element 4 (top face), one far away
nl_mine = near_list(ff.X, r, 64, targets)
[(e, collect(near_targets(nl_mine, e))) for e in 1:nelem(el) if !isempty(near_targets(nl_mine, e))]
```

## Layouts and conventions

- **Matrices** ([`near_matrix`](@ref), the blocks, [`self_matrices`](@ref), [`node_rule_block`](@ref)) are
  `(dof × targets) × (src_dim × nodes of the element)`. Rows run target by target, `dof` components each; columns
  run node by node, `src_dim` components each. This is the layout of [`kernel_matrix`](@ref). They include the
  quadrature weights, so they act on nodal density values.
- **`dof`** is `trg_dim(kernel)`. With target normals it is `trg_dim/3`: the potential dotted with the normal, as
  with `trg_normal_dot_prod`. Components `3(k-1)+1 : 3k` are dotted for the `k`-th output.
- **Node order** in [`assemble_near_correction`](@ref): element `e`'s nodes are consecutive, as in
  [`QuadElemList`](@ref).
- **Targets on a node** are matched by exact equality of coordinates. Pass the node coordinates themselves, e.g.
  `far_field_nodes(el).X`.
- **Cancellation.** The node rule here is a full-precision [`kernel_matrix`](@ref). The far field it cancels
  should be computed the same way to rounding: a direct sum ([`kernel_matrix`](@ref), [`kernel_eval`](@ref)), or
  an FMM ([`ParticleFMM`](@ref)) at the accuracy you need.

## Reference

```@docs
far_field_nodes
NearList
near_list
near_targets
npairs
ElemQuadrature
near_matrix
self_matrices
near_blocks
node_rule_block
node_rule_blocks
assemble_near_correction
near_correction
```
