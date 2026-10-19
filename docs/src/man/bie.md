# Boundary integral operators

A boundary integral operator applies a layer potential over a surface ``\Gamma``,

```math
U(x) = \int_\Gamma K(x - y)\, F(y)\, dS_y ,
```

where ``K`` is one of the [Kernel functions](@ref), ``F`` is a density sampled at the surface discretization
nodes, and ``x`` is a target either on ``\Gamma`` or off it. If ``x`` lies on or near ``\Gamma``, the integrand is
singular or nearly singular, so plain quadrature fails. SCTL's `BoundaryIntegralOp<Real,Kernel>` handles this by
combining three parts:

- a smooth quadrature at the nodes for the far field, summed by the [Fast multipole method](@ref) for Laplace and
  Stokes kernels with 40000 or more targets, and directly otherwise;
- precomputed singular and near-singular correction matrices, built by the element list for targets close to each
  element;
- a principal-value treatment of on-surface targets.

The result is a matrix-free operator, applied with [`compute_potential`](@ref). It is the building block for
solving PDEs with integral equations: Laplace, Helmholtz or Stokes problems inside or outside a closed surface,
combined with [`gmres`](@ref).

SCTL.jl wraps the element list `sctl::QuadElemList<double>` (from `sctl/experimental/quad_element.hpp`) as
[`QuadElemList`](@ref). It describes the surface as quadrilateral patches, each a tensor-product polynomial map from
``[0,1]^2`` through `order × order` Gauss-Legendre nodes. [`BoundaryIntegralOp`](@ref) supports these kernels (see
[`bie_supported_kernels`](@ref)): `Laplace3D_FxU`, `Laplace3D_DxU`, `Laplace3D_FxdU`, `Laplace3D_DxdU`,
`Helmholtz3D_FxU`, `Helmholtz3D_DxU`, `Helmholtz3D_FxdU`, `Stokes3D_FxU`, `Stokes3D_DxU` and `Stokes3D_FxT`.

For SCTL's sign conventions (``r = x - y``, ``n`` the outward normal), the Laplace double layer
``D[\sigma](x) = \int_\Gamma \frac{(x-y)\cdot n_y}{4\pi|x-y|^3}\sigma(y)\, dS_y`` satisfies

```math
D[1] = -1 \text{ inside}, \qquad D[1] = -\tfrac12 \text{ on } \Gamma \text{ (principal value)}, \qquad D[1] = 0 \text{ outside},
```

and its limits from outside and inside are ``D_\pm[\sigma] = D_{pv}[\sigma] \pm \tfrac12\sigma``.

## Tutorial

### A cubed-sphere surface

The unit sphere is the image of the six faces of a cube, each split into `ppf × ppf` elements. On each face,
`(u, v)` must be oriented so that ``\partial_u X \times \partial_v X`` points outward. SCTL takes the normal from
this cross product.

```@example bie
using SCTL, LinearAlgebra

# (outward axis n, tangents t1, t2) with t1 × t2 = n, so the normals point outward
const FACES = (([1, 0, 0], [0, 1, 0], [0, 0, 1]), ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
               ([0, 1, 0], [0, 0, 1], [1, 0, 0]), ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
               ([0, 0, 1], [1, 0, 0], [0, 1, 0]), ([0, 0, -1], [0, 1, 0], [1, 0, 0]))

# point (u, v) ∈ [0,1]² of element e of the cubed sphere with ppf × ppf elements per face
function sphere_point(e, u, v, ppf)
    f, k = divrem(e - 1, ppf^2)
    n, t1, t2 = FACES[f+1]
    a = 2 * (k ÷ ppf + u) / ppf - 1
    b = 2 * (k % ppf + v) / ppf - 1
    p = n + a * t1 + b * t2
    return p / norm(p)
end
cubed_sphere(order, ppf) = QuadElemList((e, u, v) -> sphere_point(e, u, v, ppf), order, 6ppf^2)

el = cubed_sphere(8, 2)          # 24 elements of order 8
X, N = node_coord(el)            # 3 × nnodes each
w = node_weights(el)             # surface quadrature weights
(nelem(el), order(el), nnodes(el)), abs(sum(w) - 4π), maximum(abs, N - X)
```

The constructor `QuadElemList(f, order, nelem)` evaluates `f(e, u, v)` at the nodes `(u_i, v_j)`, where `u` and
`v` take the values of [`param_nodes`](@ref)`(order)`. The weights sum to the area to about ``10^{-6}``, and the
normals match ``X`` (true for the unit sphere) to the accuracy of the order-8 geometry. [`elem_geom`](@ref)
evaluates the element map anywhere in ``[0,1]^2``:

```@example bie
g = elem_geom(el, 5, [0.0, 0.5, 1.0], [0.25])      # 3 points of element 5
maximum(abs, sqrt.(sum(abs2, g.X; dims=1)) .- 1), size(g.normal), g.area
```

### Double layer of a constant density

By default the targets are the surface nodes, so `compute_potential` returns the on-surface principal value.
[`set_target_coord!`](@ref) switches to off-surface targets:

```@example bie
D = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=1e-8)
add_elem_list!(D, el)
println("on surface:  ", maximum(abs, compute_potential(D, ones(nnodes(el))) .+ 0.5))

dirs = X[:, 1:23:end] ./ sqrt.(sum(abs2, X[:, 1:23:end]; dims=1))    # some unit vectors
Tin, Tout = 0.97 .* dirs, 1.03 .* dirs                               # near targets, both sides
set_target_coord!(D, hcat(Tin, Tout))
u = compute_potential(D, ones(nnodes(el)))
nt = size(dirs, 2)
println("inside:      ", maximum(abs, u[1:nt] .+ 1))
println("outside:     ", maximum(abs, u[nt+1:end]))
set_target_coord!(D, nothing)          # back to the surface nodes
nothing # hide
```

On the surface the identity holds to the quadrature tolerance. At the near targets (3% of the radius away), the
error of about ``10^{-5}`` comes from the geometry, not from the quadrature: each element interpolates the sphere
through its interior Gauss-Legendre nodes, so neighboring order-8 elements do not meet exactly along their shared
edges, and the discretized surface is not exactly closed. The error drops quickly with the element order (see the
convergence study below).

### Quadrature schemes

[`set_quad_scheme!`](@ref) selects the singular and near-singular rule of an element list. `:duffy` (the default)
and `:tensor_product` return the principal value on the surface. `:hedgehog` extrapolates from points off the
surface along the normal, so it returns the one-sided limit from the normal's side, here the outside, where
``D_+[1] = 0``. `:hedgehog` is required for kernels more singular than ``1/r`` along the surface, such as the
gradient of the double layer.

```@example bie
for scheme in (:duffy, :tensor_product, :hedgehog)
    Ds = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=1e-8)
    add_elem_list!(Ds, set_quad_scheme!(cubed_sphere(8, 2), scheme))   # set before adding: the op copies
    pot = compute_potential(Ds, ones(nnodes(el)))
    println(rpad(scheme, 16), "mean = ", round(sum(pot) / length(pot); digits=8))
end
```

### Green's identity

Let ``u`` be harmonic inside the sphere, for example the field of a point charge outside it. Green's
representation formula then reads ``S[\partial_n u] - D[u] = u`` inside and ``= u/2`` on the surface, where
``S`` is the single layer. The normal derivative comes from the gradient kernel:

```@example bie
x0 = [1.2, 0.9, -0.4]                                       # charge outside the sphere
uex(Y) = vec(kernel_eval(Laplace3D_FxU(), x0, [1.0], Y))
us = uex(X)
dudn = vec(sum(kernel_eval(Laplace3D_FxdU(), x0, [1.0], X) .* N; dims=1))

S = BoundaryIntegralOp(Laplace3D_FxU(); accuracy=1e-8)
add_elem_list!(S, el)
on = compute_potential(S, dudn) - compute_potential(D, us)
set_target_coord!(S, Tin); set_target_coord!(D, Tin)
ins = compute_potential(S, dudn) - compute_potential(D, us)
maximum(abs, on - us / 2) / maximum(abs, us), maximum(abs, ins - uex(Tin)) / maximum(abs, us)
```

### Solving an exterior Dirichlet problem

Find ``u`` harmonic outside the sphere with ``u = g`` on ``\Gamma`` and ``u \to 0`` at infinity. A pure
double-layer ansatz is not enough here: ``\tfrac12 I + D`` annihilates constants, since ``D_+[1] = 0``. The
combined-field representation ``u = D[\sigma] + S[\sigma]`` gives the injective second-kind equation

```math
\left(\tfrac12 I + D + S\right)\sigma = g \quad \text{on } \Gamma .
```

Take as boundary data the field of a point charge at ``x_0`` inside the sphere, so that the exact solution is that
field. Solve with [`gmres`](@ref), which applies the operator matrix-free, then evaluate ``u`` at exterior points:

```@example bie
x0 = [0.2, -0.1, 0.15]                                       # charge inside: u is harmonic outside
uex(Y) = vec(kernel_eval(Laplace3D_FxU(), x0, [1.0], Y))

function solve_exterior(order, ppf, Y; tol=1e-10)
    el = cubed_sphere(order, ppf)
    X, _ = node_coord(el)
    D = add_elem_list!(BoundaryIntegralOp(Laplace3D_DxU(); accuracy=tol), el)
    S = add_elem_list!(BoundaryIntegralOp(Laplace3D_FxU(); accuracy=tol), el)
    A!(y, σ) = (y .= 0.5 .* σ .+ compute_potential(D, σ) .+ compute_potential(S, σ))
    σ, iters = gmres(A!, uex(X); tol=1e-12)
    set_target_coord!(D, Y); set_target_coord!(S, Y)
    u = compute_potential(D, σ) + compute_potential(S, σ)
    return nnodes(el), iters, maximum(abs, u - uex(Y)) / maximum(abs, uex(Y))
end

Y = [1.5 0.0 -2.0 1.1; 0.0 3.0 1.0 1.1; 0.5 -1.0 0.0 1.1]  # exterior targets (3 × 4)
for order in (4, 8, 12)
    n, iters, err = solve_exterior(order, 2, Y)
    println("order ", lpad(order, 2), ": ", lpad(n, 4), " nodes, ", iters, " GMRES iterations, error = ", err)
end
```

The second-kind equation is well conditioned, so the number of GMRES iterations stays small and does not grow with
refinement. The error falls rapidly with the element order (spectral convergence for this smooth problem). It
stops near the quadrature tolerance and the accuracy of the polynomial geometry.

### Several element lists, scaling

An operator can hold several element lists, added under different names. The density vector concatenates their
nodes in **lexicographic order of the names** (see [`elem_list_names`](@ref)). [`sqrt_scaling!`](@ref) and
[`inv_sqrt_scaling!`](@ref) multiply and divide node data by ``\sqrt{\text{element area}}``. This symmetric
diagonal scaling is useful for iterative solvers.

```@example bie
small = QuadElemList(8, 0.3 .* node_coord(cubed_sphere(8, 1))[1] .+ [3.0, 0, 0])   # second sphere
add_elem_list!(S, small, "a_small")
elem_list_names(S), dim(S, 1) == nnodes(small) + nnodes(el)
```

## Data layouts

| Quantity | Julia shape | Notes |
|:---------|:------------|:------|
| `QuadElemList(order, coord)` input | `(3, order, order, nelem)` with `coord[:, j, i, e]` = node at `(u_i, v_j)` | **`v` varies fastest**: the second index is `j` (`v`) |
| … equivalently | `3 × (order^2 * nelem)` or a flat vector | node `(i, j)` of element `e` in column `(e-1)*order^2 + (i-1)*order + j` |
| node order (all node data) | as above, element by element | SCTL stores coordinates internally per element as all `x`, then `y`, then `z`; not exposed |
| `node_coord(el)` | `(X, N)`, each `3 × nnodes` | unit normals along ``\partial_u X \times \partial_v X`` |
| `node_weights(el)` | `nnodes` vector | area element × tensor-product Gauss-Legendre weights |
| `elem_geom(el, e, u, v)` | `X`, `normal`, `dXdu`, `dXdv`: `3 × (nu*nv)`; `area`: `nu*nv` | point `(u[a], v[b])` in column `(a-1)*nv + b`; `e` is 1-based |
| density `F` | `dim(op, 1)` values: vector or `src_dim × nnodes_total` | node order; element lists sorted by name |
| potential `U` | `dim(op, 2)` values; `dof × Nt` if `F` is a matrix | `dof = trg_dim`, or `trg_dim ÷ 3` with `trg_normal_dot_prod=true` |
| targets / target normals | `3 × Nt` | default: the surface nodes and their normals |
| complex densities (Helmholtz) | `(re, im)` pairs, `src_dim = 2` | as in [Kernel functions](@ref) |

[`dim`](@ref dim(::BoundaryIntegralOp, ::Integer)) uses `i = 1` for the input length and `i = 2` for the output
length, which correspond to SCTL's `Dim(0)` and `Dim(1)`.

## Pitfalls and limitations

- **Far-field cost.** Laplace and Stokes kernels use PVFMM for the far field once there are 40000 or more targets
  (surface nodes plus set targets). An application then costs ``O(N_\text{nodes} + N_\text{targets})``. The first
  application for a kernel and accuracy also precomputes FMM operators, cached on disk; see
  [Fast multipole method](@ref). Helmholtz kernels always use a direct ``O(N_\text{nodes} N_\text{targets})`` sum,
  because PVFMM does not pass them their wavenumber. That is fine for a few ``10^4`` nodes. Periodicity is not exposed.
- **Element order.** Elements must have order 4, 8, 12, 16 or 20 to be added to an operator. These are the orders
  that SCTL's singular quadratures are compiled for. [`QuadElemList`](@ref) itself accepts any order, e.g. for
  geometry or visualization.
- **The operator copies the element list.** [`add_elem_list!`](@ref) copies the list, so call
  [`set_quad_scheme!`](@ref) before adding it. To apply a change made afterwards, add the list again under the
  same name; that replaces the old copy.
- **Scheme vs. kernel.** `:duffy` and `:tensor_product` assume the integrand grows at most like ``1/r`` along the
  surface. For steeper kernels, e.g. `Laplace3D_DxdU`, or `Laplace3D_FxdU` without `trg_normal_dot_prod`, their
  on-surface result is wrong and SCTL prints a warning to stderr. Use `:hedgehog` for these kernels.
- **On-surface values are principal values** (`:duffy`, `:tensor_product`) or one-sided limits from the normal's
  side (`:hedgehog`). Add or subtract the jump ``\pm\tfrac12\sigma`` yourself when forming the integral equation,
  as in the exterior problem above.
- **Target normals.** With `trg_normal_dot_prod=true` and custom targets, call [`set_target_normal!`](@ref) with
  one normal per target. Without it, the default normals are the surface normals, which match only the default
  targets.
- **Accuracy.** `accuracy` (default `1e-10`) sets the quadrature tolerance. Elements are interpolated through
  interior Gauss-Legendre nodes, so neighboring elements match only approximately along their shared edges, and
  identities that need a closed surface hold only to the geometric accuracy. The final error is also limited by how
  well the polynomial elements resolve the geometry and the density. Check convergence by refining `order` or
  `ppf`, as above.
- **Setup cost.** The first `compute_potential` after a change of geometry, targets or accuracy builds the
  near-field correction matrices. [`setup!`](@ref) does this eagerly. Later applications reuse them.

## API reference

```@docs
QuadElemList
nelem
order(::QuadElemList)
nnodes
param_nodes
node_coord
node_weights
elem_geom
set_quad_scheme!
quad_scheme
write_vtk
Base.close(::QuadElemList)
BoundaryIntegralOp
bie_supported_kernels
add_elem_list!
delete_elem_list!
elem_list_names
set_target_coord!
set_target_normal!
set_accuracy!
accuracy
dim(::BoundaryIntegralOp, ::Integer)
setup!
clear_setup!
compute_potential
compute_potential!
sqrt_scaling!
inv_sqrt_scaling!
Base.close(::BoundaryIntegralOp)
```
