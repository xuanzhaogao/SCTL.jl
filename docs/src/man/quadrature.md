# Quadrature and interpolation

These are SCTL's one-dimensional building blocks: classical quadrature rules (`ChebQuadRule`, `LegQuadRule`),
*generalized* quadrature rules built for a given set of integrands (`InterpQuadRule`), and polynomial interpolation
and differentiation (`LagrangeInterp`).

```@setup quad
using SCTL, LinearAlgebra, Printf
```

## Classical rules

[`legquad(n)`](@ref legquad) is the ``n``-point Gauss-Legendre rule. It integrates polynomials of degree ``2n-1``
exactly. [`chebquad(n)`](@ref chebquad) is Fejér's first rule: its nodes are the Chebyshev points of the first kind,
``x_i = \tfrac12\left(1 - \cos\tfrac{(2i-1)\pi}{2n}\right)``, and its weights integrate the degree-``n-1``
Chebyshev interpolant exactly. Both rules are **on ``[0, 1]``** and are open, i.e. no node sits at an endpoint.
For an interval ``[a, b]``, rescale: ``\int_a^b f \approx (b-a) \sum_i w_i f(a + (b-a) x_i)``.

```@example quad
f(x) = exp(-x) * cos(4x)
exact = (1 - exp(-1) * (cos(4) - 4sin(4))) / 17          # ∫₀¹ f
for n in (4, 8, 12, 16)
    xl, wl = legquad(n)
    xc, wc = chebquad(n)
    @printf("n = %2d   Gauss-Legendre error %.1e   Fejér error %.1e\n",
            n, abs(dot(wl, f.(xl)) - exact), abs(dot(wc, f.(xc)) - exact))
end
```

Gauss-Legendre converges about twice as fast in ``n``. Fejér nodes, on the other hand, are nested and convenient for
Chebyshev interpolation.

[`legendre_poly(x, n)`](@ref legendre_poly) evaluates the Legendre polynomial ``P_n`` and its derivative on
**``[-1, 1]``**. The two rules are related: the Gauss-Legendre nodes are the roots of ``P_n``, mapped to ``[0, 1]``.

```@example quad
x, _ = legquad(7)
P, dP = legendre_poly(2 .* x .- 1, 7)
maximum(abs, P)                                           # P₇ vanishes at the nodes
```

## Generalized (Chebyshev) quadrature

`InterpQuadRule` builds a rule with as few nodes as possible that integrates a *given* set of functions
``\{\phi_j\}`` to a tolerance ``\varepsilon``. This follows Bremer, Gimbutas and Rokhlin,
[SIAM J. Sci. Comput. 32 (2010)](https://doi.org/10.1137/080737046):

1. discretize the ``\phi_j`` with a reference rule that resolves them;
2. orthonormalize them (SVD or pivoted QR) and truncate at ``\varepsilon``, leaving ``k`` basis functions;
3. pick ``k`` stable interpolation nodes;
4. compute the weights by least squares.

The result is a ``k``-point rule that integrates every function in the span of the ``\phi_j`` to about
``\varepsilon``.

Typical uses are quadratures for singular or near-singular integrands, such as ``x^m \log x``, which a polynomial
rule resolves only slowly.

### From a function

[`interp_quad_rule(f, a, b)`](@ref interp_quad_rule) adaptively discretizes the integrands on ``[a, b]`` with
composite Gauss-Legendre panels. `f(x)` receives a vector of nodes and returns a matrix with one column per
integrand, `M[i, j] = φⱼ(x[i])`:

```@example quad
# integrands: x^k and x^k log(x), k = 0..9, on [0, 1]
phis(x) = hcat([x .^ k for k in 0:9]..., [x .^ k .* log.(x) for k in 0:9]...)
exact = vcat([1 / (k + 1) for k in 0:9], [-1 / (k + 1)^2 for k in 0:9])

xq, wq, cond = interp_quad_rule(phis, 0.0, 1.0; eps = 1e-12)
@printf("%d nodes, condition number %.1f, max error %.1e\n",
        length(xq), cond, maximum(abs, phis(xq)' * wq .- exact))
```

A Gauss-Legendre rule needs far more nodes to reach the same accuracy on these integrands:

```@example quad
for n in (20, 80, 320)
    xn, wn = legquad(n)
    @printf("Gauss-Legendre, %3d nodes: max error %.1e\n", n, maximum(abs, phis(xn)' * wn .- exact))
end
```

### From a discretization

When the integrands are already sampled, pass the samples `M` (`nnodes × nfuncs`) and the reference rule
(`nodes`, `weights`) instead. [`interp_quad_rules`](@ref) builds a whole family of rules in one pass, one per
tolerance in `eps` or per node count in `order`:

```@example quad
# reference rule: 30-point Gauss-Legendre panels on [2⁻ⁱ, 2¹⁻ⁱ], i = 1..50, refined towards the log singularity
xg, wg = legquad(30)
xs = vcat(xg ./ 2^50, [2.0^-i .* (1 .+ xg) for i in 1:50]...)
ws = vcat(wg ./ 2^50, [2.0^-i .* wg for i in 1:50]...)
M = phis(xs)

for (xr, wr, c) in interp_quad_rules(M, xs, ws; eps = [1e-4, 1e-8, 1e-12])
    @printf("%2d nodes: max error %.1e\n", length(xr), maximum(abs, phis(xr)' * wr .- exact))
end
```

The keyword `interval = (a, b)` restricts the nodes to a subinterval. `use_svd = false` switches the
orthogonalization to pivoted QR.

## Lagrange interpolation and spectral differentiation

[`lagrange_interp_matrix(src, trg)`](@ref lagrange_interp_matrix) returns the matrix ``W`` of the barycentric
Lagrange interpolant through the nodes `src`, evaluated at `trg`, so that `W * f(src) ≈ f(trg)`. ``W`` has size
`length(trg) × length(src)`. [`lagrange_interp`](@ref) applies it directly.
[`lagrange_derivative(f, nodes)`](@ref lagrange_derivative) differentiates the interpolant at the nodes themselves.

On Chebyshev or Legendre nodes, both converge spectrally for smooth functions:

```@example quad
g(x) = 1 / (1 + 25 * (x - 0.5)^2)          # Runge's function, shifted to [0, 1]
dg(x) = -50 * (x - 0.5) / (1 + 25 * (x - 0.5)^2)^2
t = collect(range(0, 1; length = 1001))
for n in (8, 16, 32, 64)
    xc, _ = chebquad(n)
    e_int = maximum(abs, lagrange_interp(xc, g.(xc), t) .- g.(t))
    e_der = maximum(abs, lagrange_derivative(g.(xc), xc) .- dg.(xc))
    @printf("n = %2d   interpolation error %.1e   derivative error %.1e\n", n, e_int, e_der)
end
```

Several functions can be handled at once: pass a matrix with one column per function, and the result has the same
shape.

```@example quad
x, _ = chebquad(12)
F = [sin.(x) cos.(x) exp.(x)]                # 12 × 3
D = lagrange_derivative(F, x)                # 12 × 3
maximum(abs, D .- [cos.(x) -sin.(x) exp.(x)])
```

## Data layouts

| function | input | output |
|:-|:-|:-|
| `legquad(n)`, `chebquad(n)` | | nodes `x::Vector` (increasing, in ``(0,1)``), weights `w::Vector` |
| `legendre_poly(x, n)` | points in ``[-1,1]`` | `(P, dP)`, both like `x` |
| `interp_quad_rule(M, nodes, weights)` | `M`: `nnodes × nfuncs`, `M[i,j] = φⱼ(nodes[i])` | `(x, w, cond)` |
| `interp_quad_rule(f, a, b)` | `f(x::Vector) -> length(x) × nfuncs` matrix | `(x, w, cond)` |
| `interp_quad_rules(M, nodes, weights)` | as above, plus `eps` and/or `order` vectors | vector of `(x, w, cond)` |
| `lagrange_interp_matrix(src, trg)` | | `length(trg) × length(src)` (SCTL: row-major `Ns × Nt`) |
| `lagrange_derivative(f, nodes)` | `f`: vector, or `length(nodes) × nfuncs` matrix | same shape as `f` |

## Pitfalls

- **Intervals differ.** The quadrature rules are on ``[0, 1]``, but `legendre_poly` is on ``[-1, 1]``.
- **Rules are recomputed on every call.** `legquad(16)` takes about 1 µs. In a hot loop, compute the rule once and
  reuse it.
- **Integrands must be resolvable.** `interp_quad_rule(f, a, b)` refines until each panel reaches `eps`. Integrands
  that cannot be resolved there, such as non-integrable singularities or functions returning `NaN`/`Inf`, make the
  build fail or recurse deeply. Keep endpoint singularities integrable, as with ``\log x`` or ``x^{-1/2}``. The rule
  avoids the endpoints, because its nodes are interior.
- **Watch the condition number.** It is the third return value. A rule with `cond` ≫ 10⁴ loses digits; loosen `eps`
  or reduce the number of integrands.
- **Nodes must be distinct.** Lagrange interpolation divides by ``x_i - x_j``. Equispaced nodes work but suffer the
  Runge phenomenon at high degree; use Chebyshev or Legendre nodes.

## Reference

```@docs
legquad
chebquad
legendre_poly
interp_quad_rule
interp_quad_rules
lagrange_interp_matrix
lagrange_interp
lagrange_derivative
```
