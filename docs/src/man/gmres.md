# GMRES

SCTL ships a GMRES solver (`sctl::GMRES<double>`) and a Krylov-subspace *recycling* preconditioner
(`sctl::KrylovPrecond<double>`). SCTL.jl wraps them as [`gmres`](@ref) and [`KrylovPrecond`](@ref).

[`gmres`](@ref) solves a square, possibly non-symmetric linear system

```math
A x = b, \qquad A \in \mathbb{R}^{n \times n},
```

using only products ``y = A x``. The operator can be a matrix or a Julia function, so it can be *matrix-free*: a
fast summation, a boundary integral operator (see [Boundary integral operators](@ref)), or any other
`A!(y, x)`. SCTL's GMRES is **unrestarted**. Iteration ``k`` builds an orthonormal (Arnoldi) basis ``Q_k`` of the
Krylov space ``\mathcal{K}_k(A, r_0) = \operatorname{span}\{r_0, A r_0, \dots, A^{k-1} r_0\}`` and picks the
iterate ``x_k \in x_0 + \mathcal{K}_k`` with the smallest residual ``\|b - A x_k\|_2``. Each iteration costs one
operator application and ``O(k n)`` work for the orthogonalization. Memory grows as ``O(k n)``. GMRES suits
operators whose spectrum is clustered, such as second-kind integral equations ``(I + K)\,x = b``, which typically
converge in a few dozen iterations independent of ``n``.

When the same operator is solved with many right-hand sides (time stepping, parameter sweeps, Newton iterations),
a [`KrylovPrecond`](@ref) keeps information from previous solves. After each solve it appends a low-rank
correction built from that solve's Krylov subspace. The preconditioner is applied on the right,
``A P y = b`` with ``x = P y``, and it becomes

```math
P \leftarrow P\,(I + U Q^{\mathsf T}),
```

where the ``n \times k`` factors ``Q`` (the Arnoldi basis with the solve's Givens rotations applied) and ``U``
(built from the inverse of the ``k \times k`` Hessenberg matrix) come from a solve with ``k`` iterations. On the
directions it has already seen, ``P`` therefore approximates ``A^{-1}``. Later solves with nearby right-hand sides need fewer
iterations.

## Tutorial

### A matrix-free second-kind integral equation

Consider the Fredholm equation of the second kind on ``[0, 1]``

```math
u(x) + \int_0^1 e^{x y}\, u(y)\, dy = f(x), \qquad f(x) = e^{-x} + \frac{e^{x-1} - 1}{x - 1},
```

whose exact solution is ``u(x) = e^{-x}``. A Nyström discretization with the ``n``-point Gauss–Legendre rule
``(t_j, w_j)`` (from [`legquad`](@ref)) gives the linear system
``u_i + \sum_j e^{t_i t_j} w_j u_j = f(t_i)``. The operator is applied without ever forming the matrix:

```@example gmres
using SCTL, LinearAlgebra

n = 200
t, w = legquad(n)                  # Gauss-Legendre nodes and weights on [0, 1]

function A!(y, u)                  # y = (I + K W) u, matrix-free
    for i in eachindex(y)
        s = 0.0
        for j in eachindex(u)
            s += exp(t[i] * t[j]) * w[j] * u[j]
        end
        y[i] = u[i] + s
    end
    return y
end

f(x) = exp(-x) + expm1(x - 1) / (x - 1)
b = f.(t)

sol = gmres(A!, b; tol=1e-13)
(iters = sol.iters, max_error = maximum(abs, sol.x .- exp.(-t)))
```

The kernel is smooth, so ``K`` is numerically low-rank and GMRES converges in a handful of iterations. The Nyström
solution matches the exact solution to machine precision. For comparison, here is the same system as a dense
matrix with a direct solve. A matrix operator is applied with `mul!`:

```@example gmres
Amat = [Float64(i == j) + exp(t[i] * t[j]) * w[j] for i in 1:n, j in 1:n]
x_direct = Amat \ b
sol_mat = gmres(Amat, b; tol=1e-13)
(iters = sol_mat.iters,
 rel_diff_vs_backslash = norm(sol_mat.x - x_direct) / norm(x_direct),
 rel_residual = norm(Amat * sol_mat.x - b) / norm(b))
```

The result is a `NamedTuple` `(x, iters)`, so it can also be destructured as `x, iters = gmres(A, b)`.

### Stopping criteria and other options

By default the iteration stops when ``\|b - A x_k\| \le \mathrm{tol}\,\|b\|``. With `abstol=true` the test is
``\|b - A x_k\| \le \mathrm{tol}``. `maxiter` caps the number of iterations; the default `-1` means ``n``.
`x0` supplies an initial guess.

```@example gmres
loose = gmres(Amat, b; tol=1e-4)                      # relative tolerance
absol = gmres(Amat, 1e6 .* b; tol=1e-4, abstol=true)  # absolute tolerance on a scaled RHS
capped = gmres(Amat, b; maxiter=2)                    # stopped after 2 iterations
warm = gmres(Amat, b; x0=sol.x)                       # converged initial guess: 0 iterations
cgs = gmres(Amat, b; tol=1e-13, orthogonalization=:cgs, reorth=1)
(loose = (loose.iters, norm(Amat * loose.x - b) / norm(b)),
 absol = (absol.iters, norm(Amat * absol.x - 1e6 .* b)),
 capped = capped.iters, warm = warm.iters,
 cgs = (cgs.iters, maximum(abs, cgs.x .- sol.x)))
```

`orthogonalization=:cgs` selects classical Gram–Schmidt in the Arnoldi step instead of the default modified
Gram–Schmidt (`:mgs`). `reorth=1` adds one re-orthogonalization pass, which is the usual safeguard for CGS. In
SCTL the choice mainly matters for MPI runs, where CGS batches the reductions; SCTL.jl is MPI-free, so the default
`:mgs` is a good choice.

### Recycling Krylov subspaces across right-hand sides

A sharper Gaussian kernel ``K(x, y) = 4\,\mathcal{N}_\sigma(x - y)`` with ``\sigma = 0.02`` is far less
compressible, and GMRES needs about 20 iterations per solve. We solve a sequence of six related right-hand sides,
once from scratch and once with a shared [`KrylovPrecond`](@ref):

```@example gmres
σ = 0.02
B = [Float64(i == j) + 4 * exp(-(t[i] - t[j])^2 / (2σ^2)) / (σ * sqrt(2π)) * w[j] for i in 1:n, j in 1:n]

P = KrylovPrecond()
println("solve  plain  recycled  rank(P)  rel. residual")
for k in 1:6
    bk = cos.(k .* t) .+ t .^ 2
    plain = gmres(B, bk)
    rec = gmres(B, bk; precond=P)          # uses and then updates P
    res = norm(B * rec.x - bk) / norm(bk)
    println(lpad(k, 5), lpad(plain.iters, 7), lpad(rec.iters, 10), lpad(rank(P), 9), "  ", round(res; sigdigits=2))
end
println(P)
```

The first solve with `P` is identical to a plain solve, because a fresh `KrylovPrecond` is the identity. Each
later solve reuses all previous subspaces: the iteration count drops, while the residual still meets the
tolerance. `rank(P)` grows by the iteration count of each solve, and so does the cost of applying `P`, which is
``O(n\,\mathrm{rank}(P))``. Start a new `KrylovPrecond` when that cost stops paying off or the operator changes.

The preconditioner can also be applied directly, as `P * x` or in place with `apply!(P, x)`:

```@example gmres
v = cos.(3 .* t) .+ t .^ 2        # the third right-hand side above
y = P * v                         # out of place
y2 = apply!(P, copy(v))           # in place, same result
# P is a crude approximate inverse of B on the recycled directions: P * v is a much better approximate
# solution of B x = v than v itself (which is what the identity preconditioner would give)
(same = y == y2, residual_P = norm(B * y - v) / norm(v), residual_identity = norm(B * v - v) / norm(v))
```

### Errors inside the operator

Exceptions thrown by `A!` abort the solve and are rethrown unchanged by [`gmres`](@ref):

```@example gmres
try
    gmres((y, x) -> throw(DomainError(x[1], "operator failed")), b)
catch err
    err
end
```

## Data layouts

| Quantity | Julia type and shape | SCTL C++ counterpart |
|:--|:--|:--|
| `b`, `x0`, returned `x` | `Vector{Float64}` of length `n` | `sctl::Vector<double>` of `Dim() == n` |
| matrix operator `A` | any `AbstractMatrix` of size `n × n` (applied with `mul!(y, A, x)`) | wrapped into `GMRES::ParallelOp` |
| function operator | `A!(y, x)`: `x` and `y` are `AbstractVector{Float64}` of length `n`; overwrite `y` with `A * x` | `ParallelOp = std::function<void(Vector*, const Vector&)>` |
| return value | `NamedTuple` `(x::Vector{Float64}, iters::Int)` | `x` and `solve_iter` outputs |
| `KrylovPrecond` | `size(P) == (n, n)` (`(0, 0)` while empty), `rank(P)` = sum of the Krylov dimensions | `KrylovPrecond::Size()`, `KrylovPrecond::Rank()` |

The arguments `x` and `y` of `A!` are valid **only during the call**. For `n ≤ 512` they are Julia buffers
that are reused across calls. For larger `n` they are zero-copy views of SCTL's internal memory (`Q_k` rows and
scratch storage). Never modify `x`, and never keep a reference to either array after `A!` returns; copy them if
you need them later. `x` and `y` never alias each other.

## Pitfalls and limitations

- **No restarts.** Memory and orthogonalization cost grow linearly with the iteration count; the Arnoldi basis
  takes ``8\,n\,k`` bytes. For slowly converging problems, cap the work with `maxiter` and improve the
  formulation (e.g. use a second-kind formulation or a preconditioned operator `A!`) instead of expecting GMRES(m)
  behavior.
- **`maxiter = -1` means `n` iterations.** In exact arithmetic GMRES terminates in at most ``n`` steps, but with
  `tol` too small for the conditioning of `A` it runs all ``n`` iterations and returns the best iterate it has.
  There is no "not converged" flag; check `iters == n` or compute the residual yourself if it matters.
- **Right preconditioning only through `KrylovPrecond`.** For any other preconditioner ``M^{-1}``, solve
  ``A M^{-1} y = b`` with `A!(y, x) = A * (M \ x)` and set ``x = M^{-1} y`` afterwards.
- **`KrylovPrecond` and changing operators.** `P` is only meaningful for the operator it was built with. Using it
  with an operator of the same size but different entries is allowed, and the solve still converges to the right
  answer because `P` acts only as a preconditioner. It may slow convergence, though. Using it with a different
  size resets it on the next solve.
- **Threading.** SCTL calls `A!` only from the calling thread, never from inside an OpenMP region, so `A!` may
  use any Julia code, including Julia threads. The orthogonalization uses SCTL's OpenMP settings; see
  `SCTL.set_num_threads`.
- **`verbose=true`** prints from C via `printf` to the process' standard output. It is not captured by Julia's
  `redirect_stdout` or by Documenter.
- **Lifetime.** A `KrylovPrecond` owns C++ memory, released by the finalizer or eagerly by `close(P)`. After
  `close` every use throws an `ArgumentError`.

## API reference

```@meta
CurrentModule = SCTL
```

```@docs
gmres
KrylovPrecond
apply!
LinearAlgebra.rank(::KrylovPrecond)
Base.close(::KrylovPrecond)
```
