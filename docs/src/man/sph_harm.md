# Spherical harmonics

This module wraps `sctl::SphericalHarmonics<double>` (header `sctl/sph_harm.hpp`). It covers spectral
calculus on the unit sphere ``S^2``:

- **scalar transforms** between values on a tensor-product grid and spherical harmonic coefficients
  ([`grid2shc`](@ref), [`shc2grid`](@ref)), including ``θ``/``φ`` derivatives, plus evaluation at arbitrary points
  ([`shceval`](@ref)) and at the poles ([`shc2pole`](@ref));
- **vector transforms** for Cartesian vector fields on the sphere, expanded in vector spherical harmonics
  ([`grid2vecshc`](@ref), [`vecshc2grid`](@ref), [`vecshceval`](@ref));
- **layer potentials** of densities on the unit sphere, evaluated analytically in the spectral basis: Laplace
  single and double layer ([`laplace_eval_sl`](@ref), [`laplace_eval_dl`](@ref)) and Stokes single layer,
  double layer and traction ([`stokes_eval_sl`](@ref), [`stokes_eval_dl`](@ref), [`stokes_eval_kl`](@ref),
  [`stokes_eval_kself`](@ref)).

A real field of degree at most ``p`` is represented as

```math
f(θ,φ) = \sum_{n=0}^{p} \sum_{m=0}^{n} c_m \sqrt{4π}\, \bar P_n^m(\cos θ)\, \mathrm{Re}\big(A_{nm}\, e^{imφ}\big),
\qquad c_0 = 1,\ c_{m>0} = 2,
```

where ``\bar P_n^m(x) = \sqrt{\tfrac{2n+1}{4π}\tfrac{(n-m)!}{(n+m)!}}\, P_n^m(x)`` carries the Condon–Shortley phase, so
``Y_n^m = \bar P_n^m(\cos θ) e^{imφ}`` is orthonormal. Equivalently ``f = \sqrt{4π}\sum_{n}\sum_{m=-n}^{n} A_{nm} Y_n^m``
with ``A_{n,-m} = (-1)^m \overline{A_{nm}}``. Only ``m ≥ 0`` is stored. With this scaling ``f ≡ 1`` has
``A_{00} = 1``, the same normalization as the Fortran code SCTL's implementation follows.

The grid is the usual Gauss–Legendre × trapezoidal product grid: `Nt` latitudes with ``\cos θ_i`` the Gauss–Legendre
nodes on ``[-1,1]``, and `Np` equispaced longitudes ``φ_j = 2π j/N_p``. The transform is exact (up to rounding) for
fields of degree ``≤ p`` when `Nt ≥ p + 1` and `Np ≥ 2p + 1`. A common choice is `Nt = p + 1`, `Np = 2p + 2`.

Use this module for problems posed on a sphere: spectral differentiation and filtering, interpolation of gridded
data, and analytic (spectrally accurate, singularity-free) layer potentials on spherical boundaries, such as
vesicle and particle models.

## Tutorial

```@example sph
using SCTL, LinearAlgebra
```

### Grid ↔ coefficient round trip

[`sph_grid_angles`](@ref) returns the grid angles. Grid values are stored with `φ` fastest, so the natural Julia
array is `(Np, Nt)`:

```@example sph
p = 8
Nt, Np = p + 1, 2p + 2
θ, φ = sph_grid_angles(Nt, Np)
grid(f) = [f(θ[i], φ[j]) for j in 1:Np, i in 1:Nt]          # size (Np, Nt)

# a band-limited field of degree 3
f(t, ph) = 0.7 + cos(t) - 0.4sin(t)cos(ph) + sin(t)^2 * cos(2ph) + 0.2sin(t)^3 * sin(3ph)
X = grid(f)
S = grid2shc(X, Nt, Np, p)                                  # length shc_length(p) = (p+1)(p+2)
X2 = shc2grid(S, p, Nt, Np)                                 # flat, reshape(X2, Np, Nt)
(length(S), shc_length(p), maximum(abs, X2 .- vec(X)))
```

The coefficients can be evaluated on any other grid, which is spectral interpolation. With
`derivatives = true`, [`shc2grid`](@ref) also returns ``∂f/∂θ`` and ``∂f/∂φ``:

```@example sph
ft(t, ph) = -sin(t) - 0.4cos(t)cos(ph) + 2sin(t)cos(t)cos(2ph) + 0.6sin(t)^2 * cos(t) * sin(3ph)
fp(t, ph) = 0.4sin(t)sin(ph) - 2sin(t)^2 * sin(2ph) + 0.6sin(t)^3 * cos(3ph)

Nt2, Np2 = 20, 31                     # finer output grid
θ2, φ2 = sph_grid_angles(Nt2, Np2)
grid2(g) = vec([g(θ2[i], φ2[j]) for j in 1:Np2, i in 1:Nt2])
F, Fθ, Fφ = shc2grid(S, p, Nt2, Np2; derivatives = true)
(maximum(abs, F - grid2(f)), maximum(abs, Fθ - grid2(ft)), maximum(abs, Fφ - grid2(fp)))
```

### Spectral convergence

A smooth but not band-limited function such as ``e^{x} = e^{\sin θ \cos φ}`` converges exponentially in `p`.
Here the error is measured with [`shceval`](@ref) at 200 scattered points on a Fibonacci spiral. The points
form a `2 × N` matrix of ``(θ, φ)`` pairs:

```@example sph
g(t, ph) = exp(sin(t) * cos(ph))
N = 200
pts = [acos.(1 .- 2 .* ((1:N) .- 0.5) ./ N)'; mod.(2.399963 .* (1:N), 2π)']  # 2 × N: rows θ, φ
exact = [g(pts[1, k], pts[2, k]) for k in 1:size(pts, 2)]
function conv_err(p)
    Nt, Np = p + 1, 2p + 2
    θ, φ = sph_grid_angles(Nt, Np)
    S = grid2shc([g(θ[i], φ[j]) for j in 1:Np, i in 1:Nt], Nt, Np, p)
    return maximum(abs, shceval(S, p, pts) .- exact)
end
for p in (2, 4, 8, 12, 16)
    println("p = ", lpad(p, 2), "   max error = ", conv_err(p))
end
```

[`shc2pole`](@ref) returns the values at ``θ = 0`` and ``θ = π`` directly. The Gauss–Legendre grid does not contain
the poles:

```@example sph
S = grid2shc(X, Nt, Np, p)
shc2pole(S, p), (f(0.0, 0.0), f(π, 0.0))
```

### Coefficient layouts for a tiny order

Take ``p = 1`` and ``f = 1 + 2\cos θ + \sin θ\cos φ - 3\sin θ\sin φ``. Its coefficients are ``A_{00} = 1``,
``A_{10} = 2/\sqrt3``, ``A_{11} = (-1 - 3i)/\sqrt6``. The three [`SHCArrange`](@ref) layouts store them as follows:

```@example sph
p1 = 1
θ1, φ1 = sph_grid_angles(p1 + 1, 2p1 + 2)
h(t, ph) = 1 + 2cos(t) + sin(t)cos(ph) - 3sin(t)sin(ph)
H = [h(θ1[i], φ1[j]) for j in eachindex(φ1), i in eachindex(θ1)]
for arr in (SHC_ALL, SHC_ROW_MAJOR, SHC_COL_MAJOR_NONZERO)
    c = grid2shc(H, p1 + 1, 2p1 + 2, p1; arrange = arr)
    println(rpad(string(arr), 22), round.(c; digits = 6))
end
println("reference: A00 = 1, A10 = 2/√3 = ", round(2 / sqrt(3); digits = 6),
        ", A11 = (-1 - 3i)/√6 = ", round(-1 / sqrt(6); digits = 6), " ", round(-3 / sqrt(6); digits = 6), "i")
```

All three layouts store the same coefficients in different orders. Every routine takes the layout as the keyword
`arrange` (default `SHC_ROW_MAJOR`), and a coefficient vector must be read back with the layout that produced it.

### Several fields at once

Fields are stacked one after another (field-major) on input and in the coefficients. Point evaluations come back
point-major:

```@example sph
X3 = cat(X, grid((t, ph) -> cos(t)^2); dims = 3)            # (Np, Nt, 2)
S3 = grid2shc(X3, Nt, Np, p)                                # [coeffs of field 1; coeffs of field 2]
V = reshape(shceval(S3, p, pts[:, 1:4]), 2, 4)              # V[k, i] = field k at point i
maximum(abs, V[2, :] .- cos.(pts[1, 1:4]) .^ 2)
```

### Vector fields: the V/W/X decomposition

[`grid2vecshc`](@ref) takes the Cartesian components ``(f_x, f_y, f_z)`` of a vector field on the grid, as an
`(Np, Nt, 3)` array. It returns three coefficient blocks `(V, W, X)` for the expansion

```math
\mathbf f = \sum_{n,m} V_{nm}\big(∇_s Y_n^m - (n+1) Y_n^m \hat{\mathbf r}\big)
          + W_{nm}\big(∇_s Y_n^m + n Y_n^m \hat{\mathbf r}\big)
          + X_{nm}\, \hat{\mathbf r} × ∇_s Y_n^m ,
```

so the radial part has coefficients ``nW - (n+1)V``, the surface-gradient part has ``V + W``, and the surface-curl
part has ``X``. Below, ``\mathbf f = ∇_s \cos θ + 2\,\hat{\mathbf r}×∇_s \cos θ + 3\cos θ\,\hat{\mathbf r}``. Here
``\cos θ`` has scalar coefficient ``A_{10} = 1/\sqrt3``, so we expect gradient ``1/\sqrt3``, curl ``2/\sqrt3`` and
radial ``3/\sqrt3``:

```@example sph
er(t, ph) = [sin(t)cos(ph), sin(t)sin(ph), cos(t)]
eθ(t, ph) = [cos(t)cos(ph), cos(t)sin(ph), -sin(t)]
eφ(t, ph) = [-sin(ph), cos(ph), 0.0]
vf(t, ph) = -sin(t) .* eθ(t, ph) .- 2sin(t) .* eφ(t, ph) .+ 3cos(t) .* er(t, ph)

G = zeros(Np, Nt, 3)                                        # (Np, Nt, 3): x, y, z component grids
for i in 1:Nt, j in 1:Np
    G[j, i, :] = vf(θ[i], φ[j])
end
SV = grid2vecshc(G, Nt, Np, p)                              # length 3 * shc_length(p)
nc = shc_length(p)
k = 2 * (1 * 2 ÷ 2 + 0) + 1                                 # ROW_MAJOR index of Re A(1,0)
Vc, Wc, Xc = SV[k], SV[nc + k], SV[2nc + k]
n = 1
(gradient = (Vc + Wc) * sqrt(3), curl = Xc * sqrt(3), radial = (n * Wc - (n + 1) * Vc) * sqrt(3))
```

The inverse transform and pointwise evaluation reproduce the field:

```@example sph
err_grid = maximum(abs, vecshc2grid(SV, p, Nt, Np) .- vec(G))
U = reshape(vecshceval(SV, p, pts[:, 1:5]), 3, 5)           # 3 × N (one vector field)
err_pts = maximum(abs, U .- reduce(hcat, [vf(pts[1, i], pts[2, i]) for i in 1:5]))
(err_grid, err_pts)
```

### Laplace layer potentials on the unit sphere

Density ``Y_n^m`` gives single-layer potential ``r^n Y_n^m/(2n+1)`` inside the sphere and ``r^{-n-1}Y_n^m/(2n+1)``
outside. Its double-layer potential is ``-\frac{n+1}{2n+1} r^n Y_n^m`` inside and ``\frac{n}{2n+1} r^{-n-1} Y_n^m``
outside. The `interior` keyword is required and selects the branch for all targets:

```@example sph
p = 4
σ1 = zeros(shc_length(p)); σ1[1] = 1                        # σ ≡ 1 (A00 = 1)
xin  = [0.1 -0.3  0.5; 0.2 0.4 0.0; -0.3 0.1 0.6]           # 3 × N inside
xout = [2.0  0.0 -1.0; 0.0 3.0 1.5; 0.5 -1.0 2.0]           # 3 × N outside
rout = vec(sqrt.(sum(abs2, xout; dims = 1)))
(
    sl_in  = maximum(abs, laplace_eval_sl(σ1, p, xin; interior = true) .- 1),        # = 1
    sl_out = maximum(abs, laplace_eval_sl(σ1, p, xout; interior = false) .- 1 ./ rout), # = 1/r
    dl_in  = maximum(abs, laplace_eval_dl(σ1, p, xin; interior = true) .+ 1),        # = -1 (Gauss)
    dl_out = maximum(abs, laplace_eval_dl(σ1, p, xout; interior = false)),           # = 0
)
```

For a general density, the single layer is continuous across the sphere, and the double layer jumps by the
density, ``u^+ - u^- = σ``:

```@example sph
σ(t, ph) = 1 + cos(t)^3 - sin(t)^2 * sin(2ph)
θp, φp = sph_grid_angles(p + 1, 2p + 2)
Sσ = grid2shc([σ(θp[i], φp[j]) for j in eachindex(φp), i in eachindex(θp)], p + 1, 2p + 2, p)
ys = [0.6 -0.48 0.0; 0.0 0.6 0.8; 0.8 0.64 -0.6]            # points on |y| = 1
σy = [σ(acos(ys[3, i]), atan(ys[2, i], ys[1, i])) for i in 1:3]
(
    sl_jump = maximum(abs, laplace_eval_sl(Sσ, p, ys; interior = false) .- laplace_eval_sl(Sσ, p, ys; interior = true)),
    dl_jump = maximum(abs, laplace_eval_dl(Sσ, p, ys; interior = false) .- laplace_eval_dl(Sσ, p, ys; interior = true) .- σy),
)
```

### Stokes layer potentials on the unit sphere

The Stokes routines take the **vector** spherical harmonic coefficients of the density, as produced by
[`grid2vecshc`](@ref), and return velocities or tractions as `3 × N`. A uniform density ``\mathbf f = \mathbf e_z``
is a translating rigid sphere. Its single layer equals ``\tfrac23 \mathbf e_z`` inside. Outside it is a Stokeslet of
total force ``\mathbf F = 4π\mathbf e_z`` plus a source dipole:

```math
\mathbf u(\mathbf x) = \frac{1}{8π}\Big[\frac{I}{r} + \frac{\mathbf x\mathbf x^T}{r^3}
                       + \frac13\Big(\frac{I}{r^3} - \frac{3\mathbf x\mathbf x^T}{r^5}\Big)\Big]\mathbf F .
```

The Stokes double layer of a constant density is ``-\mathbf f`` inside and ``0`` outside:

```@example sph
Gz = zeros(Np, Nt, 3); Gz[:, :, 3] .= 1                     # f = e_z on the p = 8 grid
p = 8
Sz = grid2vecshc(Gz, Nt, Np, p)
uin = reshape(stokes_eval_sl(Sz, p, xin; interior = true), 3, :)
Fz = [0, 0, 4π]
uex(x) = (r = norm(x); ((I / r + x * x' / r^3) + (I / r^3 - 3x * x' / r^5) / 3) * Fz / 8π)
uout = reshape(stokes_eval_sl(Sz, p, xout; interior = false), 3, :)
(
    sl_in  = maximum(abs, uin .- [0, 0, 2 / 3]),
    sl_out = maximum(abs, uout .- reduce(hcat, uex.(eachcol(xout)))),
    dl_in  = maximum(abs, reshape(stokes_eval_dl(Sz, p, xin; interior = true), 3, :) .- [0, 0, -1]),
    dl_out = maximum(abs, stokes_eval_dl(Sz, p, xout; interior = false)),
)
```

[`stokes_eval_kl`](@ref) returns the traction ``(-PI + ∇\mathbf u + ∇\mathbf u^T)\boldsymbol ν`` at off-surface targets
with given normals. [`stokes_eval_kself`](@ref) returns its one-sided limits on the sphere with ``\boldsymbol ν = \hat{\mathbf r}``.
The traction jumps by the density, exterior minus interior equals ``-\mathbf f``. For the rigid translation, the
interior flow is a uniform translation with zero stress, so the interior traction is ``0`` and the exterior one is
the uniform ``-\mathbf e_z``. The total hydrodynamic force on the sphere is ``-4π\mathbf e_z``, balancing the
applied ``\mathbf F``:

```@example sph
kin  = reshape(stokes_eval_kself(Sz, p, ys; interior = true), 3, :)
kout = reshape(stokes_eval_kself(Sz, p, ys; interior = false), 3, :)
klim = reshape(stokes_eval_kl(Sz, p, ys .* (1 + 1e-9), ys; interior = false), 3, :)  # just outside, ν = r̂
(interior = maximum(abs, kin), exterior = maximum(abs, kout .- [0, 0, -1]),
 jump = maximum(abs, kout .- kin .- [0, 0, -1]), kl_vs_kself = maximum(abs, klim .- kout))
```

## Data layouts

### Grids

| Quantity | Julia shape | Memory order (SCTL C++) |
|:--|:--|:--|
| angles | `θ::Vector` (`Nt`), `φ::Vector` (`Np`) | `θ = acos.(sph_legendre_nodes(Nt-1))`, increasing; `φ_j = 2π(j-1)/Np` |
| scalar grid, `nf` fields | `(Np, Nt, nf)` | `X[(k*Nt + i)*Np + j]` (0-based): `φ` fastest, then `θ`, then field |
| vector grid, `nv` fields | `(Np, Nt, 3, nv)` | per vector field the `x`, `y`, `z` component grids one after the other |
| `shc2grid` derivatives | same as the grid | `X_θ`, `X_φ` are ``∂f/∂θ``, ``∂f/∂φ`` (not the covariant or unit-vector components) |

Inputs may be arrays of any shape with this memory order; they are flattened. Grid outputs are flat `Vector`s;
reshape them with `reshape(X, Np, Nt, nf)` or `reshape(X, Np, Nt, 3, nv)`.

### Coefficients ([`SHCArrange`](@ref))

Per scalar field of order `p` (0-based offsets; `Ar`, `Ai` are the real and imaginary parts of ``A_{nm}``):

| Layout | Length [`shc_length`](@ref) | Position of `Ar(n,m)` | Position of `Ai(n,m)` | Notes |
|:--|:--|:--|:--|:--|
| `SHC_ALL` | `2(p+1)^2` | `2(n(p+1) + m)` | `2(n(p+1) + m) + 1` | full `(p+1)×(p+1)` complex matrix, row-major; `m > n` entries are zero |
| `SHC_ROW_MAJOR` (default) | `(p+1)(p+2)` | `2(n(n+1)/2 + m)` | `2(n(n+1)/2 + m) + 1` | lower triangle, row-major, interleaved; `Ai(n,0) = 0` is stored |
| `SHC_COL_MAJOR_NONZERO` | `(p+1)^2` | — | — | `Ar(0:p,0), Ar(1:p,1), Ai(1:p,1), Ar(2:p,2), Ai(2:p,2), …, Ar(p,p), Ai(p,p)` |

- Several scalar fields: blocks of `shc_length(p, arrange)` one after another.
- Vector fields: per vector field the three blocks `V`, `W`, `X`, each of `shc_length(p, arrange)`, so
  `3 * nv * shc_length(p, arrange)` in total.

### Point inputs and outputs

| Routine | Points | Output (flat) | Reshape to |
|:--|:--|:--|:--|
| [`shceval`](@ref) | `2 × N` (`θ; φ`) or `[θ1, φ1, …]` | point-major | `(nf, N)` |
| [`vecshceval`](@ref) | `2 × N` | Cartesian, point-major | `(3, nv, N)` |
| [`shc2pole`](@ref) | — | `[north_1, south_1, north_2, …]` | `(2, nf)` |
| [`laplace_eval_sl`](@ref), [`laplace_eval_dl`](@ref) | `3 × N` (`x; y; z`) or flat | point-major | `(nf, N)` |
| [`stokes_eval_sl`](@ref), `_dl`, `_kl`, `_kself` | `3 × N` (normals for `_kl`: `3 × N`) | Cartesian, point-major | `(3, nv, N)` |

Grid outputs are field-major and point evaluations are point-major. This follows SCTL's C++ conventions:
`Vector<Real>` arrays `{X(t0,p0), X(t0,p1), …}` for grids and `{X0, Y0, Z0, X1, …}` for point values.

## Pitfalls and limitations

- **Grid resolution.** Use `Nt ≥ p + 1` and `Np ≥ 2p + 1`. Coarser grids alias, and Fourier modes above `Np ÷ 2` are
  silently dropped. A grid finer than needed just projects onto degree `≤ p`, which acts as a low-pass filter.
- **Node order.** [`sph_legendre_nodes`](@ref)`(p1)` returns `p1 + 1` nodes (note the `+1`) in *decreasing* order,
  so ``θ`` increases with the latitude index. The rule is on ``[-1, 1]`` with weights summing to 2. This differs from
  [`legquad`](@ref), whose nodes lie on ``[0, 1]``.
- **`interior` selects one expansion for all targets.** The layer-potential routines do not test ``|x| < 1``. The
  interior branch ``\propto r^n`` and the exterior branch ``\propto r^{-n-1}`` are both evaluated without complaint
  at any radius, and the wrong branch gives wrong values. Split targets by radius yourself. On the sphere, the two
  branches give the one-sided limits.
- **The origin.** Evaluating any layer potential at exactly ``x = 0`` returns `NaN`, because SCTL forms ``r^n`` as
  ``r^{n+1}/r``. Use a tiny offset, or the known value: for the single layer at the center, only the ``n = 0``
  term survives.
- **Stokes densities are vector-SH coefficients.** Pass the output of [`grid2vecshc`](@ref), not scalar
  coefficients of the Cartesian components. Unit viscosity is assumed.
- **Order limits.** SCTL's tables have `SCTL_SHMAXDEG = 1024` entries, so scalar routines need `p ≤ 1023`, vector
  routines and the Laplace evaluators `p ≤ 1022`, and Stokes `p ≤ 1021`. Grids need `Nt ≤ 1024` and `Np ≤ 1023`. The
  wrapper checks these limits, because SCTL's own `assert`s are compiled out.
- **Memory.** The first call to any routine in this module allocates about 200 MB of mostly empty lookup tables,
  sized for all orders up to 1024. Precomputed Legendre and Fourier matrices and FFT plans are cached on top of
  that, per `(p, Nt)` or `Np`. [`sph_clear_cache`](@ref) (`SphericalHarmonics::Clear()`) frees everything, but the
  tables come back on the next call.
- **Threading.** Every routine runs OpenMP-parallel internally (see [`SCTL.set_num_threads`](@ref)). The lazily
  built caches are filled inside named `omp critical` sections, so calling these routines concurrently from several
  Julia threads is safe. [`sph_clear_cache`](@ref) is **not** safe: it destroys tables that other threads may be
  reading. Call it only when no other spherical-harmonic call is in flight.
- `WriteVTK` from SCTL is not wrapped.

## API reference

```@docs
SHCArrange
shc_length
sph_grid_angles
sph_legendre_nodes
sph_legendre_weights
sph_clear_cache
grid2shc
shc2grid
shceval
shc2pole
grid2vecshc
vecshc2grid
vecshceval
laplace_eval_sl
laplace_eval_dl
stokes_eval_sl
stokes_eval_dl
stokes_eval_kl
stokes_eval_kself
```
