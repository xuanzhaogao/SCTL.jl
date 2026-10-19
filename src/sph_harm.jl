# Spherical harmonics on the unit sphere (sctl::SphericalHarmonics<double>).

export SHCArrange, SHC_ALL, SHC_ROW_MAJOR, SHC_COL_MAJOR_NONZERO, shc_length
export sph_legendre_nodes, sph_legendre_weights, sph_grid_angles, sph_clear_cache
export grid2shc, shc2grid, shceval, shc2pole
export grid2vecshc, vecshc2grid, vecshceval
export laplace_eval_sl, laplace_eval_dl
export stokes_eval_sl, stokes_eval_dl, stokes_eval_kl, stokes_eval_kself

"""
    SHCArrange

Memory layout of the spherical harmonic coefficients of one scalar field of order `p` (degrees `n = 0..p`,
orders `m = 0..n`; only `m ≥ 0` is stored since the fields are real). `A(n,m) = Ar(n,m) + i Ai(n,m)` and the
field is

```math
f(θ,φ) = \\sum_{n=0}^{p} \\sum_{m=0}^{n} c_m \\sqrt{4π}\\, \\bar P_n^m(\\cos θ)\\, \\mathrm{Re}\\big(A(n,m)\\, e^{imφ}\\big),
\\qquad c_0 = 1,\\ c_{m>0} = 2,
```

where ``\\bar P_n^m(x) = \\sqrt{\\tfrac{2n+1}{4π}\\tfrac{(n-m)!}{(n+m)!}}\\, P_n^m(x)`` (Condon-Shortley phase included,
so ``\\bar P_n^m(\\cos θ) e^{imφ} = Y_n^m`` is orthonormal on the sphere). Equivalently
``f = \\sqrt{4π} \\sum_{n} \\sum_{m=-n}^{n} A(n,m) Y_n^m`` with ``A(n,-m) = (-1)^m \\overline{A(n,m)}``;
in particular `f ≡ 1` has `A(0,0) = 1`, and ``\\cos θ`` has ``A(1,0) = 1/\\sqrt{3}``.

Values (0-based offsets into the coefficient block of one field; see [`shc_length`](@ref)):

- `SHC_ALL`: `(p+1)×(p+1)` complex numbers, row-major in `(n, m)`, as interleaved (re, im) pairs:
  `Ar(n,m)` at `2*(n*(p+1) + m)`, `Ai(n,m)` at `2*(n*(p+1) + m) + 1`; entries with `m > n` are zero.
  Length `2(p+1)^2`.
- `SHC_ROW_MAJOR` (default): the lower triangle `(p+1)(p+2)/2` complex numbers, row-major, interleaved:
  `Ar(n,m)` at `2*(n*(n+1)/2 + m)`, `Ai(n,m)` right after it (`Ai(n,0) = 0` is stored). Length `(p+1)(p+2)`.
- `SHC_COL_MAJOR_NONZERO`: `(p+1)^2` real numbers, column-major (by `m`) without the zero imaginary parts:
  `Ar(0:p,0), Ar(1:p,1), Ai(1:p,1), Ar(2:p,2), Ai(2:p,2), …, Ar(p,p), Ai(p,p)`. Length `(p+1)^2`.

Several fields are stored one after the other (field-major), so a coefficient vector has length
`nfields * shc_length(p, arrange)` (`3 * nfields * shc_length(p, arrange)` for vector fields).
"""
@enum SHCArrange::Cint SHC_ALL = 0 SHC_ROW_MAJOR = 1 SHC_COL_MAJOR_NONZERO = 2

"""
    shc_length(p, arrange=SHC_ROW_MAJOR) -> Int

Number of real coefficients of one scalar field of order `p` in the layout `arrange` (see [`SHCArrange`](@ref)):
`2(p+1)^2` (ALL), `(p+1)(p+2)` (ROW_MAJOR) or `(p+1)^2` (COL_MAJOR_NONZERO). A vector field has three times as many.
"""
function shc_length(p::Integer, arrange::SHCArrange=SHC_ROW_MAJOR)
    p >= 0 || throw(ArgumentError("order p must be non-negative"))
    arrange == SHC_ALL && return 2 * (p + 1)^2
    arrange == SHC_ROW_MAJOR && return (p + 1) * (p + 2)
    return (p + 1)^2
end

const _SH_MAXDEG = 1024  # SCTL_SHMAXDEG: size of SCTL's precomputation tables

function _check_order(p, extra=0)
    0 <= p || throw(ArgumentError("order p must be non-negative"))
    p + extra < _SH_MAXDEG || throw(ArgumentError("order p must be < $(_SH_MAXDEG - extra) (SCTL_SHMAXDEG = $_SH_MAXDEG)"))
end
function _check_grid(Nt, Np)
    1 <= Nt <= _SH_MAXDEG || throw(ArgumentError("Nt must be in 1:$_SH_MAXDEG"))
    1 <= Np < _SH_MAXDEG || throw(ArgumentError("Np must be in 1:$(_SH_MAXDEG - 1)"))
end
# number of blocks of length `block` in an array of length `n` (at least one)
function _nblocks(n, block, what)
    (n > 0 && n % block == 0) ||
        throw(DimensionMismatch("length of $what ($n) must be a positive multiple of $block"))
    return n ÷ block
end
# flat contiguous Float64 copy (or the array itself) of an input array
_flat(x::AbstractArray) = vec(_f64(x))
function _points(x::AbstractArray, dim, what)
    (x isa AbstractMatrix && size(x, 1) != dim) &&
        throw(DimensionMismatch("$what must be a $dim×N matrix or a flat vector"))
    v = _flat(x)
    _nblocks(length(v), dim, what)
    return v
end

"""
    sph_legendre_nodes(p1) -> Vector{Float64}

The `p1 + 1` Gauss-Legendre nodes on `[-1, 1]` used as ``\\cos θ`` for the spherical grids, in *decreasing* order
(so ``θ_i = \\arccos x_i`` increases from near 0 to near π). A grid with `Nt` latitudes uses
`sph_legendre_nodes(Nt - 1)`.
"""
function sph_legendre_nodes(p1::Integer)
    _check_order(p1)
    x = Vector{Float64}(undef, p1 + 1)
    _check(ccall((:sctljl_sph_legendre_nodes, libsctljl), Cint, (Ptr{Float64}, Int64), x, p1))
    return x
end

"""
    sph_legendre_weights(p1) -> Vector{Float64}

Gauss-Legendre weights on `[-1, 1]` (they sum to 2) matching [`sph_legendre_nodes`](@ref)`(p1)`; exact for
polynomials of degree `≤ 2p1 + 1`.
"""
function sph_legendre_weights(p1::Integer)
    _check_order(p1)
    w = Vector{Float64}(undef, p1 + 1)
    _check(ccall((:sctljl_sph_legendre_weights, libsctljl), Cint, (Ptr{Float64}, Int64), w, p1))
    return w
end

"""
    sph_grid_angles(Nt, Np) -> (θ, φ)

Angles of the `Nt × Np` spherical grid used by [`grid2shc`](@ref) and friends: `θ = acos.(sph_legendre_nodes(Nt-1))`
(increasing, Gauss-Legendre in ``\\cos θ``) and `φ = 2π .* (0:Np-1) ./ Np` (equispaced).

Grid values of a field `f` are stored with `φ` fastest: `X[j + Np*(i-1)] = f(θ[i], φ[j])`, i.e. the Julia array
`X = [f(θ[i], φ[j]) for j in 1:Np, i in 1:Nt]` of size `(Np, Nt)`.
"""
function sph_grid_angles(Nt::Integer, Np::Integer)
    _check_grid(Nt, Np)
    θ = acos.(clamp.(sph_legendre_nodes(Nt - 1), -1.0, 1.0))
    φ = [2π * j / Np for j in 0:Np-1]
    return θ, φ
end

"""
    sph_clear_cache()

Free SCTL's precomputed spherical harmonic matrices, FFT plans and quadrature rules (`SphericalHarmonics::Clear`).
They are rebuilt on demand. Note that the first use of any spherical harmonic routine allocates about 200 MB of
(mostly empty) lookup tables, sized for orders up to `SCTL_SHMAXDEG = 1024`; this call releases them too, but they
are allocated again on the next use.

Not thread-safe: do not call while another thread uses the spherical harmonic routines.
"""
sph_clear_cache() = _check(ccall((:sctljl_sph_clear, libsctljl), Cint, ()))

# ---------------------------------------------------------------------------------------------------------------
# scalar fields

"""
    grid2shc(X, Nt, Np, p; arrange=SHC_ROW_MAJOR) -> S::Vector{Float64}

Spherical harmonic coefficients of order `p` (layout `arrange`, see [`SHCArrange`](@ref)) of fields given by their
values on the `Nt × Np` grid of [`sph_grid_angles`](@ref) (Gauss-Legendre in ``\\cos θ`` × equispaced in `φ`).

`X` holds `nfields = length(X) ÷ (Nt*Np)` fields, one after the other; each field is stored with `φ` fastest, so
`X` may be an `(Np, Nt)` or `(Np, Nt, nfields)` array (or any array with that memory order; it is flattened). The
result has length `nfields * shc_length(p, arrange)`, field-major. The transform is exact (up to rounding) for
band-limited fields of degree `≤ p` when `Nt ≥ p + 1` and `Np ≥ 2p + 1`; the Fourier modes `m > Np ÷ 2` are dropped.
"""
function grid2shc(X::AbstractArray{<:Real}, Nt::Integer, Np::Integer, p::Integer; arrange::SHCArrange=SHC_ROW_MAJOR)
    _check_order(p)
    _check_grid(Nt, Np)
    x = _flat(X)
    nf = _nblocks(length(x), Nt * Np, "grid values X (Nt*Np per field)")
    S = Vector{Float64}(undef, nf * shc_length(p, arrange))
    @serial_blas _check(ccall((:sctljl_sph_grid2shc, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Int64, Int64, Int64, Cint),
                 S, length(S), x, length(x), Nt, Np, p, Cint(arrange)))
    return S
end

"""
    shc2grid(S, p, Nt, Np; arrange=SHC_ROW_MAJOR, derivatives=false) -> X
    shc2grid(S, p, Nt, Np; arrange=SHC_ROW_MAJOR, derivatives=true) -> (X, X_θ, X_φ)

Evaluate spherical harmonic expansions of order `p` (`nfields = length(S) ÷ shc_length(p, arrange)` fields) on the
`Nt × Np` grid of [`sph_grid_angles`](@ref). Returns a flat vector of length `nfields*Nt*Np` in the grid layout of
[`grid2shc`](@ref) (`reshape(X, Np, Nt, nfields)`). With `derivatives=true` also returns ``∂f/∂θ`` and ``∂f/∂φ``
on the grid (same layout).
"""
function shc2grid(S::AbstractArray{<:Real}, p::Integer, Nt::Integer, Np::Integer;
                  arrange::SHCArrange=SHC_ROW_MAJOR, derivatives::Bool=false)
    _check_order(p)
    _check_grid(Nt, Np)
    s = _flat(S)
    nf = _nblocks(length(s), shc_length(p, arrange), "coefficients S")
    n = nf * Nt * Np
    X = Vector{Float64}(undef, n)
    Xt = derivatives ? Vector{Float64}(undef, n) : nothing
    Xp = derivatives ? Vector{Float64}(undef, n) : nothing
    @serial_blas _check(ccall((:sctljl_sph_shc2grid, libsctljl), Cint,
                 (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64, Int64, Int64),
                 X, something(Xt, C_NULL), something(Xp, C_NULL), n, s, length(s), Cint(arrange), p, Nt, Np))
    return derivatives ? (X, Xt, Xp) : X
end

"""
    shceval(S, p, theta_phi; arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Evaluate spherical harmonic expansions of order `p` at arbitrary points `theta_phi` (a `2×N` matrix with rows
`θ ∈ [0, π]` and `φ`, or a flat vector `[θ1, φ1, θ2, φ2, …]`). With `nfields` fields in `S` the result has length
`nfields*N` and is point-major: `reshape(result, nfields, N)[k, i]` is field `k` at point `i`.
"""
function shceval(S::AbstractArray{<:Real}, p::Integer, theta_phi::AbstractArray{<:Real}; arrange::SHCArrange=SHC_ROW_MAJOR)
    _check_order(p)
    s = _flat(S)
    nf = _nblocks(length(s), shc_length(p, arrange), "coefficients S")
    tp = _points(theta_phi, 2, "theta_phi")
    X = Vector{Float64}(undef, nf * (length(tp) ÷ 2))
    @serial_blas _check(ccall((:sctljl_sph_shceval, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64, Ptr{Float64}, Int64),
                 X, length(X), s, length(s), Cint(arrange), p, tp, length(tp)))
    return X
end

"""
    shc2pole(S, p; arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Values of spherical harmonic expansions at the north (`θ = 0`) and south (`θ = π`) poles, interleaved per field:
`[north_1, south_1, north_2, south_2, …]` (`reshape(result, 2, nfields)`).
"""
function shc2pole(S::AbstractArray{<:Real}, p::Integer; arrange::SHCArrange=SHC_ROW_MAJOR)
    _check_order(p)
    s = _flat(S)
    nf = _nblocks(length(s), shc_length(p, arrange), "coefficients S")
    P = Vector{Float64}(undef, 2nf)
    @serial_blas _check(ccall((:sctljl_sph_shc2pole, libsctljl), Cint, (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64),
                 P, length(P), s, length(s), Cint(arrange), p))
    return P
end

# ---------------------------------------------------------------------------------------------------------------
# vector fields

"""
    grid2vecshc(X, Nt, Np, p; arrange=SHC_ROW_MAJOR) -> S::Vector{Float64}

Vector spherical harmonic coefficients of vector fields (not necessarily tangent) given by their Cartesian components on the
`Nt × Np` grid of [`sph_grid_angles`](@ref).

`X` holds `nvec = length(X) ÷ (3*Nt*Np)` vector fields; each one is stored as its `x`, `y`, `z` component grids
one after the other (each in the scalar grid layout, `φ` fastest), i.e. `X` may be an `(Np, Nt, 3)` or
`(Np, Nt, 3, nvec)` array. The result has length `3 * nvec * shc_length(p, arrange)`: per vector field the three
coefficient blocks `(V, W, X)` of the expansion

```math
\\mathbf f = \\sum_{n,m} V_{nm} \\mathbf V_n^m + W_{nm} \\mathbf W_n^m + X_{nm} \\mathbf X_n^m,\\quad
\\mathbf V_n^m = ∇_s Y_n^m - (n+1) Y_n^m \\hat{\\mathbf r},\\
\\mathbf W_n^m = ∇_s Y_n^m + n Y_n^m \\hat{\\mathbf r},\\
\\mathbf X_n^m = \\hat{\\mathbf r} × ∇_s Y_n^m,
```

(with the scalar normalization of [`SHCArrange`](@ref) and ``∇_s`` the surface gradient on the unit sphere), each
block in the layout `arrange`.
"""
function grid2vecshc(X::AbstractArray{<:Real}, Nt::Integer, Np::Integer, p::Integer; arrange::SHCArrange=SHC_ROW_MAJOR)
    _check_order(p, 1)
    _check_grid(Nt, Np)
    x = _flat(X)
    nv = _nblocks(length(x), 3 * Nt * Np, "grid values X (3*Nt*Np per vector field)")
    S = Vector{Float64}(undef, 3 * nv * shc_length(p, arrange))
    @serial_blas _check(ccall((:sctljl_sph_grid2vecshc, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Int64, Int64, Int64, Cint),
                 S, length(S), x, length(x), Nt, Np, p, Cint(arrange)))
    return S
end

"""
    vecshc2grid(S, p, Nt, Np; arrange=SHC_ROW_MAJOR) -> X::Vector{Float64}

Evaluate vector spherical harmonic expansions (see [`grid2vecshc`](@ref); `nvec = length(S) ÷ (3*shc_length(p, arrange))`)
on the `Nt × Np` grid. Returns the Cartesian components in the layout of [`grid2vecshc`](@ref):
`reshape(X, Np, Nt, 3, nvec)`.
"""
function vecshc2grid(S::AbstractArray{<:Real}, p::Integer, Nt::Integer, Np::Integer; arrange::SHCArrange=SHC_ROW_MAJOR)
    _check_order(p, 1)
    _check_grid(Nt, Np)
    s = _flat(S)
    nv = _nblocks(length(s), 3 * shc_length(p, arrange), "vector coefficients S")
    X = Vector{Float64}(undef, 3 * nv * Nt * Np)
    @serial_blas _check(ccall((:sctljl_sph_vecshc2grid, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64, Int64, Int64),
                 X, length(X), s, length(s), Cint(arrange), p, Nt, Np))
    return X
end

"""
    vecshceval(S, p, theta_phi; arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Evaluate vector spherical harmonic expansions at arbitrary points `theta_phi` (`2×N` matrix or flat
`[θ1, φ1, …]`). Returns the Cartesian components point-major: `reshape(result, 3, nvec, N)`.
"""
function vecshceval(S::AbstractArray{<:Real}, p::Integer, theta_phi::AbstractArray{<:Real}; arrange::SHCArrange=SHC_ROW_MAJOR)
    _check_order(p, 1)
    s = _flat(S)
    nv = _nblocks(length(s), 3 * shc_length(p, arrange), "vector coefficients S")
    tp = _points(theta_phi, 2, "theta_phi")
    X = Vector{Float64}(undef, 3 * nv * (length(tp) ÷ 2))
    @serial_blas _check(ccall((:sctljl_sph_vecshceval, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64, Ptr{Float64}, Int64),
                 X, length(X), s, length(s), Cint(arrange), p, tp, length(tp)))
    return X
end

# ---------------------------------------------------------------------------------------------------------------
# layer potentials on the unit sphere

function _laplace_eval(S, p, coord, interior, arrange, kind)
    _check_order(p, 1)
    s = _flat(S)
    nf = _nblocks(length(s), shc_length(p, arrange), "coefficients S")
    x = _points(coord, 3, "coord")
    U = Vector{Float64}(undef, nf * (length(x) ÷ 3))
    @serial_blas _check(ccall((:sctljl_sph_laplace_eval, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64, Ptr{Float64}, Int64, Cint, Cint),
                 U, length(U), s, length(s), Cint(arrange), p, x, length(x), interior, kind))
    return U
end

"""
    laplace_eval_sl(S, p, coord; interior, arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Laplace single-layer potential ``u(x) = \\int_{S^2} σ(y) / (4π|x - y|)\\, dS_y`` of densities `σ` on the unit sphere
given by their spherical harmonic coefficients `S` (order `p`), at the points `coord` (`3×N` matrix or flat
`[x1, y1, z1, …]`). Density ``Y_n^m`` gives ``r^n Y_n^m/(2n+1)`` inside and ``r^{-n-1} Y_n^m/(2n+1)`` outside, so
`σ ≡ 1` gives `u = 1` for `|x| < 1` and `u = 1/|x|` for `|x| > 1`.

`interior` (required) selects the interior (regular at the origin, for `|x| ≤ 1`) or the exterior (decaying, for
`|x| ≥ 1`) expansion for *all* points. Result is point-major: `reshape(U, nfields, N)`.
The point `x = 0` (exactly the origin) gives `NaN` in all the layer-potential evaluators (SCTL computes
``r^{n}`` as ``r^{n+1}/r``).
"""
laplace_eval_sl(S::AbstractArray{<:Real}, p::Integer, coord::AbstractArray{<:Real}; interior::Bool,
                arrange::SHCArrange=SHC_ROW_MAJOR) = _laplace_eval(S, p, coord, interior, arrange, 0)

"""
    laplace_eval_dl(S, p, coord; interior, arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Laplace double-layer potential ``u(x) = \\int_{S^2} σ(y)\\, \\partial_{n_y} \\frac{1}{4π|x - y|}\\, dS_y`` (outward
normal ``n_y = y``) of densities on the unit sphere. Density ``Y_n^m`` gives ``-\\frac{n+1}{2n+1} r^n Y_n^m``
inside and ``\\frac{n}{2n+1} r^{-n-1} Y_n^m`` outside (so `σ ≡ 1` gives `-1` inside and `0` outside). Arguments and
layout as in [`laplace_eval_sl`](@ref).
"""
laplace_eval_dl(S::AbstractArray{<:Real}, p::Integer, coord::AbstractArray{<:Real}; interior::Bool,
                arrange::SHCArrange=SHC_ROW_MAJOR) = _laplace_eval(S, p, coord, interior, arrange, 1)

function _stokes_eval(S, p, coord, normals, interior, arrange, kind)
    _check_order(p, 2)
    s = _flat(S)
    nv = _nblocks(length(s), 3 * shc_length(p, arrange), "vector coefficients S")
    x = _points(coord, 3, "coord")
    n = normals === nothing ? nothing : _points(normals, 3, "normals")
    n === nothing || length(n) == length(x) || throw(DimensionMismatch("need one normal per target point"))
    U = Vector{Float64}(undef, 3 * nv * (length(x) ÷ 3))
    @serial_blas _check(ccall((:sctljl_sph_stokes_eval, libsctljl), Cint,
                 (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64, Ptr{Float64}, Int64, Ptr{Float64}, Cint, Cint),
                 U, length(U), s, length(s), Cint(arrange), p, x, length(x), something(n, C_NULL), interior, kind))
    return U
end

"""
    stokes_eval_sl(S, p, coord; interior, arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Stokes single-layer velocity ``u(x) = \\int_{S^2} G(x - y) f(y)\\, dS_y``,
``G(r) = \\frac{1}{8π}\\big(\\frac{I}{|r|} + \\frac{r r^T}{|r|^3}\\big)`` (unit viscosity), of densities `f` on the unit
sphere given by their *vector* spherical harmonic coefficients `S` (see [`grid2vecshc`](@ref)), at the points
`coord` (`3×N` matrix or flat). `interior` selects the interior/exterior expansion for all points (as in
[`laplace_eval_sl`](@ref)). Result is point-major: `reshape(U, 3, nvec, N)`.
"""
stokes_eval_sl(S::AbstractArray{<:Real}, p::Integer, coord::AbstractArray{<:Real}; interior::Bool,
               arrange::SHCArrange=SHC_ROW_MAJOR) = _stokes_eval(S, p, coord, nothing, interior, arrange, 0)

"""
    stokes_eval_dl(S, p, coord; interior, arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Stokes double-layer velocity ``u(x) = \\int_{S^2} \\frac{3}{4π} \\frac{r\\,(r⋅n_y)(r⋅f(y))}{|r|^5}\\, dS_y`` with
``r = x - y`` and outward normal ``n_y = y``. Arguments and layout as in [`stokes_eval_sl`](@ref).
"""
stokes_eval_dl(S::AbstractArray{<:Real}, p::Integer, coord::AbstractArray{<:Real}; interior::Bool,
               arrange::SHCArrange=SHC_ROW_MAJOR) = _stokes_eval(S, p, coord, nothing, interior, arrange, 1)

"""
    stokes_eval_kl(S, p, coord, normals; interior, arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Traction ``σ(u)\\,ν = (-P I + ∇u + ∇u^T)\\,ν`` of the Stokes single-layer flow of [`stokes_eval_sl`](@ref) at the points
`coord`, contracted with the target normals `normals` (`3×N` or flat, same length as `coord`; need not be unit).
Result is point-major: `reshape(U, 3, nvec, N)`.
"""
function stokes_eval_kl(S::AbstractArray{<:Real}, p::Integer, coord::AbstractArray{<:Real}, normals::AbstractArray{<:Real};
                        interior::Bool, arrange::SHCArrange=SHC_ROW_MAJOR)
    return _stokes_eval(S, p, coord, normals, interior, arrange, 2)
end

"""
    stokes_eval_kself(S, p, coord; interior, arrange=SHC_ROW_MAJOR) -> Vector{Float64}

Traction of the Stokes single-layer flow on the unit sphere itself, with the outward normal `coord/|coord|`:
the interior (`interior=true`) or exterior limit of [`stokes_eval_kl`](@ref) at points `coord` on the sphere.
The two limits jump by the density: `exterior - interior = -f`. Layout as in [`stokes_eval_sl`](@ref).
"""
stokes_eval_kself(S::AbstractArray{<:Real}, p::Integer, coord::AbstractArray{<:Real}; interior::Bool,
                  arrange::SHCArrange=SHC_ROW_MAJOR) = _stokes_eval(S, p, coord, nothing, interior, arrange, 3)
