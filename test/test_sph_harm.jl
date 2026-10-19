using SCTL, Test, LinearAlgebra

# deterministic pseudo-random values in [-1, 1] (no Random dependency)
_prand(dims...) = reshape(sin.(1.3 .* (1:prod(dims)) .^ 1.5 .+ 0.7), dims...)

# helpers --------------------------------------------------------------------------------------------------------
const _ARRS = (SHC_ALL, SHC_ROW_MAJOR, SHC_COL_MAJOR_NONZERO)

# grid values of f(θ, φ) in SCTL's layout: (Np, Nt), φ fastest
_grid(f, θ, φ) = [f(θ[i], φ[j]) for j in eachindex(φ), i in eachindex(θ)]
# Cartesian vector field F(θ, φ) -> (Np, Nt, 3)
function _vgrid(F, θ, φ)
    A = zeros(length(φ), length(θ), 3)
    for i in eachindex(θ), j in eachindex(φ)
        A[j, i, :] = F(θ[i], φ[j])
    end
    return A
end
_er(t, f) = [sin(t) * cos(f), sin(t) * sin(f), cos(t)]
_eθ(t, f) = [cos(t) * cos(f), cos(t) * sin(f), -sin(t)]
_eφ(t, f) = [-sin(f), cos(f), 0.0]

# 1-based index of Re/Im A(n,m) in one field's coefficient block, as documented in SHCArrange
function _shc_index(n, m, part, p, arr)  # part = 0 (real) or 1 (imag)
    arr == SHC_ALL && return 2 * (n * (p + 1) + m) + part + 1
    arr == SHC_ROW_MAJOR && return 2 * (n * (n + 1) ÷ 2 + m) + part + 1
    off = 0
    for mm in 0:m-1
        off += (p + 1 - mm) * (mm == 0 ? 1 : 2)
    end
    return off + (part == 1 ? (p + 1 - m) : 0) + (n - m) + 1
end

# band-limited test function of degree 3 and its derivatives
_f(t, ph) = 0.7 + cos(t) - 0.4 * sin(t) * cos(ph) + sin(t)^2 * cos(2ph) + 0.3 * sin(t) * cos(t) * sin(ph) + 0.2 * sin(t)^3 * sin(3ph)
_ft(t, ph) = -sin(t) - 0.4 * cos(t) * cos(ph) + 2sin(t) * cos(t) * cos(2ph) + 0.3 * cos(2t) * sin(ph) + 0.6 * sin(t)^2 * cos(t) * sin(3ph)
_fp(t, ph) = 0.4 * sin(t) * sin(ph) - 2sin(t)^2 * sin(2ph) + 0.3 * sin(t) * cos(t) * cos(ph) + 0.6 * sin(t)^3 * cos(3ph)

@testset "layout and Legendre rule" begin
    @test Int(SHC_ALL) == 0 && Int(SHC_ROW_MAJOR) == 1 && Int(SHC_COL_MAJOR_NONZERO) == 2
    @test shc_length(5) == shc_length(5, SHC_ROW_MAJOR) == 42
    @test shc_length(5, SHC_ALL) == 72
    @test shc_length(5, SHC_COL_MAJOR_NONZERO) == 36

    for p1 in (0, 4, 11)
        x, w = sph_legendre_nodes(p1), sph_legendre_weights(p1)
        @test length(x) == length(w) == p1 + 1
        @test all(-1 .< x .< 1)
        @test issorted(x; rev=true)
        for k in 0:2p1+1  # exact on [-1, 1] up to degree 2p1+1
            @test dot(w, x .^ k) ≈ (iseven(k) ? 2 / (k + 1) : 0.0) atol = 1e-14
        end
        xq, wq = legquad(p1 + 1)  # same rule mapped from [0, 1]
        @test x ≈ 1 .- 2 .* xq
        @test w ≈ 2 .* wq
    end
    θ, φ = sph_grid_angles(5, 8)
    @test θ ≈ acos.(sph_legendre_nodes(4))
    @test φ ≈ 2π .* (0:7) ./ 8
end

@testset "scalar transforms" begin
    p = 6
    Nt, Np = p + 1, 2p + 2
    θ, φ = sph_grid_angles(Nt, Np)
    X = _grid(_f, θ, φ)

    for arr in _ARRS
        # known coefficients: 1 -> A(0,0) = 1, cosθ -> A(1,0) = 1/√3, x -> Ar(1,1) = -1/√6, y -> Ai(1,1) = 1/√6
        for (g, n, m, part, val) in (((t, f) -> 1.0, 0, 0, 0, 1.0), ((t, f) -> cos(t), 1, 0, 0, 1 / sqrt(3)),
                                     ((t, f) -> sin(t) * cos(f), 1, 1, 0, -1 / sqrt(6)),
                                     ((t, f) -> sin(t) * sin(f), 1, 1, 1, 1 / sqrt(6)))
            S = grid2shc(_grid(g, θ, φ), Nt, Np, p; arrange=arr)
            @test length(S) == shc_length(p, arr)
            k = _shc_index(n, m, part, p, arr)
            @test S[k] ≈ val
            @test norm(S[setdiff(eachindex(S), k)]) < 1e-13
        end

        # grid -> coefficients -> grid round trip of a band-limited function (finer output grid too)
        S = grid2shc(X, Nt, Np, p; arrange=arr)
        @test shc2grid(S, p, Nt, Np; arrange=arr) ≈ vec(X)
        θ2, φ2 = sph_grid_angles(15, 21)
        @test shc2grid(S, p, 15, 21; arrange=arr) ≈ vec(_grid(_f, θ2, φ2))

        # theta / phi derivatives
        Y, Yt, Yp = shc2grid(S, p, Nt, Np; arrange=arr, derivatives=true)
        @test Y ≈ vec(X)
        @test Yt ≈ vec(_grid(_ft, θ, φ))
        @test Yp ≈ vec(_grid(_fp, θ, φ))

        # pointwise evaluation at arbitrary points (2×N matrix or flat vector)
        tp = [π .* (1 .+ _prand(1, 9)) ./ 2; π .* (1 .+ _prand(1, 9) .^ 2)]
        @test shceval(S, p, tp; arrange=arr) ≈ [_f(tp[1, i], tp[2, i]) for i in 1:9]
        @test shceval(S, p, vec(tp); arrange=arr) ≈ shceval(S, p, tp; arrange=arr)

        # values at the poles: f(0, ⋅) = 1.7, f(π, ⋅) = -0.3
        @test shc2pole(S, p; arrange=arr) ≈ [1.7, -0.3]
    end

    # several fields: (Np, Nt, nfields) input, field-major coefficients, point-major evaluation
    g2(t, ph) = sin(t) * cos(t) * cos(ph)
    X2 = cat(X, _grid(g2, θ, φ); dims=3)
    S2 = grid2shc(X2, Nt, Np, p)
    n = shc_length(p)
    @test length(S2) == 2n
    @test S2[1:n] ≈ grid2shc(X, Nt, Np, p)
    @test S2[n+1:end] ≈ grid2shc(_grid(g2, θ, φ), Nt, Np, p)
    @test shc2grid(S2, p, Nt, Np) ≈ vec(X2)
    tp = [0.3 1.2 2.9; 0.1 4.0 5.5]
    V = reshape(shceval(S2, p, tp), 2, 3)
    @test V[1, :] ≈ [_f(tp[1, i], tp[2, i]) for i in 1:3]
    @test V[2, :] ≈ [g2(tp[1, i], tp[2, i]) for i in 1:3]
    @test shc2pole(S2, p) ≈ [1.7, -0.3, 0.0, 0.0] atol = 1e-13

    # truncation: a degree-6 grid function projected to p = 3 keeps only the degree <= 3 part
    S3 = grid2shc(X .+ _grid((t, f) -> sin(t)^5 * cos(t) * cos(5f), θ, φ), Nt, Np, 3)
    @test shceval(S3, 3, tp) ≈ [_f(tp[1, i], tp[2, i]) for i in 1:3]

    # clearing the cache does not change results
    sph_clear_cache()
    @test grid2shc(X, Nt, Np, p) ≈ S2[1:n]
end

@testset "vector transforms" begin
    p = 5
    Nt, Np = p + 2, 2p + 4
    θ, φ = sph_grid_angles(Nt, Np)
    n = shc_length(p)
    k10 = _shc_index(1, 0, 0, p, SHC_ROW_MAJOR)

    # V, W, X coefficients of ∇_s cosθ, r̂ × ∇_s cosθ and cosθ r̂ (V_n = ∇Y - (n+1)Y r̂, W_n = ∇Y + nY r̂, X_n = r̂×∇Y)
    a = 1 / sqrt(3)
    for (F, vwx) in (((t, f) -> -sin(t) .* _eθ(t, f), (a / 3, 2a / 3, 0.0)),
                     ((t, f) -> -sin(t) .* _eφ(t, f), (0.0, 0.0, a)),
                     ((t, f) -> cos(t) .* _er(t, f), (-a / 3, a / 3, 0.0)))
        G = _vgrid(F, θ, φ)
        S = grid2vecshc(G, Nt, Np, p)
        @test length(S) == 3n
        for c in 1:3
            blk = S[(c-1)*n+1:c*n]
            @test blk[k10] ≈ vwx[c] atol = 1e-14
            @test norm(blk[setdiff(1:n, k10)]) < 1e-13
        end
        @test vecshc2grid(S, p, Nt, Np) ≈ vec(G)
    end

    # round trip of a random band-limited vector field, all arrangements, two vector fields
    for arr in _ARRS
        S0 = grid2vecshc(_prand(Np, Nt, 3, 2), Nt, Np, p; arrange=arr)  # projection: band-limited coefficients
        @test length(S0) == 2 * 3 * shc_length(p, arr)
        G = vecshc2grid(S0, p, Nt, Np; arrange=arr)
        @test length(G) == 2 * 3 * Nt * Np
        @test grid2vecshc(G, Nt, Np, p; arrange=arr) ≈ S0
        # evaluation at the grid points matches the grid; layout (3, nvec, N)
        tp = vcat(repeat(θ'; inner=(1, Np)), repeat(φ', 1, Nt))
        E = reshape(vecshceval(S0, p, tp; arrange=arr), 3, 2, Nt * Np)
        Gr = reshape(G, Nt * Np, 3, 2)
        @test E ≈ permutedims(Gr, (2, 3, 1))
    end

    # analytic evaluation: F = (x, y, z) components of a smooth band-limited field
    F(t, f) = [cos(t), sin(t) * cos(f) * cos(t), 1 - sin(t)^2 * sin(f)^2]
    S = grid2vecshc(_vgrid(F, θ, φ), Nt, Np, p)
    tp = [0.4 1.7 3.0; 0.2 2.5 6.0]
    @test reshape(vecshceval(S, p, tp), 3, :) ≈ hcat([F(tp[1, i], tp[2, i]) for i in 1:3]...)
end

@testset "Laplace layer potentials" begin
    p = 4
    n = shc_length(p)
    σ1 = zeros(n); σ1[1] = 1                                # σ ≡ 1
    cz = zeros(n); cz[_shc_index(1, 0, 0, p, SHC_ROW_MAJOR)] = 1 / sqrt(3)   # σ = cosθ
    xin = [0.1 0.3 -0.5; 0.2 0.0 0.1; -0.3 0.5 0.2]
    xout = [2.0 0.0 1.0; 0.0 3.0 -1.5; 0.0 0.5 2.0]
    r(x) = vec(sqrt.(sum(abs2, x; dims=1)))
    # single layer: σ=1 -> 1 inside, 1/r outside; σ=cosθ -> z/3 inside, z/(3r^3) outside
    @test laplace_eval_sl(σ1, p, xin; interior=true) ≈ ones(3)
    @test laplace_eval_sl(σ1, p, xout; interior=false) ≈ 1 ./ r(xout)
    @test laplace_eval_sl(cz, p, xin; interior=true) ≈ xin[3, :] ./ 3
    @test laplace_eval_sl(cz, p, xout; interior=false) ≈ xout[3, :] ./ (3 .* r(xout) .^ 3)
    # double layer: σ=1 -> -1 inside, 0 outside; σ=cosθ -> -2z/3 inside, z/(3r^3) outside
    @test laplace_eval_dl(σ1, p, xin; interior=true) ≈ -ones(3)
    @test laplace_eval_dl(σ1, p, xout; interior=false) ≈ zeros(3) atol = 1e-14
    @test laplace_eval_dl(cz, p, xin; interior=true) ≈ -2 .* xin[3, :] ./ 3
    @test laplace_eval_dl(cz, p, xout; interior=false) ≈ xout[3, :] ./ (3 .* r(xout) .^ 3)
    # several densities, flat coordinates: point-major output
    U = reshape(laplace_eval_sl([σ1; cz], p, vec(xout); interior=false), 2, 3)
    @test U[1, :] ≈ 1 ./ r(xout)
    @test U[2, :] ≈ xout[3, :] ./ (3 .* r(xout) .^ 3)
end

@testset "Stokes layer potentials" begin
    p = 4
    Nt, Np = 40, 80   # fine grid for the direct quadrature check
    θ, φ = sph_grid_angles(Nt, Np)
    w = sph_legendre_weights(Nt - 1)
    S = grid2vecshc(vecshc2grid(_prand(3 * shc_length(p)), p, Nt, Np), Nt, Np, p)  # band-limited density
    F = reshape(vecshc2grid(S, p, Nt, Np), Np, Nt, 3)

    function direct(kern, x)
        u = zeros(3)
        for i in 1:Nt, j in 1:Np
            y = _er(θ[i], φ[j])
            u .+= kern(x - y, y, F[j, i, :]) .* (w[i] * 2π / Np)
        end
        return u
    end
    sl(r, y, f) = (f ./ norm(r) .+ r .* dot(r, f) ./ norm(r)^3) ./ 8π
    dl(r, y, f) = 3 / 4π .* r .* dot(r, y) .* dot(r, f) ./ norm(r)^5
    function kl(r, y, f, ν)
        R = norm(r)
        ∇u = [(-f[i] * r[j] + (i == j) * dot(r, f) + r[i] * f[j]) / R^3 - 3r[i] * r[j] * dot(r, f) / R^5 for i in 1:3, j in 1:3] ./ 8π
        P = dot(r, f) / (4π * R^3)
        return (∇u + ∇u' - P * I) * ν
    end
    ν = [0.3, -0.5, 0.8]
    for x in ([0.3, -0.2, 0.25], [1.5, 0.7, -0.9])
        int = norm(x) < 1
        @test stokes_eval_sl(S, p, x; interior=int) ≈ direct(sl, x) rtol = 1e-10
        @test stokes_eval_dl(S, p, x; interior=int) ≈ direct(dl, x) rtol = 1e-10
        @test stokes_eval_kl(S, p, x, ν; interior=int) ≈ direct((r, y, f) -> kl(r, y, f, ν), x) rtol = 1e-10
    end

    # uniform density e_z: the single layer is the rigid translation (2/3) e_z inside the sphere
    Sz = grid2vecshc(_vgrid((t, f) -> [0.0, 0.0, 1.0], θ, φ), Nt, Np, p)
    xin = [0.1 -0.4; 0.2 0.3; -0.5 0.1]
    @test reshape(stokes_eval_sl(Sz, p, xin; interior=true), 3, 2) ≈ repeat([0, 0, 2 / 3], 1, 2)

    # traction on the sphere: limits of the off-surface traction, and jump [t]_ext - [t]_int = -f
    xs = normalize.([[0.3, -0.4, 0.5], [-0.8, 0.1, -0.2]])
    X = reduce(hcat, xs)
    for int in (true, false)
        ks = stokes_eval_kself(S, p, X; interior=int)
        kl_lim = stokes_eval_kl(S, p, X .* (int ? 1 - 1e-8 : 1 + 1e-8), X; interior=int)
        @test ks ≈ kl_lim rtol = 1e-6
    end
    f = vecshceval(S, p, [acos.(X[3, :])'; atan.(X[2, :], X[1, :])'])
    @test stokes_eval_kself(S, p, X; interior=false) - stokes_eval_kself(S, p, X; interior=true) ≈ -f
end

@testset "argument errors" begin
    p, Nt, Np = 3, 4, 8
    n = shc_length(p)
    @test_throws DimensionMismatch grid2shc(ones(Nt * Np + 1), Nt, Np, p)
    @test_throws DimensionMismatch grid2shc(Float64[], Nt, Np, p)
    @test_throws ArgumentError grid2shc(ones(Nt * Np), Nt, Np, -1)
    @test_throws ArgumentError grid2shc(ones(Nt * Np), 0, Np, p)
    @test_throws ArgumentError shc2grid(ones(n), p, Nt, 2000)
    @test_throws ArgumentError shc2grid(ones(shc_length(2000)), 2000, Nt, Np)
    @test_throws DimensionMismatch shc2grid(ones(n + 1), p, Nt, Np)
    @test_throws DimensionMismatch shc2grid(ones(n), p, Nt, Np; arrange=SHC_ALL)
    @test_throws DimensionMismatch shceval(ones(n), p, ones(3, 2))     # theta_phi must be 2×N
    @test_throws DimensionMismatch shceval(ones(n), p, ones(5))
    @test_throws DimensionMismatch shc2pole(ones(n - 1), p)
    @test_throws DimensionMismatch grid2vecshc(ones(2 * Nt * Np), Nt, Np, p)
    @test_throws DimensionMismatch vecshc2grid(ones(n), p, Nt, Np)
    @test_throws DimensionMismatch vecshceval(ones(2n), p, ones(2, 2))
    @test_throws DimensionMismatch laplace_eval_sl(ones(n), p, ones(2, 4); interior=true)
    @test_throws DimensionMismatch stokes_eval_sl(ones(n), p, ones(3); interior=true)
    @test_throws DimensionMismatch stokes_eval_kl(ones(3n), p, ones(3, 2), ones(3, 3); interior=true)
    @test_throws UndefKeywordError laplace_eval_sl(ones(n), p, ones(3))
    @test_throws ArgumentError sph_legendre_nodes(-1)
    @test_throws ArgumentError sph_legendre_weights(1024)
    @test_throws ArgumentError sph_grid_angles(4, 0)
    # the C++ shim also validates (last resort, bypassing the Julia checks)
    @test_throws SCTL.SCTLError SCTL._check(ccall((:sctljl_sph_shc2pole, SCTL.libsctljl), Cint,
                                                  (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Int64),
                                                  zeros(2), 2, ones(n + 1), n + 1, Cint(1), p))
end
