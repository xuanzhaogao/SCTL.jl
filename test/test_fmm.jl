using SCTL, Test, LinearAlgebra

# 40000 targets or more make ParticleFMM use PVFMM (fewer are summed directly); low accuracy keeps the one-time
# operator precomputation short.
const NFMM = 40_000
const HAVE_PVFMM = SCTL.has_feature(:PVFMM)

_lcg(n, seed) = (x = UInt64(seed); [((x = x * 6364136223846793005 + 1442695040888963407) >> 11) / 2.0^53 for _ in 1:n])
relmax(a, b) = maximum(abs, a .- b) / maximum(abs, b)

@testset "ParticleFMM: cache and kernels" begin
    @test isdir(SCTL.fmm_cache_dir())
    @test Laplace3D_FxU in fmm_supported_kernels()
    @test_throws ArgumentError ParticleFMM(Helmholtz3D_FxU(1.0))
    @test_throws ArgumentError ParticleFMM(BiotSavart3D_FxU())
    @test_throws ArgumentError ParticleFMM(Laplace3D_FxU(); digits = 0)
end

@testset "ParticleFMM vs direct sums" begin
    Xs = reshape(_lcg(3 * 3000, 1), 3, :)
    Xt = reshape(_lcg(3 * NFMM, 2), 3, :)
    Xn = reshape(_lcg(3 * 3000, 3), 3, :) .- 0.5
    sample = 1:97:NFMM   # reference values at a subset of the targets
    cases = [(Laplace3D_FxU(), 6, 1e-5), (Laplace3D_DxU(), 6, 1e-5), (Laplace3D_FxdU(), 6, 1e-5),
             (Laplace3D_DxdU(), 6, 1e-5), (Laplace3D_Fxd2U(), 6, 1e-4),
             (Stokes3D_FxU(), 4, 1e-2), (Stokes3D_DxU(), 4, 1e-2), (Stokes3D_FxT(), 4, 5e-2)]
    for (ker, digits, tol) in cases
        @testset "$(kernel_name(ker))" begin
            F = reshape(_lcg(src_dim(ker) * 3000, 4), src_dim(ker), :) .- 0.5
            Xn_ = normal_dim(ker) > 0 ? Xn : nothing
            fmm = ParticleFMM(ker; digits = digits)
            set_sources!(fmm, Xs, Xn_)
            set_density!(fmm, F)
            set_targets!(fmm, Xt)
            U = fmm_evaluate(fmm)
            @test size(U) == (trg_dim(ker), NFMM)
            Uref = kernel_eval(ker, Xs, Xn_, F, Xt[:, sample])
            @test relmax(U[:, sample], Uref) < (HAVE_PVFMM ? tol : 1e-12)
            # new density on the same points (tree reused) and in-place output
            set_density!(fmm, -2 .* F)
            U2 = similar(U)
            @test fmm_evaluate!(U2, fmm) === U2
            @test U2 ≈ -2 .* U rtol = 1e-10
            close(fmm)
        end
    end
    # one-shot form; below the threshold this is an exact direct sum
    F = reshape(_lcg(3000, 5), 1, :)
    @test fmm_eval(Laplace3D_FxU(), Xs, F, Xt[:, 1:500]; digits = 6) ≈ kernel_eval(Laplace3D_FxU(), Xs, F, Xt[:, 1:500])
end

@testset "ParticleFMM errors" begin
    fmm = ParticleFMM(Laplace3D_DxU(); digits = 4)
    @test_throws ArgumentError fmm_evaluate(fmm)                       # nothing set
    @test_throws ArgumentError set_density!(fmm, ones(3))              # sources first
    @test_throws ArgumentError set_sources!(fmm, rand(3, 5))           # DxU needs normals
    set_sources!(fmm, rand(3, 5), rand(3, 5))
    @test_throws DimensionMismatch set_density!(fmm, ones(4))
    @test_throws DimensionMismatch set_targets!(fmm, rand(4, 2))
    close(fmm)
    @test_throws ArgumentError set_targets!(fmm, rand(3, 2))           # closed
end

if HAVE_PVFMM
    @testset "ParticleFMM periodic" begin
        # periodic in x, y, z with period 1: translating all points by a vector (modulo 1) leaves the potentials
        # unchanged; zero-mean density (the periodic Laplace sum of a net charge diverges)
        ns = 2000
        Xs = reshape(_lcg(3 * ns, 6), 3, :)
        Xt = reshape(_lcg(3 * NFMM, 7), 3, :)
        F = reshape(_lcg(ns, 8), 1, :)
        F .-= sum(F) / ns
        shift(X) = mod.(X .+ [0.37, 0.11, 0.73], 1.0)
        U1 = fmm_eval(Laplace3D_FxdU(), Xs, F, Xt; digits = 6, periodic = true)
        U2 = fmm_eval(Laplace3D_FxdU(), shift(Xs), F, shift(Xt); digits = 6, periodic = true)
        @test relmax(U2, U1) < 1e-4
        @test_throws ArgumentError fmm_eval(Laplace3D_FxdU(), Xs .+ 1, F, Xt; periodic = true)   # outside the box
        @test_throws ArgumentError fmm_direct(set_targets!(set_density!(set_sources!(
            ParticleFMM(Laplace3D_FxU(); periodic = true), Xs), F), Xt))
    end
end

@testset "BoundaryIntegralOp with 40000+ targets (FMM far field)" begin
    # cubed sphere, order 8, 24 elements; targets inside and outside the unit sphere
    faces = (((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
             ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
    function sphere_point(e, u, v; ppf = 2)
        e0 = e - 1; face = e0 ÷ ppf^2 + 1
        a = 2 * ((e0 ÷ ppf % ppf + u) / ppf) - 1; b = 2 * ((e0 % ppf + v) / ppf) - 1
        q = [faces[face][k][1] + faces[face][k][2] * a + faces[face][k][3] * b for k in 1:3]
        return q ./ norm(q)
    end
    elems = QuadElemList(sphere_point, 8, 24)
    dirs = reshape(_lcg(3 * NFMM, 9), 3, :) .- 0.5
    dirs ./= sqrt.(sum(abs2, dirs; dims = 1))
    r = vcat(fill(0.5, NFMM ÷ 2), fill(2.0, NFMM - NFMM ÷ 2))
    Xt = dirs .* r'
    inside = r .< 1

    # Laplace double layer of 1: -1 inside, 0 outside
    op = BoundaryIntegralOp(Laplace3D_DxU(); accuracy = 1e-6)
    add_elem_list!(op, elems)
    set_target_coord!(op, Xt)
    U = compute_potential(op, ones(nnodes(elems)))
    @test maximum(abs, U[inside] .+ 1) < 1e-4
    @test maximum(abs, U[.!inside]) < 1e-4

    # Laplace single layer of 1: 1 inside, 1/r outside
    op = BoundaryIntegralOp(Laplace3D_FxU(); accuracy = 1e-6)
    add_elem_list!(op, elems)
    set_target_coord!(op, Xt)
    U = compute_potential(op, ones(nnodes(elems)))
    @test maximum(abs, U .- ifelse.(inside, 1.0, 1 ./ r)) < 1e-4

    # Helmholtz: PVFMM cannot evaluate it (wavenumber), so the far field falls back to a direct sum, also above
    # the FMM threshold; single layer of 1 inside: e^{ik} sin(kr)/(kr)
    k = 1.5
    op = BoundaryIntegralOp(Helmholtz3D_FxU(k); accuracy = 1e-6)
    add_elem_list!(op, elems)
    set_target_coord!(op, Xt[:, inside])
    U = compute_potential(op, repeat([1.0, 0.0], nnodes(elems)))   # (re, im) pairs
    ρ = r[inside]
    exact = cis(k) .* sin.(k .* ρ) ./ (k .* ρ)
    @test maximum(abs, complex.(U[1:2:end], U[2:2:end]) .- exact) < 1e-4
end
