using SCTL, Test, LinearAlgebra

const ALL_KERNELS = [Laplace3D_FxU(), Laplace3D_DxU(), Laplace3D_FxdU(), Laplace3D_Fxd2U(), Laplace3D_DxdU(),
                     BiotSavart3D_FxU(), BiotSavart3D_FxdU(),
                     Helmholtz3D_FxU(1.7), Helmholtz3D_DxU(1.7), Helmholtz3D_FxdU(1.7), HelmholtzDiff3D_FxdU(1.7),
                     Stokes3D_FxU(), Stokes3D_DxU(), Stokes3D_FxT(), Stokes3D_FSxU(), Stokes3D_FxUP()]

_unit_normals(n) = (N = randn(3, n); N ./ sqrt.(sum(abs2, N; dims=1)))
_nrm(ker, n) = normal_dim(ker) == 0 ? nothing : _unit_normals(n)

@testset "kernel dimensions and names" begin
    for ker in ALL_KERNELS
        @test (coord_dim(ker), normal_dim(ker), src_dim(ker), trg_dim(ker)) == SCTL._kernel_dims_c(ker)
        @test kernel_name(ker) == replace(string(nameof(typeof(ker))), "_" => "-")
    end
end

@testset "kernel formulas" begin
    Xs = randn(3, 7)
    Xt = randn(3, 5) .+ [3.0, 0, 0]
    Xn = _unit_normals(7)
    r(t, s) = Xt[:, t] - Xs[:, s]

    K = kernel_matrix(Laplace3D_FxU(), Xs, Xt)
    @test size(K) == (5, 7)
    @test K ≈ [1 / (4π * norm(r(t, s))) for t in 1:5, s in 1:7] rtol = 1e-13

    K = kernel_matrix(Laplace3D_DxU(), Xs, Xn, Xt)
    @test K ≈ [dot(r(t, s), Xn[:, s]) / (4π * norm(r(t, s))^3) for t in 1:5, s in 1:7] rtol = 1e-13

    # gradient at the target of the single layer: trg_dim = 3, rows (t-1)*3 + j
    K = kernel_matrix(Laplace3D_FxdU(), Xs, Xt)
    @test size(K) == (15, 7)
    for t in 1:5, s in 1:7
        @test K[3t-2:3t, s] ≈ -r(t, s) / (4π * norm(r(t, s))^3) rtol = 1e-13
    end
    # ... which is the finite-difference gradient of the single layer
    F = randn(1, 7)
    h = 1e-5
    G = kernel_eval(Laplace3D_FxdU(), Xs, F, Xt)
    for j in 1:3
        e = zeros(3); e[j] = h
        dU = (kernel_eval(Laplace3D_FxU(), Xs, F, Xt .+ e) - kernel_eval(Laplace3D_FxU(), Xs, F, Xt .- e)) / 2h
        @test G[j, :] ≈ vec(dU) rtol = 1e-7
    end

    # Stokeslet: 3×3 block (target t, source s) at rows 3t-2:3t, columns 3s-2:3s
    K = kernel_matrix(Stokes3D_FxU(), Xs, Xt)
    @test size(K) == (15, 21)
    for t in 1:5, s in 1:7
        rr = r(t, s); ρ = norm(rr)
        @test K[3t-2:3t, 3s-2:3s] ≈ (I / ρ + rr * rr' / ρ^3) / (8π) rtol = 1e-13
    end

    # Helmholtz: complex values as (re, im) pairs; the 2×2 real block of e^{ikr}/(4πr) acts as multiplication
    k = 2.3
    K = kernel_matrix(Helmholtz3D_FxU(k), Xs, Xt)
    for t in 1:5, s in 1:7
        ρ = norm(r(t, s))
        g = exp(im * k * ρ) / (4π * ρ)
        @test K[2t-1:2t, 2s-1:2s] ≈ [real(g) -imag(g); imag(g) real(g)] rtol = 1e-13
    end
    Fc = randn(ComplexF64, 7)
    U = kernel_eval(Helmholtz3D_FxU(k), Xs, reinterpret(Float64, Fc), Xt)
    @test size(U) == (2, 5)
    Uc = [sum(exp(im * k * norm(r(t, s))) / (4π * norm(r(t, s))) * Fc[s] for s in 1:7) for t in 1:5]
    @test vec(reinterpret(ComplexF64, vec(U))) ≈ Uc rtol = 1e-12

    # Helmholtz double layer reduces to the Laplace one as k -> 0 (real part), and is ∂/∂n_s of the single layer
    Kd = kernel_matrix(Helmholtz3D_DxU(0.0), Xs, Xn, Xt)
    @test Kd[1:2:end, 1:2:end] ≈ kernel_matrix(Laplace3D_DxU(), Xs, Xn, Xt) rtol = 1e-12
    @test norm(Kd[2:2:end, 1:2:end]) < 1e-14
    Kd = kernel_matrix(Helmholtz3D_DxU(k), Xs, Xn, Xt)
    Ks(X) = kernel_matrix(Helmholtz3D_FxU(k), X, Xt)
    fd = zeros(10, 14)
    for s in 1:7
        Xp = copy(Xs); Xp[:, s] .+= h * Xn[:, s]
        Xm = copy(Xs); Xm[:, s] .-= h * Xn[:, s]
        fd[:, 2s-1:2s] = (Ks(Xp)[:, 2s-1:2s] - Ks(Xm)[:, 2s-1:2s]) / 2h
    end
    @test Kd ≈ fd rtol = 1e-7

    # the Helmholtz context really is the wavenumber
    @test kernel_matrix(Helmholtz3D_FxU(0.0), Xs, Xt)[1:2:end, 1:2:end] ≈ kernel_matrix(Laplace3D_FxU(), Xs, Xt)
    @test !(kernel_matrix(Helmholtz3D_FxU(1.0), Xs, Xt) ≈ kernel_matrix(Helmholtz3D_FxU(2.0), Xs, Xt))
end

@testset "kernel_eval == kernel_matrix contraction ($(kernel_name(ker)))" for ker in ALL_KERNELS
    ns, nt = 23, 19
    Xs, Xt = randn(3, ns), randn(3, nt)
    Xn = _nrm(ker, ns)
    F = randn(src_dim(ker), ns)
    K = kernel_matrix(ker, Xs, Xn, Xt)
    @test size(K) == (trg_dim(ker) * nt, src_dim(ker) * ns)
    U = kernel_eval(ker, Xs, Xn, F, Xt)
    @test size(U) == (trg_dim(ker), nt)
    @test vec(U) ≈ K * vec(F) rtol = 1e-11
    @test kernel_eval(ker, Xs, Xn, F, Xt; openmp=false) ≈ U rtol = 1e-14
    @test kernel_matrix(ker, Xs, Xn, Xt; openmp=false) == K
    # single-target and single-source code paths
    @test kernel_matrix(ker, Xs, Xn, Xt[:, 3:3]) ≈ K[trg_dim(ker)*2+1:trg_dim(ker)*3, :] rtol = 1e-14
    Xn1 = Xn === nothing ? nothing : Xn[:, 5:5]
    @test kernel_matrix(ker, Xs[:, 5:5], Xn1, Xt) ≈ K[:, src_dim(ker)*4+1:src_dim(ker)*5] rtol = 1e-14
    @test vec(kernel_eval(ker, Xs, Xn, F, Xt[:, 1:1])) ≈ vec(U[:, 1]) rtol = 1e-12
    # in place, accumulating, flat inputs
    U2 = ones(trg_dim(ker), nt)
    kernel_eval!(U2, ker, vec(Xs), Xn === nothing ? nothing : vec(Xn), vec(F), vec(Xt); accumulate=true)
    @test U2 ≈ U .+ 1 rtol = 1e-12
    kernel_eval!(U2, ker, Xs, Xn, F, Xt)
    @test U2 ≈ U rtol = 1e-14

    # self-interaction: coincident points contribute nothing
    Ks = kernel_matrix(ker, Xs, Xn, Xs)
    @test all(isfinite, Ks)
    for s in 1:ns
        @test all(iszero, Ks[trg_dim(ker)*(s-1)+1:trg_dim(ker)*s, src_dim(ker)*(s-1)+1:src_dim(ker)*s])
    end
    @test kernel_matrix(ker, Xs, Xn, Xs[:, 2:end]) ≈ Ks[trg_dim(ker)+1:end, :] rtol = 1e-14
    @test vec(kernel_eval(ker, Xs, Xn, F, Xs)) ≈ Ks * vec(F) rtol = 1e-11
end

@testset "kernel argument errors" begin
    Xs, Xt = randn(3, 4), randn(3, 2)
    @test_throws ArgumentError kernel_matrix(Laplace3D_DxU(), Xs, Xt)             # normals required
    @test_throws DimensionMismatch kernel_matrix(Laplace3D_DxU(), Xs, randn(3, 3), Xt)
    @test_throws ArgumentError kernel_matrix(Laplace3D_FxU(), Xs, randn(3, 4), Xt) # no normals taken
    @test_throws DimensionMismatch kernel_matrix(Laplace3D_FxU(), randn(2, 4), Xt)
    @test_throws DimensionMismatch kernel_matrix(Laplace3D_FxU(), randn(10), Xt)
    @test_throws DimensionMismatch kernel_eval(Stokes3D_FxU(), Xs, randn(2, 4), Xt)
    @test_throws DimensionMismatch kernel_eval!(zeros(3, 3), Stokes3D_FxU(), Xs, randn(3, 4), Xt)
    @test size(kernel_matrix(Stokes3D_FxU(), zeros(3, 0), Xt)) == (6, 0)
    @test kernel_eval(Stokes3D_FxU(), zeros(3, 0), zeros(3, 0), Xt) == zeros(3, 2)
end

@testset "kernel_matrix! (in place)" begin
    Xs, Xt = rand(3, 7), rand(3, 5) .+ 2
    for ker in (Laplace3D_FxU(), Stokes3D_FxU(), Helmholtz3D_FxU(1.5))
        K = kernel_matrix(ker, Xs, Xt)
        K2 = fill(NaN, size(K))
        @test kernel_matrix!(K2, ker, Xs, Xt) === K2
        @test K2 == K
    end
    Xn = randn(3, 7)
    K = kernel_matrix(Laplace3D_DxU(), Xs, Xn, Xt)
    @test kernel_matrix!(fill(NaN, size(K)), Laplace3D_DxU(), Xs, Xn, Xt) == K
    @test_throws DimensionMismatch kernel_matrix!(zeros(4, 7), Laplace3D_FxU(), Xs, Xt)
end
