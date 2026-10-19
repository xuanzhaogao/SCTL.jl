using SCTL, Test, LinearAlgebra

struct SDCTestError <: Exception end

# harmonic oscillator u = (cos t, sin t)
sdc_osc!(du, u) = (du[1] = -u[2]; du[2] = u[1]; nothing)
sdc_osc_exact(t) = [cos(t), sin(t)]

@testset "SDC object" begin
    S = SDC(5)
    @test order(S) == 5
    @test sprint(show, S) == "SDC(5)"
    τ = sdc_nodes(S)
    @test length(τ) == 5 && τ[1] == 0 && τ[end] ≈ 1 && issorted(τ)
    @test τ[3] ≈ 0.5
    @test_throws ArgumentError SDC(1)
    S2 = SDC(3)
    close(S2)
    close(S2)
    @test_throws ArgumentError sdc_step(S2, sdc_osc!, [1.0, 0.0], 0.1)
    @test sprint(show, S2) == "SDC(3, <freed>)"
end

@testset "sdc_step" begin
    S = SDC(6)
    # exponential decay, one step
    λ, dt = -2.0, 0.1
    s = sdc_step(S, (du, u) -> (du .= λ .* u), [1.0, 3.0], dt)
    @test s.u ≈ [1.0, 3.0] .* exp(λ * dt) rtol = 1e-8
    @test s.iters == order(S)  # default: `order` Picard iterations
    @test 0 <= s.error_interp < 1e-6
    @test 0 <= s.error_picard < 1e-6
    @test s.substeps === nothing

    # substep solutions: n × order, columns at dt * sdc_nodes
    s = sdc_step(S, sdc_osc!, [1.0, 0.0], 0.2; substeps=true)
    U = s.substeps
    @test size(U) == (2, order(S))
    @test U[:, 1] == [1.0, 0.0]
    @test U[:, end] == s.u
    for (i, τ) in enumerate(sdc_nodes(S))
        @test U[:, i] ≈ sdc_osc_exact(0.2τ) atol = 1e-9
    end

    # integer u0, scalar ODE
    @test sdc_step(SDC(4), (du, u) -> (du[1] = 1.0), [0], 0.5).u ≈ [0.5]

    # observed order of convergence (fixed steps over [0, 1])
    for p in (3, 4, 5, 6)
        Sp = SDC(p)
        errs = map((8, 16)) do N
            u = [1.0, 0.0]
            for _ in 1:N
                u = sdc_step(Sp, sdc_osc!, u, 1 / N).u
            end
            norm(u - sdc_osc_exact(1.0))
        end
        @test log2(errs[1] / errs[2]) ≈ p atol = 0.3
    end

    # Picard iteration limit and number of RHS evaluations (1 + (order-1) per iteration)
    for m in (1, 2, 3)
        ncalls = Ref(0)
        s = sdc_step(S, (du, u) -> (ncalls[] += 1; du .= -u), [1.0], 0.1; picard_iters=m)
        @test s.iters == m
        @test ncalls[] == 1 + (order(S) - 1) * m
    end
    # fewer iterations: larger Picard error
    @test sdc_step(S, sdc_osc!, [1.0, 0.0], 0.1; picard_iters=1).error_picard >
          sdc_step(S, sdc_osc!, [1.0, 0.0], 0.1; picard_iters=4).error_picard
    # early stop on picard_tol
    s = sdc_step(S, sdc_osc!, [1.0, 0.0], 0.1; picard_iters=20, picard_tol=1e-6)
    @test s.iters < 20
    @test s.error_picard < 1e-6
    @test s.u ≈ sdc_osc_exact(0.1) atol = 1e-6

    # four-argument RHS with the SDC indices
    idx = Tuple{Int,Int}[]
    s = sdc_step(S, (du, u, ci, si) -> (push!(idx, (ci, si)); du .= -u), [1.0], 0.1; picard_iters=2, indices=true)
    @test s.u ≈ [exp(-0.1)] rtol = 1e-3
    @test idx == vcat([(0, 0)], [(c, i) for c in 0:1 for i in 1:order(S)-1])

    # errors in f! propagate with their original type; the solver remains usable
    @test_throws SDCTestError sdc_step(S, (du, u) -> throw(SDCTestError()), [1.0], 0.1)
    k = Ref(0)
    @test_throws DomainError sdc_step(S, (du, u) -> ((k[] += 1) == 3 && throw(DomainError(0)); du .= u), [1.0], 0.1)
    @test sdc_step(S, (du, u) -> (du .= -u), [1.0], 0.1).u ≈ [exp(-0.1)]

    # argument checking
    @test_throws ArgumentError sdc_step(S, sdc_osc!, Float64[], 0.1)
    @test_throws ArgumentError sdc_step(S, sdc_osc!, [1.0, 0.0], 0.1; picard_iters=0)
end

@testset "sdc_solve" begin
    S = SDC(6)
    T = 10.0
    for tol in (1e-5, 1e-9)
        ts, dts, us = Float64[], Float64[], Vector{Float64}[]
        r = sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.1, T, tol; monitor=(t, dt, u) -> (push!(ts, t); push!(dts, dt); push!(us, u)))
        @test r.t ≈ T
        @test r.err <= tol
        @test norm(r.u - sdc_osc_exact(T)) <= 10 * tol
        @test length(ts) > 1
        @test issorted(ts) && allunique(ts)
        @test ts[end] ≈ T
        @test ts ≈ cumsum(dts)
        @test us[end] == r.u
        @test all(norm(us[i] - sdc_osc_exact(ts[i])) <= 10 * tol for i in eachindex(ts))
    end

    # exponential decay, without monitor
    r = sdc_solve(SDC(8), (du, u) -> (du .= -u), [2.0], 0.5, 3.0, 1e-10)
    @test r.u ≈ [2exp(-3.0)] rtol = 1e-9
    # four-argument form
    r4 = sdc_solve(SDC(8), (du, u, ci, si) -> (du .= -u), [2.0], 0.5, 3.0, 1e-10; indices=true)
    @test r4.u == r.u

    # unreachable tolerance: error by default, best effort with continue_with_errors
    @test_throws SCTL.SCTLError sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.1, T, 1e-30)
    r = redirect_stderr(devnull) do  # SCTL prints a warning to stderr (C++ std::cerr, usually not captured)
        sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.1, T, 1e-30; continue_with_errors=true)
    end
    @test r.err > 1e-30
    @test norm(r.u - sdc_osc_exact(r.t)) < 1e-8

    # errors in f! and monitor propagate with their original type
    @test_throws SDCTestError sdc_solve(S, (du, u) -> throw(SDCTestError()), [1.0], 0.1, 1.0, 1e-6)
    nmon = Ref(0)
    @test_throws SDCTestError sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.1, T, 1e-6;
                                        monitor=(t, dt, u) -> ((nmon[] += 1) == 3 && throw(SDCTestError())))
    @test nmon[] == 3

    # argument checking
    @test_throws ArgumentError sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.1, -1.0, 1e-6)
    @test_throws ArgumentError sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.0, 1.0, 1e-6)
    @test_throws ArgumentError sdc_solve(S, sdc_osc!, [1.0, 0.0], 0.1, 1.0, 0.0)
    @test_throws ArgumentError sdc_solve(S, sdc_osc!, Float64[], 0.1, 1.0, 1e-6)
end

@testset "SDC: long states (zero-copy callback path)" begin
    n = 2 * SCTL.CALLBACK_COPY_MAX + 3
    λ = collect(range(0.1, 1.0; length=n))
    u0 = cos.(1:n)
    sdc = SDC(8)
    r = sdc_step(sdc, (du, u) -> (du .= .-λ .* u), u0, 0.1)
    @test r.u ≈ u0 .* exp.(-0.1 .* λ) rtol = 1e-12
    s = sdc_solve(sdc, (du, u) -> (du .= .-λ .* u), u0, 0.1, 1.0, 1e-10)
    @test s.u ≈ u0 .* exp.(-λ) rtol = 1e-8
    @test_throws DomainError sdc_step(sdc, (du, u) -> throw(DomainError(1)), u0, 0.1)
end
