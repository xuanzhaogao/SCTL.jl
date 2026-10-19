using SCTL, Test, LinearAlgebra

@testset "quadrature rules" begin
    for (rule, exactdeg) in ((chebquad, n -> n - 1), (legquad, n -> 2n - 1))
        for n in (1, 5, 16)
            x, w = rule(n)
            @test length(x) == length(w) == n
            @test all(0 .< x .< 1)
            @test sum(w) ≈ 1
            for k in 0:exactdeg(n)
                @test dot(w, x .^ k) ≈ 1 / (k + 1) atol = 1e-13
            end
        end
    end
    @test_throws ArgumentError legquad(0)
end

@testset "legendre_poly" begin
    x = collect(range(-1, 1; length=11))
    P, dP = legendre_poly(x, 3)
    @test P ≈ (5 .* x .^ 3 .- 3 .* x) ./ 2
    @test dP ≈ (15 .* x .^ 2 .- 3) ./ 2
    P0, dP0 = legendre_poly(x, 0)
    @test all(P0 .== 1) && all(dP0 .== 0)
end

@testset "lagrange interpolation" begin
    src, _ = chebquad(8)
    trg = collect(range(0, 1; length=13))
    W = lagrange_interp_matrix(src, trg)
    @test size(W) == (13, 8)
    p(x) = 1 - 2x + 3x^3 - x^7
    @test W * p.(src) ≈ p.(trg)
    F = [p.(src) sin.(src)]
    @test lagrange_interp(src, F, trg)[:, 1] ≈ p.(trg)

    dp(x) = -2 + 9x^2 - 7x^6
    @test lagrange_derivative(p.(src), src) ≈ dp.(src)
    D = lagrange_derivative([p.(src) 2 .* p.(src)], src)
    @test size(D) == (8, 2)
    @test D[:, 2] ≈ 2 .* dp.(src)
    @test_throws DimensionMismatch lagrange_derivative(ones(3), src)
end

@testset "interp_quad_rule" begin
    # integrands: monomials x^k, k = 0..9, and log(x)-singular x^k log(x), on [0,1]
    integrands(x) = hcat([x .^ k for k in 0:9]..., [x .^ k .* log.(x) for k in 0:9]...)
    exact = vcat([1 / (k + 1) for k in 0:9], [-1 / (k + 1)^2 for k in 0:9])

    x, w, cond = interp_quad_rule(integrands, 0.0, 1.0; eps=1e-12)
    @test length(x) == length(w) < 20
    @test cond > 0
    @test maximum(abs.(integrands(x)' * w .- exact)) < 1e-9

    # from a discretization
    nd, wtd = legquad(40)
    xs = vcat(nd ./ 2^20, [2.0^-(20 - i) .* (0.5 .+ nd ./ 2) for i in 1:20]...)
    ws = vcat(wtd ./ 2^20, [2.0^-(20 - i) .* (wtd ./ 2) for i in 1:20]...)
    x2, w2, _ = interp_quad_rule(integrands(xs), xs, ws; eps=1e-10)
    @test maximum(abs.(integrands(x2)' * w2 .- exact)) < 1e-7

    rules = interp_quad_rules(integrands(xs), xs, ws; eps=[1e-4, 1e-8, 1e-12])
    @test length(rules) == 3
    @test length(rules[1][1]) < length(rules[2][1]) < length(rules[3][1])
    rules2 = interp_quad_rules(integrands(xs), xs, ws; order=[4, 6])
    @test [length(r[1]) for r in rules2] == [4, 6]

    # errors in the Julia callback propagate
    @test_throws ErrorException interp_quad_rule(x -> error("boom"), 0.0, 1.0)
    @test_throws DimensionMismatch interp_quad_rule(x -> ones(length(x) + 1, 2), 0.0, 1.0)
end

@testset "SCTL assertions become SCTLError" begin
    # bypass the Julia-side checks: 2 tolerances but 3 orders trips an SCTL_ASSERT in InterpQuadRule::Build
    x, w = legquad(8)
    M = permutedims(hcat(x, x .^ 2))
    hs = [SCTL.VecHandle{Float64}() for _ in 1:6]
    try
        px, pw = [h.ptr for h in hs[1:3]], [h.ptr for h in hs[4:6]]
        status = ccall((:sctljl_interpquad_build_multi, SCTL.libsctljl), Cint,
                       (Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                        Int64, Int64, Ptr{Float64}, Int64, Ptr{Int64}, Int64, Float64, Float64, Cint, Cint),
                       px, pw, zeros(3), 3, M, x, w, 8, 2, [1e-4, 1e-8], 2, [2, 3, 4], 3, 0.0, 0.0, 1, 0)
        err = try
            SCTL._check(status)
            nothing
        catch e
            e
        end
        @test err isa SCTL.SCTLError
        @test occursin("Assertion", sprint(showerror, err))
    finally
        foreach(SCTL.free!, hs)
    end
end

@testset "threads and features" begin
    n = SCTL.get_num_threads()
    @test n >= 1
    SCTL.set_num_threads(1)
    @test SCTL.get_num_threads() == 1
    SCTL.set_num_threads(n)
    @test SCTL.has_feature(:MPI) == false
end
