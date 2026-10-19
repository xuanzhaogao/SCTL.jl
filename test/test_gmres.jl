using SCTL, Test, LinearAlgebra

# Deterministic pseudo-random numbers in [-1/2, 1/2) (no Random dependency); `seed` selects the stream.
gmres_rand(seed, dims...) = reshape([mod(sin(12.9898 * k + 78.233 * seed) * 43758.5453, 1) - 0.5
                                     for k in 1:prod(dims)], dims...)

struct GMRESTestError <: Exception end

@testset "gmres" begin
    n = 60
    A = Matrix(1.0I, n, n) + 0.5 * gmres_rand(1, n, n) / sqrt(n)  # well conditioned, non-symmetric
    b = gmres_rand(2, n)
    xref = A \ b

    # matrix operator
    r = gmres(A, b)
    @test r.x isa Vector{Float64}
    @test 0 < r.iters <= n
    @test norm(r.x - xref) <= 1e-8 * norm(xref)
    @test norm(A * r.x - b) <= 1e-9 * norm(b)
    x, iters = r  # NamedTuple destructuring
    @test x === r.x && iters == r.iters

    # function operator: same iterates
    ncalls = Ref(0)
    A!(y, v) = (ncalls[] += 1; mul!(y, A, v))
    rf = gmres(A!, b)
    @test rf.iters == r.iters
    @test rf.x ≈ r.x
    @test ncalls[] == rf.iters  # one matvec per iteration (zero initial guess)

    # integer / non-Float64 inputs
    @test gmres(Matrix(2.0I, 4, 4), [2, 4, 6, 8]).x ≈ [1, 2, 3, 4]

    # tolerance: looser tol means fewer iterations, residual still within tol
    rl = gmres(A, b; tol=1e-4)
    @test rl.iters < r.iters
    @test norm(A * rl.x - b) <= 1e-4 * norm(b) * (1 + 1e-8)

    # maxiter limit
    for m in (1, 3, 7)
        rm = gmres(A, b; maxiter=m)
        @test rm.iters == m
    end
    @test gmres(A, b; maxiter=0).iters == 0

    # absolute tolerance
    bs = 1e3 .* b
    ra = gmres(A, bs; tol=1e-2, abstol=true)
    @test norm(A * ra.x - bs) <= 1e-2 * (1 + 1e-8)
    rr = gmres(A, bs; tol=1e-2)  # relative: much looser
    @test rr.iters < ra.iters

    # classical Gram-Schmidt (+ reorthogonalization)
    for (gs, nr) in ((:cgs, 0), (:cgs, 1), (:mgs, 1))
        rc = gmres(A, b; orthogonalization=gs, reorth=nr)
        @test norm(rc.x - xref) <= 1e-8 * norm(xref)
        @test abs(rc.iters - r.iters) <= 1
    end

    # initial guess
    @test gmres(A, b; x0=xref).iters == 0
    r0 = gmres(A, b; x0=xref .+ 1e-6 .* gmres_rand(3, n))
    @test r0.iters < r.iters
    @test norm(r0.x - xref) <= 1e-8 * norm(xref)

    # zero RHS
    rz = gmres(A, zeros(n))
    @test rz.iters == 0 && all(iszero, rz.x)
    @test gmres(zeros(0, 0), Float64[]).x == Float64[]

    # errors in the operator propagate with their original type; the solver remains usable
    @test_throws GMRESTestError gmres((y, v) -> throw(GMRESTestError()), b)
    k = Ref(0)
    @test_throws DomainError gmres((y, v) -> (k[] += 1; k[] > 3 && throw(DomainError(k[])); mul!(y, A, v)), b)
    @test k[] == 4
    @test gmres(A, b).iters == r.iters

    # argument checking
    @test_throws DimensionMismatch gmres(A, ones(n + 1))
    @test_throws DimensionMismatch gmres(ones(n, n + 1), ones(n))
    @test_throws DimensionMismatch gmres(A, b; x0=ones(2))
    @test_throws ArgumentError gmres(A, b; orthogonalization=:householder)
    @test_throws ArgumentError gmres(A, b; reorth=-1)
    @test_throws ArgumentError gmres(A, b; maxiter=-2)
end

@testset "KrylovPrecond" begin
    n = 80
    # A = I + low-rank with decaying singular values (as in the SCTL tutorial)
    A = Matrix(1.0I, n, n)
    for k in 0:n-1
        A .+= gmres_rand(100 + 2k, n) * gmres_rand(101 + 2k, n)' .* (12 * 0.7^k / n)
    end
    P = KrylovPrecond()
    @test size(P) == (0, 0)
    @test rank(P) == 0
    v = gmres_rand(6, n)
    @test P * v == v  # empty preconditioner is the identity
    @test sprint(show, P) == "KrylovPrecond(size=0, rank=0)"

    b1 = gmres_rand(7, n)
    r1 = gmres(A, b1; precond=P)
    @test norm(A * r1.x - b1) <= 1e-9 * norm(b1)
    @test size(P) == (n, n) && size(P, 1) == n
    @test rank(P) == r1.iters

    # second solve with a nearby RHS: fewer iterations, still the right answer
    b2 = b1 .+ 1e-2 .* gmres_rand(8, n)
    r2 = gmres(A, b2; precond=P)
    @test r2.iters < r1.iters
    @test norm(r2.x - A \ b2) <= 1e-8 * norm(A \ b2)
    @test rank(P) == r1.iters + r2.iters
    # the same RHS without the preconditioner takes longer
    @test gmres(A, b2).iters > r2.iters

    # applying the (now non-trivial) preconditioner: in place and out of place agree
    w = gmres_rand(9, n)
    Pw = P * w
    @test Pw != w
    w2 = copy(w)
    @test apply!(P, w2) === w2
    @test w2 == Pw
    wv = view(copy(w), :)  # non-Vector AbstractVector
    apply!(P, wv)
    @test wv == Pw
    @test_throws DimensionMismatch apply!(P, ones(n + 1))

    # reused with a different system size: reset by the next solve
    rs = gmres(Matrix(2.0I, 5, 5), ones(5); precond=P)
    @test rs.x ≈ fill(0.5, 5)
    @test size(P) == (5, 5)

    close(P)
    close(P)  # idempotent
    @test_throws ArgumentError size(P)
    @test_throws ArgumentError gmres(A, b1; precond=P)
    @test sprint(show, P) == "KrylovPrecond(<freed>)"
end

@testset "gmres: long vectors (zero-copy callback path)" begin
    n = 2 * SCTL.CALLBACK_COPY_MAX + 7
    d = collect(range(1.0, 2.0; length=n))
    b = sin.(1:n)
    seen = Ref(false)
    sol = gmres((y, x) -> (seen[] = (length(x) == n); y .= d .* x), b; tol=1e-12)
    @test seen[]
    @test sol.x ≈ b ./ d rtol = 1e-10
    # the matrix form goes through the same path
    A = Matrix(Diagonal(d)) + 0.01 * ones(n, n) / n
    @test gmres(A, b; tol=1e-12).x ≈ A \ b rtol = 1e-9
    @test_throws DomainError gmres((y, x) -> throw(DomainError(1)), b)
end
