using SCTL, Test, LinearAlgebra

# naive multi-dimensional DFT of a Julia array along all its dimensions; sign -1 forward, +1 inverse
function naive_dft(x::AbstractArray, sign)
    sz = size(x)
    y = zeros(ComplexF64, sz)
    for k in CartesianIndices(sz), n in CartesianIndices(sz)
        ph = sum((k[d] - 1) * (n[d] - 1) / sz[d] for d in 1:length(sz))
        y[k] += x[n] * cispi(2 * sign * ph)
    end
    return y
end

# naive inverse real transform: input is the first n1÷2+1 rows of the (Hermitian) spectrum of a real array
function naive_irdft(X::AbstractArray, dims)
    full = zeros(ComplexF64, dims...)
    h = size(X, 1)
    for I in CartesianIndices(full)
        k = Tuple(I) .- 1
        if k[1] < h
            full[I] = X[I]
        else  # conjugate of the reflected index
            kr = map((ki, d) -> mod(-ki, d), k, Tuple(dims))
            full[I] = conj(X[CartesianIndex(kr .+ 1)])
        end
    end
    return real.(naive_dft(full, +1))
end

@testset "FFT" begin

    @testset "c2c / c2c_inv dims=$dims" for dims in ((8,), (13,), (12,), (6, 5), (7, 4), (3, 4, 5))
        N = prod(dims)
        x = randn(ComplexF64, dims...)
        p = FFTPlan(:c2c, dims)
        pi_ = FFTPlan(:c2c_inv, dims)
        @test dim(p, 1) == dim(p, 2) == N
        @test input_size(p) == output_size(p) == dims
        y = execute(p, x)
        @test size(y) == dims
        @test y ≈ naive_dft(x, -1) rtol = 1e-12
        @test execute(pi_, x) ≈ naive_dft(x, +1) rtol = 1e-12
        @test execute(pi_, y) ≈ N .* x rtol = 1e-12    # unnormalized
        # flat vector input works too
        @test vec(execute(p, vec(x))) ≈ vec(y)
    end

    @testset "r2c / c2r dims=$dims" for dims in ((8,), (9,), (1,), (6, 5), (7, 4), (5, 3, 4))
        N = prod(dims)
        x = randn(dims...)
        p = FFTPlan(:r2c, dims)
        q = FFTPlan(:c2r, dims)
        hs = (dims[1] ÷ 2 + 1, dims[2:end]...)
        @test output_size(p) == hs && input_size(q) == hs
        @test dim(p, 1) == N && dim(p, 2) == prod(hs)
        y = execute(p, x)
        @test size(y) == hs
        @test y ≈ naive_dft(x, -1)[1:hs[1], ntuple(_ -> :, length(dims) - 1)...] rtol = 1e-12
        z = execute(q, y)
        @test eltype(z) == Float64 && size(z) == dims
        @test z ≈ N .* x rtol = 1e-12
        # c2r against a naive inverse real DFT of a random Hermitian spectrum
        Y = execute(p, randn(dims...))
        @test execute(q, Y) ≈ naive_irdft(Y, dims) rtol = 1e-12
    end

    @testset "howmany" begin
        dims, h = (6, 3), 4
        x = randn(ComplexF64, dims..., h)
        p = FFTPlan(:c2c, dims; howmany=h, nthreads=2)
        @test dim(p, 1) == prod(dims) * h
        y = execute(p, x)
        @test size(y) == (dims..., h)
        for j in 1:h
            @test y[:, :, j] ≈ naive_dft(x[:, :, j], -1) rtol = 1e-12
        end
        xr = randn(7, h)
        pr = FFTPlan(:r2c, 7; howmany=h)
        yr = execute(pr, xr)
        @test size(yr) == (4, h)
        for j in 1:h
            @test yr[:, j] ≈ naive_dft(xr[:, j], -1)[1:4] rtol = 1e-12
        end
        @test execute(FFTPlan(:c2r, 7; howmany=h), yr) ≈ 7 .* xr rtol = 1e-12
    end

    @testset "execute! and errors" begin
        p = FFTPlan(:r2c, (8, 3))
        x = randn(8, 3)
        out = Vector{ComplexF64}(undef, 15)
        @test execute!(out, p, x) === out
        @test reshape(out, 5, 3) ≈ execute(p, x)
        @test execute(p, Float32.(x)) ≈ execute(p, Float64.(Float32.(x)))   # converted input
        @test_throws DimensionMismatch execute(p, randn(25))
        @test_throws DimensionMismatch execute!(Vector{ComplexF64}(undef, 14), p, x)
        @test_throws ArgumentError execute!(Vector{Float64}(undef, 30), p, x)
        @test_throws ArgumentError execute(p, complex.(x))
        q = FFTPlan(:c2c, 5)
        @test execute(q, [1, 0, 0, 0, 0]) ≈ ones(5)     # integer input is converted
        @test_throws DimensionMismatch execute(q, ones(ComplexF64, 6))
        @test_throws ArgumentError FFTPlan(:foo, 4)
        @test_throws ArgumentError FFTPlan(:c2c, (4, 0))
        @test_throws ArgumentError FFTPlan(:c2c, ())
        @test_throws ArgumentError FFTPlan(:c2c, 4; howmany=0)
        @test_throws ArgumentError dim(q, 0)
        close(q)
        @test_throws ArgumentError execute(q, ones(5))
        close(q)  # idempotent
        # the C API's own size check
        @test_throws SCTL.SCTLError SCTL._check(ccall((:sctljl_fft_execute, SCTL.libsctljl), Cint,
            (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64), p.ptr, zeros(3), 3, zeros(30), 30))
    end
end
