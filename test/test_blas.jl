# BLAS/LAPACK of the default build: Julia's own, through libblastrampoline; library calls that can reach BLAS run with
# Julia's BLAS limited to one thread and restore the user's thread count
using LinearAlgebra
if SCTL.has_feature(:BLAS_LBT)
    @test SCTL.has_feature(:BLAS) && SCTL.has_feature(:LAPACK)
    nt0 = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(3)
        seen = Ref(0)
        SCTL._serial_blas() do
            seen[] = BLAS.get_num_threads()
            SCTL._serial_blas(() -> nothing)            # nested: still one save/restore
            @test BLAS.get_num_threads() == 1
        end
        @test seen[] == 1
        @test BLAS.get_num_threads() == 3
        # restored after an error as well
        @test_throws ErrorException SCTL._serial_blas(() -> error("boom"))
        @test BLAS.get_num_threads() == 3
        # a library call that uses BLAS (SVD in the quadrature construction, GEMM in the transforms)
        x, w, _ = interp_quad_rule(x -> hcat(one.(x), x, x .^ 2, log.(x)), 0.0, 1.0; eps = 1e-12)
        @test sum(w) ≈ 1 atol = 1e-8
        @test sum(w .* log.(x)) ≈ -1 atol = 1e-8
        @test BLAS.get_num_threads() == 3
    finally
        BLAS.set_num_threads(nt0)
    end
else
    @test_skip "library built without BLAS from libblastrampoline"
end
