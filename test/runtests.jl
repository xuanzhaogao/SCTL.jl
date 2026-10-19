using SCTL, Test

@testset "SCTL.jl" begin
    for name in ("sources", "blas", "quadrule", "sph_harm", "kernels", "fmm", "bie", "bie_pieces", "gmres", "sdc", "tree", "fft")
        @testset "$name" begin
            include("test_$name.jl")
        end
    end
end
