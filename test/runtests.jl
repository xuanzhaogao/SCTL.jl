using SCTL, Test

@testset "SCTL.jl" begin
    for name in ("quadrule", "sph_harm", "kernels", "bie", "gmres", "sdc", "tree", "fft")
        @testset "$name" begin
            include("test_$name.jl")
        end
    end
end
