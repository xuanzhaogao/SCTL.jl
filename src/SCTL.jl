"""
    SCTL

Julia wrapper for [SCTL](https://github.com/dmalhotra/SCTL), the Scientific Computing Template Library
(MPI-free build, `Float64`).
"""
module SCTL

using LinearAlgebra

include("lib.jl")
include("quadrule.jl")
include("sph_harm.jl")
include("kernels.jl")
include("fmm.jl")
include("bie.jl")
include("gmres.jl")
include("sdc.jl")
include("tree.jl")
include("fft.jl")

function __init__()
    _check_lib()
    _init_pvfmm_cache()
    # Idle libgomp threads spin after every parallel region by default, which competes with Julia's own threads.
    # Must be set before libsctljl (and with it libgomp) is loaded by the first ccall.
    haskey(ENV, "OMP_WAIT_POLICY") || (ENV["OMP_WAIT_POLICY"] = "PASSIVE")
    _init_blas()   # SCTL's BLAS/LAPACK -> Julia's (libblastrampoline); see _serial_blas
end

end # module SCTL
