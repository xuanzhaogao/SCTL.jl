"""
    SCTL

Julia wrapper for [SCTL](https://github.com/dmalhotra/SCTL), the Scientific Computing Template Library
(MPI-free build, `Float64`).
"""
module SCTL

using LinearAlgebra
using OpenBLAS32_jll   # BLAS/LAPACK of libsctljl (LP64); loaded before the first ccall

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
    # SCTL calls BLAS from inside OpenMP parallel regions; OpenBLAS32_jll's own (pthreads) thread pool there can
    # deadlock ("Detect OpenMP Loop and this application may hang"). Parallelism comes from SCTL's OpenMP instead.
    # This is libsctljl's OpenBLAS only; Julia's LinearAlgebra BLAS is a different library.
    if !haskey(ENV, "OPENBLAS_NUM_THREADS")
        ccall((:openblas_set_num_threads, OpenBLAS32_jll.libopenblas), Cvoid, (Cint,), 1)
    end
end

end # module SCTL
