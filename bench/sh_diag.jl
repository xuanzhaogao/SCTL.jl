# grid2shc(p=127) three ways inside one Julia process: through SCTL.jl, through a raw ccall into the shim, and as a
# plain C++ loop (bench/sh_loop.cpp built as a shared library). Usage: julia --project=.. sh_diag.jl DATA_DIR LIBLOOP
using SCTL, Libdl, Statistics, Printf
X = reinterpret(Float64, read(joinpath(ARGS[1], "sh_grid.bin"))) |> collect
p = 127; Nt = p + 1; Np = 2p + 2
loop = dlsym(dlopen(ARGS[2]), :sh_loop)
S = grid2shc(X, Nt, Np, p)
raw() = ccall((:sctljl_sph_grid2shc, SCTL.libsctljl), Cint, (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Int64, Int64, Int64, Cint),
              S, length(S), X, length(X), Nt, Np, p, Cint(1))
med(g) = (for _ in 1:40; g(); end; median([(@elapsed g()) for _ in 1:400]))
for r in 1:3
    @printf("  in julia: wrapper %.3e   raw ccall %.3e   C++ loop %.3e\n",
            med(() -> grid2shc(X, Nt, Np, p)), med(raw), ccall(loop, Cdouble, (Ptr{Float64}, Clong, Cint), X, length(X), 400))
end
