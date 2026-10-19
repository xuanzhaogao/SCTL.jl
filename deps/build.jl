# Compiles the C-ABI shim (csrc/) against the header-only SCTL library into deps/usr/lib/libsctljl.so.
#
# Environment variables:
#   SCTL_ROOT          path to the SCTL checkout (default: the deps/SCTL git submodule, then ../SCTL next to this package)
#   PVFMM_ROOT         path to the PVFMM checkout (default: the deps/pvfmm git submodule)
#   SCTL_JL_PVFMM      "0" builds without PVFMM (ParticleFMM and BoundaryIntegralOp then use direct sums)
#   SCTL_JL_BLAS       "0" builds without BLAS/LAPACK (default: OpenBLAS32_jll, an LP64 OpenBLAS with LAPACK)
#   CXX                C++17 compiler with OpenMP (default: g++)
#   SCTL_JL_MARCH      architecture flags (default: -march=native); e.g. "-march=x86-64-v3" for a portable build
#   SCTL_JL_CXXFLAGS   extra compiler flags, e.g. "-DSCTL_HAVE_FFTW"
#   SCTL_JL_LDFLAGS    extra linker flags, e.g. "-lfftw3"

const PKG_ROOT = normpath(joinpath(@__DIR__, ".."))

enabled(var) = get(ENV, var, "1") ∉ ("0", "false", "no", "off")

if enabled("SCTL_JL_BLAS")
    using OpenBLAS32_jll
end

function find_root(var, default, probe, what)
    candidates = String[]
    haskey(ENV, var) && push!(candidates, ENV[var])
    append!(candidates, default)
    for c in candidates
        isfile(joinpath(c, probe)) && return normpath(abspath(c))
    end
    error("$what not found (looked in: $(join(candidates, ", "))). " *
          "Run `git submodule update --init` in $PKG_ROOT, or set ENV[\"$var\"].")
end

function build()
    sctl_root = find_root("SCTL_ROOT", [joinpath(PKG_ROOT, "deps", "SCTL"), joinpath(PKG_ROOT, "..", "SCTL")],
                          joinpath("include", "sctl.hpp"), "SCTL headers")

    pvfmm_root = ""
    if enabled("SCTL_JL_PVFMM")
        pvfmm_root = find_root("PVFMM_ROOT", [joinpath(PKG_ROOT, "deps", "pvfmm")], joinpath("include", "pvfmm.hpp"),
                               "PVFMM headers")
        # SCTL versions before the ParticleFMM compatibility checks crash for kernels PVFMM cannot evaluate
        # (e.g. Helmholtz) once there are 40000 targets or more
        if !occursin("PVFMMCompatible", read(joinpath(sctl_root, "include", "sctl", "fmm-wrapper.txx"), String))
            @warn "This SCTL version lacks ParticleFMM's PVFMM compatibility checks; building without PVFMM" sctl_root
            pvfmm_root = ""
        end
    end

    blas_libdir = ""
    if enabled("SCTL_JL_BLAS")
        blas_libdir = dirname(OpenBLAS32_jll.libopenblas_path)
    end

    args = ["SCTL_ROOT=$sctl_root", "PVFMM_ROOT=$pvfmm_root", "BLAS_LIBDIR=$blas_libdir",
            "BUILDDIR=$(joinpath(PKG_ROOT, "deps", "usr"))",
            "CXX=$(get(ENV, "CXX", "g++"))",
            "MARCH=$(get(ENV, "SCTL_JL_MARCH", "-march=native"))",
            "EXTRA_CXXFLAGS=$(get(ENV, "SCTL_JL_CXXFLAGS", ""))",
            "EXTRA_LDFLAGS=$(get(ENV, "SCTL_JL_LDFLAGS", ""))"]
    @info "Building libsctljl" sctl_root pvfmm_root blas_libdir
    run(`make -C $(joinpath(PKG_ROOT, "csrc")) -j$(min(Sys.CPU_THREADS, 16)) $args`)
end

build()
