# Compiles the C-ABI shim (csrc/) against the header-only SCTL library into deps/usr/lib/libsctljl.so.
#
# Environment variables:
#   SCTL_ROOT          path to the SCTL checkout (default: the deps/SCTL git submodule, then ../SCTL next to this package,
#                      then a download of the commit pinned in deps/sources.toml)
#   PVFMM_ROOT         path to the PVFMM checkout (default: the deps/pvfmm git submodule, then a download as for SCTL)
#   SCTL_JL_PVFMM      "0" builds without PVFMM (ParticleFMM and BoundaryIntegralOp then use direct sums)
#   SCTL_JL_BLAS       "0" builds without BLAS/LAPACK (default: Julia's BLAS/LAPACK through libblastrampoline)
#   CXX                C++17 compiler with OpenMP (default: g++; on macOS the newest Homebrew g++-N)
#   SCTL_JL_MARCH      architecture flags (default: -march=native); e.g. "-march=x86-64-v3" for a portable build
#   SCTL_JL_CXXFLAGS   extra compiler flags, e.g. "-DSCTL_HAVE_FFTW"
#   SCTL_JL_LDFLAGS    extra linker flags, e.g. "-lfftw3"

const PKG_ROOT = normpath(joinpath(@__DIR__, ".."))

enabled(var) = get(ENV, var, "1") ∉ ("0", "false", "no", "off")

# Download the pinned sources of `name` (deps/sources.toml) into deps/downloads, for installations without the git
# submodules (a `Pkg.develop(url = ...)` clone does not have them).
function fetch_source(name, probe)
    src = Base.parsed_toml(joinpath(@__DIR__, "sources.toml"))[name]
    dest = joinpath(@__DIR__, "downloads", "$name-$(src["commit"])")
    isfile(joinpath(dest, probe)) && return dest
    url = "$(src["url"])/archive/$(src["commit"]).tar.gz"
    @info "Downloading $name (the git submodule is not checked out)" url
    tmp = mktempdir()
    tgz = download(url, joinpath(tmp, "src.tar.gz"))
    run(`tar -xzf $tgz -C $tmp`)
    top = only(filter(d -> isdir(joinpath(tmp, d)), readdir(tmp)))
    mkpath(dirname(dest))
    rm(dest; force = true, recursive = true)
    mv(joinpath(tmp, top), dest)
    isfile(joinpath(dest, probe)) || error("$url does not contain $probe")
    return dest
end

# The first of: ENV[var], the `default` directories, the download of `name` (unless ENV[var] is set).
function find_root(var, default, probe, what, name)
    candidates = String[]
    haskey(ENV, var) && push!(candidates, ENV[var])
    append!(candidates, default)
    for c in candidates
        isfile(joinpath(c, probe)) && return normpath(abspath(c))
    end
    haskey(ENV, var) && error("$what not found in ENV[\"$var\"] = $(ENV[var])")
    return normpath(fetch_source(name, probe))
end

# On macOS `g++` is Apple Clang, which has no -fopenmp: take the newest Homebrew GCC (g++-15, g++-14, ...) instead.
function default_cxx()
    Sys.isapple() || return "g++"
    for v in 20:-1:9
        Sys.which("g++-$v") === nothing || return "g++-$v"
    end
    @warn "No GCC found (`brew install gcc`); Apple Clang's g++ does not support -fopenmp. Set ENV[\"CXX\"] to a C++17 compiler with OpenMP."
    return "g++"
end

function build()
    sctl_root = find_root("SCTL_ROOT", [joinpath(PKG_ROOT, "deps", "SCTL"), joinpath(PKG_ROOT, "..", "SCTL")],
                          joinpath("include", "sctl.hpp"), "SCTL headers", "SCTL")

    pvfmm_root = ""
    if enabled("SCTL_JL_PVFMM")
        pvfmm_root = find_root("PVFMM_ROOT", [joinpath(PKG_ROOT, "deps", "pvfmm")], joinpath("include", "pvfmm.hpp"),
                               "PVFMM headers", "pvfmm")
        # SCTL versions before the ParticleFMM compatibility checks crash for kernels PVFMM cannot evaluate
        # (e.g. Helmholtz) once there are 40000 targets or more
        if !occursin("PVFMMCompatible", read(joinpath(sctl_root, "include", "sctl", "fmm-wrapper.txx"), String))
            @warn "This SCTL version lacks ParticleFMM's PVFMM compatibility checks; building without PVFMM" sctl_root
            pvfmm_root = ""
        end
    end

    blas = enabled("SCTL_JL_BLAS") ? "lbt" : ""

    cxx = get(ENV, "CXX", default_cxx())

    args = ["SCTL_ROOT=$sctl_root", "PVFMM_ROOT=$pvfmm_root", "BLAS=$blas", "BLAS_LIBDIR=",
            "BUILDDIR=$(joinpath(PKG_ROOT, "deps", "usr"))",
            "CXX=$cxx",
            "MARCH=$(get(ENV, "SCTL_JL_MARCH", "-march=native"))",
            "EXTRA_CXXFLAGS=$(get(ENV, "SCTL_JL_CXXFLAGS", ""))",
            "EXTRA_LDFLAGS=$(get(ENV, "SCTL_JL_LDFLAGS", ""))"]
    @info "Building libsctljl" sctl_root pvfmm_root blas cxx
    run(`make -C $(joinpath(PKG_ROOT, "csrc")) -j$(min(Sys.CPU_THREADS, 16)) $args`)
end

build()
