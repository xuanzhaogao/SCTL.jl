# Compiles the C-ABI shim (csrc/) against the header-only SCTL library into deps/usr/lib/libsctljl.so.
#
# Environment variables:
#   SCTL_ROOT          path to the SCTL checkout (default: the deps/SCTL git submodule, then ../SCTL next to this package)
#   CXX                C++17 compiler with OpenMP (default: g++)
#   SCTL_JL_MARCH      architecture flags (default: -march=native); e.g. "-march=x86-64-v3" for a portable build
#   SCTL_JL_CXXFLAGS   extra compiler flags, e.g. "-DSCTL_HAVE_FFTW"
#   SCTL_JL_LDFLAGS    extra linker flags, e.g. "-lfftw3"

const PKG_ROOT = normpath(joinpath(@__DIR__, ".."))

function find_sctl_root()
    candidates = String[]
    haskey(ENV, "SCTL_ROOT") && push!(candidates, ENV["SCTL_ROOT"])
    push!(candidates, joinpath(PKG_ROOT, "deps", "SCTL"), joinpath(PKG_ROOT, "..", "SCTL"))
    for c in candidates
        isfile(joinpath(c, "include", "sctl.hpp")) && return normpath(abspath(c))
    end
    error("SCTL headers not found (looked in: $(join(candidates, ", "))). " *
          "Run `git submodule update --init` in $PKG_ROOT, or set ENV[\"SCTL_ROOT\"] to an SCTL checkout.")
end

function build()
    sctl_root = find_sctl_root()
    args = ["SCTL_ROOT=$sctl_root", "BUILDDIR=$(joinpath(PKG_ROOT, "deps", "usr"))",
            "CXX=$(get(ENV, "CXX", "g++"))",
            "MARCH=$(get(ENV, "SCTL_JL_MARCH", "-march=native"))",
            "EXTRA_CXXFLAGS=$(get(ENV, "SCTL_JL_CXXFLAGS", ""))",
            "EXTRA_LDFLAGS=$(get(ENV, "SCTL_JL_LDFLAGS", ""))"]
    @info "Building libsctljl" sctl_root
    run(`make -C $(joinpath(PKG_ROOT, "csrc")) -j$(min(Sys.CPU_THREADS, 16)) $args`)
end

build()
