# Builds the documentation: julia --project=docs docs/make.jl
#
# The package must be built first (Pkg.build("SCTL")): the pages run their examples.
# For a quick partial build of some pages (missing docstrings / cross-references then only warn):
#   SCTL_DOCS_ONLY=man/fft.md,man/tree.md SCTL_DOCS_BUILD=/some/dir julia --project=docs docs/make.jl
using Documenter, SCTL

const PAGES = [
    "Home" => "index.md",
    "Manual" => [
        "Conventions" => "man/conventions.md",
        "Quadrature and interpolation" => "man/quadrature.md",
        "GMRES" => "man/gmres.md",
        "SDC ODE solver" => "man/sdc.md",
        "Kernel functions" => "man/kernels.md",
        "Fast multipole method" => "man/fmm.md",
        "Boundary integral operators" => "man/bie.md",
        "Near-field pieces" => "man/bie_pieces.md",
        "Spherical harmonics" => "man/sph_harm.md",
        "Point trees" => "man/tree.md",
        "FFT" => "man/fft.md",
        "Performance" => "man/performance.md",
    ],
    "Library index" => "api.md",
]

const ONLY = filter(!isempty, split(get(ENV, "SCTL_DOCS_ONLY", ""), ','))

function select(pages, keep)
    out = Any[]
    for p in pages
        v = p.second
        if v isa AbstractVector
            sub = select(v, keep)
            isempty(sub) || push!(out, p.first => sub)
        elseif v in keep || v == "index.md"
            push!(out, p)
        end
    end
    return out
end

const partial = !isempty(ONLY)

# Documenter renders every file under the source directory, so a partial build gets a copy holding just the pages it
# needs (and their examples run in isolation from pages being edited elsewhere).
function source_dir()
    partial || return joinpath(@__DIR__, "src")
    dir = mktempdir()
    for f in ("index.md", ONLY...)
        dst = joinpath(dir, f)
        mkpath(dirname(dst))
        cp(joinpath(@__DIR__, "src", f), dst)
    end
    return dir
end

makedocs(;
    sitename = "SCTL.jl",
    modules = [SCTL],
    authors = "Xuanzhao Gao",
    repo = Remotes.GitHub("xuanzhaogao", "SCTL.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://xuanzhaogao.github.io/SCTL.jl/dev",
        edit_link = partial ? nothing : "main",   # copied pages (partial build) have no git source
        size_threshold_warn = 200 * 1024,
    ),
    pages = partial ? select(PAGES, ONLY) : PAGES,
    source = source_dir(),
    build = get(ENV, "SCTL_DOCS_BUILD", joinpath(@__DIR__, "build")),
    checkdocs = :exports,
    warnonly = partial ? [:missing_docs, :cross_references] : Symbol[],
)

if !partial && get(ENV, "CI", "false") == "true"
    deploydocs(; repo = "github.com/xuanzhaogao/SCTL.jl.git", devbranch = "main", push_preview = false)
end
