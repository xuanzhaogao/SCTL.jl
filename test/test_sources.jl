# deps/sources.toml (the downloads of deps/build.jl) must pin the same commits as the git submodules
root = normpath(joinpath(@__DIR__, ".."))
srcs = Base.parsed_toml(joinpath(root, "deps", "sources.toml"))
if isdir(joinpath(root, ".git")) || isfile(joinpath(root, ".git"))
    for (name, path) in (("SCTL", "deps/SCTL"), ("pvfmm", "deps/pvfmm"))
        # the commit recorded for the submodule in the index (also valid when it is not checked out)
        line = split(readchomp(`git -C $root ls-files -s $path`))
        @test line[1] == "160000"   # a gitlink
        @test line[2] == srcs[name]["commit"]
    end
else
    @test_skip "not a git checkout"
end
