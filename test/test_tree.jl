using SCTL, Test, LinearAlgebra

# clustered points in [0,1)^DIM (dense near the center), deterministic
function cluster_pts(DIM, N; seed=1)
    X = Matrix{Float64}(undef, DIM, N)
    s = UInt64(seed)
    for i in eachindex(X)
        s = s * 6364136223846793005 + 1442695040888963407   # LCG
        u = (s >> 11) / 2.0^53
        X[i] = clamp(((2u - 1)^3) / 2 + 0.5, 0.0, prevfloat(1.0))
    end
    return X
end

node_offsets(cnt) = cumsum([0; cnt[1:end-1]])

# check the tree structure against the geometry; returns the node info
function check_tree(tree::PtTree{DIM}, X, M; periodic=false) where {DIM}
    N = size(X, 2)
    ni = node_info(tree)
    n = num_nodes(tree)
    @test size(ni.coord) == (DIM, n) && length(ni.depth) == n
    @test size(ni.children) == (2^DIM, n) && size(ni.neighbors) == (3^DIM, n)
    @test owned_range(tree) == 1:n
    @test !any(ni.ghost)
    @test ni.depth[1] == 0 && ni.parent[1] == 0 && all(ni.coord[:, 1] .== 0)
    s = 2.0 .^ -ni.depth   # box sizes

    # particle counts: only leaves hold points, at most M each, N in total
    data, cnt = get_data(tree, "pt")
    @test size(data) == (DIM, N) && length(cnt) == n
    @test sum(cnt) == N
    @test all(cnt[.!ni.leaf] .== 0)
    # (more than M points only where they coincide down to the maximum depth, 20)
    @test all(i -> !ni.leaf[i] || cnt[i] <= M || ni.depth[i] == 20, 1:n)
    # every leaf box contains its points, and the stored points are a permutation of X
    offs = node_offsets(cnt)
    ok = true
    for i in 1:n, j in offs[i]+1:offs[i]+cnt[i]
        ok &= all(ni.coord[:, i] .<= data[:, j] .< ni.coord[:, i] .+ s[i])
    end
    @test ok
    @test sortslices(data; dims=2) == sortslices(X; dims=2)

    # parent / children consistency
    ok = true
    for i in 1:n
        if ni.leaf[i]
            ok &= all(ni.children[:, i] .== 0)
        else
            for c in 1:2^DIM
                k = ni.children[c, i]
                bits = [((c - 1) >> (d - 1)) & 1 for d in 1:DIM]
                ok &= k > 0 && ni.parent[k] == i && ni.depth[k] == ni.depth[i] + 1 &&
                      ni.coord[:, k] == ni.coord[:, i] .+ bits .* (s[i] / 2)
            end
        end
        i > 1 && (ok &= ni.parent[i] > 0 && i in ni.children[:, ni.parent[i]])
    end
    @test ok
    @test count(ni.leaf) == 1 + (n - 1) * (2^DIM - 1) ÷ 2^DIM   # a full 2^DIM-ary tree

    # neighbors: same depth, geometrically adjacent (wrapping on periodic axes), symmetric, and complete
    per = periodic isa Bool ? fill(periodic, DIM) : collect(periodic)
    key(c, d) = (ntuple(k -> c[k], DIM), d)
    index = Dict(key(view(ni.coord, :, i), ni.depth[i]) => i for i in 1:n)
    ok = true
    center = (3^DIM + 1) ÷ 2
    for i in 1:n
        ok &= ni.neighbors[center, i] == i
        for j in 1:3^DIM
            o = [((j - 1) ÷ 3^(d - 1)) % 3 - 1 for d in 1:DIM]
            c = ni.coord[:, i] .+ o .* s[i]
            inside = all(per .| (0 .<= c .< 1))
            c = mod.(c, 1.0)
            expected = inside ? get(index, key(c, ni.depth[i]), 0) : 0
            k = ni.neighbors[j, i]
            ok &= k == expected
            k > 0 && (ok &= ni.neighbors[3^DIM + 1 - j, k] == i)  # symmetric
        end
    end
    @test ok
    return ni
end

# touching leaves (closed boxes intersect, with wrap-around on periodic axes) differ by at most one level
function max_level_jump(ni, DIM; periodic=false)
    leaves = findall(ni.leaf)
    s = 2.0 .^ -ni.depth
    jump = 0
    for a in leaves, b in leaves
        a < b || continue
        touch = true
        for d in 1:DIM
            lo1, hi1 = ni.coord[d, a], ni.coord[d, a] + s[a]
            lo2, hi2 = ni.coord[d, b], ni.coord[d, b] + s[b]
            t = lo1 <= hi2 && lo2 <= hi1
            periodic && (t |= (hi1 == 1 && lo2 == 0) || (hi2 == 1 && lo1 == 0))
            touch &= t
        end
        touch && (jump = max(jump, abs(ni.depth[a] - ni.depth[b])))
    end
    return jump
end

@testset "PtTree{$DIM}" for DIM in (1, 2, 3)
    N = DIM == 1 ? 300 : 1000
    M = 8
    X = cluster_pts(DIM, N; seed=DIM)
    f = vcat(sum(X; dims=1), (1:N)')             # 2 × N data, second row = original index
    tree = PtTree{DIM}()
    @test num_nodes(tree) == 1
    add_particles!(tree, "pt", X)
    add_particle_data!(tree, "f", "pt", f)
    add_particle_data!(tree, "g", "pt", X[1, :])  # vector: dof = 1
    update_refinement!(tree, X; max_pts=M)
    ni = check_tree(tree, X, M)
    @test num_nodes(tree) > 1

    # particle data round-trip in the original order, and node-order data consistent with the coordinates
    @test get_particle_data(tree, "pt") == X
    @test get_particle_data(tree, "f") == f
    @test get_particle_data(tree, "g") == reshape(X[1, :], 1, N)
    pts, cnt = get_data(tree, "pt")
    fd, cntf = get_data(tree, "f")
    @test cntf == cnt
    @test pts == X[:, Int.(fd[2, :])]             # node order is a permutation of the input order

    # refine again: data follow the particles
    update_refinement!(tree, X; max_pts=3)
    check_tree(tree, X, 3)
    @test get_particle_data(tree, "f") == f
    @test get_data(tree, "f")[1][2, :] != 1:N || N < 4  # really reordered

    # allocate + fill in node order + read back in the original order
    add_particle_data!(tree, "h", "pt", 2)
    pts, cnt = get_data(tree, "pt")
    set_data!(tree, "h", vcat(pts[1:1, :] .* 2, pts[end:end, :] .+ 1))
    @test get_particle_data(tree, "h") ≈ vcat(X[1:1, :] .* 2, X[end:end, :] .+ 1)
    @test_throws DimensionMismatch set_data!(tree, "h", zeros(2, N + 1))

    # 2:1 balance
    if DIM >= 2
        Xb = X[:, 1:300]
        tb = PtTree{DIM}()
        add_particles!(tb, "pt", Xb)
        update_refinement!(tb, Xb; max_pts=1)
        @test max_level_jump(node_info(tb), DIM) > 1     # clustered points: unbalanced without balance21
        update_refinement!(tb, Xb; max_pts=1, balance21=true)
        nb = check_tree(tb, Xb, 1)
        @test max_level_jump(nb, DIM) <= 1
        @test get_particle_data(tb, "pt") == Xb
    end

    # periodicity: neighbors wrap around; balance across the periodic boundary
    for periodic in (true, ntuple(d -> isodd(d), DIM))
        tp = PtTree{DIM}()
        add_particles!(tp, "pt", X)
        update_refinement!(tp, X; max_pts=M, balance21=DIM >= 2, periodic=periodic)
        np_ = check_tree(tp, X, M; periodic=periodic)
        periodic === true && DIM >= 2 && @test max_level_jump(np_, DIM; periodic=true) <= 1
    end
    if DIM == 2   # a box on the x = 0 face sees across to x = 1 only when x is periodic
        tq = PtTree{2}()
        add_particles!(tq, "pt", X)
        update_refinement!(tq, X; max_pts=M, periodic=(true, false))
        nq = node_info(tq)
        i = findfirst(i -> nq.depth[i] == 1 && nq.coord[:, i] == [0.0, 0.0], 1:num_nodes(tq))
        k = nq.neighbors[1 + 0 + 3 * 1, i]           # offset (-1, 0)
        @test k > 0 && nq.coord[:, k] == [0.5, 0.0]
        @test nq.neighbors[1 + 1 + 3 * 0, i] == 0    # offset (0, -1): y not periodic
    end

    # VTK output
    dir = mktempdir()
    files = write_tree_vtk(tree, joinpath(dir, "tree"))
    @test all(isfile, files) && endswith(files[1], ".vtu")
    @test occursin("UnstructuredGrid", read(files[1], String))
    @test all(isfile, write_particle_vtk(tree, joinpath(dir, "pts"), "f"))
    @test all(isfile, write_particle_vtk(tree, joinpath(dir, "pts0"), "pt"; show_ghost=true))

    # deleting data, replacing groups
    delete_particle_data!(tree, "g")
    @test_throws ArgumentError get_particle_data(tree, "g")
    Y = cluster_pts(DIM, 40; seed=7)
    add_particles!(tree, "pt", Y)                 # replacing a group empties its data sets
    @test get_particle_data(tree, "pt") == Y
    @test size(get_particle_data(tree, "f")) == (0, 40)
    add_particle_data!(tree, "f", "pt", 2 .* Y)
    update_refinement!(tree, Y; max_pts=4)
    @test get_particle_data(tree, "f") == 2 .* Y
    add_particles!(tree, "src", X)                # a second group on the same tree
    update_refinement!(tree, hcat(X, Y); max_pts=10)
    @test sum(get_data(tree, "src")[2]) == N && sum(get_data(tree, "pt")[2]) == 40
    @test get_particle_data(tree, "src") == X && get_particle_data(tree, "f") == 2 .* Y
    delete_particle_data!(tree, "pt")             # deleting a group deletes its data sets
    @test_throws ArgumentError get_particle_data(tree, "f")
    @test_throws ArgumentError get_data(tree, "pt")
    @test get_particle_data(tree, "src") == X
end

@testset "PtTree errors" begin
    tree = PtTree(2)
    X = cluster_pts(2, 50)
    @test_throws ArgumentError add_particles!(tree, "pt", hcat(X, [1.0, 0.5]))   # outside [0,1)
    @test_throws ArgumentError add_particles!(tree, "pt", hcat(X, [-0.1, 0.5]))
    @test_throws ArgumentError add_particles!(tree, "pt", hcat(X, [NaN, 0.5]))
    @test_throws DimensionMismatch add_particles!(tree, "pt", rand(3, 4))
    @test_throws DimensionMismatch add_particles!(tree, "pt", rand(5))
    add_particles!(tree, "pt", vec(X))                                           # flat AoS vector
    @test get_particle_data(tree, "pt") == X
    @test_throws ArgumentError add_particle_data!(tree, "f", "nope", ones(50))
    @test_throws DimensionMismatch add_particle_data!(tree, "f", "pt", ones(49))
    @test_throws ArgumentError add_particle_data!(tree, "pt", "pt", ones(50))
    add_particle_data!(tree, "f", "pt", ones(50))
    @test_throws ArgumentError add_particles!(tree, "f", X)
    @test_throws ArgumentError add_particles!(tree, "", X)
    @test_throws ArgumentError update_refinement!(tree, X; max_pts=0)
    @test_throws ArgumentError update_refinement!(tree, zeros(2, 0))
    @test_throws DimensionMismatch update_refinement!(tree, X; periodic=(true, false, true))
    @test_throws ArgumentError get_particle_data(tree, "nope")
    @test_throws ArgumentError delete_particle_data!(tree, "nope")
    @test_throws ArgumentError PtTree(4)
    close(tree)
    @test_throws ArgumentError num_nodes(tree)
    close(tree)
end
