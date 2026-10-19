# Near-field pieces: far-field nodes, near list, element quadratures and the assembled correction reproduce
# BoundaryIntegralOp
using LinearAlgebra, SparseArrays

# surface of the cube [-1,1]^3, n x n flat elements of order q per face
function _cube_elems(n, q)
    QuadElemList(q, 6n^2) do e, u, v
        f, r = divrem(e - 1, n^2)
        i, j = divrem(r, n)
        a, b = 2 * (i + u) / n - 1, 2 * (j + v) / n - 1
        # outward normals: (a, b, ±1) etc. with ∂X/∂u × ∂X/∂v pointing out
        f == 0 ? [a, b, 1.0] : f == 1 ? [b, a, -1.0] : f == 2 ? [1.0, a, b] :
        f == 3 ? [-1.0, b, a] : f == 4 ? [b, 1.0, a] : [a, -1.0, b]
    end
end

@testset "near-field pieces" begin
    el = _cube_elems(2, 8)
    nn = order(el)^2
    ff = far_field_nodes(el; tol = 1e-10)
    X, N = node_coord(el)
    @test ff.X == X && ff.normal == N
    @test ff.weights ≈ node_weights(el)
    @test sum(ff.weights) ≈ 24
    @test all(>(0), ff.radius)
    @test all(ff.normal[:, 1] .≈ [0, 0, 1])          # outward on the top face
    @test minimum(far_field_nodes(el; tol = 1e-4).radius) < minimum(ff.radius)   # looser: smaller radius

    # near list: SCTL's octree search agrees with brute force, and the two forms agree
    nl = near_list(el; tol = 1e-10)
    @test nl isa NearList && SCTL.nelem(nl) == nelem(el) && nl.ntrg == nnodes(el)
    Xt = [ff.X 1.05 .* ff.X[:, 1:7:end] [3.0, 0.2, 0.1]]
    nl2 = near_list(ff.X, ff.radius, nn, Xt)
    @test near_list(el, Xt; tol = 1e-10).trg == nl2.trg
    brute = [[t for t in 1:size(Xt, 2) if any(i -> norm(Xt[:, t] - ff.X[:, i]) < ff.radius[i], (e-1)*nn+1:e*nn)]
             for e in 1:nelem(el)]
    @test all(near_targets(nl2, e) == brute[e] for e in 1:nelem(el))
    @test all(e -> size(Xt, 2) ∉ near_targets(nl2, e), 1:nelem(el))     # the far target is near nothing
    @test npairs(nl2) == sum(length, brute)
    @test_throws DimensionMismatch near_list(ff.X, ff.radius, 7, Xt)
    @test_throws ArgumentError NearList([1, 3], [1], 5)

    q = ElemQuadrature(Laplace3D_DxU())
    blocks = near_blocks(q, el, nl2, Xt; tol = 1e-10)
    @test length(blocks) == nelem(el) && all(size(blocks[e]) == (length(near_targets(nl2, e)), nn) for e in 1:nelem(el))
    # an off-surface target's row is near_matrix's; an on-node target's row is the self-interaction row
    e = 1
    ts = near_targets(nl2, e)
    k = findfirst(>(nnodes(el)), ts)
    @test blocks[e][k, :] ≈ vec(near_matrix(q, el, e, Xt[:, ts[k]]; tol = 1e-10)) rtol = 1e-12
    S = self_matrices(q, el; tol = 1e-10)
    @test size(S) == (nn, nn, nelem(el))
    k = findfirst(==(3), ts)                                  # node 3 is a node of element 1
    @test blocks[e][k, :] == S[3, :, e]
    close(q)
    @test_throws ArgumentError near_matrix(q, el, 1, Xt[:, 1:1])
    @test_throws ArgumentError ElemQuadrature(Laplace3D_Fxd2U())

    # node rule + assembled correction == BoundaryIntegralOp, on and off the surface
    for (ker, dot) in ((Laplace3D_FxU(), false), (Laplace3D_DxU(), false), (Laplace3D_FxdU(), true),
                       (Stokes3D_DxU(), false), (Helmholtz3D_FxU(1.3), false))
        F = randn(src_dim(ker) * nnodes(el))
        for targets in (nothing, Xt[:, nnodes(el)+1:end])
            Nt = dot ? (targets === nothing ? nothing : repeat([0.0, 0.0, 1.0], 1, size(targets, 2))) : nothing
            C, nlc = near_correction(ker, el; targets = targets, normals = Nt, trg_normal_dot_prod = dot, tol = 1e-10)
            T = targets === nothing ? ff.X : targets
            K = node_rule_block(ker, ff.X, ff.normal, ff.weights, T; normals = dot ? (Nt === nothing ? ff.normal : Nt) : nothing)
            op = BoundaryIntegralOp(ker; trg_normal_dot_prod = dot, accuracy = 1e-10)
            add_elem_list!(op, el)
            targets === nothing || set_target_coord!(op, targets)
            dot && targets !== nothing && set_target_normal!(op, Nt)
            Uref = compute_potential(op, F)
            @test K * F + C * F ≈ Uref rtol = 1e-12
            @test nnz(C) > 0 && size(C) == (length(Uref), length(F))
        end
    end

    # assembly of user blocks: subtracting the blocks themselves leaves zeros
    Bs = [rand(length(near_targets(nl2, e)), nn) for e in 1:nelem(el)]
    Z = assemble_near_correction(nl2, Bs, nn; subtract = Bs)
    @test iszero(nonzeros(Z)) && size(Z) == (size(Xt, 2), nnodes(el))
    A = assemble_near_correction(nl2, Bs, nn)
    @test A[near_targets(nl2, 2), nn+1:2nn] == Bs[2]
    @test_throws DimensionMismatch assemble_near_correction(nl2, Bs[1:end-1], nn)
end
