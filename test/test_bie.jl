using SCTL, Test, LinearAlgebra

# Point (u, v) of element e (1-based) of the cubed sphere of radius R with ppf × ppf elements per face, twisted about
# z (as in SCTL's test-quad-elem.cpp); the normal du × dv points outward. The twist only reparametrizes the sphere.
const _FACE_MAP = (((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
                   ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
                   ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
function sphere_point(e, u, v; ppf=2, R=1.0, twist=0.3)
    e0 = e - 1
    face = e0 ÷ ppf^2 + 1
    a = 2 * ((e0 ÷ ppf % ppf + u) / ppf) - 1
    b = 2 * ((e0 % ppf + v) / ppf) - 1
    p = [_FACE_MAP[face][k][1] + _FACE_MAP[face][k][2] * a + _FACE_MAP[face][k][3] * b for k in 1:3]
    x, y, z = p .* (R / norm(p))
    c, s = cos(twist * z), sin(twist * z)
    return [x * c + y * s, -x * s + y * c, z]
end
sphere(order; ppf=2, kw...) = QuadElemList((e, u, v) -> sphere_point(e, u, v; ppf=ppf, kw...), order, 6ppf^2)

# targets: the directions of some nodes, scaled to radius rad
function radial_targets(X, rad; stride=37)
    D = X[:, 1:stride:end]
    return D ./ sqrt.(sum(abs2, D; dims=1)) .* rad
end
relerr(a, b) = norm(a - b, Inf) / max(norm(b, Inf), 1e-300)

# order 12, 2×2 elements per face (3456 nodes): geometry resolved to ~1e-8 (area) / ~1e-6 (normals)
const ORDER = 12
const EL = sphere(ORDER)
const XS, NS = node_coord(EL)
const NN = nnodes(EL)

@testset "QuadElemList" begin
    @test nelem(EL) == 24 && order(EL) == ORDER && NN == 24 * ORDER^2
    @test quad_scheme(EL) === :duffy
    t = param_nodes(ORDER)
    @test t ≈ legquad(ORDER)[1]
    # nodes on the unit sphere, outward unit normals
    @test maximum(abs.(sqrt.(sum(abs2, XS; dims=1)) .- 1)) < 1e-14
    @test maximum(abs.(NS - XS)) < 1e-5
    @test size(XS) == size(NS) == (3, NN)
    # surface area from the node weights; integral of z^2 = 4π/3
    w = node_weights(EL)
    @test length(w) == NN && all(>(0), w)
    @test sum(w) ≈ 4π rtol = 1e-7
    @test dot(w, XS[3, :] .^ 2) ≈ 4π / 3 rtol = 1e-7

    # node order: element e, node (i, j) at (u_i, v_j), column (e-1)*order^2 + (i-1)*order + j
    e, i, j = 7, 3, 5
    @test XS[:, (e-1)*ORDER^2+(i-1)*ORDER+j] ≈ sphere_point(e, t[i], t[j]) rtol = 1e-14

    # equivalent constructors
    X4 = reshape(copy(XS), 3, ORDER, ORDER, 24)
    @test X4[:, j, i, e] == XS[:, (e-1)*ORDER^2+(i-1)*ORDER+j]
    for c in (X4, XS, vec(XS))
        @test node_coord(QuadElemList(ORDER, c))[1] == XS
    end

    # geometry inside an element: at the nodes, the node data; area element sums to the element area
    g = elem_geom(EL, e, t, t)
    nodes_e = (e-1)*ORDER^2+1:e*ORDER^2
    @test g.X ≈ XS[:, nodes_e] rtol = 1e-13
    @test g.normal ≈ NS[:, nodes_e] rtol = 1e-13
    @test g.area .* kron(legquad(ORDER)[2], legquad(ORDER)[2]) ≈ w[nodes_e] rtol = 1e-13
    @test vec(sum(g.normal .* g.dXdu; dims=1)) ≈ zeros(ORDER^2) atol = 1e-13
    @test [norm(cross(g.dXdu[:, k], g.dXdv[:, k])) for k in 1:ORDER^2] ≈ g.area rtol = 1e-13
    # off-node points: interpolated geometry stays close to the sphere; (u[a], v[b]) in column (a-1)*nv + b
    u, v = [0.0, 0.25, 1.0], [0.1, 0.9]
    g = elem_geom(EL, 2, u, v)
    @test size(g.X) == (3, 6) && length(g.area) == 6
    @test maximum(abs.(sqrt.(sum(abs2, g.X; dims=1)) .- 1)) < 1e-5
    @test g.X[:, (2-1)*2+2] ≈ sphere_point(2, u[2], v[2]) rtol = 1e-5

    set_quad_scheme!(EL, :hedgehog)
    @test quad_scheme(EL) === :hedgehog
    set_quad_scheme!(EL, :duffy)
    @test_throws ArgumentError set_quad_scheme!(EL, :gauss)

    mktempdir() do dir
        f = write_vtk(EL, joinpath(dir, "sphere"), XS[3, :])
        @test isfile(f) && endswith(f, "sphere000000.vtu") && filesize(f) > 0
        @test_throws DimensionMismatch write_vtk(EL, joinpath(dir, "bad"), ones(NN + 1))
    end

    @test_throws ArgumentError QuadElemList(0, zeros(3))
    @test_throws DimensionMismatch QuadElemList(4, zeros(3, 47))
    @test_throws DimensionMismatch QuadElemList(4, zeros(3, 4, 3, 1))
    @test_throws BoundsError elem_geom(EL, 25, t, t)
    el = QuadElemList(4, zeros(3, 16))
    close(el)
    @test_throws ArgumentError nelem(el)
    close(el) # idempotent
end

@testset "BoundaryIntegralOp: Laplace" begin
    tol = 1e-8
    Xin, Xout = radial_targets(XS, 0.5), radial_targets(XS, 1.6)
    Xnear_in, Xnear_out = radial_targets(XS, 0.98), radial_targets(XS, 1.02)
    T = hcat(Xin, Xnear_in, Xnear_out, Xout)
    nt = size(Xin, 2)
    r_out = vcat(fill(1.02, nt), fill(1.6, nt))
    one_ = ones(NN)

    # double layer of density 1: -1 inside, 0 outside, -1/2 (principal value) on the surface
    D = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=tol)
    @test accuracy(D) == tol
    add_elem_list!(D, EL)
    @test elem_list_names(D) == ["elements"]
    @test dim(D, 1) == NN && dim(D, 2) == NN  # default targets: the nodes
    u = compute_potential(D, one_)
    @test maximum(abs.(u .+ 0.5)) < 1e-6
    set_target_coord!(D, T)
    @test dim(D, 2) == 4nt
    u = compute_potential(D, one_)
    @test maximum(abs.(u[1:2nt] .+ 1)) < 1e-6
    @test maximum(abs.(u[2nt+1:end])) < 1e-6
    # tensor_product: also the principal value; hedgehog: the limit from the side of the normal (outside), 0
    for (scheme, val) in ((:tensor_product, -0.5), (:hedgehog, 0.0))
        Ds = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=tol)
        add_elem_list!(Ds, set_quad_scheme!(sphere(ORDER), scheme))
        @test maximum(abs.(compute_potential(Ds, one_) .- val)) < 1e-6
    end

    # single layer of density 1 on the unit sphere: 1 inside and on the surface, 1/r outside
    S = BoundaryIntegralOp(Laplace3D_FxU(); accuracy=tol)
    add_elem_list!(S, EL)
    @test maximum(abs.(compute_potential(S, one_) .- 1)) < 1e-6
    set_target_coord!(S, T)
    setup!(S)
    u = compute_potential(S, reshape(one_, 1, NN))
    @test size(u) == (1, 4nt)
    @test maximum(abs.(u[1:2nt] .- 1)) < 1e-6
    @test maximum(abs.(u[2nt+1:end] .- 1 ./ r_out)) < 1e-6
    # in place; targets reset to the surface nodes
    U = zeros(4nt)
    @test compute_potential!(U, S, one_) === U && U ≈ vec(u)
    set_target_coord!(S, nothing)
    @test dim(S, 2) == NN

    # Green's identity for u = 1/(4π|x - x0|), x0 outside: S[∂u/∂n] - D[u] = u inside, u/2 on the surface
    x0 = [1.3, 1.2, 0.2]
    ex(X) = kernel_eval(Laplace3D_FxU(), x0, [1.0], X)
    us = vec(ex(XS))
    dudn = vec(sum(kernel_eval(Laplace3D_FxdU(), x0, [1.0], XS) .* NS; dims=1))
    clear_setup!(S)
    set_target_coord!(S, T)
    g = compute_potential(S, dudn) - compute_potential(D, us)
    @test relerr(g[1:2nt], vec(ex(T[:, 1:2nt]))) < 1e-6
    @test maximum(abs.(g[2nt+1:end])) < 1e-6 * maximum(abs, us)   # zero outside
    set_target_coord!(S, nothing); set_target_coord!(D, nothing)
    g = compute_potential(S, dudn) - compute_potential(D, us)
    @test relerr(g, us / 2) < 1e-6

    # adjoint double layer (gradient dotted with the target normal) of density 1:
    # ∂/∂n of the single layer: 0 inside, -1/r^2 outside, -1/2 on the surface (principal value)
    A = BoundaryIntegralOp(Laplace3D_FxdU(); trg_normal_dot_prod=true, accuracy=tol)
    add_elem_list!(A, EL)
    @test dim(A, 2) == NN
    @test maximum(abs.(compute_potential(A, one_) .+ 0.5)) < 1e-6
    set_target_coord!(A, T)
    @test_throws ArgumentError compute_potential(A, one_)  # target normals required
    set_target_normal!(A, T ./ sqrt.(sum(abs2, T; dims=1)))
    u = compute_potential(A, one_)
    @test maximum(abs.(u[1:2nt])) < 1e-6
    @test maximum(abs.(u[2nt+1:end] .+ 1 ./ r_out .^ 2)) < 1e-6

    # scaling by sqrt of element areas
    w = node_weights(EL)
    area = repeat([sum(w[(e-1)*ORDER^2+1:e*ORDER^2]) for e in 1:24]; inner=ORDER^2)
    V = rand(2, NN)
    V0 = copy(V)
    sqrt_scaling!(S, V)
    @test V ≈ V0 .* sqrt.(area)'
    inv_sqrt_scaling!(S, V)
    @test V ≈ V0
    @test_throws DimensionMismatch sqrt_scaling!(S, ones(NN + 1))

    # several element lists: a second sphere (radius 0.5, centered at (3,0,0)); densities in name order
    el2 = QuadElemList(ORDER, node_coord(sphere(ORDER; ppf=1))[1] .* 0.5 .+ [3.0, 0, 0])
    add_elem_list!(S, el2, "b_small")
    @test elem_list_names(S) == ["b_small", "elements"]
    @test dim(S, 1) == NN + nnodes(el2)
    set_target_coord!(S, [0.0 3.0; 0.0 0.0; 0.0 0.0])  # the two centers
    u = compute_potential(S, vcat(fill(2.0, nnodes(el2)), one_))
    @test u ≈ [1 + 2 * 0.25 / 3, 1 / 3 + 2 * 0.5] rtol = 1e-5  # el2 is coarse
    delete_elem_list!(S, "b_small")
    @test dim(S, 1) == NN && elem_list_names(S) == ["elements"]

    @test_throws DimensionMismatch compute_potential(S, ones(NN + 1))
    @test_throws DimensionMismatch compute_potential!(zeros(3), S, one_)
    @test_throws ArgumentError add_elem_list!(S, QuadElemList(5, zeros(3, 25)))
    @test_throws ArgumentError dim(S, 0)
    @test_throws ArgumentError set_accuracy!(S, 0)
    close(S)
    @test_throws ArgumentError dim(S, 1)
end

@testset "BoundaryIntegralOp: Stokes and Helmholtz" begin
    tol = 1e-8
    Xin, Xout = radial_targets(XS, 0.6), radial_targets(XS, 2.0)
    T = hcat(Xin, Xout)
    nt = size(Xin, 2)
    f = [0.3, -1.2, 0.7]
    F = repeat(f, 1, NN)

    # Stokes single layer of a constant density on the unit sphere: rigid translation 2f/3 inside;
    # double layer of a constant density: -f inside, 0 outside
    for (ker, inside, outside) in ((Stokes3D_FxU(), 2f / 3, nothing), (Stokes3D_DxU(), -f, zero(f)))
        op = BoundaryIntegralOp(ker; accuracy=tol)
        add_elem_list!(op, EL)
        set_target_coord!(op, T)
        @test dim(op, 1) == 3NN && dim(op, 2) == 6nt
        U = compute_potential(op, F)
        @test size(U) == (3, 2nt)
        @test maximum(abs.(U[:, 1:nt] .- inside)) < 1e-6
        outside === nothing || @test maximum(abs.(U[:, nt+1:end] .- outside)) < 1e-6
        # far field = the smooth node quadrature (atol: the double layer is ~1e-9 there, i.e. zero up to rounding)
        @test U[:, nt+1:end] ≈ kernel_eval(ker, XS, normal_dim(ker) > 0 ? NS : nothing, F .* node_weights(EL)', Xout) rtol = 1e-8 atol = 1e-12
    end

    # Helmholtz single layer of density 1: e^{ik} sin(kr)/(kr) inside, sin(k) e^{ikr}/(kr) outside
    k = 1.3
    S = BoundaryIntegralOp(Helmholtz3D_FxU(k); accuracy=tol)
    add_elem_list!(S, EL)
    set_target_coord!(S, T)
    Fc = repeat([1.0, 0.0], 1, NN)
    U = compute_potential(S, Fc)
    Uc = U[1, :] + im * U[2, :]
    rin, rout = 0.6, 2.0
    @test maximum(abs.(Uc[1:nt] .- exp(im * k) * sin(k * rin) / (k * rin))) < 1e-6
    @test maximum(abs.(Uc[nt+1:end] .- sin(k) * exp(im * k * rout) / (k * rout))) < 1e-6

    # Helmholtz Green's identity, u = e^{ik|x-x0|}/(4π|x-x0|) with x0 outside: S[∂u/∂n] - D[u] = u inside
    x0 = [1.3, 1.2, 0.2]
    us = kernel_eval(Helmholtz3D_FxU(k), x0, [1.0, 0.0], XS)                  # 2 × NN
    G = kernel_eval(Helmholtz3D_FxdU(k), x0, [1.0, 0.0], XS)                  # 6 × NN: (re, im) per component
    dudn = vcat(sum(G[1:2:end, :] .* NS; dims=1), sum(G[2:2:end, :] .* NS; dims=1))
    D = BoundaryIntegralOp(Helmholtz3D_DxU(k); accuracy=tol)
    add_elem_list!(D, EL)
    set_target_coord!(D, Xin)
    set_target_coord!(S, Xin)
    g = compute_potential(S, dudn) - compute_potential(D, us)
    @test relerr(g, kernel_eval(Helmholtz3D_FxU(k), x0, [1.0, 0.0], Xin)) < 1e-6

    # kernels without a BoundaryIntegralOp, and invalid options
    @test Stokes3D_FxT in bie_supported_kernels()
    @test_throws ArgumentError BoundaryIntegralOp(Laplace3D_Fxd2U())
    @test_throws ArgumentError BoundaryIntegralOp(Laplace3D_FxU(); trg_normal_dot_prod=true)
    T3 = BoundaryIntegralOp(Stokes3D_FxT(); trg_normal_dot_prod=true)
    add_elem_list!(T3, sphere(4; ppf=1))
    @test dim(T3, 1) == 3 * 6 * 16 && dim(T3, 2) == 3 * 6 * 16
end
