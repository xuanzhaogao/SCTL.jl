# A tour of SCTL.jl. Run with: julia --project=. examples/quickstart.jl
using SCTL, LinearAlgebra

# ---- quadrature and interpolation ----
x, w = legquad(10)                        # Gauss-Legendre on [0, 1]
println("∫₀¹ x⁹ dx ≈ ", dot(w, x .^ 9))
W = lagrange_interp_matrix(x, [0.25, 0.5])
println("interpolated sin: ", W * sin.(x), " vs ", sin.([0.25, 0.5]))
# generalized Gauss quadrature for x^k and x^k log(x), k = 0..9
xq, wq, _ = interp_quad_rule(t -> hcat([t .^ k for k in 0:9]..., [t .^ k .* log.(t) for k in 0:9]...), 0.0, 1.0; eps=1e-12)
println("log-singular rule with ", length(xq), " nodes: ∫ log(x) ≈ ", dot(wq, log.(xq)))

# ---- GMRES with a matrix-free operator ----
n = 50
A = I + 0.1 * [sin(i * j) for i in 1:n, j in 1:n]
b = ones(n)
sol = gmres((y, v) -> mul!(y, A, v), b; tol=1e-12)
println("GMRES: ", sol.iters, " iterations, residual ", norm(A * sol.x - b))

# ---- SDC: harmonic oscillator u'' = -u ----
sdc = SDC(8)
u, t, err = sdc_solve(sdc, (du, u) -> (du[1] = u[2]; du[2] = -u[1]), [1.0, 0.0], 0.1, 2π, 1e-10)
println("SDC: u(2π) = ", u, " (error estimate ", err, ")")

# ---- kernels: direct N-body sum ----
Xs, Xt = rand(3, 100), rand(3, 5) .+ 2
F = rand(1, 100)
U = kernel_eval(Laplace3D_FxU(), Xs, F, Xt)             # sum_j F_j / (4π |x - y_j|)
println("Laplace potential at targets: ", vec(U))

# ---- spherical harmonics ----
p = 8
θ, φ = sph_grid_angles(p + 1, 2p + 2)
X = [cos(t) + sin(t) * cos(f) for f in φ, t in θ]        # z + x on the unit sphere, (Np, Nt) grid
S = grid2shc(X, p + 1, 2p + 2, p)
println("SH evaluation at (θ, φ) = (1, 2): ", shceval(S, p, [1.0, 2.0]), " vs ", cos(1) + sin(1) * cos(2))

# ---- point tree ----
pts = rand(3, 1000)
tree = PtTree{3}()
update_refinement!(tree, pts; max_pts=20)
add_particles!(tree, "pts", pts)
info = node_info(tree)
println("PtTree: ", num_nodes(tree), " nodes, ", count(info.leaf), " leaves, max depth ", maximum(info.depth))

# ---- FFT ----
plan = FFTPlan(:r2c, 16)
println("FFT of a cosine: ", round.(abs.(execute(plan, cos.(2π * 3 * (0:15) / 16))); digits=6))

# ---- boundary integral: Laplace double layer on a sphere ----
order, nelem = 8, 6
# cubed sphere with one element per face
faces = [(u, v) -> (1.0, 2u - 1, 2v - 1), (u, v) -> (-1.0, 2v - 1, 2u - 1), (u, v) -> (2v - 1, 1.0, 2u - 1),
         (u, v) -> (2u - 1, -1.0, 2v - 1), (u, v) -> (2u - 1, 2v - 1, 1.0), (u, v) -> (2v - 1, 2u - 1, -1.0)]
elems = QuadElemList(order, nelem) do e, u, v
    q = collect(faces[e](u, v))
    q / norm(q)
end
op = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=1e-8)
add_elem_list!(op, elems)
set_target_coord!(op, [0.1 0.0; 0.2 0.0; -0.1 3.0])      # one target inside, one outside
println("double layer of σ=1 (expect -1 inside, ~0 outside): ", compute_potential(op, ones(nnodes(elems))))
