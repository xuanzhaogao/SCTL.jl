# Wrapper benchmarks, mirrored by bench/bench_native.cpp (same SCTL calls, same inputs).
#   julia --project=.. bench_julia.jl DATA_DIR      # writes the inputs to DATA_DIR, then prints "name  seconds"
using SCTL, LinearAlgebra, Random, Printf

const DATA = length(ARGS) > 0 ? ARGS[1] : mktempdir()
mkpath(DATA)
save(name, x) = write(joinpath(DATA, name * ".bin"), collect(Float64, vec(x)))

# Best wall time over repetitions (at least 3, until ~1 s or 50 runs), after one warm-up run; same rule as the C++ side.
function bench(f)
    f()
    best, total, k = Inf, 0.0, 0
    while k < 50 && (k < 3 || total < 1.0)
        t = @elapsed f()
        best = min(best, t)
        total += t
        k += 1
    end
    return best
end
report(name, t) = (@printf("%-28s %.6e\n", name, t); flush(stdout))

rng = Xoshiro(1)
sink = Ref(0.0)

# ---- inputs ----
lag_src, _ = chebquad(32); lag_trg = rand(rng, 1000)
save("lag_src", lag_src); save("lag_trg", lag_trg)

Xs, Xt = rand(rng, 3, 20_000), rand(rng, 3, 20_000) .+ [0.5, 0, 0]
F1, F3 = randn(rng, 1, 20_000), randn(rng, 3, 20_000)
save("nb_xs", Xs); save("nb_xt", Xt); save("nb_f1", F1); save("nb_f3", F3)

gm = Dict{String,Any}()
for (tag, n) in (("100", 100), ("200k", 200_000))
    d = 1 .+ rand(rng, n); b = randn(rng, n)
    gm[tag] = (d, b); save("gm_d_$tag", d); save("gm_b_$tag", b)
end
sdc_u0 = Dict("4" => randn(rng, 4), "100k" => randn(rng, 100_000))
for (k, v) in sdc_u0; save("sdc_u0_$k", v); end

p = 127; Nt, Np = p + 1, 2p + 2
θ, φ = sph_grid_angles(Nt, Np)
sh_grid = [exp(sin(t) * cos(f)) * cos(3t) for f in φ, t in θ]
save("sh_grid", sh_grid)

tree_pts = rand(rng, 3, 1_000_000) .^ 2     # non-uniform, inside [0, 1)
save("tree_pts", tree_pts)

fft_c2c = randn(rng, ComplexF64, 2^20); fft_r2c = randn(rng, 1024, 1024)
save("fft_c2c", reinterpret(Float64, fft_c2c)); save("fft_r2c", fft_r2c)

const FACE = (((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
              ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
              ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
function sphere_point(e, u, v; ppf=2)
    e0 = e - 1; face = e0 ÷ ppf^2 + 1
    a = 2 * ((e0 ÷ ppf % ppf + u) / ppf) - 1; b = 2 * ((e0 % ppf + v) / ppf) - 1
    q = [FACE[face][k][1] + FACE[face][k][2] * a + FACE[face][k][3] * b for k in 1:3]
    return q ./ norm(q)
end
order, nelem = 12, 24
un = param_nodes(order)
bie_coord = Array{Float64}(undef, 3, order, order, nelem)
for e in 1:nelem, i in 1:order, j in 1:order
    bie_coord[:, j, i, e] = sphere_point(e, un[i], un[j])
end
bie_F = randn(rng, order^2 * nelem)
save("bie_coord", bie_coord); save("bie_density", bie_F)

# ---- benchmarks ----
report("legquad_16", bench(() -> (sink[] += legquad(16)[1][1])))
report("lagrange_32x1000", bench(() -> (sink[] += lagrange_interp_matrix(lag_src, lag_trg)[1])))

lap, sto = Laplace3D_FxU(), Stokes3D_FxU()
report("laplace_eval_20k", bench(() -> (sink[] += kernel_eval(lap, Xs, F1, Xt)[1])))
Xs10, Xt10, F310 = Xs[:, 1:10_000], Xt[:, 1:10_000], F3[:, 1:10_000]
report("stokes_eval_10k", bench(() -> (sink[] += kernel_eval(sto, Xs10, F310, Xt10)[1])))
Xs2, Xt2 = Xs[:, 1:2000], Xt[:, 1:2000]
report("laplace_matrix_2k", bench(() -> (sink[] += kernel_matrix(lap, Xs2, Xt2)[1])))
K2 = Matrix{Float64}(undef, 2000, 2000)
report("laplace_matrix_2k_inplace", bench(() -> (sink[] += kernel_matrix!(K2, lap, Xs2, Xt2)[1])))

for tag in ("100", "200k")
    d, b = gm[tag]
    report("gmres_diag_$tag", bench(() -> (sink[] += gmres((y, x) -> (y .= d .* x), b; tol=1e-14, maxiter=30).x[1])))
end

sdc = SDC(8)
for tag in ("4", "100k")
    u0 = sdc_u0[tag]
    report("sdc_step_$tag", bench(() -> (sink[] += sdc_step(sdc, (du, u) -> (du .= .-u), u0, 0.1).u[1])))
end

S = grid2shc(sh_grid, Nt, Np, p)
report("grid2shc_p127", bench(() -> (sink[] += grid2shc(sh_grid, Nt, Np, p)[1])))
report("shc2grid_p127", bench(() -> (sink[] += shc2grid(S, p, Nt, Np)[1])))

report("pttree_1M", bench() do
    tree = PtTree{3}()
    update_refinement!(tree, tree_pts; max_pts=32)
    add_particles!(tree, "pts", tree_pts)
    sink[] += num_nodes(tree)
    close(tree)
end)

c2c, r2c = FFTPlan(:c2c, 2^20), FFTPlan(:r2c, (1024, 1024))
report("fft_c2c_1M", bench(() -> (sink[] += real(execute(c2c, fft_c2c)[1]))))
report("fft_r2c_1024x1024", bench(() -> (sink[] += real(execute(r2c, fft_r2c)[1]))))

elist = QuadElemList(order, bie_coord)
t_setup = Inf
for _ in 1:3
    op = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=1e-8)
    add_elem_list!(op, elist)
    global t_setup = min(t_setup, @elapsed setup!(op))
    close(op)
end
report("bie_setup_3456", t_setup)
op = BoundaryIntegralOp(Laplace3D_DxU(); accuracy=1e-8)
add_elem_list!(op, elist)
setup!(op)
report("bie_apply_3456", bench(() -> (sink[] += compute_potential(op, bie_F)[1])))
