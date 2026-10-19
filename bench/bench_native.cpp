// Native C++ baseline for bench/bench_julia.jl: the same SCTL calls on the same inputs (read from DATA_DIR, written
// by the Julia script), compiled with the flags of csrc/Makefile. Prints "name  seconds" (best of several runs).
#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

using namespace sctl;
using VecD = Vector<double>;

static std::string data_dir;

static VecD load(const std::string& name) {
  std::ifstream f(data_dir + "/" + name + ".bin", std::ios::binary | std::ios::ate);
  SCTL_ASSERT_MSG(f, "missing input " << name);
  const Long n = (Long)f.tellg() / (Long)sizeof(double);
  f.seekg(0);
  VecD v(n);
  if (n) f.read(reinterpret_cast<char*>(&v[0]), n * sizeof(double));
  return v;
}

// Best wall time of fn over repetitions (at least 3, until ~1 s or 50 runs), after one warm-up run.
static double bench(const std::function<void()>& fn) {
  fn();
  double best = 1e30, total = 0;
  for (int k = 0; k < 50 && (k < 3 || total < 1.0); k++) {
    const double t0 = omp_get_wtime();
    fn();
    const double t = omp_get_wtime() - t0;
    best = std::min(best, t);
    total += t;
  }
  return best;
}

static void report(const char* name, double t) { std::printf("%-28s %.6e\n", name, t); std::fflush(stdout); }

static volatile double sink = 0;

int main(int argc, char** argv) {
  data_dir = argc > 1 ? argv[1] : ".";

  { // quadrature: compute (what the wrapper calls) and the cached accessor
    report("legquad_16", bench([] { VecD x, w; LegQuadRule<double>::ComputeNdsWts(&x, &w, 16); sink = sink + x[0]; }));
    report("legquad_16_cached", bench([] { sink = sink + LegQuadRule<double>::nds(16)[0]; }));
  }
  { // Lagrange interpolation matrix 32 -> 1000
    const VecD src = load("lag_src"), trg = load("lag_trg");
    report("lagrange_32x1000", bench([&] { VecD w; LagrangeInterp<double>::Interpolate(w, src, trg); sink = sink + w[0]; }));
  }
  { // direct N-body sums
    const VecD Xs = load("nb_xs"), Xt = load("nb_xt"), F1 = load("nb_f1"), F3 = load("nb_f3");
    const Long Nt = Xt.Dim() / 3;
    const Laplace3D_FxU lap;
    const Stokes3D_FxU sto;
    report("laplace_eval_20k", bench([&] { VecD U(Nt); U = 0; lap.Eval<double, true>(U, Xt, Xs, VecD(), F1); sink = sink + U[0]; }));
    const VecD Xs10(Xs.Dim() / 2, (Iterator<double>)Xs.begin(), false), Xt10(Xt.Dim() / 2, (Iterator<double>)Xt.begin(), false);
    const VecD F310(F3.Dim() / 2, (Iterator<double>)F3.begin(), false);
    report("stokes_eval_10k", bench([&] { VecD U(3 * Nt / 2); U = 0; sto.Eval<double, true>(U, Xt10, Xs10, VecD(), F310); sink = sink + U[0]; }));
    const VecD Xs2(3 * 2000, (Iterator<double>)Xs.begin(), false), Xt2(3 * 2000, (Iterator<double>)Xt.begin(), false);
    report("laplace_matrix_2k", bench([&] { Matrix<double> M(2000, 2000); lap.KernelMatrix<double, true>(M, Xt2, Xs2, VecD()); sink = sink + M[0][0]; }));
  }
  { // GMRES with a cheap diagonal operator (callback overhead dominated)
    for (const char* tag : {"100", "200k"}) {
      const VecD d = load(std::string("gm_d_") + tag), b = load(std::string("gm_b_") + tag);
      const GMRES<double>::ParallelOp A = [&d](VecD* Ax, const VecD& x) {
        const Long N = x.Dim();
        if (Ax->Dim() != N) Ax->ReInit(N);
        for (Long i = 0; i < N; i++) (*Ax)[i] = d[i] * x[i];
      };
      const GMRES<double> solver(Comm::Self(), false);
      report((std::string("gmres_diag_") + tag).c_str(), bench([&] { VecD x; Long it; solver(&x, A, b, 1e-14, 30, false, &it); sink = sink + x[0]; }));
    }
  }
  { // SDC: one step of u' = -u
    const SDC<double> sdc(8);
    const SDC<double>::Fn1 F = [](VecD* dudt, const VecD& u) {
      if (dudt->Dim() != u.Dim()) dudt->ReInit(u.Dim());
      for (Long i = 0; i < u.Dim(); i++) (*dudt)[i] = -u[i];
    };
    for (const char* tag : {"4", "100k"}) {
      const VecD u0 = load(std::string("sdc_u0_") + tag);
      report((std::string("sdc_step_") + tag).c_str(), bench([&] { VecD u; sdc(&u, 0.1, u0, F); sink = sink + u[0]; }));
    }
  }
  { // spherical harmonics, p = 127
    const Long p = 127, Nt = p + 1, Np = 2 * p + 2;
    const VecD X = load("sh_grid");
    VecD S;
    SphericalHarmonics<double>::Grid2SHC(X, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
    report("grid2shc_p127", bench([&] { VecD S_; SphericalHarmonics<double>::Grid2SHC(X, Nt, Np, p, S_, SHCArrange::ROW_MAJOR); sink = sink + S_[0]; }));
    report("shc2grid_p127", bench([&] { VecD X_; SphericalHarmonics<double>::SHC2Grid(S, SHCArrange::ROW_MAJOR, p, Nt, Np, &X_); sink = sink + X_[0]; }));
  }
  { // point tree, 1e6 points
    const VecD X = load("tree_pts");
    report("pttree_1M", bench([&] {
      PtTree<double, 3> tree;
      tree.UpdateRefinement(X, 32);
      tree.AddParticles("pts", X);
      sink = sink + tree.GetNodeMID().Dim();
    }));
  }
  { // FFT
    FFT<double> c2c, r2c;
    Vector<Long> d1(1); d1[0] = 1 << 20;
    c2c.Setup(FFT_Type::C2C, 1, d1, 1);
    Vector<Long> d2(2); d2[0] = 1024; d2[1] = 1024;
    r2c.Setup(FFT_Type::R2C, 1, d2, 1);
    const VecD xc = load("fft_c2c"), xr = load("fft_r2c");
    report("fft_c2c_1M", bench([&] { VecD y; c2c.Execute(xc, y); sink = sink + y[0]; }));
    report("fft_r2c_1024x1024", bench([&] { VecD y; r2c.Execute(xr, y); sink = sink + y[0]; }));
  }
  { // boundary integral: Laplace double layer on a cubed sphere (order 12, 24 elements, 3456 nodes)
    const VecD coord = load("bie_coord"), F = load("bie_density");
    QuadElemList<double> elist(12, coord);
    const Laplace3D_DxU ker;
    double t_setup = 1e30;
    for (int k = 0; k < 3; k++) {
      BoundaryIntegralOp<double, Laplace3D_DxU> op(ker);
      op.SetAccuracy(1e-8);
      op.AddElemList(elist);
      const double t0 = omp_get_wtime();
      op.Setup();
      t_setup = std::min(t_setup, omp_get_wtime() - t0);
    }
    report("bie_setup_3456", t_setup);
    BoundaryIntegralOp<double, Laplace3D_DxU> op(ker);
    op.SetAccuracy(1e-8);
    op.AddElemList(elist);
    op.Setup();
    report("bie_apply_3456", bench([&] { VecD U; op.ComputePotential(U, F); sink = sink + U[0]; }));
  }
#ifdef SCTL_HAVE_PVFMM
  { // FMM, with the translation kernels of csrc/fmm_schemes.hpp
    const VecD Xs = load("fmm_xs"), Xt = load("fmm_xt"), F = load("fmm_f"), Fs = load("fmm_fs");
    {
      ParticleFMM<double, 3> fmm;
      const Laplace3D_FxU L;
      fmm.SetAccuracy(8);
      fmm.SetKernels(L, L, L); fmm.AddSrc("src", L, L); fmm.AddTrg("trg", L, L); fmm.SetKernelS2T("src", "trg", L);
      fmm.SetSrcCoord("src", Xs); fmm.SetTrgCoord("trg", Xt); fmm.SetSrcDensity("src", F);
      report("fmm_laplace_1M_d8", bench([&] { VecD U; fmm.Eval(U, "trg"); sink = sink + U[0]; }));
    }
    {
      ParticleFMM<double, 3> fmm;
      const Stokes3D_FxU S; const Stokes3D_FSxU SS;
      const VecD Xs2(3 * 200000, (Iterator<double>)Xs.begin(), false), Xt2(3 * 200000, (Iterator<double>)Xt.begin(), false);
      fmm.SetAccuracy(6);
      fmm.SetKernels(SS, SS, S); fmm.AddSrc("src", S, S); fmm.AddTrg("trg", SS, S); fmm.SetKernelS2T("src", "trg", S);
      fmm.SetSrcCoord("src", Xs2); fmm.SetTrgCoord("trg", Xt2); fmm.SetSrcDensity("src", Fs);
      report("fmm_stokes_200k_d6", bench([&] { VecD U; fmm.Eval(U, "trg"); sink = sink + U[0]; }));
    }
    {
      const VecD coord = load("bie_coord"), Fb = load("bie_density"), Xtb = load("bie_trg");
      QuadElemList<double> elist(12, coord);
      const Laplace3D_DxU D; const Laplace3D_FxU L;
      BoundaryIntegralOp<double, Laplace3D_DxU> op(D);
      op.SetFMMKer(D, D, D, L, L, L, L, L);
      op.SetAccuracy(1e-8);
      op.AddElemList(elist);
      op.SetTargetCoord(Xtb);
      op.Setup();
      report("bie_fmm_apply_50k", bench([&] { VecD U; op.ComputePotential(U, Fb); sink = sink + U[0]; }));
    }
  }
#endif
  return 0;
}
