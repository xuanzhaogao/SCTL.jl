#include <sctl.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>
using namespace sctl;
// Diagnostic for grid2shc (bench/cluster_bench.slurm): median seconds of `reps` Grid2SHC calls, run entirely in C++
extern "C" double sh_loop(const double* x, long n, int reps) {
  const Vector<double> X(n, (Iterator<double>)const_cast<double*>(x), false);
  const Long p = 127, Nt = p + 1, Np = 2 * p + 2;
  std::vector<double> ts;
  for (int k = 0; k < reps; k++) {
    const double t0 = omp_get_wtime();
    Vector<double> S; SphericalHarmonics<double>::Grid2SHC(X, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
    ts.push_back(omp_get_wtime() - t0);
  }
  ts.erase(ts.begin(), ts.begin() + reps / 10);
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}
#ifdef MAIN
int main(int argc, char** argv) {
  std::ifstream f(std::string(argv[1]) + "/sh_grid.bin", std::ios::binary | std::ios::ate);
  const long n = (long)f.tellg() / 8; f.seekg(0);
  std::vector<double> X(n); f.read((char*)X.data(), n * 8);
  printf("standalone process: median %.3e\n", sh_loop(X.data(), n, 400));
}
#endif
