#!/bin/bash
# SCTL.jl vs native SCTL on one exclusive cluster node. Submit from the SCTL.jl directory, e.g.
#   env -u SBATCH_RESERVATION -u SLURM_RESERVATION -u SALLOC_RESERVATION sbatch -C "genoa&rocky9" bench/cluster_bench.slurm
# The package is copied to node-local storage and both the shim and the native baseline are compiled there with
# -march=native for the node's CPU. Results: bench/results/<host>_<jobid>/. BENCH_THREADS overrides the thread counts.
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH -J sctljl-bench
#SBATCH -o bench/results/slurm_%j.out

set -euo pipefail
SRC=$SLURM_SUBMIT_DIR                                   # the SCTL.jl checkout
SCTL_ROOT=${SCTL_ROOT:-$SRC/deps/SCTL}
export JULIA=${JULIA:-$HOME/.juliaup/bin/julia}
OUT=$SRC/bench/results/$(hostname -s)_$SLURM_JOB_ID
WORK=${TMPDIR:-/tmp}/sctljl_bench_$SLURM_JOB_ID
mkdir -p "$OUT" "$WORK"
trap 'rm -rf "$WORK"' EXIT

NCORES=$(nproc)
{ echo "host: $(hostname)"; echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"; echo "cores: $NCORES"
  echo "compiler: $(g++ --version | head -1)"; echo "julia: $($JULIA --version)"
  echo "sctl: $(git -C "$SCTL_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"; } | tee "$OUT/machine.txt"

# node-local copy of the package, built for this CPU (keeps the workstation build in deps/usr untouched)
rsync -a --exclude deps/usr --exclude bench/results "$SRC/" "$WORK/SCTL.jl/"
# default configuration (PVFMM, OpenBLAS32), as users get it from Pkg.build
SCTL_ROOT="$SCTL_ROOT" "$JULIA" --project="$WORK/SCTL.jl" -e 'using Pkg; Pkg.instantiate(); Pkg.build()' > "$OUT/build.log" 2>&1
"$JULIA" --project="$WORK/SCTL.jl" -e 'using SCTL; @show SCTL.has_feature(:PVFMM) SCTL.has_feature(:BLAS)' >> "$OUT/build.log" 2>&1

BENCH_DIR=$WORK/bench "$WORK/SCTL.jl/bench/run.sh" "$SCTL_ROOT" ${BENCH_THREADS:-1 16 $NCORES} | tee "$OUT/results.txt"
cp "$WORK"/bench/*.txt "$OUT/"

# grid2shc diagnostic: the same C++ loop standalone and inside the Julia process, vs the wrapper
F="-std=c++17 -fopenmp -O3 -march=native -DNDEBUG -I$SCTL_ROOT/include"
g++ $F -fPIC -shared "$SRC/bench/sh_loop.cpp" -o "$WORK/libshloop.so"
g++ $F -DMAIN "$SRC/bench/sh_loop.cpp" -o "$WORK/sh_loop"
for t in 16 $NCORES; do
  echo "== grid2shc diagnostic, OMP_NUM_THREADS=$t"
  for r in 1 2 3; do OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=$t "$WORK/sh_loop" "$WORK/bench/data"; done
  OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=$t "$JULIA" --project="$WORK/SCTL.jl" "$SRC/bench/sh_diag.jl" "$WORK/bench/data" "$WORK/libshloop.so"
done | tee "$OUT/sh_diag.txt"
