#!/bin/bash
# Compare SCTL.jl against native C++ SCTL. Usage: bench/run.sh [SCTL_ROOT (default: deps/SCTL)] [threads...]
# Builds the native baseline with csrc/Makefile's optimization flags, writes inputs via the Julia script, and prints
# both timings side by side for each OMP_NUM_THREADS value (default: 1 and 16).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SCTL_ROOT=${1:-$HERE/../deps/SCTL}; shift || true
THREADS=${@:-1 16}
WORK=${BENCH_DIR:-$(mktemp -d)}
mkdir -p "$WORK"
# configuration of the shim build (PVFMM headers, OpenBLAS32 with LAPACK), so both sides run the same SCTL code
CONFIG=$(sed 's/ | /\n/' "$HERE/../deps/usr/obj/config.txt")
CXXFLAGS_SHIM=$(echo "$CONFIG" | head -1 | cut -d' ' -f2- | sed 's/-fvisibility[^ ]*//g; s/-fPIC//')
LDFLAGS_SHIM=$(echo "$CONFIG" | tail -1 | sed 's/-shared//')
${CXX:-g++} $CXXFLAGS_SHIM -I"$SCTL_ROOT/include" "$HERE/bench_native.cpp" -o "$WORK/bench_native" $LDFLAGS_SHIM
export PVFMM_DIR=${PVFMM_DIR:-$(${JULIA:-julia} --project="$HERE/.." -e 'using SCTL; print(SCTL.fmm_cache_dir())')}
export OMP_WAIT_POLICY=PASSIVE   # what SCTL.jl sets on load; use it for both
export OPENBLAS_NUM_THREADS=1    # likewise (SCTL calls BLAS inside OpenMP loops)
for t in $THREADS; do
  export OMP_NUM_THREADS=$t
  ${JULIA:-julia} --project="$HERE/.." "$HERE/bench_julia.jl" "$WORK/data" > "$WORK/julia_$t.txt"
  "$WORK/bench_native" "$WORK/data" > "$WORK/native_$t.txt"
  echo "== OMP_NUM_THREADS=$t"
  printf "%-28s %12s %12s %8s\n" benchmark "native [s]" "julia [s]" ratio
  awk 'NR==FNR { n[$1] = $2; next } { r = ($1 in n) ? sprintf("%8.3f", $2 / n[$1]) : "     n/a";
       printf "%-28s %12s %12.4e %s\n", $1, ($1 in n) ? sprintf("%12.4e", n[$1]) : "n/a", $2, r }' \
      "$WORK/native_$t.txt" "$WORK/julia_$t.txt"
  awk 'NR==FNR { j[$1] = 1; next } !($1 in j) { printf "%-28s %12.4e %12s\n", $1, $2, "n/a" }' \
      "$WORK/julia_$t.txt" "$WORK/native_$t.txt"
done
