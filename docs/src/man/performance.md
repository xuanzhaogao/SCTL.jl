# Performance

SCTL.jl passes Julia arrays to SCTL without copying them, so compute-heavy calls run at C++ speed. What remains is a
small fixed cost per call and per callback, and the cost of allocating fresh output arrays.

## Measured overhead

The directory `bench/` of the repository times every module through SCTL.jl. It also times the same SCTL calls made
directly from C++, on the same inputs and built with the same flags. The run below is from October 2026, on
exclusive nodes of Flatiron's Rusty cluster:
- genoa: AMD EPYC 9474F, 96 cores
- icelake: Intel Xeon Platinum 8362, 64 cores

Each entry is the time ratio SCTL.jl / C++:

| benchmark | genoa, 1 thread | genoa, 96 threads | icelake, 1 thread | icelake, 64 threads |
|:--|:--|:--|:--|:--|
| Laplace direct sum, 20k × 20k | 1.000 | 0.98 | 1.000 | 1.02 |
| Stokes direct sum, 10k × 10k | 1.000 | 0.99 | 1.000 | 1.44 (noise) |
| `kernel_matrix!`, 2000 × 2000 (reused buffer) | 1.00 | 0.97 | 1.00 | 1.01 |
| GMRES, 200k unknowns, diagonal operator | 1.00 | 0.97 | 1.00 | 1.01 |
| `grid2shc` / `shc2grid`, ``p = 127`` | 1.06 / 1.08 | 1.08 / 0.97 | 1.07 / 1.01 | 0.90 / 0.97 |
| `PtTree` refinement, 10⁶ points | 1.07 | 1.01 | 1.08 | 1.01 |
| FFT, 2²⁰ complex / 1024² real | 0.93 / 1.01 | 1.01 / 1.00 | 1.09 / 1.03 | 1.04 / 1.09 |
| boundary integral operator, setup / apply | 1.03 / 1.00 | 0.77 / 1.07 | 1.02 / 1.00 | 1.05 / 0.97 |
| SDC step, 10⁵ unknowns | 1.10 | 1.14 | 1.14 | 1.75 |
| SDC step, 4 unknowns | 2.8 | 2.9 | 2.9 | 2.6 |
| `legquad(16)` | 1.34 | 1.33 | 1.45 | 1.47 |

Single-threaded ratios are reproducible to a few percent. Calls of about 1 ms with many threads vary by 10–40%
between runs, in both C++ and Julia, because of OpenMP synchronization.

The `grid2shc` entries were rechecked with a median of 400 calls on the same icelake node, timed four ways: through
SCTL.jl, through a raw `ccall`, as a C++ loop inside the Julia process, and as a standalone C++ program. All four
agreed to within 4%. The large-state SDC result on icelake at 64 threads (1.75) was not rechecked that way.

In absolute terms, the fixed costs are:

- **Per call:** about 0.2–0.3 µs for argument checks, `ccall`, output allocation and the C++ `try`/`catch`. This
  matters only for tiny calls, such as `legquad(16)` at 0.8 µs vs 0.6 µs in C++.
- **Per callback:** about 40–60 ns for each invocation of a Julia function from SCTL (GMRES operator, SDC
  right-hand side). An 8th-order SDC step calls the right-hand side 57 times, which is why a 4-unknown step takes
  3.6 µs instead of 1.3 µs.
- **Fresh outputs:** a function that allocates its result pays for fresh memory on every call. The
  page-faulting runs single-threaded, so this shows up with many threads, e.g. `kernel_matrix` at 1.3–1.8× C++ on
  16–96 threads. The in-place variant with a reused buffer runs at C++ speed.

## Guidelines

- **Reuse output buffers in loops.** Use the in-place forms: [`kernel_matrix!`](@ref), [`kernel_eval!`](@ref),
  [`compute_potential!`](@ref) and [`execute!`](@ref).
- **Reuse objects.** Build an [`FFTPlan`](@ref), [`SDC`](@ref) or [`BoundaryIntegralOp`](@ref) once and apply it
  many times. Operator setup (singular quadratures, near-interaction lists) is typically much more expensive than
  one application.
- **Keep callbacks allocation-free.** Write into the output argument in place (`y .= …`, `mul!(y, A, x)`). For
  vectors longer than 512 entries the arguments are views of SCTL's memory, so the only cost is your own work.
- **Choose the thread count deliberately.** Many SCTL routines on small problems scale poorly beyond a few dozen
  threads; spherical harmonics at ``p = 127`` is slower on 96 threads than on 16. Use
  [`SCTL.set_num_threads`](@ref) per phase if needed.
- **Consider `OMP_WAIT_POLICY`.** SCTL.jl defaults to `PASSIVE`, which is friendly to the rest of the Julia process.
  For a benchmark loop of many short SCTL calls, `OMP_WAIT_POLICY=ACTIVE` (set before starting Julia) avoids waking
  threads up on every call.
- **Build for the machine.** The default `-march=native` lets SCTL's SIMD layer use AVX2/AVX-512. A build made on one
  CPU type may crash with "illegal instruction" or run slower on another. On a heterogeneous cluster, build on the
  node type you run on, or use `SCTL_JL_MARCH=-march=x86-64-v3`.

## Reproducing the benchmarks

```bash
bench/run.sh [SCTL_ROOT] 1 16          # on this machine, for 1 and 16 threads
# on an exclusive Rusty node (copies the package to local disk and builds there):
sbatch -C "genoa&rocky9" bench/cluster_bench.slurm
```

Results go to `bench/results/<host>_<jobid>/`: `results.txt` has the side-by-side table and `sh_diag.txt` the
median-based `grid2shc` check.
