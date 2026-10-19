# FFT

SCTL's `sctl::FFT<double>` class computes batched multi-dimensional discrete Fourier transforms. In SCTL.jl it is
wrapped as [`FFTPlan`](@ref). You set up a plan once for a transform type, a size and a batch count, and then
apply it to many inputs with [`execute`](@ref) or [`execute!`](@ref). There are four transform types: real-to-complex
(`:r2c`), complex-to-complex forward (`:c2c`), complex-to-complex inverse (`:c2c_inv`) and complex-to-real (`:c2r`).

For an array ``x`` of size ``n_1 \times \dots \times n_d`` (``N = \prod_j n_j``) the forward transforms compute

```math
X_{k} = \sum_{n} x_{n}\, \exp\Big(-2\pi i \sum_{j=1}^{d} \frac{k_j n_j}{n_j^{\text{size}}}\Big),
\qquad 0 \le k_j, n_j < n_j^{\text{size}},
```

and the inverse transforms use the same sum with ``+2\pi i``. Neither direction is normalized, so applying a
forward transform and then an inverse transform multiplies the data by ``N``. The conventions match FFTW's
`FFTW_FORWARD` and `FFTW_BACKWARD`.

The backend is chosen when the shared library is built. If it is compiled with `-DSCTL_HAVE_FFTW` and linked with
`-lfftw3`, FFTW does the work; check with `SCTL.has_feature(:FFTW)`. Otherwise SCTL's own mixed-radix Cooley–Tukey
fallback runs. The fallback handles every size, including primes, every rank up to 16 and all four transform types.
Both backends give the same results up to rounding. Use this module when you want to stay inside SCTL, for example
to match the C++ code's layout or to avoid an FFTW dependency. When FFTW.jl is available, it is the more fully
featured Julia FFT.

## Tutorial

```@example fft
using SCTL, LinearAlgebra
using Random; Random.seed!(1)   # reproducible output

# direct O(N^2) DFT along every dimension of a Julia array, for checking (sign = -1 forward, +1 inverse)
function direct_dft(x::AbstractArray, sign)
    sz = size(x)
    y = zeros(ComplexF64, sz)
    for k in CartesianIndices(sz), n in CartesianIndices(sz)
        y[k] += x[n] * cispi(2sign * sum((k[d] - 1) * (n[d] - 1) / sz[d] for d in 1:ndims(x)))
    end
    return y
end
SCTL.has_feature(:FFTW)
```

### One-dimensional complex transforms

```@example fft
n = 12
x = randn(ComplexF64, n)
fwd = FFTPlan(:c2c, n)
bwd = FFTPlan(:c2c_inv, n)
X = execute(fwd, x)
println("forward vs direct DFT:  ", maximum(abs, X - direct_dft(x, -1)))
println("inverse vs direct DFT:  ", maximum(abs, execute(bwd, x) - direct_dft(x, +1)))
println("bwd(fwd(x)) / n - x:    ", maximum(abs, execute(bwd, X) / n - x))   # unnormalized: divide by n
```

Index `k` of the output (1-based) holds frequency `k - 1`. For `k - 1 > n ÷ 2` this is the same as the negative
frequency `k - 1 - n`. A single mode ``e^{2\pi i m j/n}`` therefore shows up as one spike of height `n`:

```@example fft
m = 3
X = execute(fwd, cispi.(2m .* (0:n-1) ./ n))
(argmax(abs.(X)) - 1, round(maximum(abs, X); digits=12))
```

### Two-dimensional transforms

`dims` is the size of the Julia array, and the transform runs over all of its dimensions:

```@example fft
A = randn(ComplexF64, 6, 5)
P = FFTPlan(:c2c, size(A))
B = execute(P, A)
(size(B), maximum(abs, B - direct_dft(A, -1)))
```

### Real transforms: `:r2c` and `:c2r`

The spectrum of a real array is Hermitian: ``X_{-k} = \overline{X_k}``. `:r2c` therefore returns only the
non-redundant half of the **first** Julia dimension, `n1 ÷ 2 + 1` rows, which is the same convention as FFTW.jl's
`rfft`. `:c2r` maps such a half spectrum back to a real array. It needs a plan built with the full real size,
because `n1 ÷ 2 + 1` alone cannot tell an even `n1` from the next odd one:

```@example fft
xr = randn(9, 4)
R = FFTPlan(:r2c, size(xr))
C = FFTPlan(:c2r, size(xr))
println("input_size(R) = ", input_size(R), ",  output_size(R) = ", output_size(R))
println("input_size(C) = ", input_size(C), ",  output_size(C) = ", output_size(C))
Y = execute(R, xr)
println("r2c vs first rows of the full DFT: ", maximum(abs, Y - direct_dft(xr, -1)[1:5, :]))
z = execute(C, Y)
println("c2r(r2c(x)) / N - x:               ", maximum(abs, z / length(xr) - xr))
```

### Batched transforms (`howmany`)

With `howmany = h`, a plan applies `h` independent transforms of size `dims`. Each transform's data is contiguous,
so a batch is a Julia array of size `(dims..., h)`: the batch index is the **last** dimension. When `h > 1` the
output gets the same trailing dimension:

```@example fft
h = 3
Xb = randn(8, 5, h)                        # three independent 8 × 5 real arrays
Pb = FFTPlan(:r2c, (8, 5); howmany=h)
Yb = execute(Pb, Xb)
err = maximum(maximum(abs, Yb[:, :, j] - direct_dft(Xb[:, :, j], -1)[1:5, :]) for j in 1:h)
(size(Yb), dim(Pb, 1), dim(Pb, 2), err)
```

[`dim`](@ref dim(::FFTPlan, ::Integer)) counts Julia elements over the whole batch: `Float64` entries on a real
side and `ComplexF64` entries on a complex side. The input can have any shape with `dim(plan, 1)` elements; it is
read in memory (column-major) order. To transform along the first dimension of a matrix, for example each column
of an `n × h` matrix, use a 1D plan with `howmany = h`. To transform along any other dimension, `permutedims` the
data first.

[`execute!`](@ref) writes into a preallocated output, which avoids an allocation per call:

```@example fft
out = Vector{ComplexF64}(undef, dim(Pb, 2))   # any shape with the right length
execute!(out, Pb, Xb)
out == vec(Yb)
```

### Example: spectral differentiation

Differentiate a smooth periodic function on ``[0, 1)``: take the real transform, multiply mode ``k`` by
``2\pi i k``, and transform back. For even `n`, set the Nyquist mode `k = n/2` to zero, because its derivative is
not real. Then divide by `n` to undo the unnormalized round trip.

```@example fft
n = 48
xs = (0:n-1) ./ n
f = exp.(sin.(2π .* xs))
df_exact = 2π .* cos.(2π .* xs) .* f

r2c, c2r = FFTPlan(:r2c, n), FFTPlan(:c2r, n)
F = execute(r2c, f)                    # modes k = 0, 1, ..., n ÷ 2
k = 0:n÷2
F .*= 2π * im .* k
iseven(n) && (F[end] = 0)              # drop the Nyquist mode
df = execute(c2r, F) ./ n
maximum(abs, df - df_exact)
```

The error is at rounding level because `f` is analytic and its spectrum decays exponentially. In 2D, multiply
by ``2\pi i k_1`` for ``\partial_x``, where ``k_1 = 0, \dots, n_1/2`` along the halved first dimension, and by
``2\pi i k_2`` for ``\partial_y``, where ``k_2 = 0, \dots, n_2/2 - 1, -n_2/2, \dots, -1`` along the second.

```@example fft
n1, n2 = 32, 24
x1, x2 = (0:n1-1) ./ n1, (0:n2-1) ./ n2
u = [sin(2π * a) * cos(4π * b) for a in x1, b in x2]
uy_exact = [-4π * sin(2π * a) * sin(4π * b) for a in x1, b in x2]
R2, C2 = FFTPlan(:r2c, (n1, n2)), FFTPlan(:c2r, (n1, n2))
U = execute(R2, u)                                   # (n1 ÷ 2 + 1) × n2
k2 = [0:n2÷2-1; -n2÷2:-1]
U .*= 2π * im .* transpose(k2)
U[:, n2÷2+1] .= 0                                    # Nyquist along y
maximum(abs, execute(C2, U) ./ (n1 * n2) - uy_exact)
```

## Data layouts

| plan type  | input (Julia)                                     | output (Julia)                                    |
|:-----------|:--------------------------------------------------|:--------------------------------------------------|
| `:r2c`     | `Float64`, `(dims..., howmany)`                   | `ComplexF64`, `(dims[1]÷2+1, dims[2:end]..., howmany)` |
| `:c2c`     | `ComplexF64`, `(dims..., howmany)`                | `ComplexF64`, `(dims..., howmany)`                |
| `:c2c_inv` | `ComplexF64`, `(dims..., howmany)`                | `ComplexF64`, `(dims..., howmany)`                |
| `:c2r`     | `ComplexF64`, `(dims[1]÷2+1, dims[2:end]..., howmany)` | `Float64`, `(dims..., howmany)`              |

- The trailing `howmany` dimension appears in the output of [`execute`](@ref) only when `howmany > 1`. Inputs may
  have any shape with the right number of elements.
- **Mapping to SCTL's C++ layout.** SCTL uses row-major arrays of shape `(howmany, d_0, …, d_{r-1})`, with the
  last index varying fastest, and `R2C`/`C2R` halve `d_{r-1}`. SCTL.jl passes `reverse(dims)` as SCTL's `dim_vec`.
  The memory is therefore identical: the fastest SCTL axis is the first Julia axis, which is why the halved
  dimension is `dims[1]`.
- Complex numbers are interleaved `(re, im)` doubles in SCTL, which is exactly the memory layout of `ComplexF64`.
  SCTL's `FFT::Dim(0)`/`Dim(1)` count doubles; [`dim`](@ref dim(::FFTPlan, ::Integer))`(plan, 1|2)` counts Julia
  elements.
- Frequencies are stored in standard FFT order: index `k` (1-based) along a full dimension of size `n` is
  frequency `k - 1`, or `k - 1 - n` for `k - 1 > n ÷ 2`. Along the halved dimension, index `k` holds frequency
  `k - 1`, for `k = 1, …, n ÷ 2 + 1`.

| SCTL C++                             | SCTL.jl                                   |
|:-------------------------------------|:------------------------------------------|
| `FFT<double>::Setup(type, howmany, dim_vec, nthreads)` | `FFTPlan(type, reverse(dim_vec); howmany, nthreads)` |
| `FFT_Type::R2C`, `C2C`, `C2C_INV`, `C2R` | `:r2c`, `:c2c`, `:c2c_inv`, `:c2r`   |
| `Dim(0)`, `Dim(1)` (doubles)         | `dim(plan, 1)`, `dim(plan, 2)` (elements) |
| `Execute(in, out)`                   | `execute(plan, x)`, `execute!(out, plan, x)` |

## Pitfalls and limitations

- **No normalization.** Applying a `:c2c_inv` plan to the output of a `:c2c` plan gives `N .* x` with `N = prod(dims)`, and the batch size is not included. Divide
  by `N` yourself.
- **Inverse of a real transform.** Build the `:c2r` plan with the *real* size `dims`, not with the size of the half
  spectrum. `:c2r` assumes a Hermitian spectrum: the parts of the input that are inconsistent with a real result are
  ignored, as in FFTW's `c2r`. Examples are the imaginary part of the zero mode and, for even sizes, of the Nyquist
  mode. FFTW and the fallback may treat a non-Hermitian input differently.
- **Element types.** `:r2c` requires real input and throws `ArgumentError` for complex input. Other numeric inputs,
  such as `Float32` or integers, are converted, which copies them. Arrays that are already `Array{Float64}` or
  `Array{ComplexF64}` are passed without a copy. In an FFTW build the shim always copies the data through aligned
  buffers, because the FFTW plans assert their alignment.
- **Size errors** throw `DimensionMismatch` before SCTL is called. Invalid plan arguments, such as an unknown type,
  an empty or non-positive `dims`, or `howmany < 1`, throw `ArgumentError`.
- **Threads.** `nthreads` sets the OpenMP team for one transform; the fallback parallelizes only large transforms.
  With FFTW, `nthreads` has an effect only if SCTL was also built with `SCTL_FFTW_THREADS`. Running the same plan
  concurrently from several Julia tasks has not been tested.
- **Performance.** Plans are cheap to create: FFTW uses `FFTW_ESTIMATE`. The fallback is ``O(N \log N)`` for smooth
  sizes. For sizes with large prime factors, each such factor goes through an ``O(p^2)`` generic radix stage, so
  prefer sizes with small prime factors when speed matters.
- **Lifetime.** A plan holds a C++ object that its finalizer frees. `close(plan)` frees it immediately; after that,
  any use of the plan throws `ArgumentError`.

## API reference

```@docs
FFTPlan
execute
execute!
dim(::FFTPlan, ::Integer)
input_size
output_size
close(::FFTPlan)
```
