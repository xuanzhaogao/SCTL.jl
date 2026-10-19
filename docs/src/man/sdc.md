# SDC ODE solver

SCTL's `sctl::SDC<double>` class integrates autonomous systems of ordinary differential equations

```math
u'(t) = F(u(t)), \qquad u(0) = u_0 \in \mathbb{R}^n,
```

by **spectral deferred correction** (SDC). SCTL.jl wraps it as the [`SDC`](@ref) type. A step of size ``\Delta t``
places `order` substep nodes ``\tau_0 = 0 < \tau_1 < \dots < \tau_{p-1} = 1`` on the step: they are the
second-kind Chebyshev points (see [`sdc_nodes`](@ref)). The step solves the Picard integral equation

```math
u(\tau \Delta t) = u_0 + \int_0^{\tau \Delta t} F(u(s))\, ds
```

on these nodes. It starts from the constant initial guess ``u \equiv u_0`` and then applies Picard (deferred
correction) iterations. Each iteration sweeps the substeps with an explicit residual time-stepping update, and the integral of
``F`` is evaluated with a spectral quadrature on the nodes. Each Picard iteration raises the accuracy by one order
up to the order of the collocation scheme. With the default `order` iterations, a method of order ``p`` has a
global error of ``O(\Delta t^{p})``. The scheme is fully explicit, so it suits non-stiff problems where high
accuracy at large step sizes is wanted.

Two drivers are provided:

- [`sdc_step`](@ref) takes one step of a given size and returns the solution and SCTL's error estimates. Use it
  for fixed-step integration or your own step-size control.
- [`sdc_solve`](@ref) is SCTL's adaptive driver (`SDC::AdaptiveSolve`). It picks the step sizes to reach a
  global error tolerance on ``[0, T]`` and can report each accepted step to a monitor callback.

## Tutorial

### One step and the substep solutions

The right-hand side is an in-place function `f!(dudt, u)`. For the harmonic oscillator
``u = (\cos t, \sin t)``:

```@example sdc
using SCTL, LinearAlgebra

osc!(du, u) = (du[1] = -u[2]; du[2] = u[1]; du)
exact(t) = [cos(t), sin(t)]

S = SDC(6)
dt = 0.5
s = sdc_step(S, osc!, [1.0, 0.0], dt; substeps=true)
(iters = s.iters, error = norm(s.u - exact(dt)),
 error_interp = s.error_interp, error_picard = s.error_picard)
```

`error_interp` estimates the truncation error of the solution interpolant on the step. `error_picard` is the
size of the last Picard update multiplied by `dt`. The maximum of the two is SCTL's local error estimate.

With `substeps=true`, the solutions at all substep nodes are returned as a `length(u0) × order` matrix. Column
`i` holds the solution at `dt * sdc_nodes(S)[i]`:

```@example sdc
τ = sdc_nodes(S)
U = s.substeps
(size = size(U), nodes = round.(τ; digits=4),
 max_substep_error = maximum(norm(U[:, i] - exact(dt * τ[i])) for i in eachindex(τ)))
```

### Order of convergence

Fixed-step integration over ``[0, 2]`` with ``N = 4, 8, 16`` steps. The observed rate
``\log_2(e_N / e_{2N})`` matches the order of the method:

```@example sdc
T = 2.0
println("order  errors (N = 4, 8, 16)            observed rates")
for p in (4, 6, 8)
    Sp = SDC(p)
    errs = map((4, 8, 16)) do N
        u = [1.0, 0.0]
        for _ in 1:N
            u = sdc_step(Sp, osc!, u, T / N).u
        end
        norm(u - exact(T))
    end
    rates = [log2(errs[k] / errs[k+1]) for k in 1:2]
    println(lpad(p, 5), "  ", rpad(string(round.(errs; sigdigits=3)), 33), round.(rates; digits=2))
end
```

### Controlling the Picard iterations

`picard_iters` caps the number of Picard iterations (`-1`, the default, means `order`). The iteration stops
earlier when the Picard update falls below `picard_tol`, or when it stagnates. Each iteration costs `order - 1`
evaluations of `f!`, on top of one evaluation at ``u_0``:

```@example sdc
for m in (1, 2, 4, 6)
    ncalls = Ref(0)
    r = sdc_step(S, (du, u) -> (ncalls[] += 1; osc!(du, u)), [1.0, 0.0], dt; picard_iters=m)
    println("picard_iters = $m: iters = $(r.iters), f! calls = $(ncalls[]), error = ",
            round(norm(r.u - exact(dt)); sigdigits=3))
end
```

With a tolerance, the iteration stops once the Picard update is small enough:

```@example sdc
r = sdc_step(S, osc!, [1.0, 0.0], dt; picard_iters=20, picard_tol=1e-8)
(iters = r.iters, error_picard = r.error_picard)
```

### Adaptive solve of a Kepler orbit with a step-size monitor

A Kepler orbit with eccentricity ``e = 0.7`` has period ``2\pi`` (semi-major axis 1, ``GM = 1``). Starting at
perihelion ``(1-e, 0)`` with speed ``\sqrt{(1+e)/(1-e)}``, the solution returns to ``u_0`` after one period.
The monitor `monitor(t, dt, u)` is called after every accepted step. Here it records the step sizes and the
distance to the center:

```@example sdc
e = 0.7
function kepler!(du, u)
    x, y, vx, vy = u
    r3 = (x^2 + y^2)^1.5
    du[1] = vx; du[2] = vy; du[3] = -x / r3; du[4] = -y / r3
    return du
end
u0 = [1 - e, 0.0, 0.0, sqrt((1 + e) / (1 - e))]
energy(u) = (u[3]^2 + u[4]^2) / 2 - 1 / hypot(u[1], u[2])

ts, dts, rs = Float64[], Float64[], Float64[]
monitor(t, dt, u) = (push!(ts, t); push!(dts, dt); push!(rs, hypot(u[1], u[2])))

sol = sdc_solve(SDC(12), kepler!, u0, 0.01, 2π, 1e-10; monitor)
(t_final = sol.t, err_estimate = sol.err, error_after_one_period = maximum(abs, sol.u - u0),
 energy_drift = abs(energy(sol.u) - energy(u0)), accepted_steps = length(ts))
```

The step size adapts to the dynamics. Steps are short near perihelion, where the orbit is fast, and long near
aphelion:

```@example sdc
i_peri, i_apo = argmin(rs), argmax(rs)
(dt_near_perihelion = (r = rs[i_peri], dt = dts[i_peri]),
 dt_near_aphelion = (r = rs[i_apo], dt = dts[i_apo]),
 times_increasing = issorted(ts), last_time = ts[end])
```

The monitor receives a fresh copy of `u`, so storing it, for example with `push!(traj, u)`, is safe.

### Van der Pol oscillator

Van der Pol with ``\mu = 5`` has fast transitions between slow phases. The table compares two tolerances; their
difference measures the actual error of the looser solve:

```@example sdc
μ = 5.0
vdp!(du, u) = (du[1] = u[2]; du[2] = μ * (1 - u[1]^2) * u[2] - u[1]; du)

steps = Dict{Float64,Vector{Float64}}()
sols = map((1e-8, 1e-10)) do tol
    dts_v = Float64[]
    r = sdc_solve(SDC(10), vdp!, [2.0, 0.0], 0.01, 20.0, tol; monitor=(t, dt, u) -> push!(dts_v, dt))
    steps[tol] = dts_v
    r
end
(u_T = sols[2].u, diff_between_tolerances = norm(sols[1].u - sols[2].u),
 steps = (length(steps[1e-8]), length(steps[1e-10])),
 dt_range = extrema(steps[1e-10]))
```

### Index-aware right-hand sides

SCTL also accepts right-hand sides that receive the deferred-correction indices (`SDC::Fn0`). With
`indices=true`, `f!` is called as `f!(dudt, u, correction_idx, substep_idx)`. Both indices are 0-based, as in SCTL.
This is useful for caching, for example reusing work from the previous correction at the same substep, or for
instrumentation. The call pattern of one step with `order = 4` and two Picard iterations:

```@example sdc
calls = Tuple{Int,Int}[]
r = sdc_step(SDC(4), (du, u, c, i) -> (push!(calls, (c, i)); du .= -u), [1.0], 0.1;
             picard_iters=2, indices=true)
(calls = calls, error = abs(r.u[1] - exp(-0.1)))
```

The first call `(0, 0)` evaluates ``F(u_0)``. Correction `c` then evaluates ``F`` at substeps `1, …, order-1`.

### When the tolerance cannot be met

If SCTL's accumulated error estimate exceeds `tol`, [`sdc_solve`](@ref) throws an `SCTLError`. With
`continue_with_errors=true` it instead returns the best solution and lets you inspect `t` and `err`; SCTL also
prints a warning to stderr in that case. Low orders are especially prone to this; see the pitfalls below:

```@example sdc
try
    sdc_solve(SDC(6), kepler!, u0, 0.01, 2π, 1e-8)
catch err
    err
end
```

## Data layouts

| Quantity | Julia type and shape | SCTL C++ counterpart |
|:--|:--|:--|
| state `u0`, returned `u` | `Vector{Float64}` of length `n` | `sctl::Vector<double>` of `Dim() == n` |
| right-hand side | `f!(dudt, u)`: overwrite `dudt` (length `n`) with ``F(u)`` | `SDC::Fn1` |
| right-hand side, `indices=true` | `f!(dudt, u, correction_idx::Int, substep_idx::Int)`, 0-based | `SDC::Fn0` |
| monitor | `monitor(t, dt, u)`: new time, step just taken, copy of `u(t)` | `SDC::MonitorFn` |
| `sdc_nodes(S)` | `Vector{Float64}` of length `order`, ``\tau_i = (1 - \cos(\pi i/(p-1)))/2`` | private `SDC::nds` |
| `sdc_step(...).substeps` | `Matrix{Float64}` of size `n × order`; column `i` = ``u(\tau_i\,\Delta t)`` | `Matrix<double>` `u_substep` of size `order × n`, row-major (same memory) |
| `sdc_step` result | `NamedTuple` `(u, error_interp, error_picard, iters, substeps)` | outputs of `SDC::operator()` |
| `sdc_solve` result | `NamedTuple` `(u, t, err)` | `AdaptiveSolve` output, return value and `error` |

The first column of `substeps` is `u0` and the last is `u`. The arguments `dudt` and `u` of `f!` are valid
**only during the call**. For `n ≤ 512` they are Julia buffers reused across calls. For larger `n` they are
zero-copy views of SCTL's internal storage. Never modify `u` and never keep a reference to either array.

## Pitfalls and limitations

- **Autonomous systems only.** ``F`` does not receive the time. For ``u' = F(t, u)``, append ``t`` as an extra
  state component with derivative ``1``.
- **Explicit scheme.** SDC here is explicit, so stiff problems force tiny steps. In `sdc_solve` that appears as a
  very large number of steps or as an `SCTLError`.
- **The adaptive driver is fragile at low orders.** `AdaptiveSolve` accumulates its per-step error estimates
  into `err` and aborts (`SCTLError`) as soon as `err > tol`. It accepts a step either when the estimate is below
  the per-step budget or when the error estimate has *stagnated*, and low-order schemes often accept steps with a
  large error this way. In our tests, orders ≤ 8 regularly failed on the eccentric Kepler and Van der Pol problems,
  even at moderate tolerances, while orders 10–12 succeeded. With `continue_with_errors=true` such runs did reach
  `T`, but with `err` up to ``10^4 \times`` `tol`. Prefer `order` 10–12 with `sdc_solve`, or write your own step
  control around [`sdc_step`](@ref).
- **`picard_iters = 0` is rejected.** SCTL would read an uninitialized error value in that case. The minimum is 1.
- **Output.** `AdaptiveSolve` prints one line per step to C++ `std::cout`. SCTL.jl suppresses this unless
  `verbose=true`. SCTL's warning for `continue_with_errors=true` goes to C++ `std::cerr`, which Julia's
  `redirect_stderr` does not always capture.
- **Threading.** SCTL calls `f!` and `monitor` only from the calling thread, never from an OpenMP region.
  Exceptions they throw abort the step or solve and are rethrown unchanged.
- **Lifetime.** An `SDC` owns C++ memory, released by the finalizer or eagerly by `close(S)`. After `close`
  every use throws an `ArgumentError`.

## API reference

```@docs
SDC
order(::SDC)
sdc_nodes
sdc_step
sdc_solve
Base.close(::SDC)
```
