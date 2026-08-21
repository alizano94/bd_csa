# 8. Implementation Map

Docs 01–07 describe the **legacy Fortran** — they are the specification.
This document describes **the code that now implements it**: what every file
does, how the pieces fit, and where each legacy behaviour ended up.

For the engineering narrative — what was tried, what failed, what the numbers
were — see [DEVLOG.md](../DEVLOG.md). This is the structural reference.

## 8.1 Repository layout

```
bd_csa/
├── include/bd_csa/     public C++ headers (the API surface)
├── src/                CPU reference implementation, FP64
├── cuda/               batched GPU backend
├── apps/               bdpd CLI drop-in, bench_cuda throughput harness
├── python/             pybind11 module + Gymnasium adapter
├── tests/              4 ctest suites + the Fortran oracle builder
├── data/               frozen golden inputs + GOLDEN.md (regression targets)
├── legacy/fortran_bd/  vendored Fortran, byte-identical to upstream, read-only
└── documentation/      docs 01–07 legacy spec, 08–11 this implementation
```

Two rules the layout encodes:

* **`legacy/` is never built in place.** The oracle script copies sources out to
  `build/` before instrumenting them, so the reference tree cannot drift.
* **Tests read `data/`, never `legacy/`.** A test run cannot overwrite the
  golden inputs.

## 8.2 The dependency graph

```
                    config.hpp ──────────────┐
                    (Config, PhysicsOptions) │
                          │                  │
          ┌───────────────┼──────────────┐   │
          ▼               ▼              ▼   ▼
     forces.cpp     mobility.cpp   order_params.cpp   io.cpp
     (F on every    (D̂ lookup,    (ψ₆, C₆, R_g, RC)  (start.txt)
      particle)      ∇D̂)                │
          │               │             │
          └───────┬───────┘             │
                  ▼                     │
            simulator.cpp ──────────────┘        cuda/sim_cuda.cu
            (SimulatorCpu)                       (SimulatorCuda)
            rng.hpp (Philox)                     cuda/kernels.cuh
                  │                                     │
                  └──────────────┬──────────────────────┘
                                 ▼
                 apps/bdpd_main.cpp     python/bindings.cpp
                 (CLI drop-in)          (VecSimulator → bd_csa.Simulator)
                                                │
                                        python/bd_csa/gym_env.py
                                        (BDVectorEnv)
```

`Config` is the only thing everything shares — it replaces all 15 COMMON blocks
([04-code-reference.md](04-code-reference.md) §4.4). It is immutable after
construction, which is what makes many simulations safe to run concurrently.

## 8.3 State and the batching argument

```cpp
struct State {              // include/bd_csa/state.hpp
  std::vector<double> x;    // nm
  std::vector<double> y;    // nm
};
```

That is the *entire* mutable state. There is no `z` (the system is 2-D, every
particle pinned at `hlev`), no velocities (overdamped), no RNG state
(counter-based), no accumulated mobility (recomputed from positions).

Therefore **a step is a pure function of `(positions, seed, λ)`**. That single
property is what licenses everything else: batching thousands of environments,
running them as independent CUDA blocks, and reproducing any run exactly from
three numbers.

## 8.4 File-by-file

### `include/bd_csa/config.hpp` — `Config`, `PhysicsOptions`

`Config` holds every parameter, in the Fortran's internal units (nm, ms) after
rescaling. `Config::from_run_txt()` parses the positional `run.txt`
([05-io-formats.md](05-io-formats.md) §5.2) read-for-read against `main.f:74-156`,
*including* the two label/value pairs the Fortran reads and discards, then applies
the rescalings from `main.f:170-177` and `223-225`.

Two flags deserve attention:

| Field | Default | Effect |
|---|---|---|
| `legacy_float_literals` | `false` | Reproduce Fortran's `REAL*4` rounding of undecorated literals (`kb`, `1e18`, the field-correction polynomial). Test-only — it is what lets the differential test isolate algebra from constants. |
| `kb` | `1.380658e-23` | The pre-2019 value the Fortran hard-codes. Kept deliberately: switching to CODATA 1.380649e-23 shifts every force by 6.5e-6 relative and would put the differential gate out of reach. |

`PhysicsOptions` is the set of deliberate departures from the legacy — see §8.7.

### `src/forces.cpp` — the force model

Implements [01-physical-model.md](01-physical-model.md) §1.4 exactly: DLVO
repulsion, the asymmetric induced-dipole force, and the nDEP trap.

The dipole block is the part to read carefully. `common_x`/`common_y` are the
classical fixed-dipole force; the four `fel_{x,y}{i,j}` expressions add the
position-dependent-dipole gradient terms, with `x[j]`/`F3` for particle `i` and
`x[i]`/`F2` for particle `j`. **They are not equal and opposite, and must not be
made so.**

`grad_emag_sq()` is the closed form of `∇|E|²`, factored so the `1/ρ` singularity
at the origin cancels analytically:

```
|E| = A·ρ  with  A = (4/dg)·f(ρ/1000)
∇|E|² = 2·A·(d|E|/dρ)·(x, y)          — no division by ρ
```

`DepGradient::kFiniteDifference` retains the legacy forward difference for the
differential test only. See §8.8 for why it cannot be used in production.

### `src/mobility.cpp` — `MobilityTable`

Loads `2dtabledssnp300.txt` and offers two lookups:

* `lookup_nearest()` — piecewise constant, bit-compatible with `caldss.f`
  including the negative `R_g` stride, the `count == 0 → dssmax` rule, and the
  "more compact than tabulated → `dssmin`" rule.
* `lookup_smooth()` — smoothstep-weighted bilinear interpolation, C¹ at bin
  edges, and it returns `d(D̂)/d(distance)`.

The smooth variant exists **because of** the drift term: `∇·D` of a piecewise
constant table is zero almost everywhere with delta spikes at the bin edges, so
the Itô correction is meaningless without a differentiable interpolant. The
`resolved()` helper applies the out-of-range and unpopulated-bin rules *before*
interpolation, so the interpolant never straddles a hole.

### `src/order_params.cpp` — `compute_order_params()`

ψ₆, C₆, R_g, RC per [03-order-parameters.md](03-order-parameters.md), with the
§7.2 sparse-array defect fixed: window-passing particles are compacted into a
dense array and every loop uses the same dense count. For the shipped config
(`expbox` = the full cell) nothing is ever excluded, so the fix is a no-op and
results match the Fortran exactly — it only matters if `expbox` is ever narrowed.

The legacy synthetic measurement noise (`var * gasdev`) is omitted: `var = 0` in
the shipped config, so it only ever consumed RNG draws.

### `include/bd_csa/rng.hpp` — `Philox`

Philox 4x32-10, keyed on `(seed, env, particle, step)`. Counter-based, so the
value is a pure function of the key: any draw can be made independently, in any
order, on any thread. Replaces `ran2` + `gasdev`, whose defects are catalogued in
[07-porting-notes.md](07-porting-notes.md) §7.1.

Box–Muller rather than the polar-rejection form, because rejection consumes a
*variable* number of uniforms and would destroy the fixed key→value mapping.

### `src/simulator.cpp` — `SimulatorCpu`

The reference integrator. Deliberately scalar and readable: its job is to be
obviously correct so it can validate the GPU, not to be fast.

```cpp
dx = D*fx*fac1*dt + sqrt(D)*zx*fac2*dt  (+ drift_coef * gdx  if enabled)
```

The drift coefficient is derived without ever reconstructing `D₀` or the
viscosity. Since the random step is `√(2·D̂·D₀·dt) = √D̂·fac2·dt`, we have
`D₀·dt = (fac2·dt)²/2`, so the Itô term `∇D·dt` becomes

```cpp
drift_coef = 0.5 * (fac2*dt) * (fac2*dt);      // then × ∇D̂
```

**Documented limitation, carried in the source:** only the *local radial*
gradient of `D̂` is included. `D̂` also depends on the cluster's `R_g`, which is a
collective coordinate; that many-body contribution is omitted.

### `cuda/kernels.cuh` + `cuda/sim_cuda.cu` — `SimulatorCuda`

See §8.5.

### `apps/bdpd_main.cpp` — CLI drop-in

Same argv as the legacy (`bdpd <lambda> <seed>`), reads `run.txt`/`start.txt`
from the working directory, writes `out_param.json`, `bd_xyz1.txt` and `op1.txt`
in the legacy formats. An existing driver can call it unchanged.

Extra flags: `--legacy`, `--mobility-interval N` (§8.7).

One necessary difference: the legacy reported its *evolved* `iDummy` in the JSON
so the driver could chain the stream. A counter-based RNG has no evolving state,
so the next seed is derived deterministically from `(seed, steps)` via an LCG
mix. Chaining still works; the values differ.

### `python/bindings.cpp` — `bd_csa.Simulator`

Wraps `SimulatorCpu`/`SimulatorCuda` behind one `VecSimulator` with a `device`
switch. Positions cross as `(n_envs, np, 2)` float64 **in nanometres**.

### `python/bd_csa/gym_env.py` — `BDVectorEnv`

A thin Gymnasium adapter *on top of* the simulator, never underneath it: action
rescaling from `[-1,1]` to `λ`, observation passthrough, and lazy `gymnasium`
import. Reward and termination are left to the caller, matching the legacy
driver which imported its own `reward_function`.

## 8.5 The CUDA design

**One block per environment. One kernel launch per episode.**

```
grid  = n_envs blocks
block = 320 threads (one per particle, 20 idle tail threads)
shared memory per block ≈ 35.6 KB:
    sx, sy          FP64   2 × np × 8 B      positions, resident all episode
    red             FP64   320 × 8 B         reduction scratch
    fx, fy, mob,    FP32   7 × np × 4 B      forces, mobility, ∇D̂, list anchors
    gdx, gdy, bx, by
    nbr, nnb        u16    np × 33 × 2 B     Verlet lists + counts
```

Global memory is touched exactly twice per episode — once to load positions,
once to store them. Everything else lives in shared memory.

This works only because **there is no cross-block communication**: every
reduction the physics needs (centroid, `R_g`, list staleness) is per-environment,
hence per-block. That is a property of the model, not a trick.

### The inner loop

```
for step in 0..nsteps:
    if step % mob_interval == 0:      block-reduce centroid and R_g, refresh D̂
    if step == 0 or (step % 64 == 0 and max displacement > skin/2):
                                      rebuild Verlet lists
    force loop over each particle's neighbour list
    __syncthreads()
    position update (FP64 accumulator) + Philox noise
    __syncthreads()
```

### Why the numbers are what they are

| Constant | Value | Reason |
|---|---|---|
| `kBlock` | 320 | 256 gave threads 0–43 two particles each and the block ran at their pace — ~50 % of the force loop wasted. 320 = one particle per thread. |
| `kMaxNb` | 32 | Measured max 21 neighbours at a 5.5a list radius on the golden config; 32 leaves headroom for densification during annealing. |
| Verlet skin | 0.5a | 16.3 neighbours mean; displacement is diffusive at ~2e-3 a/step so a list survives ~15,000 steps. |
| `kRebuildCheck` | 64 | 64 steps moves a particle ~0.016a, far below the 0.25a half-skin — the check cannot be missed, and costs 1/64 of the reductions. |

`kOverflow` (0xFFFF) is stored when a particle exceeds `kMaxNb`; that particle
falls back to a full O(N) scan for the step. **Overflow degrades speed, never
correctness** — which matters because the measured max of 21 is a property of the
*initial* configuration, not a bound on the annealed one.

### Precision split

FP64 for positions and the displacement accumulator; FP32 for the force loop.
On consumer Ada, FP64 runs at 1/64 of FP32 rate, so the force loop being FP32 is
what makes the port worth doing; the accumulator being FP64 is what stops 10⁶
tiny increments washing out against coordinates of order 2×10⁴ nm.

**The separations are differenced in FP64 and only then narrowed.** Computing
`x[j]-x[i]` in FP32 would be catastrophic: each coordinate carries ~1.2e-3 nm of
representation error, and the DLVO exponential's 0.1/nm sensitivity turns that
into ~2e-4 of relative force error.

### Two optimizations that were reverted, and why

Recorded here so nobody re-attempts them:

* **`rsqrtf` for `1/r` and `r` together** — nearly doubled RMS force error
  (9.75e-6 → 1.80e-5). `r` feeds the DLVO exponential, so `rsqrtf`'s ~2.4e-7
  becomes ~7e-5 of force error over a 3000 nm separation. The final form keeps
  the correctly-rounded `sqrtf` for `r` and uses `__frcp_rn` once for `1/r`,
  which still removes six of seven divisions.
* **Accurate `expf` and Kahan summation** — changed RMS error by 0.5 % and 0.1 %
  respectively, i.e. not at all. The budget is per-term FP32 rounding, not the
  exponential and not the accumulation.

## 8.6 Where each legacy behaviour ended up

| Legacy behaviour | Ref | Disposition |
|---|---|---|
| `ran2` unseeded for positive seeds | 7.1 | **Fixed** — Philox, keyed properly |
| `gasdev` variable uniform consumption | 7.1 | **Fixed** — Box–Muller, fixed cost |
| Discarded z Gaussian draw every step | 2.1 | **Removed** — 2 draws per particle, not 3 |
| Sparse `rx`/`ry` when window excludes particles | 7.2 | **Fixed** — dense compaction |
| Missing `∇·D` drift | 7.3 | **Added** (needs `smooth_mobility`); local radial term only |
| Single-precision centroid in `caldss` | 7.4 | **Fixed** — FP64 everywhere |
| Mobility frozen for the whole episode | 7.5 | **Configurable**, defaults to every step |
| Duplicated output block, double RNG consumption | 7.6 | **Merged** |
| Uninitialised `ControP` | 7.7 | **Dropped** |
| Half-implemented periodicity | 7.8 | **Off by default**, `periodic` flag |
| Discontinuous `Fhw` contact branch | 7.9 | **Continuous capped repulsion** by default |
| Dead code (~40 % of the source) | 7.10 | **Not ported** |
| Positional `run.txt` | 7.11 | Still parsed for compatibility; `Config` is the real interface |
| Process-per-step file I/O | 7.11 | **Replaced** by the in-process batched API |

## 8.7 `PhysicsOptions` — the corrections, and how to turn them off

```cpp
struct PhysicsOptions {
  long mobility_update_interval = 1;     // legacy: once per episode
  bool enable_divD_drift        = true;  // legacy: absent
  bool smooth_mobility          = true;  // legacy: nearest bin
  bool periodic                 = false; // legacy: wrap x,y (inconsistently)
  bool continuous_overlap       = true;  // legacy: constant Fhw branch
};
```

`bdpd --legacy` sets the first three back to legacy behaviour, which is the
configuration under which the port's observable distributions are statistically
indistinguishable from the Fortran's.

**These are not cosmetic.** With the corrections on, ψ₆ shifts by −8 % relative
to the legacy simulator over a 20-seed ensemble. ψ₆ is RL observation #1, so a
policy trained against the legacy sees a shifted state distribution under the
defaults. See [11-validation-and-status.md](11-validation-and-status.md) §11.3.

## 8.8 Three things that look like micro-optimizations but are load-bearing

1. **The analytic `∇|E|²` is mandatory, not preferable.** The legacy forward
   difference uses `h = 1e-3` nm on a quantity of order 1: the two `|E|²` values
   differ in the 10th significant digit. In FP64 that leaves ~6 usable digits
   (which is why the differential test cannot gate the DEP term below 2e-6); in
   FP32 it is pure cancellation noise — the test asserts >50 % error.
2. **Pair separations must be differenced in FP64.** See §8.5.
3. **The dipole force must stay non-Newtonian.** `Σ F ≠ 0` is physically correct
   here (momentum is exchanged with the external field). The tier-2 test asserts
   `|ΣF| > 0` specifically to catch a well-meaning future "fix".
