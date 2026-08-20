# Development log

Chronological record of the Fortran → C++/CUDA port. Written to be picked up
cold: every number here was measured on this machine, and every wrong turn is
kept in rather than tidied away, because the wrong turns are where the
non-obvious constraints came from.

**Machine:** RTX 4060 Ti (Ada, cc 8.9, 8 GB), CUDA 13.2 toolkit / 13.0 driver,
g++ 13.3, gfortran 13.3, 16 cores.

**Approved plan:** `~/.claude/plans/hashed-snuggling-raccoon.md`
**Spec:** `documentation/` (7 docs, reverse-engineered from the legacy program)
**Golden data + measurements:** `data/GOLDEN.md`

---

## Orientation: what the program is

300 charged colloids in a 2-D quadrupole electrode trap. A single control
parameter `λ` (the RL action) sets the AC field strength, which simultaneously
drives dipole–dipole attraction and the negative-dielectrophoretic trap. Raising
and lowering `λ` anneals a crystal out of a disordered fluid. The observables
`(ψ₆, C₆/6)` are the RL state.

One legacy invocation = 10⁶ integration steps = 100 s simulated ≈ 5.3 min of
wall clock, single-core, and **cannot be batched** — 15 COMMON blocks of global
mutable state, `SAVE`d RNG, file-based I/O. That serial cost is the entire
motivation for the port.

---

## Phase M0 — foundations and the oracle

### What was built
- `data/` — frozen golden inputs (`run.txt`, `start.txt`, `2dtabledssnp300.txt`),
  copied so tests never read the pristine `legacy/` tree.
- `Config` (`include/bd_csa/config.hpp`) — all 15 COMMON blocks collapsed into
  one immutable struct. `from_run_txt()` mirrors `main.f:74-156` read-for-read,
  including the two label/value pairs that are read and **discarded** (`lambda`
  at run.txt lines 43-44, `idummy` at 61-62) because both actually come from argv.
- `MobilityTable` — the `2dtabledssnp300.txt` lookup, 30 R_g rows × 50 distance
  columns. Note the R_g axis has a **negative stride** (`rgdsmin = 26500`,
  `delrgdsmin = -250`): row 0 is the *largest* cluster.
- `compute_order_params()` — ψ₆, C₆, R_g, RC.
- `tests/oracle/build_force_oracle.sh` — builds instrumented copies of the
  Fortran that dump forces and stop.

### Finding 1: the documentation's regression targets are wrong

The docs cite `R_g = 21014.5 nm`, `C₆ = 4.2667`, `RC = 0.76125` for "the known
configuration". Those **do not** correspond to the shipped `start.txt`:

| | documented | actual |
|---|---|---|
| `R_g` | 21014.5 nm | **21015.90986 nm** |
| `C₆` | 4.2667 | **4.28000** |
| `RC` | 0.76125 | **0.76363** |
| `ψ₆` | — | **0.40508** |

Verified the vendored `start.txt` is byte-identical to upstream SAC3, and
reproduced R_g independently from the raw coordinates. The *algorithms* in the
docs are correct — the RC formula reproduces both its own worked example and the
true value when fed the right inputs — only the configuration differs. Recorded
in `data/GOLDEN.md`; tests use the measured values.

Also verified the t=0 row is invariant to both `λ` and seed (ran `(30,-7)`,
`(5,12345)`, `(100,1)` — identical to every printed digit), which makes it a
clean deterministic fixture.

### Finding 2: the shipped binary is the wrong architecture
`legacy/fortran_bd/bdpd` is a **Mach-O arm64** executable. Always rebuild.
Baseline on this machine: **6.23 s / 20,000 steps = ~312 µs/step**.

---

## Phase M1 — CPU FP64 reference

### What was built
- `src/forces.cpp` — DLVO + asymmetric dipole + DEP trap.
- `src/simulator.cpp` — Euler–Maruyama integrator, mobility refresh.
- `include/bd_csa/rng.hpp` — Philox 4x32-10 counter-based RNG.
- `apps/bdpd_main.cpp` — argv-compatible CLI drop-in.

### The correctness gate: 1.118e-14

This was the milestone that mattered. Getting there took three iterations, each
of which taught something that changed the design.

**Attempt 1 stalled at 2.03e-06.** That number is diagnostic. The legacy DEP
gradient is a forward difference `(|E|(x+h)² − |E|(x)²)/h` with `h = 1e-3` nm.
Both terms are ≈0.77 and differ by ≈7.7e-11 — a relative cancellation of 1e-10.
Against FP64 epsilon (2.2e-16) that leaves ~2.2e-6 of accuracy, matching the
observed error exactly.

> The approved plan claimed FP64 "retains ~9 digits" here. **It retains about 6**,
> and the residue is compiler-dependent, so it can never be reproduced across
> languages. This is the single most important numerical finding of the port.

**Fix:** split the oracle into two variants — `pair` (DEP accumulation removed)
and `full`. The pair forces carry all the physics that is easy to get wrong and
are well conditioned, so they get the strict gate.

**Attempt 2 stalled at 7.9e-08** — the signature of float32 epsilon. Fortran
defaults undecorated literals to `REAL*4`, so `kb = 1.380658E-23`, the `1e18`
unit scale, and the field-correction polynomial all carry only ~7 significant
digits despite living in `double precision` variables. Fortran also evaluates
left-to-right with promotion at each operator, so `1e18*0.75` is computed
*entirely in single precision* before meeting a double.

**Fix:** `Config::legacy_float_literals` (default **off**) reproduces that
rounding. The shipped port uses full-precision constants; the flag exists so the
differential test can prove the residue is the constants and not the algebra.

**Result: 1.118e-14** — round-off. The port's force algebra is exact.

The test also asserts Newton's third law **fails** (|ΣF| = 5.7e-5). The dipole
moments are position-dependent, so the force on i is genuinely not minus the
force on j; the legacy computes two separate expressions (`felxnew` /
`felxnew2`). Symmetrising it would be a physics bug, so the test guards against
a well-meaning future refactor.

### Performance

| | 20k steps | vs Fortran |
|---|---|---|
| Fortran oracle | 6.23 s | — |
| CPU port, mobility frozen (legacy-equivalent) | 4.74 s | **1.31×** |
| CPU port, mobility every step (fixed physics) | 4.82 s | 1.29× |

The useful result is the third row: refreshing mobility **every step costs 1.7%**,
where the legacy could only afford it once per episode. That justified defaulting
`mobility_update_interval = 1` and told us the GPU could do the same in-block
reduction essentially for free.

### Physics fixes applied
Per the plan's "fix all" decision: counter-based RNG properly seeded; the wasted
third Gaussian draw per particle (the discarded z component) removed; duplicated
output block merged; dense compaction in `conn6calc`; FP64 centroid everywhere;
periodic wrapping dropped; continuous capped repulsion; dead code not ported.

Two needed real design work rather than transliteration:

1. **Analytic `∇|E|²`** replacing the forward difference — *mandatory*, not
   optional, because in FP32 the difference is pure cancellation noise (the test
   asserts >50% error). Also removes 3 `EMAG` calls per particle per step.
2. **The `∇·D` drift term** cannot be added to a piecewise-constant lookup table
   — its gradient is zero almost everywhere with spikes at bin edges. Required
   building a C¹ smoothstep interpolation of the mobility table first, then
   deriving `∇D` from it. Implemented as
   `drift = ∇D̂ · (fac2·dt)²/2`, which follows from `D₀·dt = (fac2·dt)²/2` and so
   needs no reconstruction of D₀ or the viscosity.
   **Caveat:** only the local radial gradient is included. D̂ also depends on the
   collective R_g; that many-body term is omitted and documented in the source.

---

## Phase M2 — CUDA persistent kernel

### Architecture
One block per environment; all 300 particles resident in **shared memory** for
the whole episode (12,848 B/block). The hot loop touches global memory exactly
twice — once to load, once to store. The entire 10⁶-step episode is a **single
kernel launch**; per-step launches would cost ~5 s of pure overhead.

This works because there is no cross-block communication: every reduction the
physics needs (centroid, R_g) is per-environment, hence per-block.

Precision split: FP64 positions and displacement accumulator, FP32 force loop.
Separations are differenced **in FP64 before narrowing** — computing
`x[j]-x[i]` in FP32 would be catastrophic, since coordinates are ~2×10⁴ nm and
the DLVO exponential's 0.1/nm sensitivity turns the resulting 1.2e-3 nm
representation error into ~2e-4 of relative force error.

### Three build/environment problems

1. **Wrong GPU architecture, silently.** `cuobjdump` showed an **sm_75** cubin on
   an sm_89 device. `CMAKE_CUDA_ARCHITECTURES` must be set **before**
   `enable_language(CUDA)` — mine came after, so CMake's default won.
2. **Toolkit newer than driver.** nvcc 13.2 vs a driver supporting 13.0. PTX JIT
   requires driver ≥ toolkit, so the fallback path failed with *"the provided PTX
   was compiled with an unsupported toolchain"*. Fixed by targeting `89-real` —
   SASS only, no embedded PTX, nothing ever JIT'd. Override with
   `-DBD_CSA_CUDA_ARCH=<arch>-real` on other hardware.
3. **`option()` coerced a path to OFF.** `option(BD_CSA_ENABLE_CUDA "..."
   ${CMAKE_CUDA_COMPILER})` cached OFF, because a filepath is not a boolean
   constant. Now computed explicitly.

### Finding: my error metric was wrong

The force error looked like 1.4e-3, 10× over budget. I guessed two causes —
`__expf` accuracy on the dominant DLVO term, and FP32 accumulation over 299
partially-cancelling terms — implemented both, and **neither changed anything**:

| variant | RMS error |
|---|---|
| `__expf`, plain summation (kept) | 9.751e-06 |
| accurate `expf` | 9.798e-06 |
| accurate `expf` + Kahan summation | 9.812e-06 |

The actual cause was the metric. Particle 175 sits near a **force null** where
the inward DEP trap cancels the pair repulsion, carrying 1/99 of the mean force;
normalising by its own magnitude inflated an absolute error that is irrelevant to
the dynamics. Normalised to the RMS force scale the real budget is
**max 9.06e-5, RMS 9.70e-6**. Both optimizations were reverted and the
measurements recorded in the source so they are not re-attempted.

### Cross-validation
Over 2000 steps GPU and CPU agree on R_g to 2.5e-10 and C₆ exactly — the Philox
streams coincide by construction, leaving only the float-vs-double Box–Muller
difference. GPU runs are bit-identical for a fixed seed.

### Throughput (20k steps, pre-optimization)

| n_envs | µs/env-step | vs Fortran |
|---|---|---|
| 1 | 381.9 | **0.8×** |
| 16 | 24.1 | 12.9× |
| 64 | 7.55 | 41.3× |
| 1024 | 7.28 | 42.8× |
| 4096 | 6.41 | **48.6×** |

The single-environment case being *slower* than the Fortran is exactly what the
plan predicted: this is a throughput win, not a latency win. Saturation arrives
by ~64 environments, earlier than the ≥128 blocks estimated.

**48× is well short of the ~500× the plan projected.** See M3.

### Optimizations, one of which backfired

- **Block size 256 → 320.** With `np = 300`, a 256-thread block gave threads
  0–43 two particles and everyone else one, so every block ran at the pace of the
  double-loaded threads — wasting ~50% of the force loop. 320 gives one particle
  per thread with only 20 idle tail threads. Required replacing the
  power-of-two reduction tree with a warp-shuffle reduction.
- **`rsqrtf` — reverted.** Hoisting a reciprocal-sqrt to replace the sqrt and
  seven divisions nearly doubled RMS error (9.75e-6 → 1.80e-5) and broke the
  gate. Asymmetric sensitivity is the reason: `r` feeds the DLVO exponential, so
  `rsqrtf`'s ~2.4e-7 becomes ~7e-5 of force error over a 3000 nm separation,
  while `1/r` merely scales a direction and enters linearly. Final form keeps the
  correctly-rounded `sqrtf` for `r` and uses `__frcp_rn` once for the reciprocal
  — accuracy restored (RMS 9.70e-6) with six of seven divisions still gone.

---

## Phase M3 — closing the gap to the projected speedup

M2 ended at 48.6×, against a plan that projected ~500×. Two hypotheses were
tested; **the first was wrong and the second was right**, and the difference
between them is the main lesson of this phase.

### Hypothesis 1 (wrong): the FP64 subtractions dominate

Only ~15 of 299 partners lie inside the 5a cutoff (measured on the golden
configuration — a 20× redundancy), yet the loop paid two FP64 subtractions per
pair to compute the separation, and consumer Ada runs FP64 at 1/64 of FP32 rate.
Two FP64 ops ≈ 128 FP32-equivalents, apparently dwarfing the ~50 FP32 ops of the
pair term itself.

So I added FP32 shadow copies of the positions and rejected far pairs before
ever touching the FP64 arrays, with a 1 nm margin covering the ~2.4e-3 nm
shadow error (the margin can only admit extra pairs, never drop real ones, so
the physics is provably unchanged — confirmed by bit-identical force errors).

**Measured: 6.41 → 5.52 µs/env-step. A 14% gain, not the predicted 15×.**

### Hypothesis 2 (right): the loop trip count dominates

The 14% result falsifies hypothesis 1 and points at the answer: the cost was
never the work *inside* the loop, it was the 299 iterations themselves — load,
subtract, compare, branch — which the cheap rejection does not avoid. The only
fix is to stop iterating over non-neighbours at all.

**Verlet neighbour lists**, sized from measurement rather than guesswork:

| list radius | mean nb | max nb | shared for 300 particles |
|---|---|---|---|
| 5.25a | 15.7 | 20 | 14.1 KB |
| **5.5a (0.5a skin)** | **16.3** | **21** | **14.1 KB** |
| 6.0a | 21.0 | 28 | 18.8 KB |

Displacement is diffusive at ~2e-3 a/step, so a 0.5a skin survives ~15,000
steps. Staleness is checked with an actual max-displacement reduction (not a
fixed interval) every 64 steps — 64 steps moves a particle ~0.016a, far below
the 0.25a half-skin, so the check cannot be missed while costing 1/64 of the
reductions.

Capacity is 32 with **correct overflow handling**: a particle whose true
neighbour count exceeds 32 stores a sentinel and falls back to a full O(N) scan
for that step. Overflow degrades speed, never correctness — which matters
because the cluster densifies during annealing and the measured max of 21 is a
property of the *initial* configuration, not a bound.

Shared memory 18.2 → 35.6 KB, so occupancy drops from ~4 to ~2-3 blocks/SM. Worth
it against a 14× trip-count reduction.

### Result

| n_envs | µs/env-step | vs Fortran |
|---|---|---|
| 1 | 34.70 | **9.0×** |
| 16 | 2.06 | 151× |
| 64 | 1.01 | 310× |
| 1024 | 0.97 | 321× |
| 4096 | **0.94** | **331×** |

Cumulative: **48.6× → 331×**, a 6.8× gain across M3. Force errors stayed
bit-for-bit identical (max 9.055e-05, RMS 9.702e-06) throughout, which is the
strongest evidence the lists are physics-neutral.

**The single-environment framing has changed.** In M2 one environment ran at
0.8× — slower than the Fortran — and the honest summary was "throughput, not
latency". It is now **9×**, so a single episode is genuinely faster too. Batching
still buys another ~37× on top.

### Validation tier 5 — statistical trajectory

`tests/compare_trajectories.py` runs an ensemble of seeds through both the
Fortran oracle and the port and compares observable distributions with a
two-sample KS test (implemented inline, no scipy dependency).

Two arms, and the distinction is the point:
- **legacy mode** (`bdpd --legacy`: frozen mobility, nearest-bin lookup, no
  drift) — the only difference from the Fortran is the RNG. This is the
  correctness check.
- **fixed mode** (defaults) — quantifies how much the deliberate physics
  corrections actually move the observables.

Seeds are **negative** so the legacy `ran2` initialises its shuffle table
properly, giving the Fortran its best case rather than exercising the §7.1
defect.

#### Result: 20 seeds × 50,000 steps (5 s simulated) at λ = 30

**Legacy mode — the correctness check. PASSED.**

| observable | Fortran | bd_csa | KS D | p |
|---|---|---|---|---|
| ψ₆ | 0.6367 | 0.6253 | 0.250 | 0.497 |
| C₆ | 5.1107 | 5.1080 | 0.200 | 0.771 |
| R_g (nm) | 19644.6 | 19633.9 | 0.200 | 0.771 |

All three are statistically indistinguishable. With the physics held at the
legacy behaviour, the only difference is the random stream — so this says the
port reproduces the legacy *dynamics*, not merely the legacy *forces*. Combined
with the 1.1e-14 force differential, the deterministic and stochastic halves are
now both pinned.

**Fixed mode — the physics corrections DO move the observables. Significantly.**

| observable | Fortran | bd_csa fixed | change | p |
|---|---|---|---|---|
| ψ₆ | 0.6367 | 0.5849 | **−8.1%** | 0.023 |
| C₆ | 5.1107 | 5.0150 | −1.9% | 0.001 |
| R_g (nm) | 19644.6 | 19704.2 | +0.3% | 0.000 |

This is the single most consequential result for downstream use, and it is why
the two-arm design was worth the extra runtime. The corrections — mobility
refreshed every step, the Itô `∇·D` drift term, C¹ interpolation of the mobility
table, continuous contact repulsion — produce clusters that are measurably
**less ordered and slightly larger**.

The direction is physically sensible. The legacy omitted the `∇·D` drift, which
biases particles toward low-mobility regions (the cluster centre); restoring it
removes that spurious inward pull, so R_g rises and the lattice is marginally
less well formed. The two legacy omissions cancelled only because mobility was
frozen — refreshing it without the drift term would have made the bias real,
which is exactly the trap flagged in the plan.

**Implication:** ψ₆ is RL observation #1, and it shifts by 8%. **A policy trained
against the legacy simulator will see a shifted state distribution under the
fixed-physics default.** Either retrain, or run with `--legacy` (equivalently
`PhysicsOptions` with frozen mobility, nearest-bin lookup and drift off) to
reproduce what previously trained policies experienced. This is a real decision,
not a formality.

---

## Phase M4 — Python bindings

`python/bindings.cpp` exposes `Config`, `PhysicsOptions`, `OrderParams` and a
batched `Simulator` with a `cpu`/`cuda` backend switch. Positions cross the
boundary as `(n_envs, np, 2)` float64 **in nanometres** — the legacy text format
uses multiples of `a`, and `read_start_txt` converts on read so the unit
mismatch cannot be made silently.

Per the plan, the core knows nothing about Gym; an adapter belongs on top in
Python.

**Known limitation, deliberately surfaced rather than hidden:** the CUDA backend
takes one `lambda` per launch, so a per-environment lambda currently raises
rather than silently serialising. For RL — where each environment has its own
action — this is the next thing to fix; it needs `lambda` moved into a
per-block array argument, which is a small kernel change.

**Blocked:** the build fetches pybind11 via `FetchContent`, and that CMake
configure was denied by a sandbox rule because it downloads from GitHub. The
code and CMake wiring are complete; the module builds once that command is
approved:

```sh
cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release   # fetches pybind11
cmake --build build/cmake -j16
PYTHONPATH=python python3 -c "import bd_csa; print(bd_csa.cuda_available())"
```

There is no pip and no pybind11 on this machine, so fetching (or a system
install of `pybind11-dev`) is required either way.

`python/bd_csa/gym_env.py` provides `BDVectorEnv`, a thin Gymnasium adapter that
imports `gymnasium` lazily so the rest of the package works without it. Reward
and termination are left to the caller, matching the legacy driver which
imported its own `reward_function`.

---

## Where things stand

| | status |
|---|---|
| M0 foundations, golden data, oracle | done |
| M1 CPU FP64 reference, force differential at 1.1e-14 | done |
| M2 CUDA persistent kernel, tier-4 accuracy | done |
| M3 Verlet lists, 331× throughput | done |
| M3 tier-5 statistical validation | done — legacy mode matches the Fortran (p ≥ 0.50) |
| M4 Python bindings | written; build blocked on a network fetch |

Test suite: 4 ctest suites, all passing
(`tier1_constants`, `tier2_forces`, `integrator`, `tier4_cuda`).

`legacy/fortran_bd/` has been verified byte-identical to upstream SAC3
throughout — every build is out-of-tree.

### Next steps, in the order I would take them

1. **Approve the pybind11 fetch** and build the module (one command, above).
2. **Per-environment lambda on the GPU.** Currently one `lambda` per launch,
   which blocks the main RL use case. Move it to a per-block array argument —
   small kernel change, large payoff.
3. **Investigate the remaining performance gap.** 331× against a ~500×
   projection. The two things I would measure next, with `ncu` rather than by
   guessing (this session's guesses were wrong twice): shared-memory bank
   conflicts in the neighbour-list gather, and whether occupancy is limited by
   the 35.6 KB shared footprint (dropping `kMaxNb` to 24 with the overflow
   fallback would buy a third block per SM).
4. **Order parameters on device.** Currently computed host-side after download;
   fine while episodes are long, but it forces a full position copy per action.
5. **The collective `∇·D` term.** Only the local radial gradient is included;
   D̂ also depends on R_g, which is a many-body coupling.

### Things that would be easy to get wrong later

- **Do not symmetrise the dipole force.** It is genuinely non-Newtonian; the
  tier-2 test asserts |ΣF| > 0 to catch exactly that "fix".
- **Do not use the legacy finite-difference DEP gradient on GPU.** In FP32 it is
  pure cancellation noise (>50% error) — the test asserts this.
- **Do not compute pair separations in FP32.** Coordinates are ~2e4 nm and the
  DLVO exponential amplifies 0.1 per nm.
- **`CMAKE_CUDA_ARCHITECTURES` before `enable_language(CUDA)`,** and keep
  `-real` so no PTX is embedded while the driver trails the toolkit.
- **The documented `R_g`/`C₆`/`RC` values are for a different configuration.**
  Use `data/GOLDEN.md`.
