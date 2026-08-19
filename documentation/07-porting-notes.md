# 7. Porting Notes: Defects, Quirks, and the Rewrite Plan

This is the list of things that will bite during the C++/CUDA port. Each entry
says what the legacy code does, whether it is a bug, and what the rewrite should
do. Items marked **verified** were reproduced by running the code, not inferred.

---

## 7.1 RNG: positive seeds are not seeded — **verified, real bug**

`ran2.f` follows Numerical Recipes exactly, including the convention that the
shuffle table is initialized **only when `idum ≤ 0`**:

```fortran
if (idummy.le.0) then          ! ran2.f:17 — initialization branch
```

`bd_env.py` always passes a **non-negative** seed
(`np.random.randint(low=0, high=max_epochs+1)`, then the chained value from
`out_param.json`, which is a large positive integer). So on the first call the
shuffle table `iv` is all zeros and `iy = 0`, and the generator runs from an
uninitialized state whose early output is governed by the hard-coded
`idum2 = 123456789` rather than by the seed.

**Measured consequence:** with a standalone driver, seeds `1`, `7`, `999`, and
`123456` produce **byte-identical first 8 deviates**; the streams diverge only at
draw 9. A negative seed (`-7`) initializes correctly and differs from draw 1.

```
seed 1:       0.65541640  0.20099520  0.89362246  0.28188657  0.52500042
seed 7:       0.65541640  0.20099520  0.89362246  0.28188657  0.52500042
seed 999:     0.65541640  0.20099520  0.89362246  0.28188657  0.52500042
seed 123456:  0.65541640  0.20099520  0.89362246  0.28188657  0.52500042
seed -7:      0.45206039  0.88851301  0.31787409  0.59642567  0.61300603
```

Over 10⁶ steps this is dynamically irrelevant, but it means **every RL epoch
begins with the same handful of random displacements**, and it makes the seed
argument misleading. Fix in the port; do not replicate.

**Also:** `gasdev` caches the second Box–Muller deviate in a `SAVE`d variable and
its rejection loop consumes a variable number of uniforms, so the mapping from
"step number" to "position in the uniform stream" is not fixed.

**Rewrite:** counter-based RNG (cuRAND Philox / Random123), keyed on
`(global_seed, env_id, particle_id, step, component)`. This gives
per-particle independent streams, bitwise reproducibility regardless of thread
scheduling, and no state to carry. Keep the "integer in / integer out" signature
if `bd_env.py` chaining is to survive unchanged, or replace it with an explicit
RNG-state object (preferred).

---

## 7.2 `conn6calc.f` array indexing — latent bug, currently masked

Particles are filtered into the working arrays **by their original index `i`**:

```fortran
do i=1,np
   IF (in window) THEN
      rx(i) = ...  ;  ry(i) = ...          ! indexed by i
      NPTEMP = NPTEMP + 1                  ! but counted into NPTEMP
   ENDIF
enddo
```

The subsequent ψ₆ and C₆ loops then run `do i = 1, NPTEMP`. If any particle were
excluded, `rx`/`ry` would be **sparse** — the loops would read uninitialized
zeros for the gaps and would skip the highest-indexed particles entirely.

It is masked because `expbox = 63.415 a` equals the full cell, so `NPTEMP = np`
always. **A rewrite that changes `expbox`, adds particles, or lets the cluster
drift will expose this.** The fix is to compact into a dense array with its own
counter.

Related: the global ψ₆ sum divides by `np` while the per-particle loop uses
`NPTEMP` (`conn6calc.f:79-84`) — the same latent inconsistency.

---

## 7.3 Missing `∇·D` drift term — physics approximation

`D̂` depends on position (through the radial-distance bin) and on the global
configuration (through `R_g`), but the integrator uses the plain

`Δr = D F Δt/kT + √(2DΔt) ξ`

without the `∇·D Δt` term required by the Ermak–McCammon / Itô formulation for
position-dependent diffusivity. Strictly, this biases particles toward
low-mobility regions (the cluster centre).

In practice the effect is suppressed because **`D̂` is only refreshed at output
steps** (§7.5) — within an episode it is a *constant* per particle, for which no
drift term is needed. The two omissions cancel. If the port starts updating `D̂`
every step (the physically correct thing), it **must** add the drift term, or the
resulting bias will be a real, silent change in the equilibrium density profile.

---

## 7.4 Single-precision centroid in `caldss.f` — real precision loss

`caldss.f` declares its double-precision arrays but **not** `xmean`, `ymean`,
`disttemp`, or `calcudss`. With no `implicit none`, these default to `REAL*4`.
Coordinates are ~10⁴–10⁵ nm, so a single-precision centroid carries ~7 significant
digits — about 0.01 nm resolution. The result only selects a 1435-nm-wide bin, so
it is harmless *here*, but it means `caldss` and `conn6calc` compute **different
centroids from the same coordinates**. Use double everywhere in the port.

`calcudss = 0.5*(dssmax+dssmin)` is computed and never used.

---

## 7.5 Mobility and order parameters are refreshed only on output steps

`CONN6CALC` and `caldss` sit inside `if (mod(l,iprint).eq.0 .or. t.eq.0)`. With
`iprint = nstep = 10⁶`, `dsscalcu` is computed **once from the initial
configuration** and held constant for the whole 100-second episode.

This is almost certainly not what a reader would assume. It is also a *feature*
for the GPU port — no global reduction inside the inner loop — and it is what the
existing trained policies experienced. **Make it an explicit configuration
option** (`mobility_update_interval`) rather than an accident of `iprint`.

---

## 7.6 Duplicated work in the output block

`main.f:271-283` contains two consecutive, textually near-identical `if` blocks
with the same condition. Both call `CONN6CALC` and `caldss`; the first also calls
`writcn`, the second `QuadControl`. So on every output step the O(N²) order
parameters are computed **twice**, and `CONN6CALC` consumes `2 × np` extra
`gasdev` draws (it calls `gasdev` unconditionally even though `var = 0`).

Merge into one block. Note this changes the RNG stream — another reason
bit-exact reproduction of the legacy code is not a sensible goal.

---

## 7.7 `ControP` is never assigned

`main.f` declares `integer ControP`, passes it to `QuadControl`, and prints it to
`op*.txt` — but nothing ever assigns it. The shipped `op1.txt` shows `*****`
(`i5` overflow on garbage). Drop it, or make it a real diagnostic.

Similarly, `QuadControl.f` declares `Controlpolicy(N)` with `N = 50*120 = 6000`
and never touches it — the remains of a tabulated feedback controller that the RL
agent replaced.

---

## 7.8 Periodic boundaries are half-implemented

`main.f:330-338` wraps `x` and `y` into `[-dg/2, dg/2]`, but the minimum-image
convention in the force loop is **commented out** (`forces.f:88-92`). If a
particle ever wrapped, its forces would be computed against unwrapped neighbours
— an instant, silent blow-up.

It never happens: the nDEP trap holds the cluster at `R_g ≈ 21 µm` inside a
90 µm box. **Decide explicitly in the rewrite** — either drop periodicity
entirely (correct for a trapped finite cluster, and what the physics actually is)
or implement it consistently in both places. Do not leave it half-done.

Note `conn6calc.f` *does* apply minimum-imaging for ψ₆ neighbours, so the three
places disagree with each other.

---

## 7.9 Overlap force is weaker than the contact repulsion

For `r ≤ 2a` the pair force is the constant `Fhw = 0.417`; for `r` just above
`2a` the DLVO branch gives **1.334**. The force therefore *drops* by a factor of
3 on entering overlap, and the dipole force is switched off at the same point.
This is a discontinuous, non-monotonic, non-conservative pair force.

**Verified never exercised:** minimum pair separation over a run was `2.052 a`.
The 10 nm Debye screening stops particles well before contact. In the rewrite,
replace with a continuous capped repulsion (e.g. clamp the exponential at a
maximum force) so the branch cannot misbehave if parameters change.

---

## 7.10 Dead code inventory

Roughly 40 % of the source is inert. Do not port any of it.

| Item | Location | Status |
|---|---|---|
| `DDPOT` | `ddpot.f` (entire file) | never called; uses COMMON blocks nothing else declares |
| `pwexact` | `pwexct.f` (entire file) | never called; call site commented out; contains a typo (`12420*v**2 + 5654*v**2`) |
| `writ_head` | `writcn.f:2-19` | never called |
| Finite-difference force code | `forces.f` — ~150 commented lines | abandoned alternative to the analytic forces |
| Exact quadrupole field | `emag.f:30-52` | computed every call, then overwritten |
| Gravity / wall forces | `forces.f:341-376` | commented out (system is 2-D) |
| Histograms | `rghist`, `psihist`, `conhist` + 3 filenames | allocated, zeroed, never accumulated or written |
| Unused parameters | `phi`, `istart`, `pwfactor`, `Fgrav`, `pfpw`, `t` (input), `RGMIN`, `DELRG`, `DELPSI`, `delcon` | parsed, never used meaningfully |
| Unused variables | `m`, `dt_m`, `m_2`, `nout`, `u`, `ud`, `t_min_isdt`, `nbond`, `bonds`, `vol`, `n`, `delta`, `rdepc`, `calcudss`, `rgarraybin` | declared/computed, unused |
| Legacy `check='o'` path | `readcn.f:44-53` | reads a 5-column format nothing writes |

---

## 7.11 Structural obstacles specific to the port

| Obstacle | Impact | Resolution |
|---|---|---|
| 15 COMMON blocks as global mutable state | no threading, no multiple instances in one process | `Config` (immutable) + `State` (per-env) structs |
| Inconsistent COMMON declarations (`/radius/`, `/boxlen_xy/`, `/ORDER_PAR/` truncated in some units) | silent aliasing hazards | eliminated by the struct refactor |
| No `implicit none` anywhere | typos become variables; silent single precision | C++ removes the class of bug |
| `SAVE`d RNG state | not thread-safe, not batchable | counter-based RNG (§7.1) |
| Process-per-step, file-based interface | ~4.7 min/step, no batching | in-process vectorized API ([06-rl-integration.md](06-rl-integration.md) §6.5) |
| Positional `run.txt` | any edit shifts every parameter | TOML/JSON config with a `from_run_txt()` shim |
| Fortran unit 40 never closed | `op*.txt` flushed only at exit | irrelevant once file I/O leaves the hot path |

---

## 7.12 Recommended porting sequence

1. **CPU C++ reference.** Straight transliteration into structs, double
   precision, `implicit none` equivalent — but with the RNG replaced and the
   duplicated block removed. Target: reproduce the Fortran's `Ψ₆`, `C₆`, `R_g`
   trajectories *statistically* (ensemble means over ~20 seeds), since bit-exact
   is off the table once the RNG changes.
2. **Regression harness first.** Freeze the shipped `start.txt`, `run.txt`, and
   `2dtabledssnp300.txt` as golden inputs. Assert:
   * `fac1`/`fac2` derivation reproduces `0.011709` and `5.797`
     ([02-numerical-methods.md](02-numerical-methods.md) §2.3);
   * `R_g = 21014.5 nm` and `RC = 0.76125` for the known configuration
     ([03-order-parameters.md](03-order-parameters.md));
   * single-step forces match the Fortran to ~1e-12 with the noise term disabled
     (this is the highest-value test — do it before touching CUDA).
3. **Force-kernel validation.** Compare the analytic dipole force against a
   central-difference of the potential in `ddpot.f`; the asymmetric `i`/`j`
   gradient terms ([01-physical-model.md](01-physical-model.md) §1.4a) are the
   easiest thing in this codebase to get subtly wrong.
4. **CUDA kernels.** Batched N-body force kernel (`[n_env][np]`), fused DEP,
   Philox noise, fused position update. Order parameters as a separate kernel on
   output steps only.
5. **pybind11 bindings.** Zero-copy via `__cuda_array_interface__`/DLPack;
   `Config` from TOML; a `BDVectorEnv` Gym adapter last.
6. **Physics options, explicitly flagged.** `mobility_update_interval`, the
   `∇·D` drift term (§7.3), periodicity (§7.8), and a continuous overlap force
   (§7.9) each change results — expose them as config with the legacy behaviour
   as the default, so old policies stay reproducible.
