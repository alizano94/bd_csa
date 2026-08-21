# 11. Validation and Project Status

Where the port stands, what has been proven and on which machine, and what is
left. [DEVLOG.md](../DEVLOG.md) is the chronological record; this is the
point-in-time assessment.

**Last audited:** 2026-08-20, on macOS 15 / arm64 (Apple silicon), AppleClang
21.0.0, CMake 4.4.0, gfortran 16.1.0, **no CUDA device**.
The development machine (RTX 4060 Ti, Linux/x86-64, g++ 13.3, CUDA 13.2) is the
source of all GPU numbers; they could not be re-checked here.

## 11.1 Milestone status

| Milestone | Status | Evidence |
|---|---|---|
| **M0** foundations, golden data, Fortran oracle | **done** | `data/GOLDEN.md`; oracle rebuilt and run during this audit |
| **M1** CPU FP64 reference | **done** | force differential at round-off (§11.2) |
| **M2** CUDA persistent kernel | **done** (not re-verified here) | tier-4 suite; DEVLOG M2 |
| **M3** Verlet lists, 331× throughput | **done** (not re-verified here) | DEVLOG M3 |
| **M3** statistical validation vs Fortran | **done** | 20-seed KS test, legacy arm p ≥ 0.50 |
| **M4** Python bindings | **written, never built** | see §11.5 |

The honest one-line summary: **the simulator is finished and validated; the
Python packaging is the only substantive gap.** Everything the RL work needs
exists in C++ and has been checked against the Fortran; nobody has yet compiled
the `.so` that lets Python call it.

## 11.2 What was verified in this audit (macOS/arm64)

The DEVLOG's numbers come from one Linux machine with one compiler. Repeating
the CPU-side checks on a different OS, architecture and compiler is what turns
"it worked once" into "it is portable".

| Check | Result |
|---|---|
| Configure + build, CUDA absent, `-DBD_CSA_BUILD_PYTHON=OFF` | **clean**, no warnings surfaced |
| Fortran oracle rebuilt and run (after a one-line portability patch, §11.4) | **ok** |
| **tier2_forces** — pair forces vs the Fortran | **PASS, max rel err 6.916e-14** (DEVLOG reports 1.118e-14 on Linux/g++; both are round-off) |
| tier2 — non-Newtonian assertion `\|ΣF\| > 0` | **PASS**, `\|ΣF\| = 5.7e-5` |
| tier2 — full forces incl. legacy finite-difference DEP | **PASS**, 3.390e-07 |
| tier2 — analytic `∇\|E\|²` vs central difference | **PASS**, 2.640e-09 |
| tier2 — FP32 forward difference is unusable | **PASS**, `fd32 = -3.58e-4` vs exact `-1.08e-4` (>50 % error) |
| **tier1_constants** | **PASS** |
| **integrator** | **1 assertion FAILS** — a 1-ulp RNG equality check, §11.4 |
| tier4_cuda | not run (no device) |
| CLI golden row from `start.txt` | `4.28000  21015.90986  0.40508  0.76363` — **exact match** to `data/GOLDEN.md` |

### Performance on this machine (same session, same inputs, 20,000 steps)

| | wall | µs/step | vs Fortran |
|---|---|---|---|
| Legacy Fortran (`gfortran -O`) | 5.67 s | 284 | 1× |
| bd_csa CPU, corrected physics | 4.75 s | 238 | **1.19×** |
| bd_csa CPU, `--legacy` | 4.65 s | 233 | 1.22× |

Consistent with the 1.31× measured on Linux/x86-64. The corrected physics —
mobility refreshed every step plus the drift term — costs **2 %**, where the
legacy could only afford to refresh once per episode.

> The GPU figures (9× single-env, 331× at 4096 envs) are from the development
> machine and were **not** re-measured. Treat them as reported, not as verified
> on arbitrary hardware.

## 11.3 The result that matters most for RL

From the 20-seed × 50,000-step ensemble comparison (`tests/compare_trajectories.py`):

**Legacy-physics arm — the correctness check. Passed.**

| observable | Fortran | bd_csa | KS D | p |
|---|---|---|---|---|
| ψ₆ | 0.6367 | 0.6253 | 0.250 | 0.497 |
| C₆ | 5.1107 | 5.1080 | 0.200 | 0.771 |
| R_g (nm) | 19644.6 | 19633.9 | 0.200 | 0.771 |

With the physics pinned to legacy behaviour the only remaining difference is the
random stream, and the distributions are statistically indistinguishable. Taken
together with the round-off force differential, both the deterministic and
stochastic halves of the port are pinned.

**Corrected-physics arm — the corrections move the observables. Significantly.**

| observable | Fortran | bd_csa default | change | p |
|---|---|---|---|---|
| ψ₆ | 0.6367 | 0.5849 | **−8.1 %** | 0.023 |
| C₆ | 5.1107 | 5.0150 | −1.9 % | 0.001 |
| R_g (nm) | 19644.6 | 19704.2 | +0.3 % | 0.000 |

The direction is physically sensible: the legacy omitted the `∇·D` drift, which
biases particles toward low-mobility regions (the cluster centre). Restoring it
removes a spurious inward pull, so `R_g` rises and the lattice is marginally less
well formed. The two legacy omissions — frozen mobility and missing drift —
cancelled each other; refreshing mobility *without* the drift term would have
made the bias real.

> **ψ₆ is RL observation #1.** A policy trained against the legacy simulator will
> see a shifted state distribution under the corrected defaults. Retrain, or run
> with legacy physics ([10-usage-guide.md](10-usage-guide.md) §10.5). This is a
> decision, not a formality.

## 11.4 Open defects found in this audit

### (a) The RNG bit-reproducibility assertion fails on AppleClang — **test issue, not a physics issue**

`tests/test_integrator.cpp` asserts that `Philox::normal2` returns bit-identical
values for the same key regardless of call order, with tolerance exactly `0`. On
macOS/arm64 at `-O3` the **second** Box–Muller component differs by **5.55e-17**
(1 ulp). The first component matches exactly.

Reproducible and deterministic in the full test binary; it does **not** reproduce
when the same calls are extracted into a small program, with or without FMA
contraction (`-ffp-contract=off`) and with or without vectorization — so it is a
per-call-site codegen difference in the `sin`/`cos` tail of Box–Muller, not a
logic error. All statistical properties, the determinism check, and the
legacy-defect regression pass.

*Impact:* none on results — 1 ulp of a Gaussian deviate. But the claim in
`rng.hpp` that the RNG gives "bitwise reproducibility" holds **within a build**,
not across compilers, and the test as written encodes the stronger claim.

*Fix:* give that one check a tolerance of a few ulp (`1e-15`), or gate exact
equality on the reference toolchain.

### (b) `tests/oracle/build_force_oracle.sh` is not portable to macOS

The awk program embeds single quotes as `\x27`. BSD awk parses `\x` escapes
greedily, so `\x27f` consumes the following `f` and emits
`open(77,file=orce_dump.txt')`, and the oracle build fails with
*"Invalid value for FILE specification"*.

*Fix:* use the portable octal escape `\047` instead of `\x27` (verified — with
that one substitution both oracle variants build and run on macOS), or require
`gawk`.

### (c) `PhysicsOptions::continuous_overlap` is not wired to anything

The flag is declared, set by `bdpd --legacy`, and exposed to Python — but no code
reads it. The real switch is `ForceOptions::legacy_overlap`, which only the tests
and `SimulatorCuda::forces_once` ever set; `SimulatorCuda`'s episode kernel
hard-codes `legacy_overlap = 0`, and `bdpd_main` never calls
`set_force_options()`.

*Impact:* none on results today — the contact branch is unreachable (minimum pair
separation measured at **2.046 a** in this audit, 2.052 a previously). But
`bdpd --legacy` does not restore the legacy contact force as it claims, and the
Python flag silently does nothing.

*Fix:* have `SimulatorCpu` derive `ForceOptions::legacy_overlap` from
`cfg.physics.continuous_overlap`, and pass the same through to `DevConfig`.

## 11.5 Open work, in the order I would take it

1. **Build the Python module.** It is the only thing between the current state
   and usable RL. Needs pybind11 — one network fetch at configure time, or a
   system `pybind11-dev`. Nothing else about M4 is untested-by-choice; it is
   untested because it has never been compiled. *Until this happens, the API in
   [09-api-reference.md](09-api-reference.md) §9.1 is unexercised code.*
2. **Per-environment λ on the GPU.** The kernel takes one λ per launch, so
   `Simulator.step` **raises** on `device="cuda"` when given a per-env array.
   Every RL algorithm gives each environment its own action, so this blocks the
   primary use case. It is a small change — move λ into a per-block array
   argument.
3. **Fix the three defects in §11.4.** All are small; (a) and (b) affect anyone
   building on macOS.
4. **Order parameters on device.** Currently computed host-side after a full
   position download. Fine while episodes are long (one download per action),
   but it forces a copy of `(n_envs, 300, 2)` doubles per action.
5. **Investigate the remaining performance gap.** 331× against a ~500×
   projection. Measure with `ncu` rather than guessing — this project's two
   guesses were both wrong. The candidates: shared-memory bank conflicts in the
   neighbour-list gather, and whether occupancy is limited by the 35.6 KB shared
   footprint (dropping `kMaxNb` to 24 would buy a third block per SM, and the
   overflow fallback makes that safe).
6. **The collective `∇·D` term.** Only the local radial gradient is included;
   `D̂` also depends on `R_g`, a many-body coupling.

## 11.6 Invariants that must not be broken

Collected from the source comments and tests, because each one has a test
guarding it and each is a plausible "improvement" someone would attempt:

* **Do not symmetrise the dipole force.** It is genuinely non-Newtonian; tier-2
  asserts `|ΣF| > 0`.
* **Do not use the finite-difference DEP gradient on GPU.** In FP32 it is pure
  cancellation noise; tier-2 asserts >50 % error.
* **Do not compute pair separations in FP32.** Coordinates are ~2×10⁴ nm and the
  DLVO exponential amplifies 0.1 per nm.
* **Set `CMAKE_CUDA_ARCHITECTURES` before `enable_language(CUDA)`**, and keep
  `-real` so no PTX is embedded.
* **`data/GOLDEN.md` holds the regression targets** —
  `ψ₆ = 0.40508`, `C₆ = 4.28`, `R_g = 21015.90986 nm`, `RC = 0.76363` for the
  shipped `start.txt` at t = 0, invariant to λ and seed.
* **`legacy/fortran_bd/` is read-only** and byte-identical to upstream SAC3.
  Every oracle build is out-of-tree.
