# bd_csa — Documentation

Two halves:

* **01–07 — the specification.** Reverse-engineered documentation of the legacy
  Fortran 77 simulator: the physics, the maths, the file formats, and every
  defect that had to be decided about. This is what the rewrite implements.
* **08–11 — the implementation.** What the C++/CUDA/Python code actually does,
  how to use it, and where the project stands.

The legacy source is vendored read-only at [`legacy/fortran_bd/`](../legacy/fortran_bd/)
(12 `.f` files, ~1,100 lines), byte-identical to upstream `SAC3/sac3/fortran_bd`.
For the engineering narrative — what was tried and what failed — see
[DEVLOG.md](../DEVLOG.md). For regression targets see
[data/GOLDEN.md](../data/GOLDEN.md).

## What the program simulates

300 charged colloidal spheres (radius `a = 1435 nm`) confined to a **2-D plane**
inside a **quadrupole electrode cell** (electrode gap `dg = 91 µm`). An AC electric
field induces a dipole in every particle. The particles interact through
induced-dipole/induced-dipole forces, a stiff screened-electrostatic (DLVO)
repulsion, and a **negative dielectrophoretic (nDEP)** force that traps the
assembly at the field null in the centre of the cell. Motion is overdamped and
Brownian; hydrodynamic hindrance is applied as a scalar, position-dependent
mobility read from a pre-tabulated Stokesian-dynamics table.

The physical question is **directed self-assembly**: by modulating the field
strength `λ` (the single control input), a crystalline colloidal cluster can be
annealed out of a disordered fluid. The program is driven episode-by-episode from
`sac3/bd_env.py`, a Gym environment, which makes `λ` the RL action and the pair
(`ψ₆`, `C₆/6`) the RL state.

## Document index

### The specification (legacy Fortran)

| File | Contents |
|---|---|
| [01-physical-model.md](01-physical-model.md) | The physics: geometry, field, every force term, full derivations |
| [02-numerical-methods.md](02-numerical-methods.md) | Ermak–McCammon integrator, unit system, calibration constants, RNG, mobility table |
| [03-order-parameters.md](03-order-parameters.md) | ψ₆, C₆, R_g, RC — definitions and exact algorithms |
| [04-code-reference.md](04-code-reference.md) | File-by-file, routine-by-routine, COMMON-block and variable reference |
| [05-io-formats.md](05-io-formats.md) | CLI, `run.txt`, `start.txt`, `bd_xyz1.txt`, `out_param.json`, mobility table |
| [06-rl-integration.md](06-rl-integration.md) | How `bd_env.py` drove the binary — the contract the rewrite preserves |
| [07-porting-notes.md](07-porting-notes.md) | Bugs, quirks, dead code, numerical-fidelity traps |

### The implementation (bd_csa)

| File | Contents |
|---|---|
| [08-implementation-map.md](08-implementation-map.md) | Architecture, file-by-file, the CUDA design, where each legacy behaviour ended up |
| [09-api-reference.md](09-api-reference.md) | Python API, C++ API, the `bdpd` CLI, `bench_cuda` |
| [10-usage-guide.md](10-usage-guide.md) | Build, test, run, use from Python and from RL; troubleshooting |
| [11-validation-and-status.md](11-validation-and-status.md) | What is proven and on which machine, open defects, what is left |
| [12-image-state-representation.md](12-image-state-representation.md) | **Plan (proposal):** unsupervised image-based state representation to replace the hand-crafted (psi6, C6) RL state |

**Start with [11-validation-and-status.md](11-validation-and-status.md)** for the
current state, then [10-usage-guide.md](10-usage-guide.md) to run something.

## Quick facts

| Quantity | Value | Where set |
|---|---|---|
| Particles `np` | 300 | `run.txt` |
| Particle radius `a` | 1435 nm | `run.txt` |
| Electrode gap `dg` | 63.415 a = 90,000 nm ≈ 91 µm | `run.txt` |
| Time step `dt` | 0.1 ms | `run.txt` |
| Steps per invocation `nstep` | 1,000,000 (= 100 s simulated) | `run.txt` |
| Dimensionality | 2-D (z frozen at `h = 1.0663 a`) | `readcn.f` |
| Debye length | `a/κ` = 10.0 nm | `run.txt` (`kappa = 143.5`) |
| Reduced mobility `D̂` | 0.10 – 0.40 | `2dtabledssnp300.txt` |
| Control input | `λ` (argv[1]) | `main.f` |
| Observables written | `ψ₆`, `C₆`, seed, step, `λ` | `out_param.json` |
| Runtime (1 M steps, 1 core, Apple M-series) | **≈ 4.7 min** | measured |

## Verification of the legacy spec (docs 01–07)

Everything below was checked against the actual Fortran program, not inferred.
The corresponding audit of the *implementation* is in
[11-validation-and-status.md](11-validation-and-status.md) §11.2.

* The source was rebuilt from scratch with `gfortran 16.1.0 -O` — it compiles
  clean with no modifications.
* A 20,000-step run took **5.66 s** (283 µs/step, 6.3 ns per particle pair),
  giving the performance baseline quoted above.
* `R_g` and `RC` printed by the program were reproduced from the raw coordinates
  using the formulas in [03-order-parameters.md](03-order-parameters.md). For the
  shipped `start.txt` at t = 0: `R_g = 21015.90986 nm`, `C₆ = 4.28`,
  `ψ₆ = 0.40508`, `RC = 0.76363`. **[data/GOLDEN.md](../data/GOLDEN.md) is the
  authoritative list of regression targets** — an earlier draft of these docs
  quoted values taken from a configuration one run removed from `start.txt`.
* The unit-conversion constants `fac1` and `fac2` in `run.txt` were shown to
  correspond to `D₀ = k_BT/(6πηa)` with `η = 0.890 mPa·s`, agreeing to 0.03 %
  ([02-numerical-methods.md](02-numerical-methods.md)).
* The RNG seeding defect (all positive seeds share their first 8 deviates) was
  reproduced with a standalone driver ([07-porting-notes.md](07-porting-notes.md)).
* Minimum pair separation over a run was 2.052 a, confirming the hard-overlap
  branch of `forces.f` is never exercised in practice.
