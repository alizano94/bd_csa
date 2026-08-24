# bd_csa — Brownian Dynamics for Colloidal Self-Assembly

A batched, GPU-accelerated rewrite of a legacy Fortran 77 Brownian-dynamics
simulator for **directed colloidal self-assembly in a quadrupole electrode cell**,
exposed to Python as a vectorized reinforcement-learning environment.

> **Status: the simulator is finished and validated; the Python packaging is the
> only substantive gap.** CPU reference and CUDA backend both implemented and
> checked against the legacy Fortran — pair forces to round-off, and observable
> distributions statistically indistinguishable over 20-seed ensembles. **331× the
> Fortran's throughput** at 4096 batched environments, 9× on a single one.
>
> Remaining: the Python bindings are written but have never been compiled (they
> need pybind11, one network fetch at configure time), and the CUDA kernel takes
> one `λ` per launch — which blocks per-environment RL actions on GPU.
>
> [documentation/11-validation-and-status.md](documentation/11-validation-and-status.md)
> is the current, audited assessment; [DEVLOG.md](DEVLOG.md) is the engineering
> record; [data/GOLDEN.md](data/GOLDEN.md) holds the regression targets.

## Quick start

```sh
# Build (CUDA auto-detected; add -DBD_CSA_BUILD_PYTHON=OFF to skip the fetch)
cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j16
ctest --test-dir build/cmake            # 4 validation suites

# Build the Fortran oracle used by the differential tests
tests/oracle/build_force_oracle.sh

# Run the drop-in CLI (same argv as the legacy bdpd; reads run.txt from the CWD)
mkdir -p run1 && cd run1 && cp ../data/{run.txt,start.txt,2dtabledssnp300.txt} .
../build/cmake/bdpd 30.0 -7 && cd ..

# Throughput sweep against the Fortran baseline
./build/cmake/bench_cuda 5000

# Python: run a trajectory and render snapshots (needs the venv, see below)
.venv/bin/python scripts/run_and_plot.py                  # 10 images -> frames/
.venv/bin/python scripts/run_and_plot.py --device cuda --images 20 --dashboard

# Statistical validation vs the Fortran (tier 5)
python3 tests/compare_trajectories.py 20 50000
```

## Measured results

Linux / x86-64, RTX 4060 Ti, g++ 13.3 (development machine):

| | µs/env-step | vs Fortran |
|---|---|---|
| Fortran (baseline) | 311.5 | 1× |
| CPU port, single env | 237 | 1.31× |
| CUDA, 1 env | 34.7 | 9.0× |
| CUDA, 64 envs | 1.01 | 310× |
| CUDA, 4096 envs | **0.94** | **331×** |

Independently re-measured on macOS / arm64, AppleClang 21 (no GPU):

| | µs/step | vs Fortran |
|---|---|---|
| Fortran (baseline) | 284 | 1× |
| CPU port | 238 | 1.19× |

Accuracy: pair forces reproduce the Fortran to **1.1e-14** on Linux/g++ and
**6.9e-14** on macOS/AppleClang — round-off on both. The FP32 GPU force kernel
sits at max 9.1e-5 / RMS 9.7e-6 against the FP64 reference. Over 20-seed ensembles
the port's ψ₆/C₆/R_g distributions are statistically indistinguishable from the
Fortran's when run in legacy-physics mode (KS p ≥ 0.50).

> **Read this before training a policy.** The default (corrected) physics shifts
> ψ₆ by **−8%** relative to the legacy simulator — see
> [DEVLOG.md](DEVLOG.md#result-20-seeds--50000-steps-5-s-simulated-at-λ--30).
> Pass `--legacy` to reproduce what previously trained policies experienced.

## What it simulates

300 charged colloidal spheres (radius `a = 1435 nm`) confined to a 2-D plane
inside a quadrupole electrode cell (gap ≈ 91 µm). An AC field induces a dipole in
every particle; the particles interact through induced-dipole/induced-dipole
forces, a stiff screened-electrostatic (DLVO) repulsion, and a negative
dielectrophoretic (nDEP) trap that holds the assembly at the field null. Motion is
overdamped and Brownian, with hydrodynamic hindrance applied as a scalar,
position-dependent mobility from a pre-tabulated Stokesian-dynamics table.

Modulating the field strength `λ` — the single control input — anneals a
crystalline cluster out of a disordered fluid. That makes `λ` the RL action and
the bond-orientational order parameters (`ψ₆`, `C₆/6`) the RL state.

## Why the rewrite

The legacy simulator is driven one episode at a time through `subprocess` and
text files:

| | Legacy (`fortran_bd`) | Target (`bd_csa`) |
|---|---|---|
| One env step (10⁶ integration steps = 100 s simulated) | ≈ 4.7 min, one core | batched on GPU |
| Parallel environments | 1 | thousands (`[n_env][np]`) |
| Interface | process spawn + `run.txt` / `start.txt` / `out_param.json` | in-process, numpy arrays |
| Global state | 15 COMMON blocks | `Config` (immutable) + `State` (per-env) |

A 100-step episode currently takes ~8 hours, and modern RL wants 10⁴–10⁶ env
steps. The bottleneck is not that a 300-particle BD step is slow — it is that the
environment cannot be batched. See
[documentation/06-rl-integration.md](documentation/06-rl-integration.md) §6.3.

## API

```python
import bd_csa

cfg = bd_csa.Config.from_run_txt("data/run.txt")
x0  = bd_csa.read_start_txt("data/start.txt", cfg)   # (300, 2) in nm

sim = bd_csa.Simulator(cfg, "data/2dtabledssnp300.txt", n_envs=4096, device="cuda")
sim.reset(x0)                            # (np,2) broadcasts, or pass (n_envs,np,2)
sim.step(30.0, n_steps=1_000_000, seed=7)
obs = sim.observations()                 # (n_envs, 2) = [ψ₆, C₆/6]
pos = sim.positions()                    # (n_envs, np, 2) float64 nm, a copy
```

A step is a pure function of `(positions, seed, λ)` — no hidden state — which is
what makes batching sound. The Gymnasium adapter (`BDVectorEnv`) sits *on top* of
this, never underneath: the simulator core does not know about Gym.

Full reference: [09-api-reference.md](documentation/09-api-reference.md).
Recipes: [10-usage-guide.md](documentation/10-usage-guide.md).

## Documentation

The [`documentation/`](documentation/README.md) directory is the specification for
this rewrite. It was reverse-engineered from the legacy source and **verified by
running it** (rebuild, timing, reproduction of `R_g`/`RC` from dumped coordinates,
reproduction of the RNG seeding defect).

**The specification** — reverse-engineered from the legacy Fortran:

| Document | Contents |
|---|---|
| [01-physical-model.md](documentation/01-physical-model.md) | Geometry, field, every force term, full derivations |
| [02-numerical-methods.md](documentation/02-numerical-methods.md) | Ermak–McCammon integrator, units, calibration constants, RNG, mobility table |
| [03-order-parameters.md](documentation/03-order-parameters.md) | `ψ₆`, `C₆`, `R_g`, `RC` — definitions and exact algorithms |
| [04-code-reference.md](documentation/04-code-reference.md) | File-by-file, routine-by-routine, COMMON-block reference |
| [05-io-formats.md](documentation/05-io-formats.md) | CLI, `run.txt`, `start.txt`, `bd_xyz1.txt`, `out_param.json`, mobility table |
| [06-rl-integration.md](documentation/06-rl-integration.md) | How `bd_env.py` drove the binary — the contract preserved |
| [07-porting-notes.md](documentation/07-porting-notes.md) | Bugs, quirks, dead code, numerical-fidelity traps |

**The implementation** — what this repository contains:

| Document | Contents |
|---|---|
| [08-implementation-map.md](documentation/08-implementation-map.md) | Architecture, file-by-file, the CUDA design, where each legacy behaviour ended up |
| [09-api-reference.md](documentation/09-api-reference.md) | Python API, C++ API, the `bdpd` CLI, `bench_cuda` |
| [10-usage-guide.md](documentation/10-usage-guide.md) | Build, test, run, use from Python and from RL; troubleshooting |
| [11-validation-and-status.md](documentation/11-validation-and-status.md) | What is proven and on which machine, open defects, what is left |

**Read [07-porting-notes.md](documentation/07-porting-notes.md) before changing the
physics** — it documents the RNG seeding bug, the missing `∇·D` drift term, the
single-precision centroid, the half-implemented periodic boundaries and the ~40 %
dead code, all of which the port decided about deliberately
([08-implementation-map.md](documentation/08-implementation-map.md) §8.6).

## Layout

```
bd_csa/
├── documentation/          spec (01-07) + implementation docs (08-11)
├── include/bd_csa/         public C++ headers — Config, State, Simulator, Philox
├── src/                    CPU reference implementation (C++20, FP64)
├── cuda/                   batched persistent kernel, one block per environment
├── apps/                   bdpd CLI drop-in, bench_cuda throughput sweep
├── python/                 pybind11 bindings + BDVectorEnv Gym adapter
├── tests/                  4 ctest suites + the Fortran oracle builder
├── data/                   golden inputs + GOLDEN.md regression targets
└── legacy/fortran_bd/      vendored Fortran, read-only, byte-identical to upstream
```

Build: CMake (+ CUDA when available), pybind11 fetched at configure time.
See [10-usage-guide.md](documentation/10-usage-guide.md) §10.1.

## Status and what is left

The port is complete through the CUDA backend and its validation. The remaining
work, in priority order (details in
[11-validation-and-status.md](documentation/11-validation-and-status.md) §11.5):

1. **Build the Python module.** It is the only thing between the current state
   and usable RL. Needs pybind11 — one network fetch at configure time, or a
   system `pybind11-dev`. The code and CMake wiring are complete; nobody has
   compiled it yet, so the Python API is unexercised.
2. **Per-environment `λ` on the GPU.** The kernel takes one `λ` per launch, so a
   per-env action array *raises* on `device="cuda"`. Every RL algorithm needs
   this. Small kernel change: move `λ` into a per-block array argument.
3. **Three small defects** found in the 2026-08-20 audit: a 1-ulp RNG equality
   assertion that fails on AppleClang, a BSD-awk incompatibility in the oracle
   build script, and `PhysicsOptions::continuous_overlap` not being wired to
   anything.
4. **Order parameters on device**, to avoid a full position download per action.
5. **The remaining performance gap** — 331× against a ~500× projection. Measure
   with `ncu`; this project's two guesses were both wrong.
6. **The collective `∇·D` term** — only the local radial gradient is included.

## Legacy source

The reference implementation is vendored, frozen, at
[`legacy/fortran_bd/`](legacy/fortran_bd/) — 12 `.f` files (~1,100 lines) built by
`makefile` into the executable `bdpd`, plus the shipped `run.txt`, `start.txt`,
`2dtabledssnp300.txt`, and reference outputs. It builds clean and unmodified:

```sh
cd legacy/fortran_bd && make          # gfortran -O -> ./bdpd
./bdpd <lambda> <seed>                # writes bd_xyz1.txt, op1.txt, out_param.json
```

Treat it as read-only. The frozen copies in [`data/`](data/) are what the tests
read — do not regenerate anything in place, and note that
`tests/oracle/build_force_oracle.sh` builds its instrumented variants
out-of-tree for exactly this reason. The upstream copy lives in the `SAC3`
repository at `sac3/fortran_bd`.
