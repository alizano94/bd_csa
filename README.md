# bd_csa — Brownian Dynamics for Colloidal Self-Assembly

A batched, GPU-accelerated rewrite of a legacy Fortran 77 Brownian-dynamics
simulator for **directed colloidal self-assembly in a quadrupole electrode cell**,
exposed to Python as a vectorized reinforcement-learning environment.

> **Status: working.** CPU reference and CUDA backend both implemented and
> validated against the legacy Fortran. **331× the Fortran's throughput** at 4096
> batched environments, and 9× on a single one. See [DEVLOG.md](DEVLOG.md) for
> the full engineering record and [data/GOLDEN.md](data/GOLDEN.md) for measured
> reference values.
>
> Remaining: Python bindings are written but not yet built (they need one
> approved network fetch — see [DEVLOG.md](DEVLOG.md) Phase M4).

## Quick start

```sh
# Build (CUDA auto-detected; add -DBD_CSA_BUILD_PYTHON=OFF to skip the fetch)
cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j16
ctest --test-dir build/cmake            # 4 validation suites

# Build the Fortran oracle used by the differential tests
tests/oracle/build_force_oracle.sh

# Run the drop-in CLI (same argv as the legacy bdpd)
cd build/cli_test && ../cmake/bdpd 30.0 -7

# Throughput sweep against the Fortran baseline
./build/cmake/bench_cuda 5000

# Statistical validation vs the Fortran (tier 5)
python3 tests/compare_trajectories.py 20 50000
```

## Measured results

| | µs/env-step | vs Fortran |
|---|---|---|
| Fortran (baseline, this machine) | 311.5 | 1× |
| CPU port, single env | 237 | 1.31× |
| CUDA, 1 env | 34.7 | 9.0× |
| CUDA, 64 envs | 1.01 | 310× |
| CUDA, 4096 envs | **0.94** | **331×** |

Accuracy: pair forces reproduce the Fortran to **1.1e-14** (round-off); the FP32
GPU force kernel sits at max 9.1e-5 / RMS 9.7e-6 against the FP64 reference. Over
20-seed ensembles the port's ψ₆/C₆/R_g distributions are statistically
indistinguishable from the Fortran's when run in legacy-physics mode (KS p ≥ 0.50).

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
| Interface | process spawn + `run.txt` / `start.txt` / `out_param.json` | in-process, zero-copy DLPack |
| Global state | 15 COMMON blocks | `Config` (immutable) + `State` (per-env) |

A 100-step episode currently takes ~8 hours, and modern RL wants 10⁴–10⁶ env
steps. The bottleneck is not that a 300-particle BD step is slow — it is that the
environment cannot be batched. See
[documentation/06-rl-integration.md](documentation/06-rl-integration.md) §6.3.

## Target API

```python
import bd_csa

sim = bd_csa.Simulator(
    config = bd_csa.Config.from_toml("configs/quadrupole_300.toml"),
    n_envs = 4096,
    device = "cuda",
)

sim.reset(positions=x0, seeds=seeds)     # (n_envs, np, 2)
obs = sim.step(lam, n_steps=1_000_000)   # lam: (n_envs,) -> obs: (n_envs, 2) = [ψ₆, C₆/6]
pos = sim.positions()                    # zero-copy, __cuda_array_interface__ / DLPack
```

The Gym/Gymnasium adapter (`BDVectorEnv`) sits *on top* of this, not underneath —
the simulator core does not know about Gym.

## Documentation

The [`documentation/`](documentation/README.md) directory is the specification for
this rewrite. It was reverse-engineered from the legacy source and **verified by
running it** (rebuild, timing, reproduction of `R_g`/`RC` from dumped coordinates,
reproduction of the RNG seeding defect).

| Document | Contents |
|---|---|
| [01-physical-model.md](documentation/01-physical-model.md) | Geometry, field, every force term, full derivations |
| [02-numerical-methods.md](documentation/02-numerical-methods.md) | Ermak–McCammon integrator, units, calibration constants, RNG, mobility table |
| [03-order-parameters.md](documentation/03-order-parameters.md) | `ψ₆`, `C₆`, `R_g`, `RC` — definitions and exact algorithms |
| [04-code-reference.md](documentation/04-code-reference.md) | File-by-file, routine-by-routine, COMMON-block reference |
| [05-io-formats.md](documentation/05-io-formats.md) | CLI, `run.txt`, `start.txt`, `bd_xyz1.txt`, `out_param.json`, mobility table |
| [06-rl-integration.md](documentation/06-rl-integration.md) | How `bd_env.py` drives the binary — the contract to preserve |
| [07-porting-notes.md](documentation/07-porting-notes.md) | Bugs, quirks, dead code, numerical-fidelity traps, CUDA and pybind11 plan |

**Read [07-porting-notes.md](documentation/07-porting-notes.md) before writing any
code.** It documents a real RNG seeding bug, a missing `∇·D` drift term, a
single-precision centroid, half-implemented periodic boundaries, and ~40 % dead
code that should not be ported.

## Planned layout

```
bd_csa/
├── documentation/          reverse-engineered spec (this is what exists today)
├── include/bd_csa/         public C++ headers — Config, State, Simulator
├── src/                    CPU reference implementation (C++20, double precision)
├── cuda/                   batched force / noise / order-parameter kernels
├── python/bd_csa/          pybind11 module + BDVectorEnv Gym adapter
├── tests/                  regression harness (golden inputs, force comparisons)
├── configs/                TOML configs, with a from_run_txt() shim
└── data/                   golden inputs: start.txt, run.txt, 2dtabledssnp300.txt
```

Build: CMake + CUDA, Python packaging via scikit-build-core + pybind11.

## Roadmap

Sequenced per [07-porting-notes.md](documentation/07-porting-notes.md) §7.12:

1. **Regression harness first.** Freeze the shipped `start.txt`, `run.txt`, and
   `2dtabledssnp300.txt` as golden inputs. Assert `fac1`/`fac2` derivation
   (`0.011709`, `5.797`), `R_g = 21014.5 nm`, `RC = 0.76125`, and single-step
   forces matching Fortran to ~1e-12 with noise disabled.
2. **CPU C++ reference.** Structs instead of COMMON blocks, double precision
   throughout, counter-based RNG, duplicated output block merged. Target:
   reproduce the Fortran `ψ₆`/`C₆`/`R_g` trajectories *statistically* (ensemble
   means over ~20 seeds) — bit-exact is off the table once the RNG changes.
3. **Force-kernel validation.** Check the analytic dipole force against a central
   difference of the potential; the asymmetric `i`/`j` gradient terms are the
   easiest thing here to get subtly wrong.
4. **CUDA kernels.** Batched N-body forces `[n_env][np]`, fused DEP, Philox noise,
   fused position update; order parameters as a separate kernel on output steps.
5. **pybind11 bindings.** Zero-copy interop, `Config` from TOML, `BDVectorEnv` last.
6. **Physics options, explicitly flagged.** `mobility_update_interval`, the `∇·D`
   drift term, periodicity, and a continuous overlap force each change results —
   expose them as config with the legacy behaviour as the default, so previously
   trained policies stay reproducible.

## Legacy source

The reference implementation is vendored, frozen, at
[`legacy/fortran_bd/`](legacy/fortran_bd/) — 12 `.f` files (~1,100 lines) built by
`makefile` into the executable `bdpd`, plus the shipped `run.txt`, `start.txt`,
`2dtabledssnp300.txt`, and reference outputs. It builds clean and unmodified:

```sh
cd legacy/fortran_bd && make          # gfortran -O -> ./bdpd
./bdpd <lambda> <seed>                # writes bd_xyz1.txt, op1.txt, out_param.json
```

Treat it as read-only. The inputs and reference outputs in that directory are the
golden data for the regression harness (roadmap step 1) — do not regenerate them
in place. The upstream copy lives in the `SAC3` repository at
`sac3/fortran_bd`.
