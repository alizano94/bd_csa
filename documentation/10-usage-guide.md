# 10. Usage Guide

Task-oriented. API details are in [09-api-reference.md](09-api-reference.md).

## 10.1 Build

### Requirements

| | Minimum | Notes |
|---|---|---|
| CMake | 3.24 | |
| C++ compiler | C++20 | verified: g++ 13.3 (Linux/x86-64), AppleClang 21 (macOS/arm64) |
| CUDA toolkit | optional | auto-detected; without it only the CPU path builds |
| Python 3 + headers | optional | needed for the bindings |
| gfortran | optional | only to build the Fortran oracle for the differential test |

### Standard build

```bash
cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j
```

CUDA is enabled automatically when `nvcc` is found. The CPU library always
builds standalone, so the test suite runs on machines without a GPU.

### Without the Python bindings

The bindings pull pybind11 via `FetchContent`, which needs network access at
**configure** time. To skip it:

```bash
cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release -DBD_CSA_BUILD_PYTHON=OFF
```

### On a different GPU

`CMAKE_CUDA_ARCHITECTURES` defaults to `89-real` (Ada, cc 8.9). Override:

```bash
cmake -B build/cmake -DBD_CSA_CUDA_ARCH=86-real     # Ampere, cc 8.6
```

Keep the `-real` suffix. It emits SASS only, with no embedded PTX. Shipping PTX
makes the driver JIT it at load time, and **JIT requires driver ≥ toolkit** —
a newer toolkit than driver fails at launch with *"the provided PTX was compiled
with an unsupported toolchain"*.

### Build targets

| Target | Output |
|---|---|
| `bd_csa_core` | CPU static library |
| `bd_csa_cuda` | CUDA backend (only when enabled) |
| `bdpd` | CLI drop-in for the legacy binary |
| `bench_cuda` | throughput sweep (CUDA only) |
| `_bd_csa` | Python extension, emitted into `python/bd_csa/` |
| `test_constants`, `test_forces`, `test_integrator`, `test_cuda` | ctest suites |

## 10.2 Test

```bash
ctest --test-dir build/cmake --output-on-failure
```

| Suite | Gates | Needs |
|---|---|---|
| `tier1_constants` | `run.txt` parsing, `fac1`/`fac2` calibration, mobility table, golden order parameters | — |
| `tier2_forces` | forces vs the Fortran oracle to 1e-12 | the oracle (below); **skips** (exit 77) without it |
| `integrator` | RNG statistics, free-diffusion MSD, determinism, short-run sanity | — |
| `tier4_cuda` | FP32 kernel vs FP64 reference; GPU/CPU trajectory agreement | a CUDA device; **skips** without one |

### Building the Fortran oracle (needed for `tier2_forces`)

```bash
tests/oracle/build_force_oracle.sh
```

This copies `legacy/fortran_bd/*.f` into `build/`, inserts a force dump just
before the time loop, builds two variants (`full` and `pair`), and runs each
once. `legacy/` is never modified.

> **macOS:** the script uses GNU-awk `\x27` hex escapes, which BSD awk parses
> greedily — it produces a broken `open(77,file=orce_dump.txt')` and the build
> fails. Either install `gawk` and put it first on `PATH`, or replace `\x27`
> with the portable octal `\047`. See §11.4.

### Statistical validation against the Fortran

```bash
python3 tests/compare_trajectories.py 20 50000     # 20 seeds, 50k steps each
```

Runs both the Fortran oracle and the port over an ensemble and compares ψ₆, C₆
and `R_g` distributions with a two-sample KS test (implemented inline — no scipy
dependency). Two arms: `--legacy` physics (the correctness check) and the
corrected defaults (which quantify how much the corrections move the
observables). Seeds are **negative** so the legacy `ran2` initialises properly,
giving the Fortran its best case.

## 10.3 Run the CLI

Fully argv-compatible with the legacy binary. It reads `run.txt` and `start.txt`
from the **working directory**, so run it in a directory containing them:

```bash
mkdir -p run1 && cd run1
cp ../data/run.txt ../data/start.txt ../data/2dtabledssnp300.txt .
../build/cmake/bdpd 30.0 -7
```

Outputs `op1.txt`, `bd_xyz1.txt`, `out_param.json`. To reproduce legacy physics:

```bash
../build/cmake/bdpd 30.0 -7 --legacy
```

This is the fastest way to sanity-check a build: with the shipped `start.txt`,
the t = 0 row of `op1.txt` must read

```
     0.00000     4.28000    21015.90986     0.40508     0.76363
```

(columns: `t[s]`, `C₆`, `R_g[nm]`, `ψ₆`, `RC`) — those are the golden values in
[data/GOLDEN.md](../data/GOLDEN.md), and they are invariant to both `λ` and seed.

## 10.4 Use from Python

### Install / import

The extension is emitted into `python/bd_csa/`, so no install step is needed:

```bash
export PYTHONPATH=$PWD/python
python3 -c "import bd_csa; print(bd_csa.cuda_available())"
```

### Single environment

```python
import bd_csa

cfg = bd_csa.Config.from_run_txt("data/run.txt")
x0  = bd_csa.read_start_txt("data/start.txt", cfg)      # (300, 2) in nm

sim = bd_csa.Simulator(cfg, "data/2dtabledssnp300.txt", n_envs=1, device="cpu")
sim.reset(x0)
sim.step(lam=30.0, n_steps=100_000, seed=7)

print(sim.observations())        # [[psi6, C6/6]]
print(sim.order_parameters())    # [{'psi6':..., 'c6':..., 'rg':..., 'rc':...}]
```

### Batched on GPU

```python
sim = bd_csa.Simulator(cfg, "data/2dtabledssnp300.txt",
                       n_envs=4096, device="cuda")
sim.reset(x0)                    # (np,2) broadcasts to all 4096
sim.step(30.0, 1_000_000, seed=7)
obs = sim.observations()         # (4096, 2)
```

Every environment gets an independent random stream — Philox is keyed on the
environment index, so one `seed` is enough.

To start each environment from a *different* configuration, pass the full
`(n_envs, np, 2)` array to `reset`.

### Choosing `n_envs`

Throughput saturates around 64 environments on an RTX 4060 Ti. Below ~16 the GPU
is underused, and at `n_envs = 1` the GPU is only ~9× the Fortran — this is a
throughput win, not a latency win. Use the largest batch your algorithm can
consume.

### Reinforcement learning

```python
from bd_csa.gym_env import make_from_run_txt

env = make_from_run_txt("data/run.txt", "data/start.txt",
                        "data/2dtabledssnp300.txt",
                        n_envs=64, lambda_range=(1.0, 60.0), device="cuda")

obs, info = env.reset(seed=0)
for _ in range(100):
    action = policy(obs)                       # in [-1, 1]
    obs, reward, term, trunc, info = env.step(action)
    reward = my_reward_function(obs, info)     # reward is yours to define
```

Three things to know before you train:

1. **Reward and termination are not provided.** `BDVectorEnv` returns zeros and
   `False` — supply your own, exactly as the legacy driver imported its own
   `reward_function`.
2. **The default physics is not the legacy physics.** ψ₆ shifts by −8 % relative
   to the Fortran. A policy trained against the legacy simulator will see a
   shifted state distribution. Either retrain, or set the legacy options:
   ```python
   cfg.physics.mobility_update_interval = 0
   cfg.physics.smooth_mobility          = False
   cfg.physics.enable_divD_drift        = False
   ```
3. **One action shares one λ across the batch on CUDA.** `BDVectorEnv.step`
   currently takes `action[0]` for the whole batch. For per-environment actions
   use `device="cpu"` until the kernel change lands (§11.5).

## 10.5 Choosing physics options

| Goal | Settings |
|---|---|
| Reproduce the legacy simulator | `mobility_update_interval=0`, `smooth_mobility=False`, `enable_divD_drift=False` (or `bdpd --legacy`) |
| Best available physics | the defaults |
| Isolate one correction | change one flag at a time and compare — that is what the flags are for |

Do **not** set `enable_divD_drift=True` with `smooth_mobility=False`. The drift
term is silently disabled in that combination (both the CPU and CUDA backends
require both flags), because `∇D` of a piecewise-constant table is zero almost
everywhere with spikes at the bin edges.

Do **not** enable `periodic` unless you also implement minimum-image in the force
loop. The legacy's half-implemented periodicity is documented in
[07-porting-notes.md](07-porting-notes.md) §7.8.

## 10.6 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| CMake configure hangs or fails on a network fetch | pybind11 `FetchContent` | `-DBD_CSA_BUILD_PYTHON=OFF`, or install `pybind11-dev` |
| `the provided PTX was compiled with an unsupported toolchain` | toolkit newer than driver | keep `-real` in `BD_CSA_CUDA_ARCH` and set it to your actual cc |
| Kernel launch fails with `invalid device function` | wrong `CMAKE_CUDA_ARCHITECTURES` | check with `cuobjdump -sass`; the arch must be set *before* `enable_language(CUDA)` |
| `tier2_forces` reports `[SKIP]` | oracle not built | run `tests/oracle/build_force_oracle.sh` |
| Oracle build fails with `Invalid value for FILE specification` | BSD awk | use `gawk`, or swap `\x27` → `\047` (§11.4) |
| `device='cuda' requested but no CUDA device found` | CPU-only build or no GPU | check `bd_csa.cuda_available()` |
| `per-environment lambda is not yet supported on the CUDA backend` | known limitation | scalar λ, or `device="cpu"` (§11.5) |
| `cannot open run.txt` | CLI reads the CWD | run `bdpd` from a directory containing the inputs |
| Order parameters all read 0 | every particle fell outside `expbox` (`n_in_window == 0`) | check `expbox` against your configuration's extent |
| `PhysicsOptions.continuous_overlap` appears to do nothing | it is **not wired up** — known gap, §11.5 | harmless today: the contact branch is never reached |
