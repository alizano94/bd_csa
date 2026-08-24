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

## 10.4a Saving and reusing runs

The legacy interface passed state through text files (`start.txt` in,
`bd_xyz1.txt` + `op1.txt` + `out_param.json` out). That is still supported for
interoperability, but it is not the recommended way to keep results.

### Trajectories: one HDF5 file per run

```python
out  = visualize.snapshot_series(sim, lam=30.0, n_frames=9, steps_per_frame=10_000)
traj = out["trajectory"]
traj.save("runs/anneal.h5")
```
or from the script: `--save-trajectory runs/anneal.h5`.

Every array has a leading time axis:

| dataset | shape | meaning |
|---|---|---|
| `positions` | `(T, np, 2)` | coordinates, **nm** |
| `psi6`, `c6`, `rg`, `rc` | `(T,)` | global order parameters |
| `psi6_local` | `(T, np)` | per-particle \|ψ₆⁽ⁱ⁾\| |
| `neighbours` | `(T, np)` int32 | neighbours within `rmin` |
| `time_s` | `(T,)` | simulated seconds |
| `lam` | `(T,)` | field strength — see below |
| `step` | `(T,)` int64 | cumulative integration steps |

**The λ convention.** `lam[k]` is the field strength applied *to reach* state
`k`, not the one applied from it. Frame 0 is the initial configuration, which no
action produced, so `lam[0]` is **NaN**. This matches what RL expects: action
`lam[k]` maps state `k-1` to state `k`.

`meta` embeds the whole configuration — `np`, `a`, `dt`, every physics flag,
seed, device and package version — so a file records *which physics produced it*.
That matters because the corrected physics shifts ψ₆ by ~8% against the legacy
behaviour; without the metadata you cannot tell two runs apart later.

Size is small: 300 particles × every field ≈ 7 kB per frame, so a 5-frame run is
0.07 MB gzip-compressed.

### Replaying a run exactly

`seed[k]` records the RNG seed of the step that produced frame `k` (`-1` on
frame 0, matching the `lam` NaN convention). Together with `positions[0]`, `lam`,
the step counts and the **embedded `run.txt`**, a trajectory is sufficient to
reconstruct itself:

```python
from bd_csa import Trajectory, replay
t = Trajectory.load("runs/anneal.h5")
replay(t, "data/2dtabledssnp300.txt")
# replay EXACT: max |dx| = 0.000e+00 nm, max |dpsi6| = 0.000e+00 on cpu
```

Verified: **max position error 0.0 nm**, bit-for-bit.

The full `run.txt` (~9 kB) is embedded because `Config` carries many fields the
Python bindings do not expose as setters — `from_run_txt` is the only way to
rebuild it faithfully, so the text has to travel with the data.

Three things will legitimately break a replay, and all three are recorded in
`meta` so you can tell which applies:

* **Device.** CUDA computes forces in FP32, CPU in FP64; the two diverge
  chaotically. `replay` defaults to the device the run used.
* **Physics flags.** Restored from `meta`; a mismatch changes the dynamics.
* **Package version.** A change to the force kernel or RNG will show up here —
  which is the point of keeping the check.

Trajectories written before seeds were recorded raise a clear error rather than
replaying wrongly.

### Resuming

```python
from bd_csa import Trajectory
traj = Trajectory.load("runs/anneal.h5")
sim.reset(traj.state())        # last frame; state(0) or state(k) for any other
```
or `--start runs/anneal.h5` (with `--frame k` to pick a different one).

Verified: resuming from a saved trajectory reproduces the final state **exactly**
— the continued run's frame 0 matches the original's last frame to every digit.

### Single configurations

```python
bd_csa.save_configuration(positions, "init.npy", cfg.a)   # exact, compact
bd_csa.save_configuration(positions, "init.txt", cfg.a)   # legacy, for bdpd
bd_csa.save_configuration(positions, "init.h5",  cfg.a)   # with metadata

x0 = bd_csa.load_configuration("init.npy", cfg.np, cfg.a)
```
or `--save-initial init.npy` from the script.

Measured round-trip error: `.h5` and `.npy` are **exact** (0 nm); `.txt` loses
**7×10⁻³ nm** because the legacy format writes 5 decimals in units of `a`. Use
`.txt` only to feed the Fortran or the C++ CLI, never to checkpoint.

### Why not pickle

Pickle stores the Python object graph, so it is Python-only, breaks silently
when a class definition changes, and executes arbitrary code on load. HDF5
stores data with a schema — readable from any language, survives refactors of
`trajectory.py`, and supports partial reads of files larger than memory.

## 10.4b Visualization

`bd_csa.visualize` renders particle configurations and order-parameter traces
straight from the in-memory simulator. Nothing is read or written except the
PNGs you ask for. It needs `matplotlib` (see `requirements.txt`) and is imported
lazily, so simulations run without it.

### The ready-made script

`scripts/run_and_plot.py` runs one trajectory and writes N images. It inserts
`python/` on `sys.path` itself, so no `PYTHONPATH` is needed:

```sh
.venv/bin/python scripts/run_and_plot.py                     # 10 images, CPU
.venv/bin/python scripts/run_and_plot.py --device cuda --images 20
.venv/bin/python scripts/run_and_plot.py --dashboard --lam 45
.venv/bin/python scripts/run_and_plot.py --legacy            # legacy physics
```

`--images` is the **total** written, counting the t=0 frame. `--start` accepts
either a `start.txt` (exactly `np` rows) or a multi-frame trajectory such as
`bd_xyz1.txt`, in which case it resumes from the **last** frame — the same
chaining the legacy driver used:

```sh
.venv/bin/python scripts/run_and_plot.py --start run1/bd_xyz1.txt --images 20
```

`--save-history out.npz` also stores the order-parameter traces for later
re-plotting without re-running.

Measured with the defaults (10 images, 10,000 steps apart = 9 s simulated):
**23.5 s on CPU**, **1.3 s for 4 images on CUDA** — past a few thousand steps per
frame the GPU path is dominated by rendering, not simulation.

### A trajectory as a sequence of snapshots

"Every step" is not achievable — an episode is 10⁶ steps. Sample instead: step
in chunks and render after each.

```python
import bd_csa
from bd_csa import visualize

cfg = bd_csa.Config.from_run_txt("data/run.txt")
x0  = bd_csa.read_start_txt("data/start.txt", cfg)
sim = bd_csa.Simulator(cfg, "data/2dtabledssnp300.txt", n_envs=1, device="cpu")
sim.reset(x0)

out = visualize.snapshot_series(
    sim, lam=30.0,
    n_frames=20, steps_per_frame=10_000,   # 20 frames over 20 s simulated
    out_dir="frames",
)
# out["history"] holds psi6/c6/rg/rc; out["time_s"] the time axis
```

Cost: `n_frames * steps_per_frame` integration steps. At ~240 µs/step on CPU the
example above is ~48 s. A full 10⁶-step episode in 100 frames is ~4 min.

### The two views

| function | shows |
|---|---|
| `plot_configuration(pos, psi6_local)` | particles as true-to-scale circles of radius `a`, coloured by **local** order |
| `plot_order_parameters(history)` | global ψ₆, C₆, R_g, RC versus time, stacked |
| `plot_dashboard(pos, history, psi6_local)` | both, with a marker on the traces at the current frame |

Each takes plain numpy arrays, so they work on data from anywhere — not only
from a live simulator.

### Local vs global order — they are different quantities

`sim.local_psi6(env)` returns |ψ₆⁽ⁱ⁾| per particle; `sim.observations()` returns
the global |⟨ψ₆⁽ⁱ⁾⟩|. On the shipped initial configuration the local mean is
**0.75** while the global value is **0.41**, because a polycrystal of
well-oriented grains has high local order whose phases cancel in the average.
Colouring by the local field is precisely what makes grain structure visible;
plotting the global value alone would not show it.

The colour scale is fixed to [0, 1] rather than autoscaled per frame — otherwise
the colours would not be comparable across an annealing sequence and the bar
would misrepresent progress.

Particles with no neighbour inside `rmin` correctly report |ψ₆⁽ⁱ⁾| = 0 and render
at the bottom of the scale. That is physical, not a defect: the shipped
configuration has exactly one such particle, 21.2 a from the centroid with its
nearest neighbour 2.80 a away, just outside the 2.634 a cutoff.

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
