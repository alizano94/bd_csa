# 9. API Reference

Complete reference for the C++ and Python interfaces. Structure and rationale
are in [08-implementation-map.md](08-implementation-map.md); task-oriented
recipes are in [10-usage-guide.md](10-usage-guide.md).

**Units, once and for all:** positions are **nanometres** everywhere in both
APIs. Time is **milliseconds**. Forces are in the reduced unit
`10¹⁸ · k_BT / a[nm]` ≈ 2.819×10⁻⁶. The legacy *text files* store positions in
multiples of the particle radius `a`; `read_start_txt` converts on read, and
`Config.a` is exposed so you can convert back explicitly.

---

## 9.1 Python API

### `bd_csa.Config`

```python
cfg = bd_csa.Config.from_run_txt("data/run.txt")   # or Config() for defaults
```

| Attribute | Type | Meaning |
|---|---|---|
| `np` | int | particle count (300) |
| `nstep` | int | default steps per episode (10⁶) |
| `a` | float | particle radius, nm (1435) |
| `dt` | float | time step, ms (0.1) |
| `tempr` | float | temperature, °C (20) |
| `dg` | float | electrode gap, nm (= 63.415·a) |
| `rcut`, `re` | float | DLVO / dipole cutoffs, nm (both 5a) |
| `rmin` | float | ψ₆ neighbour cutoff, nm (3780) |
| `physics` | `PhysicsOptions` | the correction switches |
| `temperature_K` | float, read-only | `273 + tempr` — note 273, not 273.15 |

`from_run_txt(path)` parses the legacy positional file. It is **strictly
positional**: inserting or removing a line shifts every later parameter
([05-io-formats.md](05-io-formats.md) §5.2).

### `bd_csa.PhysicsOptions`

```python
cfg.physics.mobility_update_interval = 0     # 0 = freeze for the episode
cfg.physics.enable_divD_drift        = False
cfg.physics.smooth_mobility          = False
```

| Attribute | Default | Legacy value | Notes |
|---|---|---|---|
| `mobility_update_interval` | `1` | `0` (once per episode) | costs 1.7 % on CPU, ~free on GPU |
| `enable_divD_drift` | `True` | `False` | **requires `smooth_mobility`** |
| `smooth_mobility` | `True` | `False` | C¹ interpolation vs nearest bin |
| `periodic` | `False` | `True` (inconsistently) | leave off |
| `continuous_overlap` | `True` | `False` | branch is never reached in practice |

To reproduce the legacy exactly, set the first three to their legacy values.

### `bd_csa.Simulator`

```python
sim = bd_csa.Simulator(config, mobility_table, n_envs=1, device="cpu")
```

| Parameter | Type | Notes |
|---|---|---|
| `config` | `Config` | copied; later mutation of `cfg` has no effect |
| `mobility_table` | str | path to `2dtabledssnp300.txt` |
| `n_envs` | int | independent simulations; must be > 0 |
| `device` | str | `"cpu"` or `"cuda"`; `"cuda"` raises if unavailable |

| Method | Signature | Notes |
|---|---|---|
| `reset` | `(positions)` | `(np, 2)` broadcasts to all envs; `(n_envs, np, 2)` sets each. float64 nm. |
| `step` | `(lam, n_steps, seed=0)` | `lam` scalar or length-`n_envs`. **See the CUDA caveat below.** |
| `positions` | `() -> (n_envs, np, 2)` | float64 nm, a **copy** |
| `observations` | `() -> (n_envs, 2)` | `[ψ₆, C₆/6]`, both in `[0,1]` — the RL observation |
| `order_parameters` | `() -> list[dict]` | `psi6`, `c6`, `rg`, `rc` per env |
| `n_envs`, `config`, `device` | properties | |

> **CUDA caveat.** The kernel takes one `λ` per launch. A per-environment `λ`
> **raises** on `device="cuda"` rather than silently serialising. This is the
> main open item for RL use — see
> [11-validation-and-status.md](11-validation-and-status.md) §11.5.

`step()` seeds Philox with `(seed, env_index, particle, step)`, so a single
`seed` still gives every environment an independent stream.

### `bd_csa.read_start_txt(path, config) -> (np, 2) float64`

Reads a legacy `start.txt` and converts to nm. The `z` column is discarded —
every particle sits at the fixed levitation height.

### `bd_csa.cuda_available() -> bool`

True only if the module was built with CUDA *and* a device is present.

### `bd_csa.gym_env.BDVectorEnv`

```python
from bd_csa.gym_env import BDVectorEnv, make_from_run_txt

env = make_from_run_txt("data/run.txt", "data/start.txt",
                        "data/2dtabledssnp300.txt",
                        n_envs=64, device="cuda")
```

| Parameter | Default | Notes |
|---|---|---|
| `lambda_range` | `(1.0, 60.0)` | actions in `[-1,1]` map linearly onto this |
| `n_steps` | `1_000_000` | one action = 100 s simulated at `dt = 0.1 ms` |
| `device` | `"cuda"` | |

`step(action)` returns the Gymnasium 5-tuple. **`reward` is all zeros and
`terminated`/`truncated` are all `False` by design** — reward and termination are
problem-specific, matching the legacy driver which imported its own
`reward_function`. `info` carries `order_parameters` and the applied `lambda`.

`rescale(action)` applies the legacy formula
`0.5·high·(a+1) + 0.5·low·(1−a)` after clipping to `[-1,1]`.

---

## 9.2 C++ API

All in `namespace bd_csa`. Header-per-concern under `include/bd_csa/`.

### `Config` / `PhysicsOptions` — `config.hpp`

```cpp
Config cfg = Config::from_run_txt("run.txt");
double fs  = cfg.force_scale();        // 1e18 * kb * T / a
double D0  = cfg.D0_nm2_per_s();       // Stokes-Einstein, eta = 0.890 mPa.s
```

`D0_nm2_per_s()` is a cross-check only — the simulation uses `fac1`/`fac2`,
which are read from `run.txt`, not derived.

### `State` — `state.hpp`

```cpp
State s(cfg.np);   // s.x, s.y : std::vector<double>, nm
```

### `read_start_txt` — `io.hpp`

```cpp
State s = read_start_txt("start.txt", cfg);
```

### `MobilityTable` — `mobility.hpp`

```cpp
auto table = MobilityTable::load("2dtabledssnp300.txt", cfg);
double d = table.lookup_nearest(rg_nm, dist_nm);         // legacy, piecewise const
double grad;
double d2 = table.lookup_smooth(rg_nm, dist_nm, &grad);  // C1, plus d(D̂)/d(dist)
```

`rows()`, `cols()`, `at(r,c)`, `count_at(r,c)` expose the raw table for tests.

### `compute_forces` / `emag` / `grad_emag_sq` — `forces.hpp`

```cpp
ForceOptions opt;
opt.dep_gradient  = DepGradient::kAnalytic;   // kFiniteDifference / kNone
opt.legacy_overlap = false;
compute_forces(cfg, lambda, x, y, fx, fy, np, opt);   // fx,fy OVERWRITTEN
```

| `DepGradient` | Use |
|---|---|
| `kAnalytic` | production; required on GPU |
| `kFiniteDifference` | test-only, reproduces the legacy arithmetic and its ~2e-6 error |
| `kNone` | omit the trap, isolating the well-conditioned pair forces |

### `compute_order_params` / `compute_rc` — `order_params.hpp`

```cpp
OrderParams op = compute_order_params(cfg, s.x.data(), s.y.data(), cfg.np);
// op.psi6, op.c6, op.rg, op.rc, op.n_in_window
```

### `Philox` — `rng.hpp`

```cpp
double z0, z1;
Philox::normal2(seed, env, particle, step, z0, z1);
```

Stateless and order-independent: the value is a pure function of the key.

> **Portability note.** Bitwise reproducibility holds within a given build. It
> is *not* guaranteed across compilers or optimization levels — an exact-equality
> assertion on the second Box–Muller component fails by 1 ulp (5.55e-17) under
> AppleClang 21 at `-O3`. See
> [11-validation-and-status.md](11-validation-and-status.md) §11.4.

### `SimulatorCpu` — `simulator.hpp`

```cpp
SimulatorCpu sim(cfg, table);
sim.step(s, lambda, n_steps, seed, /*env=*/0);   // mutates s in place
OrderParams op = sim.order_params(s);

sim.set_force_options(opt);
sim.refresh_mobility_now(s);
const std::vector<double>& d = sim.mobility();   // per-particle D̂
```

### `SimulatorCuda` — `sim_cuda.hpp`

```cpp
if (SimulatorCuda::available()) {
  SimulatorCuda gpu(cfg, table, n_envs);
  gpu.upload(s, env);
  gpu.step(lambda, n_steps, seed);          // ONE lambda for the whole batch
  gpu.download(s, env);
  std::vector<float> f = gpu.forces_once(lambda);   // interleaved (fx,fy)
  size_t bytes = gpu.shared_bytes();
}
```

Non-copyable (it owns device allocations). `forces_once` runs the kernel in
force-probe mode: one evaluation, no integration — this is what the tier-4 test
compares against the FP64 reference.

---

## 9.3 CLI — `bdpd`

Drop-in for the legacy binary.

```sh
bdpd <lambda> <seed> [--legacy] [--mobility-interval N]
```

Reads `run.txt` and `start.txt` **from the working directory**; writes
`out_param.json`, `bd_xyz1.txt`, `op1.txt` there in the legacy formats
([05-io-formats.md](05-io-formats.md)).

| Flag | Effect |
|---|---|
| `--legacy` | frozen mobility, nearest-bin lookup, no drift, legacy contact branch |
| `--mobility-interval N` | `0` freezes for the episode; `1` refreshes every step |

Differences from the legacy binary, all deliberate:

* `op1.txt` has **5 columns**, not 7 — the legacy's uninitialised `ControP` and
  the echoed `lambda` are dropped.
* The `seed` in `out_param.json` is derived from `(seed, steps)` by an LCG mix
  rather than being an evolved generator state. Chaining still works; values
  differ.
* Only two records are written (t = 0 and the end), matching the shipped
  `iprint = nstep`.

## 9.4 Bench — `bench_cuda`

```sh
./bench_cuda [n_steps]        # n_steps defaults to 20000
```

Sweeps `n_envs` over `{1, 16, 64, 256, 1024, 2048, 4096}`, reporting wall clock,
µs per environment-step, env-steps/s, and the speedup against a hard-coded
Fortran baseline of 311.5 µs/step. Requires a CUDA device; exits 1 without one.

> The baseline is a constant in the source (`apps/bench_cuda.cpp`), measured on
> the development machine. Re-measure it on yours before quoting the speedup —
> §11.2 has the numbers for this Mac.
