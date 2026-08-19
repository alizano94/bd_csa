# 6. Reinforcement-Learning Integration

The simulator is already an RL environment — via files and `subprocess`.
`sac3/bd_env.py` (`class BD_ENV_OP(gym.Env)`) wraps it. Understanding this
contract is what determines the API of the C++/Python rewrite.

## 6.1 The current loop

```
BD_ENV_OP.reset()
    copy <init_data>/bd_sim/run.txt          -> fortran_bd/run.txt
    sample an initial configuration           -> fortran_bd/start.txt
    state <- (psi6, c6/6) from the sampled row of initial_states.csv

BD_ENV_OP.step(action)
    a_scaled = rescale(action)                                  # -> lambda
    idummy   = (epoch 1) random int, or previous epoch's "seed" from JSON
    chdir(fortran_bd)
    subprocess.run([bdpd, str(lambda), str(idummy)])             # ~4.7 min
    chdir(back)
    results = json.load(out_param.json)
    state   = np.array([results['psi6'], results['c6']/6])
    reward  = reward_function(self)
    copy_last_n_lines(bd_xyz1.txt -> start.txt, 300)             # hand off state
```

### The three pieces of state that cross the boundary

| Carrier | Content | Direction |
|---|---|---|
| `start.txt` / `bd_xyz1.txt` | 300 particle positions (last 300 lines of the output become the next input) | out → in |
| `out_param.json` `seed` | evolved `iDummy`, chained so the RNG stream continues | out → in |
| `out_param.json` `psi6`,`c6` | the observation | out |

There is **no other hidden state** — the mobility table is recomputed from the
configuration, and all other parameters are constants. This is the property that
makes an in-process API straightforward: a step is a pure function of
`(positions, seed, λ)`.

## 6.2 Spaces

**Action** — one continuous scalar, the reduced dipole strength `λ`:

```python
a_scaled = 0.5*high*(action+1) + 0.5*low*(1-action)     # action clipped to [-1,1]
```
A discrete mode also exists (`action_map` lookup). `λ` is the applied AC field
strength squared, in units of `k_BT` — physically, the "annealing temperature"
knob, inverted: large `λ` = strong attraction + tight trap.

**Observation** — `[Ψ₆, C₆/6]`, both in `[0,1]`, either continuous (`Box`) or
binned to a `Discrete` index via `state_to_discrete`.

**Termination** — max epochs reached, or the state comes within
`tolerance` of `goal_state` (continuous) / equals it (discrete).

## 6.3 Cost of the current design

Per environment step:

| Item | Cost |
|---|---|
| 10⁶ integration steps | ≈ 283 s ≈ **4.7 min** on one core |
| process spawn, `run.txt`/`start.txt` copies, JSON parse | ≈ ms |
| parallelism | **none** — one core, one env, fully serial |

A 100-step episode is ~8 hours. Any modern RL algorithm wants 10⁴–10⁶ env steps.
**This is the entire motivation for the rewrite**: not that a 300-particle
BD step is slow, but that the environment cannot be batched or vectorized.

## 6.4 What the rewrite must preserve

Numerically identical results are impossible (different RNG), so define the
contract at the level of *statistics and interface*:

1. **Same observables.** `Ψ₆` and `C₆` computed exactly as in
   [03-order-parameters.md](03-order-parameters.md), including the phase-coherent
   global average and the `0.32` connection threshold. A policy trained against
   these is not transferable to a differently-defined `Ψ₆`.
2. **Same action semantics.** `λ` scales both the dipole attraction and the DEP
   trap; do not decouple them without flagging it as a physics change.
3. **Same episode length in simulated time.** 10⁶ steps × 0.1 ms = 100 s per
   action. If `dt` changes, hold the *simulated duration* fixed, not the step
   count.
4. **Same units at the boundary.** Coordinates in the file interface are in
   multiples of `a`; keep that (or convert explicitly) so existing
   `initial_states/*.txt` and `initial_states.csv` remain usable.
5. **Reproducibility from an explicit seed.** Fix the `ran2` defect
   ([07-porting-notes.md](07-porting-notes.md) §7.1) but keep "one integer in,
   one integer out" so the chaining logic in `bd_env.py` still works — or replace
   it with an explicit RNG-state object.

## 6.5 Target API for the rewrite

The natural shape, given §6.1, is a stateful object with a vectorized step —
this is what unlocks both CUDA and modern RL:

```python
import bd_csa

sim = bd_csa.Simulator(
    config = bd_csa.Config.from_toml("quadrupole_300.toml"),
    n_envs = 4096,                 # batched episodes on one GPU
    device = "cuda",
)

sim.reset(positions=x0, seeds=seeds)          # (n_envs, np, 2) float32/64

obs = sim.step(lam, n_steps=1_000_000)        # lam: (n_envs,) -> obs: (n_envs, 2)

pos = sim.positions()                         # zero-copy DLPack / __cuda_array_interface__
```

Design notes that follow directly from this document:

* **Batch dimension outermost.** Store `[n_env][np][3]`. At `np = 300` a single
  system cannot saturate a GPU; a batch of thousands can
  ([02-numerical-methods.md](02-numerical-methods.md) §2.6).
* **No file I/O in the hot path.** Positions and observations stay on device;
  expose them through `__cuda_array_interface__` / DLPack so PyTorch consumes
  them without a copy.
* **Config as a struct, not positional text.** Replace `run.txt` with TOML/JSON
  and a validated `Config` struct; keep a `from_run_txt()` shim so existing
  experiment directories still load.
* **Keep a CPU reference implementation.** A scalar C++ path that reproduces the
  Fortran to round-off is the only practical way to validate the CUDA kernels;
  budget for it explicitly.
* **Expose the order parameters separately** (`sim.order_parameters()`) so reward
  shaping can use `R_g` and `RC` without a second pass.
* **Gym/Gymnasium wrapper on top, not underneath.** The vectorized simulator
  should not know about Gym; provide `BDVectorEnv` as a thin adapter so
  `bd_env.py`'s existing reward/termination logic ports over unchanged.
