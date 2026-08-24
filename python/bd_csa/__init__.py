"""Brownian dynamics for colloidal self-assembly.

A batched, GPU-accelerated rewrite of a legacy Fortran 77 simulator for directed
colloidal self-assembly in a quadrupole electrode trap.

Quick start
-----------
    import bd_csa

    cfg = bd_csa.Config.from_run_txt("data/run.txt")
    x0 = bd_csa.read_start_txt("data/start.txt", cfg)   # (np, 2) in nm

    sim = bd_csa.Simulator(cfg, "data/2dtabledssnp300.txt",
                           n_envs=1024, device="cuda")
    sim.reset(x0)                       # broadcast one config to all envs
    sim.step(lam=30.0, n_steps=1_000_000, seed=7)
    obs = sim.observations()            # (n_envs, 2) = [psi6, C6/6]

Units
-----
Positions are in **nanometres** everywhere in this API. The legacy text files
store multiples of the particle radius `a`; :func:`read_start_txt` converts on
read, and ``cfg.a`` is exposed so you can convert back explicitly.

What a "step" means
-------------------
One RL action corresponds to ``n_steps = 1_000_000`` integration steps at
``dt = 0.1 ms``, i.e. 100 s of simulated time. If you change ``dt``, hold the
simulated *duration* fixed rather than the step count.
"""

from .trajectory import (  # noqa: F401
    Trajectory,
    TrajectoryRecorder,
    load_configuration,
    replay,
    save_configuration,
)
from ._bd_csa import (  # noqa: F401
    Config,
    OrderParams,
    PhysicsOptions,
    Simulator,
    cuda_available,
    read_start_txt,
)

__all__ = [
    "Config",
    "OrderParams",
    "PhysicsOptions",
    "Simulator",
    "cuda_available",
    "read_start_txt",
    "Trajectory",
    "TrajectoryRecorder",
    "save_configuration",
    "load_configuration",
    "replay",
    "visualize",
]


def __getattr__(name):
    """Expose `bd_csa.visualize` lazily.

    Importing it eagerly would make matplotlib a hard dependency of the whole
    package; this way simulations run without it and the import error only
    appears if you actually try to plot.

    Must use importlib rather than `from . import visualize`: the latter is an
    attribute lookup on this package, which re-enters __getattr__ and recurses
    until the stack blows. Caching in globals() also stops __getattr__ being
    consulted on subsequent accesses.
    """
    if name == "visualize":
        import importlib

        mod = importlib.import_module(f"{__name__}.visualize")
        globals()["visualize"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
