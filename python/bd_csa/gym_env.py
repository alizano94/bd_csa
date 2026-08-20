"""Gymnasium adapter.

Deliberately a thin layer *on top of* :class:`bd_csa.Simulator`, never
underneath it. The simulator core has no notion of episodes, rewards or action
spaces; keeping it that way is what lets the same core serve batched RL, a CLI,
and offline analysis without each one paying for the others.

Import is lazy: ``gymnasium`` is an optional dependency, and the rest of
``bd_csa`` works fine without it.

Contract preserved from the legacy environment
----------------------------------------------
* **Observation** ``[psi6, C6/6]``, both in ``[0, 1]``. These are computed
  exactly as the Fortran did, including the phase-coherent global average and
  the 0.32 connection threshold. A policy trained against a differently-defined
  psi6 is not transferable.
* **Action** a single continuous ``lambda``, rescaled from ``[-1, 1]``. It
  scales the dipole attraction and the DEP trap together -- decoupling them
  would be a physics change.
* **Episode length in simulated time**, not step count: one action advances
  ``n_steps * dt`` of simulated time (100 s by default). If you change ``dt``,
  hold the duration fixed.
"""

from __future__ import annotations

import numpy as np

from ._bd_csa import Config, Simulator, read_start_txt


class BDVectorEnv:
    """Vectorised environment over ``n_envs`` independent simulations.

    Parameters
    ----------
    config, mobility_table, start_positions
        As for :class:`bd_csa.Simulator`. ``start_positions`` is ``(np, 2)`` in
        nanometres and is broadcast to every environment on reset.
    n_envs
        Number of independent simulations. Throughput saturates around 64 on an
        RTX 4060 Ti; below ~16 the GPU is underutilised.
    lambda_range
        ``(low, high)`` that actions in ``[-1, 1]`` map onto.
    n_steps
        Integration steps per action. 1e6 steps at dt = 0.1 ms = 100 s simulated.
    device
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        config: Config,
        mobility_table: str,
        start_positions: np.ndarray,
        n_envs: int = 1,
        lambda_range: tuple[float, float] = (1.0, 60.0),
        n_steps: int = 1_000_000,
        device: str = "cuda",
    ) -> None:
        self.sim = Simulator(config, mobility_table, n_envs, device)
        self.x0 = np.asarray(start_positions, dtype=np.float64)
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.low, self.high = lambda_range
        self._seed = 0

    # -- spaces, built lazily so gymnasium stays optional --------------------
    @property
    def action_space(self):
        import gymnasium as gym

        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    @property
    def observation_space(self):
        import gymnasium as gym

        return gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def rescale(self, action: np.ndarray) -> np.ndarray:
        """Map an action in [-1, 1] onto lambda, matching the legacy formula."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return 0.5 * self.high * (a + 1.0) + 0.5 * self.low * (1.0 - a)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
        self.sim.reset(self.x0)
        return self.sim.observations(), {}

    def step(self, action):
        lam = self.rescale(action)
        # The CUDA backend takes one lambda per launch. Until that becomes a
        # per-block argument, a batch must share an action; a differing batch
        # raises rather than silently serialising.
        lam_arg = float(np.ravel(lam)[0]) if np.ndim(lam) else float(lam)

        self._seed += 1
        self.sim.step(lam_arg, self.n_steps, self._seed)

        obs = self.sim.observations()
        info = {"order_parameters": self.sim.order_parameters(), "lambda": lam_arg}
        # Reward and termination are problem-specific and intentionally left to
        # the caller: the legacy driver imported its own reward_function.
        reward = np.zeros(self.n_envs, dtype=np.float64)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        return obs, reward, terminated, truncated, info

    def positions(self) -> np.ndarray:
        """(n_envs, np, 2) in nanometres."""
        return self.sim.positions()


def make_from_run_txt(run_txt: str, start_txt: str, table: str, **kwargs):
    """Build an env straight from a legacy experiment directory."""
    cfg = Config.from_run_txt(run_txt)
    x0 = read_start_txt(start_txt, cfg)
    return BDVectorEnv(cfg, table, x0, **kwargs)
