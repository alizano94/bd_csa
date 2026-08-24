"""Trajectory storage and configuration I/O.

Replaces the legacy text-file interface (`start.txt` in, `bd_xyz1.txt` +
`op1.txt` + `out_param.json` out) with a single self-describing HDF5 file
holding everything about a run.

Why HDF5 and not pickle
-----------------------
A pickle stores the Python object graph, which means it is Python-only, breaks
silently when a class definition changes, and executes arbitrary code on load.
HDF5 stores *data* with a schema: it is readable from any language, survives
refactors of this module, and supports partial reads of files larger than
memory. For results you intend to keep, that matters more than convenience.

What a trajectory holds
-----------------------
Every array has a leading time axis ``T`` of length ``n_frames``:

===============  ====================  ==================================
name             shape                 meaning
===============  ====================  ==================================
``positions``    ``(T, np, 2)``        coordinates, **nanometres**
``psi6``         ``(T,)``              global |<psi6_i>|
``c6``           ``(T,)``              mean hexatic connectivity, [0,6]
``rg``           ``(T,)``              radius of gyration, nm
``rc``           ``(T,)``              composite crystallinity
``psi6_local``   ``(T, np)``           per-particle |psi6_i|, [0,1]
``neighbours``   ``(T, np)`` int32     neighbour count within rmin
``time_s``       ``(T,)``              simulated time, seconds
``lam``          ``(T,)``              field strength -- see below
``step``         ``(T,)`` int64        cumulative integration steps
``seed``         ``(T,)`` int64        RNG seed that produced this frame
===============  ====================  ==================================

``seed[k]`` is the RNG seed passed to the step that produced frame ``k``.
Together with ``positions[0]``, ``lam`` and the embedded ``run_txt`` this makes a
trajectory **replayable**: :func:`replay` re-runs it and checks the positions
come back identical. Frame 0 has ``seed = -1`` for the same reason ``lam`` is
NaN there -- no step produced it.

**The λ convention matters.** ``lam[k]`` is the field strength applied *to
reach* state ``k``, not the one applied from it. Frame 0 is the initial
configuration, which no action produced, so ``lam[0]`` is NaN. This is the
convention reinforcement learning expects: action ``lam[k]`` maps state ``k-1``
to state ``k``.

``meta`` carries the full configuration -- particle count, radius, dt, physics
flags, seed, device, package version -- so a file records which physics produced
it. That matters here because the corrected physics shifts psi6 by ~8% relative
to the legacy behaviour.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

_ARRAYS_2D = ("positions",)
_ARRAYS_1D = ("psi6", "c6", "rg", "rc", "time_s", "lam", "step", "seed")
_ARRAYS_PARTICLE = ("psi6_local", "neighbours")

# File-format version, bumped if the schema changes incompatibly.
SCHEMA_VERSION = 1


def _h5py():
    try:
        import h5py

        return h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "h5py is required for trajectory storage. Install it with:\n"
            "    .venv/bin/pip install -r requirements.txt"
        ) from exc


@dataclass
class Trajectory:
    """An entire run: positions, order parameters, time and control input."""

    positions: np.ndarray
    psi6: np.ndarray
    c6: np.ndarray
    rg: np.ndarray
    rc: np.ndarray
    time_s: np.ndarray
    lam: np.ndarray
    step: np.ndarray
    seed: np.ndarray | None = None
    psi6_local: np.ndarray | None = None
    neighbours: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.positions)

    @property
    def n_particles(self) -> int:
        return self.positions.shape[1]

    def __repr__(self) -> str:
        return (
            f"<Trajectory {len(self)} frames x {self.n_particles} particles, "
            f"t = {self.time_s[0]:.2f}..{self.time_s[-1]:.2f} s, "
            f"psi6 {self.psi6[0]:.3f} -> {self.psi6[-1]:.3f}>"
        )

    # -- resuming ----------------------------------------------------------
    def state(self, index: int = -1) -> np.ndarray:
        """Positions of one frame, ``(np, 2)`` in nm -- feed to ``sim.reset``.

        Defaults to the final frame, which is what "continue this run" means.
        """
        return np.ascontiguousarray(self.positions[index])

    # -- persistence -------------------------------------------------------
    def save(self, path: str, compression: str | None = "gzip") -> str:
        """Write to HDF5. Returns the path."""
        h5py = _h5py()
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        opts = {"compression": compression} if compression else {}
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["format"] = "bd_csa.Trajectory"
            # Metadata as JSON rather than individual attributes: it keeps
            # nested physics options intact and avoids HDF5 type juggling.
            f.attrs["meta"] = json.dumps(self.meta, default=str)
            # Units are recorded in the file so it stays self-describing even if
            # this module is not around to explain it.
            f.attrs["units"] = json.dumps(
                {"positions": "nm", "time_s": "s", "rg": "nm",
                 "lam": "dimensionless", "psi6": "[0,1]", "c6": "[0,6]"}
            )
            f.attrs["lam_convention"] = (
                "lam[k] is the field strength applied to reach state k; "
                "lam[0] is NaN because frame 0 is the initial configuration"
            )

            for name in _ARRAYS_2D + _ARRAYS_1D + _ARRAYS_PARTICLE:
                arr = getattr(self, name, None)
                if arr is None:
                    continue
                f.create_dataset(name, data=np.asarray(arr), **opts)
        return path

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        """Read a trajectory written by :meth:`save`."""
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            fmt = f.attrs.get("format", "")
            if fmt != "bd_csa.Trajectory":
                raise ValueError(
                    f"{path}: not a bd_csa trajectory (format={fmt!r})"
                )
            version = int(f.attrs.get("schema_version", 0))
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: schema version {version} is newer than this "
                    f"package understands ({SCHEMA_VERSION}); upgrade bd_csa"
                )
            data = {k: f[k][...] for k in f.keys()}
            meta = json.loads(f.attrs.get("meta", "{}"))
        return cls(meta=meta, **data)


class TrajectoryRecorder:
    """Accumulate frames during a run, then :meth:`build` or :meth:`save`.

    Frames are held in memory: a 300-particle frame is ~7 kB with every field,
    so even 10,000 frames is well under 100 MB. If that ever stops being true,
    this is the place to switch to appending directly into an HDF5 dataset.
    """

    def __init__(self, sim, env: int = 0, meta: dict | None = None,
                 run_txt_path: str | None = None):
        self._sim = sim
        self._env = env
        self._frames: list[dict] = []
        cfg = sim.config
        self._meta = {
            "np": cfg.np,
            "a_nm": cfg.a,
            "dt_ms": cfg.dt,
            "temperature_C": cfg.tempr,
            "dg_nm": cfg.dg,
            "rcut_nm": cfg.rcut,
            "re_nm": cfg.re,
            "rmin_nm": cfg.rmin,
            "device": getattr(sim, "device", "unknown"),
            "physics": {
                "mobility_update_interval": cfg.physics.mobility_update_interval,
                "smooth_mobility": cfg.physics.smooth_mobility,
                "enable_divD_drift": cfg.physics.enable_divD_drift,
                "periodic": cfg.physics.periodic,
                "continuous_overlap": cfg.physics.continuous_overlap,
            },
        }
        try:
            from . import __version__

            self._meta["bd_csa_version"] = __version__
        except ImportError:  # pragma: no cover
            pass
        # Embed the parameter file verbatim. Only ~9 kB, and it is the
        # difference between a trajectory that describes a run and one that can
        # reconstruct it: Config carries many fields the Python bindings do not
        # expose as setters, so from_run_txt is the only way to rebuild it.
        if run_txt_path and os.path.exists(run_txt_path):
            with open(run_txt_path) as fh:
                self._meta["run_txt"] = fh.read()
            self._meta["run_txt_path"] = os.path.abspath(run_txt_path)
        if meta:
            self._meta.update(meta)

    def record(self, *, lam: float | None, time_s: float, step: int,
               seed: int | None = None) -> dict:
        """Capture the current state.

        ``lam`` and ``seed`` describe the step that *reached* this state; pass
        None for both on the initial frame, where no step has been taken.
        """
        if seed is not None and not (0 <= int(seed) < 2**63):
            raise ValueError(
                f"seed {seed} outside the int64 range storable in HDF5")
        sim, env = self._sim, self._env
        op = sim.order_parameters()[env]
        self._frames.append(
            {
                "positions": np.asarray(sim.positions()[env], dtype=np.float64),
                "psi6_local": np.asarray(sim.local_psi6(env), dtype=np.float64),
                "neighbours": np.asarray(sim.neighbour_counts(env), dtype=np.int32),
                "psi6": op["psi6"],
                "c6": op["c6"],
                "rg": op["rg"],
                "rc": op["rc"],
                "time_s": float(time_s),
                "lam": float("nan") if lam is None else float(lam),
                "step": int(step),
                # -1 marks "no step produced this frame"; 0 is a legal seed.
                "seed": -1 if seed is None else int(seed),
            }
        )
        return op

    def build(self) -> Trajectory:
        if not self._frames:
            raise ValueError("no frames recorded")
        stack = lambda k: np.stack([f[k] for f in self._frames])  # noqa: E731
        col = lambda k, dt: np.array([f[k] for f in self._frames], dtype=dt)  # noqa: E731
        return Trajectory(
            positions=stack("positions"),
            psi6_local=stack("psi6_local"),
            neighbours=stack("neighbours"),
            psi6=col("psi6", np.float64),
            c6=col("c6", np.float64),
            rg=col("rg", np.float64),
            rc=col("rc", np.float64),
            time_s=col("time_s", np.float64),
            lam=col("lam", np.float64),
            step=col("step", np.int64),
            seed=col("seed", np.int64),
            meta=dict(self._meta),
        )

    def save(self, path: str, **kw) -> str:
        return self.build().save(path, **kw)


def replay(traj: "Trajectory", table_path: str, *, device: str | None = None,
           run_txt: str | None = None, verbose: bool = True) -> dict:
    """Re-run a recorded trajectory and report how exactly it reproduces.

    This is the check that a file really is sufficient to reconstruct its run.
    It rebuilds the configuration from the embedded ``run_txt`` (or the path
    given), restores the recorded physics flags, resets to ``positions[0]`` and
    replays every chunk with the stored ``(lam, seed, n_steps)``.

    Returns ``{"max_position_error_nm", "max_psi6_error", "exact", ...}``.

    Caveats that will make a replay differ, all of them recorded in ``meta`` so
    you can tell which applies:

    * **Device.** The CUDA backend computes forces in FP32 and the CPU backend
      in FP64, so the two diverge chaotically. A replay must run on the device
      the trajectory was produced on -- that is the default.
    * **Physics flags.** Restored from ``meta``; a mismatch changes the dynamics.
    * **Package version.** A change to the force kernel or RNG between versions
      will show up here, which is precisely the point of keeping the check.
    """
    import tempfile

    from ._bd_csa import Config, Simulator

    if traj.seed is None:
        raise ValueError(
            "this trajectory has no seed array and cannot be replayed; it was "
            "written before seeds were recorded")

    meta = traj.meta or {}
    embedded = meta.get("run_txt")
    if run_txt is None and not embedded:
        raise ValueError(
            "no run.txt embedded in the trajectory and none supplied; pass "
            "run_txt=<path>")

    tmp = None
    try:
        if run_txt is None:
            tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
            tmp.write(embedded)
            tmp.close()
            run_txt = tmp.name
        cfg = Config.from_run_txt(run_txt)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)

    phys = meta.get("physics", {})
    for k, v in phys.items():
        if hasattr(cfg.physics, k):
            setattr(cfg.physics, k, v)

    dev = device or meta.get("device", "cpu")
    sim = Simulator(cfg, table_path, 1, dev)
    sim.reset(traj.state(0))

    # Chunk lengths come from the cumulative step counter, so a run with a
    # varying schedule replays correctly too.
    n_steps = np.diff(np.asarray(traj.step, dtype=np.int64))

    worst_pos = 0.0
    worst_psi6 = 0.0
    for k in range(1, len(traj)):
        sim.step(float(traj.lam[k]), int(n_steps[k - 1]), int(traj.seed[k]))
        got = sim.positions()[0]
        worst_pos = max(worst_pos, float(np.abs(got - traj.positions[k]).max()))
        worst_psi6 = max(
            worst_psi6,
            abs(sim.order_parameters()[0]["psi6"] - float(traj.psi6[k])))
        if verbose:
            print(f"  frame {k:4d}  max |dx| = {worst_pos:.3e} nm")

    result = {
        "frames": len(traj) - 1,
        "device": dev,
        "max_position_error_nm": worst_pos,
        "max_psi6_error": worst_psi6,
        "exact": worst_pos == 0.0,
    }
    if verbose:
        verdict = "EXACT" if result["exact"] else "differs"
        print(f"replay {verdict}: max |dx| = {worst_pos:.3e} nm, "
              f"max |dpsi6| = {worst_psi6:.3e} on {dev}")
    return result


# --------------------------------------------------------------------------
# Single-configuration I/O
# --------------------------------------------------------------------------
# Positions are always **nanometres** in memory. The legacy text format stores
# multiples of the particle radius, so .txt conversions are explicit.

_LEGACY_Z_OVER_A = 1.0663  # hlev / a; the z column the Fortran writes


def save_configuration(positions: np.ndarray, path: str, a_nm: float,
                       fmt: str | None = None) -> str:
    """Write one configuration so it can be reused as a starting point.

    Format is taken from the extension unless ``fmt`` overrides it:

    ``.npy``
        Raw ``(np, 2)`` float64 array in nm. Compact and exact.
    ``.txt``
        Legacy four-column layout (index, x/a, y/a, z/a) that the Fortran
        ``bdpd`` and the C++ CLI both read. Use this for interoperability, not
        for precision: it round-trips at 5 decimal places in units of a, i.e.
        ~0.01 nm.
    ``.h5``
        A one-frame trajectory, so provenance metadata travels with it.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must be (np, 2); got {pos.shape}")

    ext = (fmt or os.path.splitext(path)[1].lstrip(".")).lower()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    if ext == "npy":
        np.save(path, pos)
    elif ext == "txt":
        with open(path, "w") as fh:
            for i, (x, y) in enumerate(pos / a_nm, start=1):
                fh.write(f"{i:7d}{x:16.5f}{y:16.5f}{_LEGACY_Z_OVER_A:16.5f}\n")
    elif ext in ("h5", "hdf5"):
        n = pos.shape[0]
        Trajectory(
            positions=pos[None, ...],
            psi6=np.full(1, np.nan), c6=np.full(1, np.nan),
            rg=np.full(1, np.nan), rc=np.full(1, np.nan),
            time_s=np.zeros(1), lam=np.full(1, np.nan),
            step=np.zeros(1, dtype=np.int64),
            meta={"np": n, "a_nm": a_nm, "note": "single configuration"},
        ).save(path)
    else:
        raise ValueError(f"unsupported configuration format {ext!r}; "
                         "use npy, txt or h5")
    return path


def load_configuration(path: str, n_particles: int, a_nm: float,
                       index: int = -1) -> np.ndarray:
    """Read a configuration from any supported source. Returns ``(np, 2)`` nm.

    Accepts:

    * ``.npy`` -- as written by :func:`save_configuration`
    * ``.h5`` / ``.hdf5`` -- a trajectory; ``index`` selects the frame,
      defaulting to the last, which is what "resume this run" means
    * ``.txt`` -- legacy layout. A file holding several concatenated frames
      (``bd_xyz1.txt``) is accepted and the frame chosen by ``index``.
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")

    if ext == "npy":
        pos = np.load(path)
        if pos.shape != (n_particles, 2):
            raise ValueError(
                f"{path}: expected ({n_particles}, 2), got {pos.shape}")
        return np.ascontiguousarray(pos, dtype=np.float64)

    if ext in ("h5", "hdf5"):
        traj = Trajectory.load(path)
        if traj.n_particles != n_particles:
            raise ValueError(
                f"{path}: holds {traj.n_particles} particles, config expects "
                f"{n_particles}")
        return traj.state(index)

    if ext in ("txt", "dat", ""):
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    rows.append((float(parts[1]), float(parts[2])))
        if len(rows) < n_particles or len(rows) % n_particles:
            raise ValueError(
                f"{path}: {len(rows)} particle rows is not a whole number of "
                f"{n_particles}-particle frames")
        frames = np.asarray(rows, dtype=np.float64).reshape(-1, n_particles, 2)
        return np.ascontiguousarray(frames[index]) * a_nm  # units of a -> nm

    raise ValueError(f"unsupported configuration format {ext!r}")
