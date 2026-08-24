"""Visualization for bd_csa: particle configurations and order-parameter traces.

Everything here is driven from the in-memory simulator -- no text files are
read or written unless you ask for a PNG. The plotting functions take plain
numpy arrays, so they also work on data loaded from anywhere else.

Two views, answering different questions:

* :func:`plot_configuration` draws each particle as a true-to-scale circle of
  radius ``a``, coloured by its **local** hexatic order |psi6_i|. Crystalline
  domains show up as patches of uniform colour; the fluid rim stays dark. Because
  the circles are to scale, contact and packing are read directly off the figure.
* :func:`plot_order_parameters` plots the global psi6, C6, R_g and RC against
  time -- the quantitative view of the same run.

Local vs global order
---------------------
``sim.local_psi6()`` is |psi6_i| per particle; ``sim.observations()`` gives the
global |<psi6_i>|. These differ, and the difference is physical: a polycrystal of
well-formed grains at random orientations has **high local** and **low global**
order because the phases cancel in the average. On the shipped initial
configuration the local mean is 0.75 while the global value is 0.41. Colouring by
the local field is what lets you see that distinction.

Requires matplotlib (see requirements.txt); it is imported lazily so the rest of
``bd_csa`` works without it.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# Particle radius used when the caller does not pass a Config. The shipped
# system is monodisperse at a = 1435 nm.
_DEFAULT_A_NM = 1435.0


def _mpl():
    """Import matplotlib with a message that says what to install."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # file output; no display needed
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for bd_csa.visualize. Install it with:\n"
            "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        ) from exc


def format_globals(op: dict, time_s: float | None = None,
                   lam: float | None = None, seed: int | None = None,
                   horizontal: bool = False) -> str:
    """Multi-line label of the GLOBAL order parameters and the control input.

    These are cluster-wide scalars, distinct from the per-particle field used for
    colouring: ``psi6`` here is |<psi6_i>| over all particles, whereas the colour
    shows |psi6_i| for each particle individually. Both appear on the same figure
    on purpose -- the gap between them is what reveals grain structure.

    ``lam`` and ``seed`` describe the step that REACHED this state, so both are
    undefined for the initial frame and are omitted there. Showing the seed on
    the figure means a render carries everything needed to identify the run that
    produced it, not just its outcome.
    """
    lines = []
    if time_s is not None:
        lines.append(f"t = {time_s:.2f} s")
    if lam is not None:
        lines.append(rf"$\lambda$ = {lam:g}")
    if seed is not None:
        lines.append(f"seed = {int(seed)}")
    if "psi6" in op:
        lines.append(rf"$\psi_6$ = {op['psi6']:.4f}")
    if "c6" in op:
        lines.append(rf"$C_6$ = {op['c6']:.3f}")
    if "rg" in op:
        lines.append(rf"$R_g$ = {op['rg']:.0f} nm")
    if "rc" in op:
        lines.append(f"RC = {op['rc']:.4f}")

    if not horizontal:
        return "\n".join(lines)

    # Two rows for the header above the plot: run state on top, order
    # parameters below. Keeping it to two lines stops the header stealing
    # vertical space from the configuration.
    n_state = (time_s is not None) + (lam is not None) + (seed is not None)
    head, tail = lines[:n_state], lines[n_state:]
    rows = [s for s in ("     ".join(head), "     ".join(tail)) if s]
    return "\n".join(rows)


def plot_configuration(
    positions: np.ndarray,
    psi6_local: np.ndarray | None = None,
    *,
    a_nm: float = _DEFAULT_A_NM,
    box_a: float = 30.0,
    title: str | None = None,
    order_params: dict | None = None,
    time_s: float | None = None,
    lam: float | None = None,
    seed: int | None = None,
    annotation: str | None = "outside",
    save_path: str | None = None,
    dpi: int = 200,
    cmap: str = "viridis",
    colorbar: bool = True,
    ax=None,
):
    """Draw one particle configuration as true-to-scale circles.

    Parameters
    ----------
    positions
        ``(np, 2)`` in **nanometres**, as returned by ``sim.positions()[env]``.
    psi6_local
        ``(np,)`` local order in [0,1] from ``sim.local_psi6(env)``. If omitted,
        all particles are drawn in a single colour.
    order_params
        Dict with any of ``psi6``, ``c6``, ``rg``, ``rc`` -- the **global**
        cluster values, as returned by ``sim.order_parameters()[env]``. Drawn as
        an annotation box on the figure. Note these are cluster-wide scalars,
        not the per-particle field used for the colouring.
    time_s
        Simulated time, shown alongside the globals when given.
    lam, seed
        Field strength and RNG seed of the step that reached this state. Both
        undefined for the initial frame, so pass None there. The seed makes a
        render self-identifying: it names the run that produced the picture.
    annotation
        Where to put the global order parameters: ``"outside"`` (default) puts
        them in a header above the axes so they can never overlap particles,
        ``"inside"`` uses a boxed overlay in the top-left corner, ``None``
        omits them.
    a_nm
        Particle radius in nm. Circles are drawn at this radius in data
        coordinates, so the figure is geometrically faithful -- touching circles
        really are touching particles.
    box_a
        Half-width of the view in units of ``a``. The default of 30 matches the
        SAC3 convention so figures are comparable with existing ones.
    save_path
        If given, write a PNG here (parent directories are created).
    ax
        Draw into an existing axes instead of creating a figure. Used by
        :func:`plot_dashboard`.

    Returns the axes, so callers can annotate further.
    """
    plt = _mpl()
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must be (np, 2); got {pos.shape}")

    # Work in units of a: the circle radius becomes exactly 1, which keeps the
    # true-to-scale property obvious and matches the axis convention of the
    # legacy figures.
    xy = pos / a_nm

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(figsize=(8, 8))

    circles = [Circle((x, y), radius=1.0) for x, y in xy]
    coll = PatchCollection(circles, linewidths=0.4, edgecolors="black")

    if psi6_local is not None:
        vals = np.asarray(psi6_local, dtype=float).ravel()
        if vals.size != xy.shape[0]:
            raise ValueError(
                f"psi6_local has {vals.size} entries but there are {xy.shape[0]} particles"
            )
        coll.set_array(vals)
        coll.set_cmap(cmap)
        # Fixed scale: |psi6_i| is bounded in [0,1] by construction, and a fixed
        # range keeps colours comparable across frames of an annealing run.
        # Autoscaling per frame would make the colour bar lie about progress.
        coll.set_clim(0.0, 1.0)
    else:
        coll.set_facecolor("#1f77b4")

    ax.add_collection(coll)
    ax.set_xlim(-box_a, box_a)
    ax.set_ylim(-box_a, box_a)
    ax.set_aspect("equal")          # circles must not become ellipses
    ax.set_xlabel("x / a")
    ax.set_ylabel("y / a")
    if title:
        ax.set_title(title)

    # Global order parameters. Default is a header ABOVE the axes: an in-plot
    # box inevitably collides with particles once the cluster expands (at low
    # lambda the trap weakens and R_g grows), and a legend that hides data is
    # worse than one that costs a little margin.
    if order_params and annotation:
        if annotation == "outside":
            ax.text(
                0.0, 1.015,
                format_globals(order_params, time_s, lam, seed, horizontal=True),
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=10, family="monospace", linespacing=1.5,
            )
        elif annotation == "inside":
            ax.text(
                0.02, 0.98, format_globals(order_params, time_s, lam, seed),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=11, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="0.6", alpha=0.85),
            )
        else:
            raise ValueError(
                f"annotation must be 'outside', 'inside' or None; got {annotation!r}")

    if colorbar and psi6_local is not None:
        cb = ax.figure.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"local order $|\psi_6^{(i)}|$")

    if save_path:
        _save(ax.figure, save_path, dpi)
        if owns_fig:
            plt.close(ax.figure)
    return ax


def plot_order_parameters(
    history: dict[str, Sequence[float]],
    *,
    time: Sequence[float] | None = None,
    time_label: str = "simulated time (s)",
    save_path: str | None = None,
    dpi: int = 200,
    mark_index: int | None = None,
    axes=None,
):
    """Plot psi6, C6, R_g and RC against time in a single stacked figure.

    Parameters
    ----------
    history
        Keys among ``psi6``, ``c6``, ``rg``, ``rc``; each a sequence of equal
        length. Missing keys are skipped, so partial histories work.
    time
        X values. Defaults to the sample index.
    mark_index
        Draw a vertical marker at this sample -- used to tie a time series to a
        particular configuration frame.

    The panels are stacked rather than overlaid because the quantities have
    incompatible ranges: psi6 and RC are in [0,1], C6 in [0,6], and R_g is tens
    of thousands of nanometres. Forcing them onto one axis would either need
    normalisation (hiding the absolute values) or a log scale (hiding the shape).
    """
    plt = _mpl()

    panels = [
        ("psi6", r"$\psi_6$", (0.0, 1.0)),
        ("c6", r"$C_6$", (0.0, 6.0)),
        ("rg", r"$R_g$ (nm)", None),
        ("rc", "RC", (0.0, 1.0)),
    ]
    present = [p for p in panels if p[0] in history and len(history[p[0]])]
    if not present:
        raise ValueError("history contains none of: psi6, c6, rg, rc")

    n = len(history[present[0][0]])
    t = np.arange(n) if time is None else np.asarray(time, dtype=float)

    owns_fig = axes is None
    if owns_fig:
        fig, axes = plt.subplots(
            len(present), 1, figsize=(8, 2.2 * len(present)), sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    for i, (ax, (key, label, ylim)) in enumerate(zip(axes, present)):
        y = np.asarray(history[key], dtype=float)
        ax.plot(t[: len(y)], y, lw=1.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if mark_index is not None and 0 <= mark_index < len(t):
            ax.axvline(t[mark_index], color="crimson", lw=1.2, alpha=0.8)
        # Only the bottom panel carries tick labels: the panels share an x axis
        # by construction, and repeating the labels four times is just noise.
        # (sharex= is unavailable here because plot_dashboard supplies axes that
        # belong to a gridspec it built itself.)
        if i < len(present) - 1:
            ax.tick_params(labelbottom=False)

    axes[len(present) - 1].set_xlabel(time_label if time is not None else "sample")

    if save_path:
        _save(fig, save_path, dpi)
        if owns_fig:
            plt.close(fig)
    return axes


def plot_dashboard(
    positions: np.ndarray,
    history: dict[str, Sequence[float]],
    psi6_local: np.ndarray | None = None,
    *,
    a_nm: float = _DEFAULT_A_NM,
    box_a: float = 30.0,
    time: Sequence[float] | None = None,
    mark_index: int | None = None,
    title: str | None = None,
    order_params: dict | None = None,
    time_s: float | None = None,
    lam: float | None = None,
    seed: int | None = None,
    annotation: str | None = "outside",
    save_path: str | None = None,
    dpi: int = 200,
):
    """Configuration alongside the order-parameter traces, in one figure.

    The traces show the whole run so far with a marker at the frame being drawn,
    which is what makes a sequence of these readable as an annealing sequence.
    """
    plt = _mpl()

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1.0])
    ax_cfg = fig.add_subplot(gs[:, 0])
    ax_ops = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    plot_configuration(
        positions, psi6_local, a_nm=a_nm, box_a=box_a, ax=ax_cfg, colorbar=True,
        title=title, order_params=order_params, time_s=time_s, lam=lam,
        seed=seed, annotation=annotation,
    )
    plot_order_parameters(history, time=time, mark_index=mark_index, axes=ax_ops)

    if save_path:
        _save(fig, save_path, dpi)
        plt.close(fig)
    return fig


def snapshot_series(
    sim,
    lam: float,
    *,
    n_frames: int = 20,
    steps_per_frame: int = 10_000,
    seed: int = 0,
    env: int = 0,
    out_dir: str = "frames",
    dashboard: bool = False,
    dpi: int = 200,
    box_a: float = 30.0,
    record: bool = True,
    annotation: str | None = "outside",
    run_txt_path: str | None = None,
    verbose: bool = True,
):
    """Run a trajectory and write one PNG per frame, keeping state in memory.

    This is the intended entry point for "plot the configuration as it evolves".
    The simulator is stepped in chunks; after each chunk the positions and order
    parameters are read straight out of memory and rendered. Nothing is written
    except the PNGs.

    Returns a dict with ``history`` (the order-parameter traces), ``time_s``,
    ``frames`` (PNG paths) and, when ``record`` is set, ``trajectory`` -- a
    :class:`~bd_csa.trajectory.Trajectory` holding every frame's positions,
    per-particle order, neighbour counts, time and lambda. Save it with
    ``out["trajectory"].save("run.h5")`` and resume later from
    ``Trajectory.load("run.h5").state()``.

    Note ``sim`` must already have been ``reset``. Total simulated time is
    ``n_frames * steps_per_frame * dt``; with the defaults and dt = 0.1 ms that
    is 20 s over 20 frames.
    """
    cfg = sim.config
    a_nm = cfg.a
    dt_ms = cfg.dt

    recorder = None
    if record:
        from .trajectory import TrajectoryRecorder

        recorder = TrajectoryRecorder(
            sim, env=env, run_txt_path=run_txt_path,
            meta={"lambda": lam, "seed_base": seed,
                  "steps_per_frame": steps_per_frame})

    history: dict[str, list[float]] = {"psi6": [], "c6": [], "rg": [], "rc": []}
    times: list[float] = []
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    def capture(frame: int) -> dict:
        t_s = frame * steps_per_frame * dt_ms / 1e3  # ms -> s
        times.append(t_s)
        if recorder is not None:
            # lam and seed describe the step that REACHED this state, so frame 0
            # has neither. The per-chunk seed is seed + frame, matching the
            # sim.step call below.
            op = recorder.record(lam=None if frame == 0 else lam,
                                 seed=None if frame == 0 else seed + frame,
                                 time_s=t_s, step=frame * steps_per_frame)
        else:
            op = sim.order_parameters()[env]
        for k in history:
            history[k].append(op[k])
        return op

    # Frame 0 is the initial configuration, before any dynamics.
    op = capture(0)
    for frame in range(n_frames + 1):
        if frame > 0:
            # Seed varies per chunk so the stream advances; the counter-based
            # RNG makes each chunk independent and reproducible.
            sim.step(lam, steps_per_frame, seed + frame)
            op = capture(frame)

        pos = sim.positions()[env]
        local = sim.local_psi6(env)
        t_s = times[frame]
        path = os.path.join(out_dir, f"frame_{frame:04d}.png")

        if dashboard:
            plot_dashboard(
                pos, history, local, a_nm=a_nm, box_a=box_a, time=times,
                mark_index=frame, order_params=op, time_s=t_s,
                lam=(None if frame == 0 else lam),
                seed=(None if frame == 0 else seed + frame),
                annotation=annotation, save_path=path, dpi=dpi,
            )
        else:
            plot_configuration(
                pos, local, a_nm=a_nm, box_a=box_a,
                order_params=op, time_s=t_s,
                lam=(None if frame == 0 else lam),
                seed=(None if frame == 0 else seed + frame),
                annotation=annotation, save_path=path, dpi=dpi,
            )
        paths.append(path)
        if verbose:
            print(f"  {path}  psi6={history['psi6'][frame]:.4f} "
                  f"C6={history['c6'][frame]:.3f} Rg={history['rg'][frame]:.1f}")

    result = {"history": history, "time_s": times, "frames": paths}
    if recorder is not None:
        result["trajectory"] = recorder.build()
    return result


def _save(fig, save_path: str, dpi: int) -> None:
    parent = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
