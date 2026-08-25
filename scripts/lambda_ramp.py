#!/usr/bin/env python3
"""Lambda-ramp experiment: continuous ordering spectrum, or discrete phases?

Sweeps the field strength lambda slowly up and then back down, holding at each
value long enough to relax, and records the order parameters along both
branches. See documentation/12-image-state-representation.md §12.5.

Lambda range and spacing
------------------------
Theory puts lambda in [0, 20]. Values used in the related literature are
0.2209, 0.8744, 1.9674 and 19.9373 -- spanning nearly two decades, so the sweep
defaults to **logarithmic** spacing. A linear 0-20 sweep with 40 points would
put three of those four values in its first four samples, wasting almost the
whole schedule on the high-lambda end where little changes. The literature
values are injected into the schedule exactly (--literature, on by default) so
results are directly comparable with prior work.

It answers two questions that the representation work depends on:

1. **Continuous or discrete?**  If (psi6, C6, R_g) move smoothly with lambda,
   the state space is a continuous spectrum and a fixed SOM grid is defensible.
   If they jump, or if the up and down branches disagree (hysteresis), there is
   genuine phase structure and a growing SOM (GSOM / neural gas) is the safer
   choice.

2. **Is lattice orientation signal or nuisance?** (§12.4)  The quadrupole pins
   the lab frame, so the crystal's orientation relative to the electrode axes
   may be real physics rather than a symmetry to be removed. This records the
   *phase* of the global psi6, not just its magnitude, and reports whether it
   concentrates or is uniformly distributed.

A caveat worth taking seriously
-------------------------------
Hysteresis in a ramp can be **kinetic** (you swept faster than the system
relaxes) rather than thermodynamic (a real first-order transition). The two look
identical in a single run. The control is to repeat at a different rate: pass
--steps-per-point twice as large and see whether the loop area shrinks. If it
does, the hysteresis is a rate artefact. `--rate-check` runs both for you.

Usage
-----
    .venv/bin/python scripts/lambda_ramp.py --seeds 5 --points 40 \\
        --steps-per-point 200000 --device cuda --out runs/ramp.h5

    .venv/bin/python scripts/lambda_ramp.py --analyze runs/ramp.h5

Cost: n_seeds * 2 * points * steps_per_point integration steps. The defaults
(5 seeds, 40 points, 2e5 steps) are 8e7 env-steps -- roughly 45 min on CUDA at
the measured 34.7 us/step for a single environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Field strengths used in the related literature. Note 0.8744 is also the value
# sitting in run.txt line 44 -- the Fortran reads and discards it because lambda
# comes from argv, but it confirms the scale.
LITERATURE_LAMBDAS = (0.2209, 0.8744, 1.9674, 19.9373)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

try:
    import bd_csa
    from bd_csa.trajectory import load_configuration
except ImportError as exc:
    sys.exit(
        f"cannot import bd_csa: {exc}\n"
        "Build the extension first -- see documentation/10-usage-guide.md §10.1"
    )


# --------------------------------------------------------------------------
# psi6 phase
# --------------------------------------------------------------------------
# The lattice orientation comes straight from the C++ core, which now returns
# the real and imaginary parts of <psi6_i> alongside the magnitude.
#
# This script used to recompute psi6 in numpy to recover the phase, guarded by a
# cross-check against the C++ magnitude. That guard fired on the first real run:
# compute_order_params filters particles to the expbox measurement window before
# averaging, and the numpy version did not. Starting from a dispersed
# configuration, particles sit outside the window and the two disagreed by 1.5%.
# The duplication is now deleted rather than patched -- one implementation
# cannot diverge from itself.

def psi6_per_particle(pos_nm: np.ndarray, rmin_nm: float) -> np.ndarray:
    """Complex psi6_i per particle, for COLOURING SAVED CONFIGURATIONS ONLY.

    Used by the plotting paths, which read positions back from an HDF5 file and
    have no simulator to ask. Note it does not apply the expbox window, so it
    is not a substitute for the core's psi6 -- it is a per-particle field for
    display, never a source of recorded numbers.
    """
    d = pos_nm[None, :, :] - pos_nm[:, None, :]
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    nbr = r2 <= rmin_nm ** 2
    z = np.exp(6j * np.arctan2(d[..., 1], d[..., 0])) * nbr
    counts = nbr.sum(1)
    return np.where(counts > 0, z.sum(1) / np.maximum(counts, 1), 0.0)


def circular_concentration(phases_deg: np.ndarray) -> float:
    """Mean resultant length of the lattice angle, in [0, 1].

    0 = uniformly distributed (orientation is a free nuisance variable),
    1 = perfectly aligned (orientation is pinned, i.e. it is signal).
    Computed on 6*theta so the 60-degree period maps onto the full circle.
    """
    a = np.radians(np.asarray(phases_deg, dtype=float) * 6.0)
    return float(np.abs(np.exp(1j * a).mean()))


# --------------------------------------------------------------------------
# the ramp
# --------------------------------------------------------------------------

def rotate_about_centroid(pos: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a configuration about its own centroid.

    Used to give each seed a DIFFERENT initial lattice orientation. Without
    this every seed inherits start.txt's orientation, and the orientation test
    in §12.4 degenerates into "did the lattice rotate away from where it
    started" -- which it barely does on these timescales, producing a spurious
    concentration of 1.0.

    Note this is a rotation of the *initial condition*, not a symmetry
    operation: continuous rotation is not a symmetry of the quadrupole (§12.4),
    so each rotated start is a genuinely different -- but equally valid --
    physical state. That is exactly what we want to sample.
    """
    c = pos.mean(0)
    t = np.radians(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (pos - c) @ R.T + c


def build_schedule(lam_min: float, lam_max: float, points: int, down: bool,
                   spacing: str = "log",
                   include: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """lambda values and a branch label (0 = up, 1 = down).

    `spacing` is "log" by default because the physically interesting lambdas
    span two decades (see module docstring); linear spacing wastes most of the
    schedule. `include` forces specific values into the sweep so they are
    sampled exactly rather than approximately.
    """
    if spacing == "log":
        if lam_min <= 0:
            raise SystemExit(
                "--lam-min must be > 0 for log spacing (log(0) is undefined). "
                "Use --spacing linear, or add 0 via --include-lam 0.")
        up = np.logspace(np.log10(lam_min), np.log10(lam_max), points)
    elif spacing == "linear":
        up = np.linspace(lam_min, lam_max, points)
    else:
        raise SystemExit(f"unknown spacing {spacing!r}")

    if include:
        extra = [v for v in include if lam_min <= v <= lam_max or v == 0.0]
        up = np.unique(np.concatenate([up, np.asarray(extra, float)]))

    if not down:
        return up, np.zeros(len(up), np.int8)

    # The down branch MIRRORS the up branch rather than being rebuilt from
    # `points`. Two reasons: forced values (--include-lam / the literature
    # lambdas) extend `up`, so a rebuilt down branch would be a different
    # length -- desynchronising `lam` from `branch` -- and the hysteresis test
    # compares the two branches at matched lambda, which only works if the same
    # values are visited on the way back down.
    dn = up[::-1][1:]      # drop the duplicated turning point at lam_max
    lam = np.concatenate([up, dn])
    branch = np.concatenate([np.zeros(len(up), np.int8),
                             np.ones(len(dn), np.int8)])
    assert len(lam) == len(branch), (len(lam), len(branch))
    return lam, branch


def run_ramp(cfg, table, x0, lam_sched, seed, steps_per_point, device,
             equilibrate, samples_per_point=1, verbose=True):
    """One seed through the whole schedule. Returns a dict of arrays.

    Each hold is split into `samples_per_point` equal chunks and the state is
    read after each. This costs NO extra simulation -- the same total steps are
    taken -- but it turns "is the hold long enough?" from an inference into a
    direct observation: if the order parameters are still drifting across the
    samples within one hold, that hold has not relaxed.

    Scalars are recorded per sample, shape (n_points, n_samples). Positions are
    kept only for the LAST sample of each hold: storing every sample would be
    ~100x the bytes for a question that is about the order parameters, not the
    configurations.
    """
    sim = bd_csa.Simulator(cfg, table, 1, device)
    sim.reset(x0)

    if equilibrate:
        sim.step(float(lam_sched[0]), equilibrate, seed)

    n, m = len(lam_sched), max(1, samples_per_point)
    per_sample = max(1, steps_per_point // m)
    out = {k: np.zeros((n, m)) for k in ("psi6", "phase", "c6", "rg", "rc")}
    pos = np.zeros((n, cfg.np, 2))

    for i, lam in enumerate(lam_sched):
        for j in range(m):
            # (seed, i, j) determines this chunk entirely -> reproducible.
            sim.step(float(lam), per_sample, seed * 100_003 + i * m + j)

            op = sim.order_parameters()[0]
            p = sim.positions()[0]

            out["psi6"][i, j] = op["psi6"]
            # Phase from the same code path as the magnitude -- see the note at
            # the top of this file for why this is not recomputed here.
            out["phase"][i, j] = op["psi6_phase"]
            out["c6"][i, j] = op["c6"]
            out["rg"][i, j] = op["rg"]
            out["rc"][i, j] = op["rc"]
        pos[i] = p                                  # last sample of the hold

        if verbose and (i % 10 == 0 or i == n - 1):
            print(f"    point {i+1:3d}/{n}  lam={lam:7.3f}  "
                  f"psi6={out['psi6'][i,-1]:.4f}  C6={out['c6'][i,-1]:.3f}  "
                  f"Rg={out['rg'][i,-1]:.0f}")

    out["steps_per_sample"] = per_sample
    out["positions"] = pos
    return out


def save(path, lam_sched, branch, seeds, results, cfg, args, init_angles=None):
    import h5py

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "bd_csa.LambdaRamp"
        f.attrs["schema_version"] = 1
        f.attrs["schedule"] = json.dumps({
            "lam_min": args.lam_min, "lam_max": args.lam_max,
            "points": args.points, "down": bool(args.down),
            "spacing": args.spacing,
            "literature_lambdas": list(LITERATURE_LAMBDAS) if args.literature else [],
            "steps_per_point": args.steps_per_point,
            "samples_per_point": args.samples_per_point,
            "steps_per_sample": int(results[0]["steps_per_sample"]),
            "equilibrate": args.equilibrate, "device": args.device,
            "rotate_init": bool(args.rotate_init),
            "dt_ms": cfg.dt,
            "hold_time_s": args.steps_per_point * cfg.dt / 1e3,
        })
        f.attrs["physics"] = json.dumps({
            "mobility_update_interval": cfg.physics.mobility_update_interval,
            "smooth_mobility": cfg.physics.smooth_mobility,
            "enable_divD_drift": cfg.physics.enable_divD_drift,
            "periodic": cfg.physics.periodic,
        })
        f.attrs["phase_convention"] = (
            "psi6_phase is the hexatic director in [0,60) deg: "
            "psi6 = |psi6|*exp(6i*theta). Electrode axes at 0 and 90 deg map to "
            "0 and 30 deg under this period.")
        with open(args.run) as fh:
            f.attrs["run_txt"] = fh.read()

        f.create_dataset("lam", data=lam_sched)
        f.create_dataset("branch", data=branch)
        f.create_dataset("seeds", data=np.asarray(seeds, np.int64))
        if init_angles is not None:
            f.create_dataset("initial_rotation_deg", data=np.asarray(init_angles))
        for k in ("psi6", "c6", "rg", "rc"):
            f.create_dataset(k, data=np.stack([r[k] for r in results]))
        f.create_dataset("psi6_phase", data=np.stack([r["phase"] for r in results]))
        f.create_dataset("positions",
                         data=np.stack([r["positions"] for r in results]),
                         compression="gzip")
    return path


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def plot_hysteresis(path: str, out_png: str | None = None) -> str:
    """Draw the hysteresis loops: order parameters vs lambda, both branches.

    Up and down are drawn as separate curves so a loop is visible as an
    enclosed area. The band is +/-1 std across seeds -- a gap that stays inside
    the band is not evidence of hysteresis.
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(path, "r") as f:
        lam = f["lam"][...]
        branch = f["branch"][...]
        sched = json.loads(f.attrs["schedule"])
        d = {k: f[k][...] for k in ("psi6", "c6", "rg", "rc")}
    # Files written with --samples-per-point carry a trailing sample axis; the
    # curves use the last sample of each hold, i.e. the most relaxed state.
    d = {k: (v[..., -1] if v.ndim == 3 else v) for k, v in d.items()}
    out_png = out_png or os.path.splitext(path)[0] + "_hysteresis.png"

    up, dn = branch == 0, branch == 1
    panels = [("psi6", r"$\psi_6$"), ("c6", r"$C_6$"),
              ("rg", r"$R_g$ (nm)"), ("rc", "RC")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for ax, (key, label) in zip(axes.ravel(), panels):
        m, sd = d[key].mean(0), d[key].std(0)
        for mask, name, colour, mark in ((up, "up-ramp", "tab:blue", "o"),
                                         (dn, "down-ramp", "tab:red", "s")):
            if not mask.any():
                continue
            ax.plot(lam[mask], m[mask], marker=mark, ms=3.5, lw=1.5,
                    color=colour, label=name)
            ax.fill_between(lam[mask], m[mask] - sd[mask], m[mask] + sd[mask],
                            color=colour, alpha=0.18, lw=0)
        for v in sched.get("literature_lambdas") or []:
            ax.axvline(v, color="0.6", lw=0.8, ls=":", zorder=0)
        if sched.get("spacing") == "log":
            ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)

    n_seeds = d["psi6"].shape[0]
    fig.suptitle(
        f"lambda hysteresis  |  {n_seeds} seeds, "
        f"{sched['hold_time_s']:.0f} s hold per point, "
        f"{sched.get('spacing','?')} spacing  |  band = +/-1 std across seeds\n"
        "dotted lines: literature lambda values", fontsize=10)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_configurations(path: str, seed_index: int = 0, n_cols: int = 6,
                        out_png: str | None = None) -> str:
    """Contact sheet of configurations across lambda, up branch over down.

    The same lambda appears in both rows, so hysteresis is visible as a
    *structural* difference between the two -- which is what a step in the
    order-parameter curve actually means. Colour is per-particle |psi6_i|,
    recomputed from the stored positions.

    The view box is sized from the data rather than fixed: a dispersed cluster
    can exceed any guess, and silently clipping particles would misrepresent
    exactly the states this figure exists to show.
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(REPO, "python"))
    from bd_csa.visualize import plot_configuration

    with h5py.File(path, "r") as f:
        lam, branch = f["lam"][...], f["branch"][...]
        pos = f["positions"][seed_index]
        sched = json.loads(f.attrs["schedule"])
        run_txt = f.attrs["run_txt"]
    a_nm, rmin = 1435.0, 3780.0
    for line_no, line in enumerate(run_txt.splitlines()):
        if line_no == 13:                       # run.txt line 14 = radius
            a_nm = float(line.split()[0])
        if line_no == 51:                       # line 52 = rmin (nm)
            rmin = float(line.split()[0])

    box = float(np.abs(pos).max()) / a_nm * 1.05

    # Sample lambdas common to both branches, always including the literature
    # values so the sheet lines up with the curves.
    up_lam = lam[branch == 0]
    lit = [v for v in (sched.get("literature_lambdas") or []) if v in set(up_lam)]
    picks = sorted(set(list(lit) + list(up_lam[np.linspace(
        0, len(up_lam) - 1, max(n_cols - len(lit), 2)).astype(int)])))[:n_cols]

    rows = [("up-ramp", 0), ("down-ramp", 1)] if (branch == 1).any() else [("up-ramp", 0)]
    fig, axes = plt.subplots(len(rows), len(picks),
                             figsize=(2.6 * len(picks), 2.8 * len(rows)),
                             squeeze=False, constrained_layout=True)

    for r, (label, b) in enumerate(rows):
        for c, lv in enumerate(picks):
            cand = np.where(branch == b)[0]
            i = cand[np.abs(lam[cand] - lv).argmin()]
            ax = axes[r][c]
            plot_configuration(pos[i], np.abs(psi6_per_particle(pos[i], rmin)),
                               a_nm=a_nm, box_a=box, ax=ax, colorbar=False,
                               annotation=None)
            ax.set_title(rf"$\lambda$ = {lam[i]:.3g}", fontsize=9)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle(f"configurations across the ramp (seed index {seed_index}), "
                 rf"coloured by local $|\psi_6^{{(i)}}|$", fontsize=11)
    out_png = out_png or os.path.splitext(path)[0] + "_configs.png"
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_frames(path: str, out_dir: str, seed_index: int = 0) -> int:
    """Write one configuration PNG per lambda point, like frames_lam0/.

    Same renderer as bd_csa.visualize, so these match the trajectory snapshots:
    true-to-scale circles coloured by local |psi6_i|, with the globals in the
    header. One image per point of the schedule (both branches), for a single
    seed -- rendering every seed and every intra-hold sample would be thousands
    of files for little added insight.
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(REPO, "python"))
    from bd_csa.visualize import plot_configuration

    with h5py.File(path, "r") as f:
        lam, branch = f["lam"][...], f["branch"][...]
        pos = f["positions"][seed_index]
        seed = int(f["seeds"][seed_index])
        run_txt = f.attrs["run_txt"]
        g = {k: f[k][...] for k in ("psi6", "c6", "rg", "rc")}
    g = {k: (v[seed_index, :, -1] if v.ndim == 3 else v[seed_index])
         for k, v in g.items()}

    a_nm, rmin = 1435.0, 3780.0
    for i, line in enumerate(run_txt.splitlines()):
        if i == 13:
            a_nm = float(line.split()[0])
        if i == 51:
            rmin = float(line.split()[0])

    box = float(np.abs(pos).max()) / a_nm * 1.05     # sized from data, never clips
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(lam)):
        tag = "up" if branch[i] == 0 else "dn"
        plot_configuration(
            pos[i], np.abs(psi6_per_particle(pos[i], rmin)),
            a_nm=a_nm, box_a=box,
            order_params={k: float(g[k][i]) for k in ("psi6", "c6", "rg", "rc")},
            lam=float(lam[i]), seed=seed,
            save_path=os.path.join(out_dir, f"frame_{i:04d}_{tag}.png"), dpi=170)
    return len(lam)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze(path: str, plot: bool = True) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != "bd_csa.LambdaRamp":
            raise SystemExit(f"{path}: not a lambda-ramp file")
        lam = f["lam"][...]
        branch = f["branch"][...]
        sched = json.loads(f.attrs["schedule"])
        d = {k: f[k][...] for k in ("psi6", "c6", "rg", "rc", "psi6_phase")}

    # Files may be (n_seeds, n_points) or (n_seeds, n_points, n_samples).
    samples = {k: v for k, v in d.items() if v.ndim == 3}
    # The curves use the LAST sample of each hold -- the most relaxed state.
    d = {k: (v[..., -1] if v.ndim == 3 else v) for k, v in d.items()}

    n_seeds = d["psi6"].shape[0]
    up, dn = branch == 0, branch == 1
    print(f"{path}")
    print(f"  {n_seeds} seeds, {len(lam)} points, lambda "
          f"{sched['lam_min']}..{sched['lam_max']} "
          f"({sched.get('spacing','?')} spacing), "
          f"hold {sched['hold_time_s']:.1f} s per point")
    lit = sched.get("literature_lambdas") or []
    if lit:
        print("  values at the literature lambdas:")
        for v in lit:
            j = int(np.abs(lam - v).argmin())
            print(f"    lam={v:8.4f}  psi6={d['psi6'][:, j].mean():.4f} "
                  f"+/- {d['psi6'][:, j].std():.4f}   "
                  f"C6={d['c6'][:, j].mean():.3f}   "
                  f"R_g={d['rg'][:, j].mean():.0f} nm")
    print()

    if samples:
        n_s = next(iter(samples.values())).shape[2]
        print(f"-- 0. did each hold equilibrate? ({n_s} samples per hold) --")
        any_bad = False
        for key, label in (("psi6", "psi6"), ("c6", "C6"), ("rg", "R_g")):
            v = samples[key]                       # (seeds, points, samples)
            half = n_s // 2
            if half < 1:
                continue
            # Systematic drift within a hold = still relaxing. Compare the
            # second half of the samples against the first.
            drift = v[..., half:].mean(-1) - v[..., :half].mean(-1)
            spread = v[..., -1].std(0).mean()      # seed-to-seed noise
            frac = float((np.abs(drift).mean(0) > spread).mean())
            flag = "STILL DRIFTING" if frac > 0.3 else "settled"
            any_bad |= frac > 0.3
            print(f"  {label:5s} |drift| exceeds seed noise at "
                  f"{frac*100:5.1f}% of lambda points  -> {flag}")
        print("  (drift = mean of the last half of a hold minus the first half)")
        if any_bad:
            print("  ACTION: holds are too short. Increase --steps-per-point;")
            print("  hysteresis measured here would be kinetic, not physical.")
        print()

    print("-- 1. continuous or discrete? --")
    for key, label in (("psi6", "psi6"), ("c6", "C6"), ("rg", "R_g")):
        m = d[key].mean(0)
        step = np.abs(np.diff(m[up]))
        # A transition that is sharp relative to the rest of the sweep shows up
        # as one step far larger than the median step.
        ratio = step.max() / max(np.median(step), 1e-12)
        at = lam[up][1:][step.argmax()]
        verdict = "SHARP" if ratio > 5 else "smooth"
        print(f"  {label:5s} largest step {step.max():.4f} at lam={at:5.2f}  "
              f"({ratio:5.1f}x median)  -> {verdict}")

    if dn.any():
        print("\n-- 2. hysteresis (up vs down at matched lambda) --")
        for key, label in (("psi6", "psi6"), ("c6", "C6"), ("rg", "R_g")):
            m = d[key].mean(0)
            # Interpolate the down branch onto the up-branch lambdas.
            lu, ld = lam[up], lam[dn][::-1]
            gap = m[up] - np.interp(lu, ld, m[dn][::-1])
            spread = d[key][:, up].std(0).mean()      # seed-to-seed noise
            sig = "SIGNIFICANT" if np.abs(gap).max() > 2 * spread else "within noise"
            print(f"  {label:5s} max gap {np.abs(gap).max():.4f} at "
                  f"lam={lu[np.abs(gap).argmax()]:5.2f}  "
                  f"(seed spread {spread:.4f})  -> {sig}")
        print("  NOTE: rerun with a larger --steps-per-point. If the gap shrinks,")
        print("        the hysteresis is kinetic, not a phase transition.")

    print("\n-- 3. is lattice orientation signal or nuisance? --")
    ph = d["psi6_phase"]                                 # (n_seeds, n_points)
    if n_seeds < 3:
        print(f"  only {n_seeds} seeds: not enough independent samples to say "
              "anything. Use >= 5.")
        return
    if not sched.get("rotate_init", False):
        print("  WARNING: this run did not randomise the initial lattice")
        print("  orientation across seeds, so every seed inherited the same")
        print("  starting angle. The statistic below is NOT a valid test of")
        print("  whether the field pins orientation. Rerun without")
        print("  --no-rotate-init.")

    # Across seeds at fixed lambda the samples are independent. Pooling all
    # (seed, point) pairs would inflate the concentration, because successive
    # points in one ramp are strongly correlated -- the lattice reorients far
    # more slowly than the sweep advances.
    per_lambda = np.array([circular_concentration(ph[:, i])
                           for i in range(ph.shape[1])])
    R = float(per_lambda.mean())
    R_pooled = circular_concentration(ph.ravel())
    print(f"  concentration across seeds, averaged over lambda: R = {R:.3f}")
    print(f"  (pooled over all samples: {R_pooled:.3f} -- inflated by temporal "
          "correlation, shown only for contrast)")
    print(f"  expected R for {n_seeds} uniformly random angles: "
          f"~{1.0/np.sqrt(n_seeds):.3f} by chance alone")

    chance = 1.0 / np.sqrt(n_seeds)
    if R < 1.5 * chance:
        print("  -> consistent with uniform: orientation is a NUISANCE variable.")
    elif R > 3 * chance:
        print("  -> concentrated well beyond chance: orientation is SIGNAL and")
        print("     must not be canonicalised or augmented away (§12.4).")
    else:
        print("  -> inconclusive; add seeds.")

    final = ph[:, -1]
    hist, edges = np.histogram(ph.ravel(), bins=6, range=(0, 60))
    print("  angle histogram (deg): " +
          "  ".join(f"[{edges[i]:2.0f}-{edges[i+1]:2.0f}) {hist[i]:4d}"
                   for i in range(6)))
    print(f"  final angles per seed: "
          + ", ".join(f"{a:.1f}" for a in final))
    print("  electrode axes sit at 0 and 30 deg under this 60-deg period.")

    if plot:
        try:
            print(f"\n  hysteresis curves    -> {plot_hysteresis(path)}")
            print(f"  configuration sheet  -> {plot_configurations(path)}")
        except ImportError:
            print("\n  (matplotlib not available; skipped the figures)")


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Lambda-ramp experiment (documentation §12.5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--analyze", metavar="PATH",
                   help="analyse an existing ramp file and exit")
    p.add_argument("--start", default=os.path.join(REPO, "data", "start.txt"))
    p.add_argument("--run", default=os.path.join(REPO, "data", "run.txt"))
    p.add_argument("--table",
                   default=os.path.join(REPO, "data", "2dtabledssnp300.txt"))
    p.add_argument("--lam-min", type=float, default=0.2,
                   help="must be > 0 for log spacing")
    p.add_argument("--lam-max", type=float, default=20.0,
                   help="theory puts lambda in [0, 20]")
    p.add_argument("--spacing", choices=("log", "linear"), default="log",
                   help="log by default: the interesting lambdas span two decades")
    p.add_argument("--no-literature", dest="literature", action="store_false",
                   help="do NOT force the literature lambdas "
                        f"{LITERATURE_LAMBDAS} into the schedule")
    p.add_argument("--include-lam", type=float, nargs="*", default=[],
                   metavar="L", help="extra lambda values to sample exactly")
    p.add_argument("--points", type=int, default=40,
                   help="lambda values per branch")
    p.add_argument("--steps-per-point", type=int, default=200_000,
                   help="integration steps held at each lambda (20 s at dt=0.1ms)")
    p.add_argument("--equilibrate", type=int, default=100_000,
                   help="steps at lam_min before the sweep starts")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--no-down", dest="down", action="store_false",
                   help="up-ramp only; disables the hysteresis test")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--out", default="runs/lambda_ramp.h5")
    p.add_argument("--no-rotate-init", dest="rotate_init", action="store_false",
                   help="do NOT randomise each seed's initial lattice "
                        "orientation. Off by default because sharing one "
                        "orientation across seeds invalidates the §12.4 test")
    p.add_argument("--samples-per-point", type=int, default=10,
                   help="state readouts within each hold. Costs no extra "
                        "simulation and gives a direct equilibration check")
    p.add_argument("--save-frames", metavar="DIR",
                   help="also write individual configuration PNGs (like "
                        "frames_lam0/) -- one per lambda point")
    p.add_argument("--frames-seed", type=int, default=0,
                   help="which seed index --save-frames renders")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="skip writing the hysteresis figure")
    p.add_argument("--rate-check", action="store_true",
                   help="also run at 2x steps-per-point into <out>.slow.h5, the "
                        "control that separates kinetic from thermodynamic "
                        "hysteresis")
    args = p.parse_args()

    if args.analyze:
        analyze(args.analyze, plot=args.plot)
        if args.save_frames:
            n = save_frames(args.analyze, args.save_frames, args.frames_seed)
            print(f"  individual frames    -> {args.save_frames}/ ({n} PNGs)")
        return 0

    cfg = bd_csa.Config.from_run_txt(args.run)
    if args.device == "cuda" and not bd_csa.cuda_available():
        raise SystemExit("--device cuda requested but no CUDA device found")
    x0 = load_configuration(args.start, cfg.np, cfg.a)

    forced = tuple(args.include_lam) + (LITERATURE_LAMBDAS if args.literature else ())
    lam_sched, branch = build_schedule(args.lam_min, args.lam_max, args.points,
                                       args.down, args.spacing, forced)
    seeds = [args.seed_base + k for k in range(args.seeds)]
    total = args.seeds * (len(lam_sched) * args.steps_per_point + args.equilibrate)

    n_up = int((branch == 0).sum())
    print(f"lambda ramp  {args.lam_min} -> {args.lam_max}"
          f"{' -> ' + str(args.lam_min) if args.down else ''}"
          f"  ({args.spacing} spacing, {n_up} points per branch)")
    if args.literature:
        print(f"  literature values forced into the schedule: "
              f"{list(LITERATURE_LAMBDAS)}")
    print(f"  {args.seeds} seeds x {len(lam_sched)} points x "
          f"{args.steps_per_point:,} steps = {total:,} env-steps")
    print(f"  hold {args.steps_per_point * cfg.dt / 1e3:.1f} s simulated per point")
    print(f"  device {args.device}   -> {args.out}\n")

    t0 = time.time()
    results = []
    rng = np.random.default_rng(args.seed_base)
    init_angles = (np.zeros(len(seeds)) if not args.rotate_init
                   else rng.uniform(0.0, 360.0, len(seeds)))
    for k, s in enumerate(seeds):
        print(f"  seed {s} ({k+1}/{len(seeds)})  "
              f"initial rotation {init_angles[k]:6.1f} deg")
        results.append(run_ramp(cfg, args.table,
                                rotate_about_centroid(x0, init_angles[k]),
                                lam_sched, s, args.steps_per_point,
                                args.device, args.equilibrate,
                                args.samples_per_point))
        # Write after every seed. A crash in seed 4 should not discard seeds
        # 0-3; the first real run lost a completed seed to a late exception.
        save(args.out, lam_sched, branch, seeds[:len(results)], results, cfg,
             args, init_angles=init_angles[:len(results)])
    save(args.out, lam_sched, branch, seeds, results, cfg, args,
         init_angles=init_angles)
    print(f"\nwrote {args.out} in {time.time()-t0:.1f} s "
          f"({os.path.getsize(args.out)/1e6:.2f} MB)\n")
    analyze(args.out, plot=args.plot)
    if args.save_frames:
        n = save_frames(args.out, args.save_frames, args.frames_seed)
        print(f"  individual frames    -> {args.save_frames}/ ({n} PNGs)")

    if args.rate_check:
        slow = args.out.replace(".h5", "") + ".slow.h5"
        print(f"\n{'='*70}\nrate control: 2x hold time -> {slow}\n{'='*70}")
        args.steps_per_point *= 2
        results = [run_ramp(cfg, args.table,
                            rotate_about_centroid(x0, init_angles[k]),
                            lam_sched, s, args.steps_per_point, args.device,
                            args.equilibrate, args.samples_per_point)
                   for k, s in enumerate(seeds)]
        save(slow, lam_sched, branch, seeds, results, cfg, args,
             init_angles=init_angles)
        analyze(slow, plot=args.plot)
        print("\nCompare the hysteresis gaps between the two files: if the slow "
              "run's gap is materially smaller, the hysteresis is kinetic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
