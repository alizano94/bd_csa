#!/usr/bin/env python3
"""Run a single trajectory from an existing configuration and render N images.

    scripts/run_and_plot.py                       # 10 images from data/start.txt
    scripts/run_and_plot.py --images 20 --lam 45
    scripts/run_and_plot.py --start frames/last.txt --dashboard

Everything stays in memory: the simulator is stepped in chunks and each frame is
rendered straight from `sim.positions()` / `sim.local_psi6()`. The only files
written are the PNGs (and the optional history .npz).

Run it with the project venv so the compiled extension is importable:

    PYTHONPATH=python .venv/bin/python scripts/run_and_plot.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Allow running as `python scripts/run_and_plot.py` without setting PYTHONPATH.
sys.path.insert(0, os.path.join(REPO, "python"))

try:
    import bd_csa
    from bd_csa import visualize
    from bd_csa.trajectory import load_configuration, save_configuration
except ImportError as exc:
    sys.exit(
        f"cannot import bd_csa: {exc}\n\n"
        "The compiled extension is missing. Build it with:\n"
        "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
        "  cmake -B build/cmake -DCMAKE_BUILD_TYPE=Release "
        "-DBD_CSA_BUILD_PYTHON=ON -DPython3_EXECUTABLE=$PWD/.venv/bin/python\n"
        "  cmake --build build/cmake -j16\n"
        "and run this script with .venv/bin/python."
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run one trajectory and render a sequence of snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default=os.path.join(REPO, "data", "start.txt"),
                   help="initial configuration (start.txt or a bd_xyz trajectory)")
    p.add_argument("--run", default=os.path.join(REPO, "data", "run.txt"),
                   help="parameter file")
    p.add_argument("--table", default=os.path.join(REPO, "data", "2dtabledssnp300.txt"),
                   help="mobility table")
    p.add_argument("--images", type=int, default=10,
                   help="TOTAL images to write, including the t=0 frame")
    p.add_argument("--steps-per-frame", type=int, default=10_000,
                   help="integration steps between frames (dt = 0.1 ms)")
    p.add_argument("--lam", type=float, default=30.0,
                   help="field strength lambda; higher anneals harder")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", default="frames", help="output directory for PNGs")
    p.add_argument("--dashboard", action="store_true",
                   help="render configuration + order-parameter traces per frame")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--annotation", choices=("outside", "inside", "none"),
                   default="outside",
                   help="where to draw the global order parameters; 'outside' "
                        "keeps them clear of the particles")
    p.add_argument("--box", type=float, default=30.0,
                   help="half-width of the view, in units of a")
    p.add_argument("--legacy", action="store_true",
                   help="legacy physics (frozen mobility, nearest-bin lookup, no "
                        "div.D drift) -- use to compare against old Fortran runs")
    p.add_argument("--save-trajectory", metavar="PATH",
                   help="write the full run (positions, order parameters, "
                        "per-particle order, neighbour counts, time, lambda "
                        "and config metadata) to an HDF5 file")
    p.add_argument("--save-initial", metavar="PATH",
                   help="export the starting configuration for reuse; format "
                        "from the extension (.npy, .txt, .h5)")
    p.add_argument("--frame", type=int, default=-1,
                   help="which frame to take when --start holds several "
                        "(default: the last, i.e. resume)")
    args = p.parse_args()

    if args.images < 1:
        raise SystemExit("--images must be at least 1")

    cfg = bd_csa.Config.from_run_txt(args.run)
    if args.legacy:
        cfg.physics.mobility_update_interval = 0
        cfg.physics.smooth_mobility = False
        cfg.physics.enable_divD_drift = False

    x0 = load_configuration(args.start, cfg.np, cfg.a, index=args.frame)
    if args.save_initial:
        save_configuration(x0, args.save_initial, cfg.a)
        print(f"initial    -> {args.save_initial}")

    if args.device == "cuda" and not bd_csa.cuda_available():
        raise SystemExit("--device cuda requested but no CUDA device is available")

    sim = bd_csa.Simulator(cfg, args.table, 1, args.device)
    sim.reset(x0)

    total_steps = (args.images - 1) * args.steps_per_frame
    print(f"start      {args.start}")
    print(f"physics    {'legacy' if args.legacy else 'corrected (default)'}"
          f"   device {args.device}   lambda {args.lam}")
    print(f"schedule   {args.images} images, {args.steps_per_frame:,} steps apart"
          f"  ->  {total_steps:,} steps = {total_steps * cfg.dt / 1e3:.1f} s simulated")
    print(f"output     {args.out}/\n")

    t0 = time.time()
    # snapshot_series renders n_frames+1 images (frame 0 is the initial state),
    # so ask for one fewer than the number of images requested.
    out = visualize.snapshot_series(
        sim, lam=args.lam,
        n_frames=args.images - 1,
        steps_per_frame=args.steps_per_frame,
        seed=args.seed,
        out_dir=args.out,
        dashboard=args.dashboard,
        dpi=args.dpi,
        box_a=args.box,
        annotation=None if args.annotation == "none" else args.annotation,
        run_txt_path=args.run,
    )
    elapsed = time.time() - t0

    h = out["history"]
    print(f"\nwrote {len(out['frames'])} images in {elapsed:.1f} s"
          f" ({elapsed / max(len(out['frames']), 1):.2f} s/frame)")
    print(f"  psi6  {h['psi6'][0]:.4f} -> {h['psi6'][-1]:.4f}")
    print(f"  C6    {h['c6'][0]:.3f} -> {h['c6'][-1]:.3f}")
    print(f"  R_g   {h['rg'][0]:.1f} -> {h['rg'][-1]:.1f} nm")
    print(f"  RC    {h['rc'][0]:.4f} -> {h['rc'][-1]:.4f}")

    if args.save_trajectory:
        traj = out["trajectory"]
        path = traj.save(args.save_trajectory)
        size_mb = os.path.getsize(path) / 1e6
        print(f"\ntrajectory -> {path}  ({size_mb:.2f} MB, {len(traj)} frames)")
        print(f"  resume with: --start {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
