#!/usr/bin/env python3
"""Validation tier 5: statistical trajectory comparison against the Fortran.

Bit-exact agreement is impossible and not a goal -- the port replaces the
legacy `ran2`/`gasdev` generator (which had a real seeding defect, see
documentation/07-porting-notes.md 7.1), so the random stream differs from the
first draw. The contract is therefore statistical: over an ensemble of seeds,
the distributions of the observables that define the RL problem must agree.

Two comparisons are run, and the distinction matters:

  legacy mode   bd_csa with frozen mobility, nearest-bin lookup and no div.D
                drift -- i.e. the legacy physics. Any difference here is
                attributable to the RNG alone. THIS is the correctness check.

  fixed mode    bd_csa defaults (mobility every step, C1 interpolation, Ito
                drift). Difference from the Fortran here is RNG *plus* the
                deliberate physics corrections, so it quantifies how much those
                corrections actually move the observables.

Uses a two-sample Kolmogorov-Smirnov test. With ~20 samples per arm the test
only detects gross distributional shifts; that is the appropriate power here,
since the aim is to catch a broken port, not to resolve a 1% bias.

Usage:  tests/compare_trajectories.py [n_seeds] [n_steps]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ORACLE = REPO / "build" / "oracle" / "bdpd"
PORT = REPO / "build" / "cmake" / "bdpd"


def ks_2samp(a, b):
    """Two-sample KS statistic and p-value, without a scipy dependency."""
    a, b = sorted(a), sorted(b)
    na, nb = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / na - j / nb))
    en = (na * nb / (na + nb)) ** 0.5
    lam = (en + 0.12 + 0.11 / en) * d
    # Kolmogorov distribution tail
    p = 2.0 * sum((-1) ** (k - 1) * pow(2.718281828459045, -2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return d, max(0.0, min(1.0, p))


def make_rundir(tmp, n_steps):
    """A working directory with run.txt shortened to n_steps."""
    d = Path(tempfile.mkdtemp(dir=tmp))
    for f in ("start.txt", "2dtabledssnp300.txt"):
        shutil.copy(DATA / f, d / f)
    lines = (DATA / "run.txt").read_text().splitlines()
    lines[3] = str(n_steps)   # nstep
    lines[5] = str(n_steps)   # iprint -> emit at the final step
    (d / "run.txt").write_text("\n".join(lines) + "\n")
    return d


def run(exe, d, lam, seed, extra=()):
    subprocess.run([str(exe), str(lam), str(seed), *extra], cwd=d,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Final row of op1.txt: time, C6, R_g, psi6, RC
    row = (d / "op1.txt").read_text().strip().splitlines()[-1].split()
    return {"c6": float(row[1]), "rg": float(row[2]), "psi6": float(row[3])}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    lam = 30.0

    for exe, what in ((ORACLE, "Fortran oracle"), (PORT, "bd_csa bdpd")):
        if not exe.exists():
            print(f"[SKIP] {what} not built at {exe}")
            return 77

    print(f"tier 5: {n_seeds} seeds x {n_steps} steps at lambda={lam}")
    print(f"        ({n_steps * 1e-4:.1f} s simulated per episode)\n")

    with tempfile.TemporaryDirectory() as tmp:
        arms = {"fortran": [], "legacy": [], "fixed": []}
        for k in range(n_seeds):
            # Negative seeds so the legacy ran2 actually initialises its shuffle
            # table -- this gives the Fortran its best case rather than
            # exercising the 7.1 defect.
            seed = -(k + 1)
            d = make_rundir(tmp, n_steps)
            arms["fortran"].append(run(ORACLE, d, lam, seed))
            arms["legacy"].append(run(PORT, d, lam, seed, ("--legacy",)))
            arms["fixed"].append(run(PORT, d, lam, seed))
            print(f"  seed {seed:>4}  "
                  f"fortran psi6={arms['fortran'][-1]['psi6']:.4f}  "
                  f"legacy psi6={arms['legacy'][-1]['psi6']:.4f}  "
                  f"fixed psi6={arms['fixed'][-1]['psi6']:.4f}")

    print()
    failures = 0
    for arm in ("legacy", "fixed"):
        print(f"-- bd_csa ({arm} physics) vs Fortran --")
        for key, label in (("psi6", "psi6"), ("c6", "C6"), ("rg", "R_g (nm)")):
            fa = [r[key] for r in arms["fortran"]]
            fb = [r[key] for r in arms[arm]]
            ma, mb = sum(fa) / len(fa), sum(fb) / len(fb)
            d, p = ks_2samp(fa, fb)
            # p > 0.05: the ensembles are indistinguishable at 5%.
            ok = p > 0.05
            flag = "ok  " if ok else "DIFF"
            if arm == "legacy" and not ok:
                failures += 1
            print(f"   [{flag}] {label:9s} mean {ma:10.4f} -> {mb:10.4f}   "
                  f"KS D={d:.3f} p={p:.3f}")
        print()

    if failures:
        print(f"FAILED: {failures} observable(s) differ in legacy mode, where only "
              f"the RNG should differ")
        return 1
    print("legacy-mode ensembles are statistically indistinguishable from the "
          "Fortran: the port reproduces the legacy dynamics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
