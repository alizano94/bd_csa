# 4. Code Reference

## 4.1 Build

`makefile`, GNU make implicit-rule style:

```makefile
FC = gfortran
FFLAGS = -O
2d = caldss.o conn6calc.o ddpot.o emag.o forces.o gasdev.o main.o \
     QuadControl.o pwexct.o ran2.o readcn.o writcn.o
bdpd: $(2d) ; $(FC) $(FFLAGS) -o $@ $(2d)
```

Fixed-form Fortran 77 with Fortran 90 extensions (`allocatable`,
`COMMAND_ARGUMENT_COUNT`). Verified to build clean and unmodified with
`gfortran 16.1.0 -O` on macOS/arm64. `submit.sh` is a legacy PBS job script
(`./bdpd > out.txt`) and is not used by the RL driver.

**No file uses `implicit none`.** Every undeclared name silently becomes
`REAL*4` (or `INTEGER` for `i`–`n`). This causes real precision loss in
`caldss.f` and hides several unused variables — see
[07-porting-notes.md](07-porting-notes.md).

## 4.2 Call graph

```
SBD (main.f)
├── readcn        (readcn.f)     load initial coordinates
├── loop over cycleindex = 1..cyclenum
│   └── loop over l = 1..nstep
│       ├── [only when mod(l,iprint)==0 or t==0]
│       │   ├── CONN6CALC (conn6calc.f)  → psi6, conn6avg, rgmean, RC   [calls gasdev]
│       │   ├── caldss    (caldss.f)     → dsscalcu(1:np)
│       │   ├── writcn    (writcn.f)     append coordinates to unit 20
│       │   ├── CONN6CALC        ← DUPLICATE CALL
│       │   ├── caldss           ← DUPLICATE CALL
│       │   └── QuadControl (QuadControl.f) → out_param.json
│       ├── per-particle: gasdev (gasdev.f) → ran2 (ran2.f)
│       ├── forces (forces.f)
│       │   └── EMAG (emag.f)            2 calls per particle per step
│       └── Euler–Maruyama position update + periodic wrap
```

Compiled but **never called**: `ddpot.f` (`DDPOT`), `pwexct.f` (`pwexact`),
`writ_head` (in `writcn.f`).

## 4.3 File-by-file

### `main.f` — `PROGRAM SBD` (368 lines)

Driver. Responsibilities:

1. Read `λ` and the RNG seed from the command line (`argv[1]`, `argv[2]`).
2. Parse `run.txt` (strictly positional; see [05-io-formats.md](05-io-formats.md)).
3. Load the mobility table from `rgdsfile` into `dssarray`/`dsscount`.
4. Rescale inputs to internal units:
   ```fortran
   lambda = lambda*dpf      pfpp = pfpp*a       rcut = rcut*a
   re     = re*a            dg   = dg*a         hlev = a + a*hlev
   expbox = expbox*a        boxlenx = boxleny = dg
   fac1   = fac1/a          fac2 = fac2*sqrt((273+tempr)/a)/sqrt(dt)
   ```
5. Allocate `F, D, randisp, r0, u0, u, ud, r` (`np3`) and `nxyz(3,np)`; build the
   index map.
6. Set radii — uniform `a` for `polymono='mono'`, else read `pdfile` and reset
   `a` to the mean radius.
7. Run `cyclenum` independent episodes; each opens `<par_out><n>.txt` (unit 20)
   and `op<n>.txt` (unit 40), reloads the initial configuration, and integrates
   `nstep` steps.
8. Report CPU time as `HH:MM:SS`.

Key hard-coded value: `Fhw = 0.417` (`main.f:233`), the overlap force.

### `forces.f` — `SUBROUTINE forces(F, r, nxyz)` (379 lines)

Computes the full force array. Structure: an `i<j` double loop for pair forces,
then a single loop for the DEP force. About 60 % of the file is commented-out
code: an abandoned finite-difference force implementation (variables `Udx`,
`Uppdx`, `fddpx`, `term1dx` …, and step sizes `step2`, `step3`) kept alongside the
analytic version that is actually used. `rdepc = 16.5*a` is assigned and never
used. See [01-physical-model.md](01-physical-model.md) §1.4 for the mathematics.

### `emag.f` — `FUNCTION EMAG(RX, RY)`

Electric-field magnitude. Computes the exact four-line-charge quadrupole
solution into `EX`/`EY`, then **unconditionally overwrites the result** with the
near-axis linear form `EMAG = 4*RT/DG`, optionally scaled by the empirical
quartic correction when `ecorrectflag = 1`. The exact-solution block (lines
30–52) is therefore dead arithmetic executed twice per particle per step — a free
~15 % speedup for the port, and worth keeping as an option if the linear
approximation is ever to be relaxed.

### `caldss.f` — `SUBROUTINE caldss(...)`

Mobility lookup, described in [02-numerical-methods.md](02-numerical-methods.md)
§2.4. Note: `calcudss = 0.5*(dssmax+dssmin)` is computed and never used; the
centroid variables are implicitly single precision; its `COMMON /ORDER_PAR/`
declares four names where `main.f` declares five (`RC` missing) — harmless as
long as the trailing variable is not touched, but strictly non-conforming.

### `conn6calc.f` — `SUBROUTINE CONN6CALC(R, NXYZ, IDUMMY)`

All four order parameters. Three O(N²) passes (ψ₆, then C₆, then a cheap centroid
pass). Fully documented in [03-order-parameters.md](03-order-parameters.md).

### `QuadControl.f` — `SUBROUTINE QuadControl(l, Psi6, lambda, ControP, C6, flag, IDUMMY)`

Once a lookup-table feedback controller (`Controlpolicy(N)`, `N = 50×120`), now
gutted: the array is declared and never touched, and the routine's only remaining
job is to clamp `Psi6` to `[0,1]` and `C6` to `[0,6]` **in place** (so the clamped
values propagate back to `main.f`) and write `out_param.json`.

`ControP` is never assigned anywhere in the program yet is printed to `op*.txt`
by `main.f` — the shipped `op1.txt` shows `*****` (integer overflow on output of
an uninitialized value). This is the vestige of the RL handoff: control now comes
from Python, not from this routine.

### `readcn.f` / `writcn.f` — I/O

`readcn` reads unit 10 in one of two layouts (`check = 'n'` or `'o'`), scales x,y
by `a`, and **forces z to `hlev` for all particles** in the `'n'` path.
`writcn` appends `i, x/a, y/a, z/a` for all particles to unit 20 in format
`(i7,3f14.5)`. `writ_head` writes a metadata header and is never called.

### `gasdev.f` / `ran2.f` — RNG

Numerical Recipes; see [02-numerical-methods.md](02-numerical-methods.md) §2.5.

### `ddpot.f` — `FUNCTION DDPOT(...)` — **dead**

The dipole–dipole *potential* (as opposed to force). Uses a completely different
set of COMMON blocks (`/FIELD/`, `/BXLN/`, `/DDMM/`, `/RAD/`, `/cyclenum/`) that
nothing else declares, and treats `DPF` as an array indexed by cycle. It is a
leftover from a Monte-Carlo version of this code. Compiled, never linked to.
Useful only as a cross-check on the force derivation.

### `pwexct.f` — `SUBROUTINE pwexact(pw, n1, r, nxyz)` — **dead**

Particle–wall mobility tensor (parallel `Aw`, perpendicular `Bw`) for a sphere
near a plane. Only meaningful for the 3-D version. The call site in `main.f:299`
is commented out. Its `Aw` expression contains an obvious typo —
`(12420*v**2 + 5654*v**2 + 100*v)` sums two `v**2` terms — further evidence it is
abandoned.

## 4.4 COMMON blocks (the global state)

There are no modules; all cross-routine state is in unnamed-layout COMMON.
This is the single biggest structural obstacle to threading and to a clean C++
port: **every routine mutates shared globals.**

| Block | Contents | Declared in |
|---|---|---|
| `/num_par/` | `np` | main, forces, conn6calc, caldss, readcn, writcn, pwexct |
| `/np_3/` | `np3` | main, forces, conn6calc, readcn, writcn, pwexct |
| `/num_tot/` | `n` (unused) | main |
| `/boxlen_xy/` | `boxlenx, boxleny` | main, forces, conn6calc; **`readcn.f` declares it as a single `boxlen`** |
| `/radius/` | `a, radii(1000)` | main, forces, writcn; **`readcn.f` and `pwexct.f` declare only `a`** |
| `/delta_fn/` | `delta` (unused) | main |
| `/seed/` | `iDummy` | main |
| `/grav_pot/` | `Fgrav, Fhw, rcut, tempr` | main, forces |
| `/es_pot/` | `pfpp, pfpw, kappa` | main, forces |
| `/f_pot/` | `lambda, fcm, dg, re` | main, forces, emag |
| `/ORDER_PAR/` | `CONN6AVG, RMIN, PSI6, RGMEAN, RC` | main, conn6calc; **`caldss.f` omits `RC`** |
| `/EXP_PARA/` | `expbox(2), VAR` | main, conn6calc |
| `/calds/` | `rgdsmin, delrgdsmin, distmin, deldist` | main, caldss |
| `/calds2/` | `rgdssbin, distdssbin` | main, caldss |
| `/ecorr/` | `ecorrectflag` | main, emag |

The truncated declarations flagged above are legal Fortran (a COMMON block may be
declared shorter in one unit) but are exactly the kind of latent hazard that makes
the rewrite worthwhile. In the C++ version these become one `SimulationConfig`
struct (immutable after setup) plus one `SimulationState` struct.

## 4.5 Fortran unit numbers

| Unit | File | Purpose |
|---|---|---|
| 1 | `run.txt` | parameters (opened, read, never closed) |
| 2 | `out_param.json` | results for the Python driver |
| 10 | `par_in` = `start.txt` | initial coordinates (`cnunit1` in `readcn.f`) |
| 20 | `<par_out><n>.txt` = `bd_xyz1.txt` | trajectory dump |
| 40 | `op<n>.txt` = `op1.txt` | order-parameter time series |
| 131 | `rgdsfile` = `2dtabledssnp300.txt` | mobility table (never closed) |
| 1111 | `pdfile` | polydisperse radii (only when `polymono='poly'`) |

Unit 1 is reused for `run.txt` and never closed, and `main.f:346` closes unit 10
(already closed by `readcn`) rather than unit 40 — so `op*.txt` is flushed only at
program exit. Harmless for a process-per-step driver; a bug for an in-process
library. See [07-porting-notes.md](07-porting-notes.md).
