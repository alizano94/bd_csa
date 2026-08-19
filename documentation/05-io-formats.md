# 5. Input / Output Formats

Everything crosses the process boundary as files in the **current working
directory**. `bd_env.py` therefore `chdir`s into `fortran_bd` before every call.
Reproducing or replacing this contract is the core of the Python-binding work.

## 5.1 Command line

```
./bdpd <lambda> <seed>
```

| Arg | Type | Meaning |
|---|---|---|
| `argv[1]` | double | `λ`, the field-strength control parameter. Multiplied by `dpf` from `run.txt`. **The RL action.** |
| `argv[2]` | integer | RNG seed / state (`iDummy`), passed straight to `ran2`. |

Both are read with `READ(args(i),*)` into a `CHARACTER(len=12)` buffer — values
longer than 12 characters are silently truncated. Note `seed` values chained from
`out_param.json` can reach 10 digits, which fits, but only just.

There is **no argument validation**: a missing argument aborts on the
`args(1)` access.

## 5.2 `run.txt` — the parameter file

**Strictly positional.** Every value line is preceded by a description line that
`main.f` skips with a bare `read(1,*)`. Inserting, removing, or reordering a line
silently shifts every subsequent parameter. Reproduced below with the exact read
order from `main.f:74-156`.

| run.txt line | Variable | Shipped value | Meaning / internal rescaling |
|---:|---|---|---|
| 2 | `np` | 300 | particle count |
| 4 | `nstep` | 1000000 | steps per invocation (= 100 s at `dt = 0.1 ms`) |
| 6 | `iprint` | 1000000 | output interval — also the mobility-refresh interval (§2.4) |
| 8 | `istart` | 0 | equilibration steps to ignore — **computed but unused** |
| 10 | `par_in` | `start.txt` | initial coordinates |
| 12 | `par_out` | `bd_xyz` | trajectory basename → `bd_xyz1.txt` |
| 14 | `a` | 1435.0 | particle radius (nm) |
| 16 | `tempr` | 20 | temperature (°C); used as `273 + tempr` |
| 18 | `phi` | 0.001 | area fraction — **only ever written to a header that is never written** |
| 20 | `dt` | 0.1 | time step (ms) |
| 22 | `t` | 0 | initial time — **overwritten with 0.0 at `main.f:256`** |
| 24 | `check` | `'n'` | `'n'` = new coordinate layout, `'o'` = continuation layout |
| 26 | `fac1` | 5.9582E+07 | mobility constant; → `fac1/a`. **Second number on the line is ignored** |
| 28 | `fac2` | 40.5622 | noise constant; → `fac2·√((273+T)/a)/√dt`. Second number ignored |
| 30 | `pwfactor` | 0.4079 | parallel diffusion factor — **read, never used** |
| 32 | `Fgrav` | 43.5488 | gravity — **read, only used in commented-out code** |
| 34 | `rcut` | 5.0 | DLVO cutoff in radii → `rcut·a` |
| 36 | `re` | 5.0 | dipole cutoff in radii → `re·a` |
| 38 | `kappa` | 143.5 | `κa`; Debye length = `a/κ` = 10.0 nm |
| 40 | `pfpp` | 2.2975 | `B_pp/kT/a` → `pfpp·a` |
| 42 | `pfpw` | 6628.1036 | particle–wall prefactor — **read, never used** |
| 44 | *(skipped)* | 0.8744 | a `λ` value that is **ignored** — `λ` comes from `argv[1]` |
| 46 | `fcm` | −0.4667 | Clausius–Mossotti factor; negative ⇒ nDEP trap |
| 48 | `dg` | 63.415 | electrode gap in radii → `dg·a` = 90,000 nm ≈ 91 µm |
| 50 | `hlev` | 0.0663 | levitation height → `z = a(1 + hlev)` = 1530.1 nm |
| 52 | `rmin` | 3780 | ψ₆ neighbour cutoff (nm, absolute) |
| 54 | `expbox(1:2)` | 63.415 63.415 | observation window in radii → `·a`. **Trailing numbers on the line ignored** |
| 56 | `var` | 0 | synthetic measurement noise (nm) |
| 58 | `polymono` | `mono` | `poly` ⇒ read radii from `pdfile` and redefine `a` as their mean |
| 60 | `pdfile` | `raddist.txt` | polydisperse radii (file absent — unused for `mono`) |
| 62 | *(skipped)* | 1 | a seed value that is **ignored** — seed comes from `argv[2]` |
| 64 | `RGHISTFILE, DELRG, RGMIN` | `bd_rghist.txt 50 2000` | **dead** — histogram never written |
| 66 | `PSIHISTFILE, DELPSI` | `bd_psihist.txt 0.05` | **dead** |
| 68 | `conhistfile, delcon` | `bd_conhist.txt 0.2` | **dead** |
| 70 | `cyclenum` | 1 | independent episodes per invocation |
| 72 | `rgdsfile` | `2dtabledssnp300.txt` | mobility table (read with `'(a150)'`) |
| 74 | `rgdsmin, delrgdsmin, rgdssbin` | 26500 −250 30 | `R_g` binning — **note the negative stride** |
| 76 | `distmin, deldist, distdssbin` | 0 1435 50 | radial-distance binning |
| 78 | `dssmin, dssmax` | 0.10 0.4 | mobility clamps for out-of-table lookups |
| 80 | `dpf` | 1 | `λ` multiplier |
| 82 | `ecorrectflag` | 1 | enable the empirical field correction in `emag.f` |

## 5.3 `start.txt` — initial configuration (input, unit 10)

`check = 'n'` layout — 4 whitespace-separated columns, one row per particle:

```
index    x/a           y/a           z/a
    1    -9.86383      -15.18930     1.06630
```

`readcn.f` multiplies `x` and `y` by `a` and **discards the `z` column**,
setting `z = hlev` for every particle.

`check = 'o'` layout — 5 columns (`time, index, x, y, z`), values taken as
absolute nm with no scaling. **This path is currently unusable**: the matching
5-column write in `writcn.f` is commented out, so nothing produces the format.

## 5.4 `bd_xyz1.txt` — trajectory (output, unit 20)

Appended by `writcn.f`, format `(i7,3f14.5)`, one block of `np` rows per output
step, coordinates divided by `a`:

```
      1     -12.49171     -18.99506       1.06630
```

There is **no time column and no block separator** — consumers must know `np`.
`bd_env.py` exploits this by simply taking the **last 300 lines** as the final
configuration and copying them over `start.txt` for the next epoch. With
`iprint = nstep` the file contains exactly two blocks (initial and final),
600 lines.

## 5.5 `op1.txt` — order-parameter log (output, unit 40)

`main.f:274`, format `(5f12.5,i5,f12.5)`:

| Column | Value |
|---|---|
| 1 | `t/1000` — time in **seconds** |
| 2 | `conn6avg` (C₆) |
| 3 | `rgmean` (R_g, nm) |
| 4 | `psi6` (Ψ₆) |
| 5 | `RC` |
| 6 | `ControP` — **uninitialized**; prints as `*****` when the garbage value overflows `i5` |
| 7 | `lambda` |

Example (shipped file):
```
   0.00000     1.52000 25988.13092     0.00560     0.06752*****     2.34097
  99.99990     4.28000 21015.90996     0.40508     0.76363*****     2.34097
```

Unit 40 is never explicitly closed (`main.f:346` closes unit 10 instead), so the
file is flushed only at program exit.

## 5.6 `out_param.json` — the RL result channel (output, unit 2)

Written by `QuadControl.f` on every output step; the last write wins. Produced
with list-directed `write` statements, so it is heavily padded but **valid JSON**
(verified with `json.load`):

```json
 {
       "step":         200 ,
       "lambda":  0.87439999999999996      ,
       "psi6":  0.40531116141727186      ,
       "c6":   4.2666666666666666      ,
       "seed":  1749401541
 }
```

| Key | Meaning |
|---|---|
| `step` | the loop counter `l` at the time of writing |
| `lambda` | the applied `λ` (echo of `argv[1] × dpf`) |
| `psi6` | Ψ₆, clamped to `[0,1]` |
| `c6` | C₆, clamped to `[0,6]` |
| `seed` | **the evolved `iDummy` after the run, not the input seed** |

That last row is load-bearing: `bd_env.py` reads `seed` back out and feeds it in
as `argv[2]` on the next epoch, which is how the random stream is chained across
a trajectory. See [06-rl-integration.md](06-rl-integration.md).

## 5.7 `2dtabledssnp300.txt` — mobility table (input, unit 131)

`rgdssbin × distdssbin` = 30 × 50 = 1500 rows, read in row-major order
(`i` outer, `j` inner). Six columns:

```
   i    j    Rg(nm)        dist(nm)      D_hat       count
   1    1    26375.00000   717.50000     0.28580     4
```

Columns 1–2 (indices) and 3–4 (bin centres) are read into throwaway variables —
**the binning is recomputed from `run.txt`, not taken from the file**, so the two
must agree. Column 5 is the reduced mobility `D̂`; column 6 is the number of
samples that produced it. `count = 0` marks an unsampled bin, which the lookup
replaces with `dssmax`.

The `300` in the filename refers to `np = 300`: **this table is specific to a
300-particle system** and must be regenerated for any other particle count.

## 5.8 Files present but unused

| File | Status |
|---|---|
| `raddist.txt` | referenced by `pdfile`, absent; only opened when `polymono = 'poly'` |
| `bd_rghist.txt`, `bd_psihist.txt`, `bd_conhist.txt` | filenames parsed, histograms never accumulated or written |
| `submit.sh` | legacy PBS batch script, unused by the RL driver |
| `bdpd` | prebuilt binary checked into the tree |
