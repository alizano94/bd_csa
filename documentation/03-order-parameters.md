# 3. Order Parameters

All order parameters are computed in `conn6calc.f` (ψ₆, C₆, R_g, RC). They are the
**observation space of the RL problem**, so their exact definitions are part of
the interface contract, not an implementation detail.

## 3.0 The measurement window and synthetic noise

Before anything is computed, `conn6calc.f` copies particles into working arrays,
filtering to an "experimental" window and optionally adding Gaussian measurement
noise:

```fortran
IF (ABS(x_i) .LE. 0.5*EXPBOX(1) .AND. ABS(y_i) .LE. 0.5*EXPBOX(2)) THEN
    rx(i) = x_i + var*gasdev(idummy)
    ry(i) = y_i + var*gasdev(idummy)
    rz(i) = z_i
    NPTEMP = NPTEMP + 1
ENDIF
```

This models a finite microscope field of view and camera localization error, so
the RL agent sees the same degraded observable a real experiment would.

* `expbox = 63.415 a × 63.415 a` = the full cell → **no particle is ever excluded**
  and `NPTEMP = np = 300` in the shipped configuration.
* `var = 0` → **no noise is added**, but `gasdev` is still called twice per
  particle and the RNG stream still advances.

Both facts matter for a port: see [07-porting-notes.md](07-porting-notes.md) §7.2
for the array-indexing bug that this configuration happens to hide.

## 3.1 ψ₆ — bond-orientational order

Per-particle complex hexatic order parameter over neighbours within
`rmin = 3780 nm` (= 2.63 a; contact is 2 a):

$$ \psi_6^{(i)} \;=\; \frac{1}{N_b(i)}\sum_{j\in \mathcal N(i)} e^{\,i\,6\theta_{ij}},
\qquad \theta_{ij}=\operatorname{atan2}(y_j-y_i,\;x_j-x_i) $$

Neighbour separations here **do** use the minimum-image convention
(`conn6calc.f:56-58`), unlike the force loop. Particles with no neighbours keep
`ψ₆ = 0`.

The reported scalar is the modulus of the **global average**:

$$ \Psi_6 \;=\; \left|\frac{1}{N}\sum_{i=1}^{N}\psi_6^{(i)}\right| $$

Two details worth preserving exactly:

* The per-particle loop runs over `NPTEMP`, but the global sum runs over `np`
  (`conn6calc.f:79`). Identical here because `NPTEMP = np`.
* This is `|⟨ψ₆⟩|`, **not** `⟨|ψ₆|⟩`. It is phase-coherent — it goes to zero for a
  polycrystal with randomly oriented grains, and to 1 only for a single crystal.
  This is deliberate: it is the quantity that distinguishes a defect-free crystal
  from a grainy one, which is the whole point of the control problem.

Range: `[0, 1]`. `QuadControl.f` clamps it to `[0, 1]` before reporting.

## 3.2 C₆ — average number of "hexatically connected" neighbours

For each neighbour pair within `rmin`, the phase coherence of their local order
parameters is tested:

$$ \chi_{ij} \;=\; \frac{\operatorname{Re}\!\big(\psi_6^{(i)}\,\overline{\psi_6^{(j)}}\big)}
{\big|\psi_6^{(i)}\,\overline{\psi_6^{(j)}}\big|} \;=\; \cos\!\big(\arg\psi_6^{(i)} - \arg\psi_6^{(j)}\big) $$

In code:

```fortran
numer = psir(i)*psir(j) + psii(i)*psii(j)
denom = sqrt(numer**2 + (psii(i)*psir(j) - psii(j)*psir(i))**2)
testv = numer/denom
if (testv .ge. ctestv) con(i) = con(i) + 1        ! ctestv = 0.32
```

A neighbour counts as *connected* when `χ_ij ≥ 0.32`, i.e. the two local lattices
are aligned to within ≈ 71.3°. Then

$$ C_6 \;=\; \frac{1}{N}\sum_i \mathrm{con}(i) $$

Range: `[0, 6]` for a 2-D hexagonal lattice (`QuadControl.f` clamps to `[0,6]`).
`C₆ = 6` is a perfect defect-free crystal; a fluid gives `C₆ ≈ 0–2`.

`C₆` and `Ψ₆` are complementary: `C₆` is *local* (survives grain boundaries),
`Ψ₆` is *global* (destroyed by them). Together they separate "many small crystals"
from "one big crystal" — which is why both are in the RL state.

> `denom` is `|ψ_i||ψ_j|` written the long way. It is zero when either particle
> has no neighbours, producing a NaN; in practice `con(i)` is only incremented
> for pairs that *are* neighbours, so both moduli are non-zero. A port should
> still guard the division.

## 3.3 R_g — radius of gyration

Standard 2-D radius of gyration about the instantaneous centroid:

$$ R_g \;=\; \sqrt{\frac{1}{N}\sum_i \Big[(x_i-\bar x)^2 + (y_i-\bar y)^2\Big]} $$

Computed **without** minimum-imaging (`conn6calc.f:116-140`), which is correct
here because the cluster never straddles a boundary.

`R_g` is not part of the RL state, but it **is** the row index into the mobility
table (§2.4) and it feeds `RC`. Typical value for the shipped start
configuration: **21,014.5 nm** ≈ 14.6 a. *(Reproduced exactly from the dumped
coordinates while writing these docs.)*

Note `caldss.f` recomputes its own centroid in **single precision** (the variables
`xmean`, `ymean`, `disttemp` are undeclared and default to `REAL*4`), while
`conn6calc.f` uses double. See [07-porting-notes.md](07-porting-notes.md) §7.4.

## 3.4 RC — the composite crystallinity coordinate

A single scalar blending compactness and connectivity, used for logging (column 5
of `op*.txt`). It is *not* currently read by `bd_env.py`, but it encodes the
domain expert's notion of "how crystalline is this", so it is a natural reward
shaping term.

```
RgLB = 18526.0        RgUB = 26500.0                     (nm, hard-coded)

Rg'  = min(Rg, RgUB)
Ra   = 1 - (Rg' - RgLB)/(RgUB - RgLB)          compactness, 1 = maximally compact
Crc  = C6 / 5.6                                connectivity, ~1 = crystalline
Wrc  = 1 / ( exp(18*(Crc - 0.5)) + 1 )         logistic switch, midpoint Crc = 0.5
RC   = Wrc*Ra + (1 - Wrc)*Crc
```

The logistic weight `Wrc` is a **soft switch between two regimes**:

* while the system is disordered (`Crc < 0.5`, i.e. `C₆ < 2.8`), `Wrc → 1` and
  `RC` tracks **compactness** — "first pull the cluster together";
* once it starts ordering (`Crc > 0.5`), `Wrc → 0` and `RC` tracks
  **connectivity** — "now perfect the lattice".

The steepness 18 makes the crossover sharp (10–90 % over `ΔCrc ≈ 0.24`).
The normalizer 5.6 (rather than 6) and the `R_g` bounds are empirical constants
tuned for `np = 300`; **they will not transfer to other particle counts** and
should become explicit configuration in the rewrite.

*Worked check (from a real run):* `R_g = 21014.5`, `C₆ = 4.2667` →
`Ra = 0.68793`, `Crc = 0.76191`, `Wrc = 0.008889`, `RC = 0.76125` — matching the
program's printed `0.76125`.

## 3.5 Summary

| Symbol | Code name | Range | Role |
|---|---|---|---|
| `Ψ₆` | `psi6` | [0,1] | RL observation 1 — global hexatic order |
| `C₆` | `conn6avg` | [0,6] | RL observation 2 (reported as `C₆/6`) |
| `R_g` | `rgmean` | nm | mobility-table row index; input to `RC` |
| `RC` | `RC` | ~[0,1] | composite crystallinity, logged only |
