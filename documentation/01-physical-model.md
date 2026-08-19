# 1. Physical Model

## 1.1 Geometry and state

The system is `np = 300` spheres of radius `a`. The state is the position array
`r(1:3*np)` in **nanometres**, interleaved as `x₁ y₁ z₁ x₂ y₂ z₂ …`. The index map
is precomputed once in `main.f`:

```
nxyz(1,i) = 3(i-1)+1     nxyz(2,i) = 3(i-1)+2     nxyz(3,i) = 3(i-1)+3
```

`readcn.f` sets `z_i = hlev = a(1 + h/a) = 1530.1 nm` for **every** particle, so
all `z` are equal. Because every force term that could move `z` is proportional to
`r_ij,z` (which is therefore identically zero), the simulation is strictly 2-D.
The z slot is carried through the arrays but never changes.

A square periodic box of side `boxlenx = boxleny = dg = 90,000 nm` wraps positions
in `main.f`, but the pair loop in `forces.f` has the minimum-image convention
**commented out** (`forces.f:88-92`). This is consistent only because the cluster
occupies a region far smaller than the box (`R_g ≈ 21,000 nm`) and the DEP trap
prevents particles reaching the boundary — the wrap never fires. See
[07-porting-notes.md](07-porting-notes.md).

## 1.2 The applied electric field

The cell is a planar quadrupole electrode (four electrodes, gap `dg`). `emag.f`
first evaluates the exact four-line-charge quadrupole solution (Huang & Pethig,
*Meas. Sci. Tech.* 1991) — and then **discards it**, overwriting the result with
the small-`r` linear expansion valid near the field null:

$$ |\mathbf E|(x,y) \;=\; \frac{4\rho}{d_g}, \qquad \rho = \sqrt{x^2+y^2} $$

The corresponding dimensionless field vector, used throughout `forces.f`, is

$$ \mathbf E_i \;=\; \left(-\frac{4x_i}{d_g},\; \frac{4y_i}{d_g},\; 0\right) $$

Note `|E_i|` from these components equals `4ρ/dg`, so the vector form and the
magnitude form are consistent. The sign pattern `(−x, +y)` is the saddle
characteristic of a quadrupole: the field vanishes at the origin and grows
linearly outward.

### Empirical field correction

When `ecorrectflag = 1` (the shipped default), `emag.f` multiplies the magnitude
by a fitted quartic in `u = ρ/1000` (i.e. ρ in µm):

$$ f_{\text{corr}}(u) = 2.081\!\times\!10^{-7}u^4 - 1.539\!\times\!10^{-9}u^3 + 8.341\!\times\!10^{-5}u^2 + 1.961\!\times\!10^{-5}u + 1.028 $$

This is a calibration against the measured field of the real device. At the
origin it is 1.028; at `ρ = 30 µm` it is ≈ 1.10. **It is applied only to the
magnitude used for the DEP force, not to the vector `E_i` used for the dipole
forces** — an asymmetry the rewrite must either preserve or deliberately fix.

## 1.3 Induced dipoles

Each particle acquires an induced dipole proportional to the local field. In the
reduced units of this code the dipole moment *is* the dimensionless field vector
`E_i`, and the interaction strength is carried by the single control parameter

$$ \lambda \;=\; \frac{\pi \varepsilon_m a^3 f_{CM}^2 E_0^2}{k_B T} $$

`λ` is supplied on the command line (`argv[1]`), multiplied by `dpf` (= 1 in the
shipped `run.txt`), and is the **only quantity the RL agent controls**. The
Clausius–Mossotti factor `fcm = −0.4667` is negative (particle less polarizable
than the medium), which makes the DEP force point inward — a trap rather than a
repeller.

## 1.4 Force terms

`forces.f` builds `F(1:3np)` from four contributions. Total:

$$ \mathbf F_i = \mathbf F_i^{\text{dip}} + \mathbf F_i^{\text{DLVO}} + \mathbf F_i^{\text{DEP}} \;(+\; \mathbf F^{\text{ovlp}}) $$

All forces are expressed in the code's reduced force unit; see
[02-numerical-methods.md](02-numerical-methods.md) §2.3.

### (a) Dipole–dipole interaction — the assembly driver

The pair potential is the standard time-averaged induced-dipole potential
(Aubry & Singh, *Europhys. Lett.* 2006; Furst & Gast, *PRL* 1998):

$$ U_{ij} \;=\; -\tfrac{1}{2}\,\lambda\,k_BT \left(\frac{2a}{r}\right)^{3}
   \Big[\,3(\mathbf E_i\!\cdot\!\hat{\mathbf r})(\mathbf E_j\!\cdot\!\hat{\mathbf r}) - \mathbf E_i\!\cdot\!\mathbf E_j\,\Big] $$

with `r_ij = r_j − r_i`, `r = |r_ij|`, `r̂ = r_ij/r`. The code names the three
invariants

```
F1 = E_i · E_j          (forces.f:218)
F2 = E_i · r̂           (forces.f:220)
F3 = E_j · r̂           (forces.f:223)
```

and defines the force prefactor

```
Fo = 1e18 · 0.75 · λ · k_B · (273 + tempr) / a          (forces.f:62)
```

**Crucially, the dipole moments are position-dependent** (`E_i` depends on `r_i`),
so `∇U` picks up terms beyond the textbook fixed-dipole result, and
**the forces on `i` and `j` are not equal and opposite**. This is physically
correct — momentum is exchanged with the external field — and it is why
`forces.f` computes two separate expressions (`felxnew` for `i`, `felxnew2` for
`j`) rather than applying Newton's third law.

**Derivation.** Write `C = −½λk_BT(2a)³` so `U = C r^{−3}(3F₂F₃ − F₁)`.

*Part A — variation through `r̂` and `r` only (fixed dipoles):*

$$ \mathbf F_i^{A} = -\,F_o\!\left(\frac{2a}{r}\right)^{4}\!
\Big[F_1\hat{\mathbf r} + \mathbf E_i F_3 + \mathbf E_j F_2 - 5F_2F_3\hat{\mathbf r}\Big] $$

which is the classical dipolar force, since `3C/r⁴ = −F_o(2a/r)⁴`.
This is the bracket appearing on `forces.f:235-240` and `:244-249`.

*Part B — variation of the particle's own dipole,* using
`∂E_i/∂x_i = (−4/d_g, 0)` and `∂E_i/∂y_i = (0, +4/d_g)`:

$$ F_{i,x}^{B} = -F_o\!\left(\frac{2a}{r}\right)^{4}\!\left[\frac{4F_3\,r_{ij,x}}{d_g} + \frac{r}{3}\frac{16x_j}{d_g^{2}}\right] $$

which is exactly the pair of extra terms in `felxnew`:

```fortran
felxnew = Fo*(2a/rijsep)**4 * ( F1*rij(1)/rijsep + Exi*F3 + Exj*F2
                                - 5*F2*F3*rij(1)/rijsep
                                + rijsep*16*r(nxyz(1,j))/dg**2/3.0     ! -(r/3)(4/dg)E_jx
                                - F3*4*(-rij(1))/dg )                  ! +4 F3 r_ij,x /dg
```

and the mirror expression for `j` (`felxnew2`) carries `x_i` in place of `x_j`
with the opposite sign, plus `F2` in place of `F3`. **These derivations were
checked term by term against the source; the code is correct.** The application
is

```fortran
F(x_i) -= felxnew        F(x_j) += felxnew2
F(y_i) -= felynew        F(y_j) += felynew2
F(z_i) -= felz           F(z_j) += felz          ! always 0 in 2-D
```

The `z` component `felz` uses only Part A (there is no `E_z` and no `z`-gradient),
and it is applied antisymmetrically. It evaluates to zero because `r_ij,z ≡ 0`.

**Range.** This term is active only for `(a_i + a_j) < r < re`, with `re = 5a`
(`run.txt`). Outside that window all `fel*` are set to zero.

### (b) Screened electrostatic (DLVO) repulsion

For `(a_i + a_j) < r < rcut` (`rcut = 5a`):

$$ F^{\text{DLVO}}(r) \;=\; 10^{18}\,\frac{k_BT\,\kappa\,B_{pp}}{a}\;
   \exp\!\left[-\frac{\kappa\,(r - a_i - a_j)}{a}\right] $$

(`forces.f:111`). `κ = κa = 143.5` is the reduced inverse Debye length, so the
decay length is `a/κ = 10.0 nm` — the repulsion is a **stiff wall of ~10 nm
thickness** at contact, not a long-range interaction. `B_pp = pfpp` is read as
`bpp/kT/a = 2.2975` and rescaled to `pfpp·a = 3296.9` in `main.f:170`.

At contact (`r = 2a`) this evaluates to **1.334** reduced force units, roughly
7 × 10⁵ times the dipole force there — the particles behave as near-hard spheres.
The nominal cutoff `rcut = 5a` is irrelevant: the exponential has already
underflowed to zero by `r = 2.1a`.

The force is applied antisymmetrically along `r̂` (`Fss` in `forces.f:273, 287-293`).

### (c) Hard-overlap force

If `r ≤ a_i + a_j`, the DLVO branch is replaced by a **constant** repulsive
magnitude `Fhw = 0.417` (set in `main.f:233`), and all dipole terms are zeroed.

Note this is *smaller* than the DLVO force at contact (1.334), so the pair force
is discontinuous and **non-monotonic** across contact. In practice this branch is
never taken: the minimum pair separation measured over a run was 2.052 a. Flagged
in [07-porting-notes.md](07-porting-notes.md).

### (d) Dielectrophoretic trap

Single-particle force from the field-intensity gradient (`forces.f:301-351`):

$$ \mathbf F_i^{\text{DEP}} \;=\; 2\times10^{18}\,\frac{k_BT\,\lambda}{f_{CM}}\;
   \nabla|\mathbf E|^2 \;\cdot\;\left(\frac{a_i}{a}\right)^{3} $$

`∇|E|²` is evaluated by **forward finite difference** with step `STEP = 1e-3` nm:

```fortran
dE2x = ( EMAG(x+STEP, y)**2 - EMAG(x, y)**2 ) / STEP
dE2y = ( EMAG(x, y+STEP)**2 - EMAG(x, y)**2 ) / STEP
```

With the uncorrected field `|E|² = 16(x²+y²)/dg²` the analytic answer is
`∇|E|² = 32(x,y)/dg²` — this is written out as a commented check on
`forces.f:314`. With `ecorrectflag = 1` the finite difference is the *only*
way to differentiate the corrected magnitude, which is why it is done numerically.

Because `fcm < 0`, `λ/fcm < 0`, and `∇|E|²` points radially outward, the DEP
force points **inward**: a trap centred on the field null. Its strength scales
linearly with the control parameter `λ`, so `λ` simultaneously sets the
dipole attraction *and* the confinement — this coupling is what makes `λ` an
effective single-knob annealing control.

The `(a_i/a)³` factor is the polydispersity weighting; with `polymono = 'mono'`
all radii equal `a` and it is 1.

### (e) Gravity / particle–wall forces

Present in the source but **entirely commented out** (`forces.f:348-376`), because
the system is 2-D. `Fgrav`, `pfpw`, `Fvdw`, and the whole `pwexct.f` routine are
dead. See [04-code-reference.md](04-code-reference.md).

## 1.5 Summary of active interactions

| Term | Range | Pairwise? | Newton's 3rd law? | Scales with λ |
|---|---|---|---|---|
| Dipole–dipole | `2a < r < 5a` | yes | **no** (position-dependent dipoles) | yes |
| DLVO repulsion | `2a < r < 5a`, effective to `2.1a` | yes | yes | no |
| Hard overlap | `r ≤ 2a` (never reached) | yes | yes | no |
| nDEP trap | single particle | no | n/a | yes |
| Brownian noise | single particle | no | n/a | no |
