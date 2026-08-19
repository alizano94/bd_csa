# 2. Numerical Methods

## 2.1 The integrator

The dynamics are overdamped Langevin (no inertia), advanced with the
**Ermak–McCammon** scheme:

$$ \mathbf r_i(t+\Delta t) \;=\; \mathbf r_i(t) \;+\; \frac{D_i}{k_BT}\,\mathbf F_i\,\Delta t \;+\; \sqrt{2D_i\Delta t}\;\boldsymbol\xi_i,
\qquad \langle \xi_\alpha \xi_\beta\rangle = \delta_{\alpha\beta} $$

The mobility is a **scalar per particle** (isotropic, no tensor, no off-diagonal
hydrodynamic coupling), and the **thermal drift term `∇·D` is omitted** even
though `D` is position-dependent. See [07-porting-notes.md](07-porting-notes.md) §7.3.

### As written in `main.f`

```fortran
do j = 1, np                                      ! per particle
  do k = 1, 3
    r0(nxyz(k,j)) = r(nxyz(k,j))
    D(nxyz(k,j))  = dsscalcu(j)                   ! scalar reduced mobility
    temp = gasdev(idummy)                         ! N(0,1)
    if (k.ne.3) then
      randisp(nxyz(k,j)) = temp * sqrt(1.0/dsscalcu(j))
    else
      randisp(nxyz(k,j)) = 0.0                    ! z is frozen
    end if
  end do
end do

call forces(F, r, nxyz)

do j = 1, np3                                     ! per degree of freedom
  u0(j) = D(j) * ( F(j)*fac1 + randisp(j)*fac2 )
  r(j)  = r0(j) + u0(j)*dt
  ... periodic wrap in x and y ...
end do
```

The apparently odd `randisp = ξ·√(1/D)` followed by multiplication by `D`
collapses to `√D · ξ`, giving the correct `√(2DΔt)` scaling:

$$ \Delta r^{\text{rand}} = D\cdot\xi\sqrt{1/D}\cdot \texttt{fac2}\cdot\Delta t = \sqrt{D}\,\xi\,\texttt{fac2}\,\Delta t $$

and since `fac2` carries a factor `1/√Δt` (see below), this is `∝ √(D Δt)`.

**A single `gasdev` call is consumed for the z component every step even though
the result is discarded.** This matters for stream-for-stream reproducibility of
a port; see [07-porting-notes.md](07-porting-notes.md) §7.1.

## 2.2 Unit system

| Quantity | Unit | Note |
|---|---|---|
| Length (internal) | nm | `start.txt` is in units of `a` and is scaled by `a` on read |
| Length (files) | multiples of `a` | `readcn.f` multiplies by `a`, `writcn.f` divides by `a` |
| Time | ms | `dt = 0.1`, `t` printed as `t/1000` (seconds) |
| Temperature | °C in `run.txt`, converted as `273 + tempr` | note: 273, not 273.15 |
| Mobility `D̂` | dimensionless, relative to bulk `D₀` | 0.10 – 0.40 |
| Force | reduced: `10¹⁸ · k_BT / a[nm]` ≈ 2.819×10⁻⁶ | see §2.3 |

## 2.3 The calibration constants `fac1` and `fac2`

These are the least self-documenting part of the program and the most important to
get right in a port. They are read from `run.txt` and transformed in `main.f`:

```fortran
fac1 = fac1 / a                                        ! main.f:223
fac2 = fac2 * sqrt((273+tempr)/a)                      ! main.f:224
fac2 = fac2 / sqrt(dt)                                 ! main.f:225
```

Shipped input values: `fac1 = 5.9582E+07`, `fac2 = 40.5622`.

> Both lines in `run.txt` contain **two** numbers
> (`5.9582E+07  4.0809E+04` and `40.5622  18.3254`). Fortran list-directed input
> reads one variable, so **only the first is used**; the second is a leftover for
> a different particle size.

### What they mean

Matching the code's update against the Ermak–McCammon form shows

$$ \texttt{fac1}_{\text{used}} = \frac{D_0}{k_BT}\ \text{in code units},\qquad
   \texttt{fac2}_{\text{used}} = \sqrt{2D_0}\Big/\sqrt{\Delta t}\ \text{in code units} $$

**Verified numerically.** With `a = 1435 nm`, `T = 293 K`, `dt = 0.1 ms`:

*Deterministic step* — code gives
`D̂ · (fac1/a) · Fo_prefactor · dt = D̂ · 41520.6 · 2.81905×10⁻⁶ · 0.1 = 0.011706 D̂` nm
per unit reduced force. The Stokes–Einstein prediction
`D̂ · D₀/(k_BT) · (10¹⁸k_BT/a) · dt = D̂ · D₀·10¹⁸/a · dt` equals **0.011709 D̂** nm
for `D₀ = k_BT/(6πηa)` with **η = 0.890 mPa·s**. Agreement: 0.03 %.

*Random step* — code gives `√D̂ · fac2_used · dt = 5.796 √D̂` nm, with
`fac2_used = 40.5622·√(293/1435)/√0.1 = 57.96`. Prediction
`√(2 D̂ D₀ Δt) = √(2·1.6803×10⁵·10⁻⁴) √D̂ = **5.797 √D̂**` nm. Agreement: 0.02 %.

Both constants are therefore consistent with **one** physical assumption:

$$ D_0 = \frac{k_BT}{6\pi\eta a} = 1.680\times10^{-13}\ \mathrm{m^2/s} = 1.680\times10^{5}\ \mathrm{nm^2/s},
\qquad \eta = 0.890\ \mathrm{mPa\,s} $$

Note `η = 0.890 mPa·s` is the viscosity of water at **25 °C**, whereas `run.txt`
specifies `tempr = 20 °C` (used for `k_BT`). This inconsistency is baked into the
calibration constants; the rewrite should either reproduce it exactly or make
`η` an explicit input.

### Recommended treatment in the port

Do **not** carry `fac1`/`fac2` forward as opaque magic numbers. Compute them:

```cpp
const double kT  = kB * (273.0 + tempr_C);            // J
const double D0  = kT / (6.0*M_PI*eta*a_m);           // m^2/s
const double D0_ = D0 * 1e18 / 1e3;                   // nm^2/ms

// deterministic:  dr = D_hat * fac1_used * F_code * dt_ms
const double fac1_used = D0 / kT * 1e-3;              // = 41520.6 for the shipped config
// random:         dr = sqrt(D_hat) * fac2_used * dt_ms
const double fac2_used = std::sqrt(2.0 * D0_ / dt_ms);// = 57.96 for the shipped config
```
and validate against the legacy values (`0.011709` and `5.797` for the shipped
configuration) with a regression test.

### Step-size sanity

Per-step RMS Brownian displacement at maximum mobility:
`√(2·0.4·D₀·Δt) = 3.67 nm`. The DLVO decay length is 10 nm. The integrator is
therefore only marginally resolved at contact — reducing `dt` changes results.
Any GPU port that changes `dt` for throughput reasons will change the physics.

## 2.4 Position-dependent mobility — `caldss.f`

Instead of solving hydrodynamics, the code looks the reduced mobility up in a
precomputed 2-D table (`2dtabledssnp300.txt`, from earlier Stokesian-dynamics
work) indexed by:

* **row** — the cluster's radius of gyration `R_g` (a *global* property);
* **column** — that particle's radial distance from the cluster centroid.

```fortran
rgbinindex   = int((rgmean  - rgdsmin) / delrgdsmin) + 1     ! rgdsmin=26500, delrgdsmin=-250
distbinindex = int((disttemp - distmin) / deldist  ) + 1     ! distmin=0,   deldist=1435
```

Note `delrgdsmin` is **negative**: row 1 corresponds to the *largest* `R_g`
(26,500 nm) and row 30 to the smallest (19,125 nm), stepping down by 250 nm.
Columns are 50 bins of one particle radius each, out to 71 µm.

Resolution rules:

| Condition | Result |
|---|---|
| `rgbinindex ≥ rgdssbin` (30) — cluster more compact than the table | `D̂ = dssmin = 0.10` |
| in range, `dsscount ≥ 1` (bin has samples) | `D̂ = dssarray(row,col)` |
| in range, `dsscount = 0` (empty bin) | `D̂ = dssmax = 0.40` |
| `distbinindex` out of range | `D̂ = dssmax = 0.40` |
| `rgbinindex ≤ 0` | clamped to 1 |

Physically: a **compact cluster is hydrodynamically hindered** (`D̂ → 0.1`), a
**dispersed one moves nearly freely** (`D̂ → 0.4`). It is a mean-field stand-in for
many-body hydrodynamic interactions — cheap, and crucially for the GPU port,
**embarrassingly parallel** (each particle needs only `R_g` and its own radius).

### Critical: `caldss` is called only on output steps

`caldss` (and `CONN6CALC`, which supplies `rgmean`) run inside
`if (mod(l,iprint).eq.0 .or. t.eq.0)`. With the shipped `iprint = nstep = 10⁶`
this fires at `l = 1` and `l = 10⁶` only. **The mobility array `dsscalcu` is
therefore computed once from the initial configuration and held fixed for the
entire 100 s episode.** The rewrite must decide deliberately whether to keep this
(cheap, and what the trained RL policies were trained against) or to update `D̂`
every step (physically better, changes behaviour).

## 2.5 Random numbers

Two Numerical Recipes routines, both with `SAVE`d internal state:

* **`ran2.f`** — L'Ecuyer two-stream combined LCG with a 32-entry Bays–Durham
  shuffle. Period ≈ 2.3×10¹⁸. Returns uniform `(0,1)`.
* **`gasdev.f`** — Marsaglia polar (rejection) Box–Muller. Generates deviates in
  pairs and caches the second in a `SAVE`d `gset`, so **consecutive calls are
  stateful and the number of `ran2` draws per `gasdev` call is variable**
  (rejection loop).

Both are strictly serial and stateful. They cannot be used on a GPU; see
[07-porting-notes.md](07-porting-notes.md) §7.1 for the seeding defect and the
recommended replacement (counter-based Philox, one stream per particle).

## 2.6 Cost model (for the CUDA port)

Measured: **283 µs/step** for 300 particles on one Apple M-series core;
44,850 pairs/step → **6.3 ns per pair**. Breakdown by algorithmic weight:

| Work | Complexity | Per step (np=300) | GPU strategy |
|---|---|---|---|
| Pair forces (dipole + DLVO) | O(N²) | 44,850 pairs | tile/shared-memory N-body kernel; cutoff `5a` makes a cell list worthwhile for large N |
| DEP force (2 × `EMAG`, 2 fin. diff.) | O(N) | 300 | fuse into the same kernel |
| Gaussian deviates | O(N) | 600 used, 900 drawn | cuRAND Philox, per-particle offset |
| Position update + wrap | O(N) | 300 | trivial |
| `CONN6CALC` (ψ₆, C₆) | O(N²) | only on output steps | separate kernel, two O(N²) passes |
| `caldss` | O(N) | only on output steps | trivial after an `R_g` reduction |

At `np = 300` the O(N²) pair kernel is small enough that a single GPU is
launch-latency-bound. The real win for RL is **batching many independent
episodes** (different `λ`, different seeds) into one kernel launch — an
"ensemble of environments" layout — rather than accelerating one 300-particle
system. This should drive the C++ data-structure design: store
`[n_env][np][3]`, not `[np][3]`.
