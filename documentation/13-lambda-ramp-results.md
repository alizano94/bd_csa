# 13. λ-Ramp Experiment — Results

**Script:** `scripts/lambda_ramp.py`

| run | date | data | starts |
|---|---|---|---|
| **A** exploratory | 2026-08-24 | `runs/ramp.h5`, `runs/ramp.slow.h5` (40 s rate control) | 5 seeds, one dispersed configuration, rotated |
| **B** definitive | 2026-08-25 | `runs/ramp20.h5` | 20 seeds, **distinct** configurations from a filtered pool |

**Read §13.5 first if you want the conclusions.** Run A's starting state was
outside the mobility table's range and far from equilibrium, which inflated two
of its findings; run B fixes that and **overturns run A's conclusion about
R_g**. Run A is kept because its rate control (the 40 s repeat) is the only
evidence on rate dependence, and because the way its artifacts appeared is
instructive.

## 13.1 Motivation

The state-representation plan ([§12.5](12-image-state-representation.md))
depends on a question that had only been answered by intuition: **do the
assembly states form a continuous ordering spectrum, or discrete phases?**

It matters concretely. A self-organising map over a continuous spectrum can use
a fixed grid; genuine discrete phases, or bistability, argue for a growing
variant (GSOM / neural gas) that does not have to guess the topology in advance.
The prior evidence was indirect — a Siamese network had been needed because plain
CNN classification blurred similar configurations, hinting at fine-grained,
possibly discontinuous structure.

A secondary question rode along at no extra cost
([§12.4](12-image-state-representation.md)): **is lattice orientation signal or
nuisance?** The quadrupole pins the lab frame, so the crystal's orientation
relative to the electrode axes may be real physics rather than a symmetry to
canonicalise away.

## 13.2 Procedure (run A; run B differs only as noted in §13.5)

λ was swept **up and back down** through the physically meaningful range,
holding at each value to let the system relax, across independent seeds.

| | |
|---|---|
| λ range | 0.2 → 20 → 0.2, **logarithmically spaced** |
| points | 44 per branch (40 log points + the 4 literature λ forced in exactly) |
| literature λ | 0.2209, 0.8744, 1.9674, 19.9373 — sampled exactly on both branches |
| hold | 200,000 steps = **20 s simulated** per λ |
| sampling | 10 readouts *within* each hold (no extra cost; equilibration check) |
| seeds | 5, each with a **randomly rotated initial configuration** |
| initial state | the fully dispersed end of the λ=0 null run (ψ₆ = 0.042, R_g = 31,386 nm) |
| device | CUDA |
| total | 87.5 M env-steps (~51 min), plus a 175 M-step rate control |

**Rate control.** The whole experiment was repeated at **40 s holds**
(`--rate-check`). This is the only way to separate *kinetic* hysteresis — you
swept faster than the system relaxes — from *thermodynamic* hysteresis. A single
ramp cannot distinguish them.

Seeds start from randomly rotated copies of the same configuration so that
initial lattice orientations are decorrelated; without this the orientation test
degenerates (see §12.4).

## 13.3 Results

### Equilibration — the holds were not long enough for R_g

| observable | 20 s holds | 40 s holds |
|---|---|---|
| ψ₆ | 0.0% of λ points still drifting | 0.0% |
| C₆ | 11.5% | 19.5% |
| **R_g** | **32.2%** | **40.2%** |

ψ₆ settles within a hold everywhere. **R_g does not**, and — importantly — the
figure got *worse* with longer holds. That is not a contradiction: the
diagnostic compares drift within a hold against seed-to-seed noise, and for a
quantity relaxing diffusively rather than exponentially, more time simply means
more accumulated drift. R_g is not approaching a plateau on these timescales.

### ψ₆ and C₆ — clear, classic hysteresis loops

Both show sigmoidal branches that saturate at both ends with a large horizontal
offset between them. Taking the λ at which ψ₆ crosses 0.5:

| | up-ramp | down-ramp | loop width |
|---|---|---|---|
| 20 s holds | λ = 3.61 | λ = 0.835 | **4.32×** in λ |
| 40 s holds | λ = 2.62 | λ = 0.989 | **2.65×** in λ |

The system orders at a much higher λ than it melts at — the signature of a
first-order-like transition with a nucleation barrier on the ordering side.

Peak ordering reached ψ₆ ≈ 0.89, C₆ ≈ 5.5 (near-perfect hexagonal, max 6).

### Rate dependence — the loops shrink, but do not close

| gap (up − down at matched λ) | 20 s | 40 s | change |
|---|---|---|---|
| ψ₆ | 0.556 | 0.507 | **−9%** |
| C₆ | 2.015 | 1.199 | **−40%** |
| R_g | 7014 nm | 5850 nm | −17% |
| loop width in λ (ψ₆) | 4.32× | 2.65× | −39% |

Doubling the hold time narrowed every loop, so **part of the hysteresis is
kinetic**. But the effect is very uneven: C₆'s gap fell 40% while ψ₆'s fell only
9%, and ψ₆'s residual gap (0.507) is still **4× the seed spread** (0.12).

Two doublings is not enough to extrapolate to the infinitely-slow limit. What
can be said: the loops are rate-dependent, and a substantial ψ₆ loop survives a
2× slower sweep.

### R_g — this is not a hysteresis loop

The R_g panel looks different from the others because it *is* different, and the
distinction matters.

```
up-branch:   31,074 nm  ──(compacts monotonically)──►  19,308 nm
down-branch: 19,308 nm  ──(expands, incompletely)───►  24,059 nm
```

Three things to notice:

1. **Neither branch saturates at low λ.** The up branch is still falling steeply
   at λ = 0.2; the down branch is still rising. ψ₆ and C₆ both flatten at both
   ends. A hysteresis loop needs two *stable* states at the same λ — here there
   are two *transient* ones.
2. **The two directions were not given comparable time.** The up-branch starting
   point came from **1000 s** of free diffusion in the λ=0 null run. Each
   down-branch point gets 20–40 s. You cannot re-disperse in 20 s what took 1000 s
   to disperse.
3. **The physics is intrinsically asymmetric.** Compaction is *driven* — the trap
   and the dipole attraction actively pull inward. Expansion at low λ is
   *diffusive*, spreading as √t with almost no force behind it. The two
   directions have fundamentally different timescales, so an offset is expected
   even at equilibrium-ish sweep rates.

So the R_g gap is dominated by initial-condition memory and the driven/diffusive
asymmetry, **not** by bistability. It is the least trustworthy of the four
panels.

**One more caveat specific to R_g:** the mobility table covers R_g ∈
[19,000, 26,500] nm. At 10 of 87 points (6 of 87 in the slow run) R_g exceeded
the top of the table, where the bin index is clamped and mobility is held at the
table edge. The early up-ramp therefore runs on extrapolated mobility. This is
the same clamping the legacy code did, not new behaviour — but those points are
not on firm ground.

### RC — the shape is real, the low-λ flat is an artifact

RC tracks ψ₆/C₆ closely and shows the same loop. But the up-branch is pinned at
**exactly 0** for λ ≲ 0.5, which is not physics — it is the RC formula clamping.

RC blends compactness and connectivity:
`Ra = 1 − (min(R_g, 26500) − 18526)/7974`. Once R_g exceeds
**RgUB = 26,500 nm**, `Ra` clamps to 0, and in the disordered regime the
logistic weight `Wrc ≈ 1`, so RC ≈ 0 regardless of what the cluster is doing.

Your instinct that more points at low λ would sharpen RC is right about the
sampling but won't fix this: **RC is functionally blind above R_g = 26,500 nm**.
Its constants were tuned for compact 300-particle clusters and do not extend to
the dispersed regime. For the representation work, RC should be treated as
informative only once the cluster is inside its calibration range.

### Lattice orientation — inconclusive, leaning nuisance

Circular concentration across seeds, averaged over λ: **R = 0.633**, against
**0.447** expected by chance for 5 seeds. That is 1.42× chance — below the 1.5×
threshold the script uses to call it uniform, so it reports "nuisance", but it is
genuinely borderline rather than settled.

The pooled angle histogram is not obviously uniform either: 68% of samples fall
in 20–40°, bracketing the 30° that the electrode axes map to under the 60°
hexatic period. That is suggestive of alignment, but pooled samples are
temporally correlated and 5 seeds is thin.

**Verdict: not resolved.** Rerunning with 15–20 seeds would settle it, and the
answer changes §12.4: if orientation is pinned by the field it is signal and must
not be canonicalised or augmented away.

## 13.4 Conclusions (run A — see §13.6 for the revised set)

1. **The state space is not a smooth continuum.** ψ₆ and C₆ show genuine
   hysteresis loops with a 2.6–4.3× separation in λ between ordering and melting,
   far outside seed noise. There is a nucleation-limited ordering transition.
2. **This argues for a growing SOM over a fixed grid.** Bistability means two
   distinct configurations can share the same λ, and a fixed grid must have its
   topology guessed in advance. A GSOM / neural gas can discover it.
3. **Part of the hysteresis is kinetic and the loops are not converged.** All
   gaps shrank with 2× holds. The experiment establishes *that* loops exist at
   accessible sweep rates, not their infinitely-slow limit.
4. ~~**R_g's apparent loop is not evidence of bistability**~~ — **SUPERSEDED by §13.6.4.** True of run A specifically; run B, started inside the mobility
   table, shows a genuine closed loop.
5. **RC is unusable in the dispersed regime** — it saturates at 0 above
   R_g = 26,500 nm by construction.
6. ~~**Orientation is unresolved**~~ — **RESOLVED in §13.6.6:** it is a nuisance
   variable (R = 0.180 vs 0.224 by chance, 20 seeds).

### What this means for the representation work

* Use a **growing SOM**, per conclusion 2.
* The evaluation set in [§12.7](12-image-state-representation.md) now has a
  natural source of hard cases: **configurations at matched λ on opposite
  branches**. They have the same control input and different structure, which is
  precisely the discrimination a learned representation must demonstrate over
  (ψ₆, C₆).
* Do not use RC as an evaluation target outside its calibration range.

## 13.5 Run B — 20 seeds, pooled starting configurations

### What changed

Two of run A's limitations were addressed at once:

* **Starting states inside the mobility table.** Configurations were drawn from
  run A's own saved positions, filtered to R_g ∈ [22,000, 26,500] nm and
  ψ₆ < 0.15 — dispersed enough to watch ordering happen, but inside the
  calibrated range. Run A started at R_g = 31,386 nm, above the top of the
  table, so its first 10 points ran on clamped, extrapolated mobility.
* **Genuinely distinct starts.** Each of the 20 seeds got a *different*
  configuration (69 candidates qualified), not 20 rotations of one. Provenance
  for every seed is recorded in the file.

Everything else was identical: λ = 0.2 → 20 → 0.2, log-spaced, 44 points per
branch, 20 s holds, 10 intra-hold samples, random initial rotations.
350 M env-steps, ~3.4 h on CUDA.

### Equilibration — now clean

| still drifting | run A (5 seeds) | **run B (20 seeds)** |
|---|---|---|
| ψ₆ | 0.0% | 0.0% |
| C₆ | 11.5% | **2.3%** |
| R_g | **32.2%** | **0.0%** |

R_g went from the worst-behaved observable to fully settled. This is the
starting state, not the seed count: run A was relaxing out of a far-from-
equilibrium, out-of-table configuration for the whole up-ramp.

### Correction: R_g **does** show a genuine hysteresis loop

Run A concluded that R_g's gap was "not evidence of bistability" — initial-
condition memory plus the driven-vs-diffusive asymmetry. **That conclusion was
right about run A and wrong as a general statement.**

| | run A | run B |
|---|---|---|
| up-branch start | 31,074 nm | 24,567 nm |
| down-branch end | 24,059 nm | 24,077 nm |
| **branches meet at low λ?** | **no** (7,015 nm apart) | **yes** (490 nm apart) |
| max gap | 7,014 nm | **2,056 nm** |
| seed spread | 113 nm | 229 nm |
| gap / spread | 62× | **9×** |
| points above the mobility table | 10/87 | **0/87** |

In run B the two branches **close at both ends** — meeting near 24,000 nm at low
λ and 19,300 nm at high λ — and separate in between. That is the topology of a
real hysteresis loop, and it is what run A could not show because its up-branch
never returned to its starting point.

So R_g hysteresis is real, but run A overstated it by ~3.4×: roughly 70% of that
7,014 nm gap was initial-condition memory, ~30% is physical.

### Correction: RC is fine once R_g stays in range

Run A found RC pinned at exactly 0 for λ ≲ 0.5 and attributed it to the formula
clamping above RgUB = 26,500 nm. Run B confirms the diagnosis by removing the
cause: **0 of 44 up-branch points are pinned**, against 9 of 44 in run A, and RC
is informative across the whole sweep.

RC is not broken — it is calibrated for compact clusters and must not be used
above R_g = 26,500 nm.

### Resolved: lattice orientation is a **nuisance** variable

This was the primary reason for 20 seeds, and it is now settled.

| | run A (5 seeds) | run B (20 seeds) |
|---|---|---|
| concentration R | 0.633 | **0.180** |
| chance level (1/√n) | 0.447 | **0.224** |
| R / chance | 1.42 (borderline) | **0.80** |

R falls **below** the chance level, which is exactly what a uniform distribution
does. Final lattice angles across the 20 seeds span the full period —
5.6°, 7.1°, 9.5°, 11.1°, … 49.9°, 50.5°, 51.0° — with no clustering near the
0°/30° that the electrode axes map to. Run A's apparent concentration in 20–40°
was a small-sample artifact.

**The quadrupole does not pin the hexatic director.** For
[§12.4](12-image-state-representation.md) this means orientation carries no
state information and should not be allowed to consume encoder capacity.

The practical recommendation from §12.4 is unchanged and now has evidence behind
it: since orientation is uniformly distributed *in the data*, a large training
set automatically contains all orientations, and the encoder can learn
approximate invariance without any explicit canonicalisation — which is
fortunate, because rotation canonicalisation remains numerically unusable
(clusters are 1–3% anisotropic) and continuous rotation is still not an exact
symmetry.

### Hysteresis in ψ₆ and C₆ — confirmed, slightly narrower

| | run A | run B |
|---|---|---|
| ψ₆ = 0.5 crossing, up | λ = 3.61 | λ = 3.57 |
| ψ₆ = 0.5 crossing, down | λ = 0.835 | λ = 0.928 |
| **loop width** | **4.32×** | **3.85×** |
| ψ₆ max gap / seed spread | 0.556 / 0.112 = 5.0× | 0.522 / 0.125 = **4.2×** |
| C₆ max gap / seed spread | 2.015 / 0.098 = 21× | 1.621 / 0.136 = **12×** |

The loop narrowed modestly with better starts but remains far outside noise
-- 4x the seed spread for psi6, 12x for C6. The up-branch ψ₆ shows a clear plateau at
≈0.05 until λ ≈ 2 followed by a rapid rise — the signature of a nucleation
barrier.

### A small feature worth noting

On the up-ramp, R_g *rises* slightly before it falls — 24,567 → 24,880 nm over
λ = 0.2 → 0.3 (+2,112 nm/decade) — before the field takes hold and compaction
begins. At the lowest λ the trap is too weak to hold the cluster against
diffusion. It is small but reproducible across 20 seeds and visible in the
figure.

## 13.6 Revised conclusions

Superseding §13.4 where they conflict:

1. **The state space is not a smooth continuum.** ψ₆ and C₆ show hysteresis
   loops with a 3.9× separation in λ between ordering and melting, at 4–12× the
   seed spread. Nucleation-limited ordering transition. *(unchanged)*
2. **Use a growing SOM, not a fixed grid.** *(unchanged, and now stronger —
   three of four order parameters show closed loops)*
3. **Part of the hysteresis is kinetic.** Doubling holds narrowed every loop in
   run A. Run B did not repeat the rate control, so the infinitely-slow limit
   remains unmeasured. *(unchanged)*
4. **R_g shows a genuine loop.** ✱ **Corrects §13.4 conclusion 4.** With a
   starting state inside the mobility table the branches close at both ends and
   a 2,056 nm gap survives at 9× the seed spread.
5. **RC is usable inside its calibration range.** ✱ **Refines §13.4 conclusion
   5.** It is blind above R_g = 26,500 nm by construction, not broken.
6. **Lattice orientation is a nuisance variable.** ✱ **Resolves §13.4
   conclusion 6.** R = 0.180 against 0.224 by chance across 20 seeds.

### What this means for the representation work

* **Growing SOM**, per conclusion 2.
* **Matched-λ opposite-branch pairs** are the hard-case evaluation set for
  [§12.7](12-image-state-representation.md): same control input, different
  structure — exactly the discrimination a learned representation must
  demonstrate over (ψ₆, C₆). Run B's data contains these at every λ between
  0.3 and 10.
* **Do not spend model capacity on orientation** (conclusion 6), and do not use
  RC outside R_g < 26,500 nm (conclusion 5).
* **Dataset generation should start from pooled, in-range configurations**, as
  run B did — the R0/R1 generator in §12.6 should reuse `--start-pool`'s
  filtering rather than a single seed configuration.

## 13.7 Limitations and what to run next

| limitation | status |
|---|---|
| ~~Orientation inconclusive (5 seeds)~~ | **done** — run B, 20 seeds |
| ~~Up-branch start outside the mobility table~~ | **done** — `--start-pool` filters to R_g < 26,500 nm |
| ~~One initial configuration, rotated~~ | **done** — 20 distinct pooled starts |
| ~~Holds too short for R_g~~ | **done** — 0% drift in run B |
| **Loops not converged in sweep rate** | **open** — run B has no rate control; only run A's 20 s/40 s pair exists |
| Low-λ region coarsely sampled where melting occurs | open — down-branch crossing sits at λ ≈ 0.93 |
| Starts inherit run A's history | open — not an equilibrium ensemble, so seed spread is not a clean physical error bar |

The single highest-value follow-up remains the **third rate point** — run B's
configuration at 80 s holds. It is the one thing standing between "loops exist
at accessible sweep rates" and a defensible claim about a thermodynamic loop.
At 20 seeds that is ~14 h; at 8 seeds, ~5.5 h, which is enough to measure loop
width since that statistic converged well before 20 seeds (4.32× → 3.85×).
