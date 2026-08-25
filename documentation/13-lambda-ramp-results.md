# 13. λ-Ramp Experiment — Results

**Run date:** 2026-08-24 · **Script:** `scripts/lambda_ramp.py` ·
**Data:** `runs/ramp.h5` (20 s holds), `runs/ramp.slow.h5` (40 s holds)

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

## 13.2 Procedure

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

## 13.4 Conclusions

1. **The state space is not a smooth continuum.** ψ₆ and C₆ show genuine
   hysteresis loops with a 2.6–4.3× separation in λ between ordering and melting,
   far outside seed noise. There is a nucleation-limited ordering transition.
2. **This argues for a growing SOM over a fixed grid.** Bistability means two
   distinct configurations can share the same λ, and a fixed grid must have its
   topology guessed in advance. A GSOM / neural gas can discover it.
3. **Part of the hysteresis is kinetic and the loops are not converged.** All
   gaps shrank with 2× holds. The experiment establishes *that* loops exist at
   accessible sweep rates, not their infinitely-slow limit.
4. **R_g's apparent loop is not evidence of bistability** — it is initial-condition
   memory plus the driven-vs-diffusive asymmetry, on top of clamped mobility.
5. **RC is unusable in the dispersed regime** — it saturates at 0 above
   R_g = 26,500 nm by construction.
6. **Orientation is unresolved** and needs more seeds.

### What this means for the representation work

* Use a **growing SOM**, per conclusion 2.
* The evaluation set in [§12.7](12-image-state-representation.md) now has a
  natural source of hard cases: **configurations at matched λ on opposite
  branches**. They have the same control input and different structure, which is
  precisely the discrimination a learned representation must demonstrate over
  (ψ₆, C₆).
* Do not use RC as an evaluation target outside its calibration range.

## 13.5 Limitations and what to run next

| limitation | fix |
|---|---|
| Holds too short for R_g; loops not converged | third rate point at 80–160 s holds, then extrapolate loop width vs rate |
| Orientation inconclusive (5 seeds) | 15–20 seeds; only the final λ point per seed is needed, so this is cheap |
| Up-branch start far from equilibrium and outside the mobility table | start from a *moderately* dispersed state inside R_g < 26,500 nm, or equilibrate at λ_min for longer |
| Low-λ region coarsely sampled where the melting transition sits | more points below λ = 1 — the down-branch crossing is at λ ≈ 0.84–0.99 |
| One initial configuration, rotated | genuinely independent initial configurations |

The single highest-value follow-up is the **third rate point**: it converts
"loops exist at these rates" into a defensible statement about whether a
thermodynamic loop survives.
