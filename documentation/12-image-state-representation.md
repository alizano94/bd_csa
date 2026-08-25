# 12. Image-Based State Representation — MAE → SOM

**Status: proposal for review.** This replaces an earlier draft of this file that
I wrote before seeing the architecture discussion in
`claude.ai/code/session_018Kus7QbUnMYbkN4SoR8vD3`. That draft proposed
contrastive learning into UMAP/HDBSCAN and was wrong on essentially every
architectural choice. This version starts from what you actually landed on.

## 12.1 The architecture, as I understand it

Restating so you can correct me before anything gets built:

```
configurations ──► render ──► MAE pretraining ──► global embedding ──► SOM ──► RL state
   (bd_csa)                   (ViT, masked          (CLS or                    (DQN)
                               reconstruction)       mean-pool)
```

* **MAE self-supervised pretraining**, replacing the current CNN-features →
  HDBSCAN route, whose features were learned somewhat circularly. DINO was the
  considered alternative.
* **Tokenisation is the crux**: standard ViT linear patch embedding versus a
  **conv stem** (strided convs downsampling to patch-grid resolution). The conv
  stem restores locality bias — valuable on a few-thousand-image dataset — but
  mixes information across patch boundaries, so signal about a masked patch
  leaks in through unmasked neighbours. Manageable, not free.
* **Masking ratio** is an open lever. MAE's 75% was tuned for natural images.
* **Embedding extraction** kept separate from pretraining: MAE objective stays
  pure reconstruction; CLS-vs-mean-pool is a downstream decision.
* **SOM downstream**, not HDBSCAN — topology-preserving. Fixed grid vs a growing
  variant (GSOM / growing neural gas) to avoid guessing grid size.
* **Scope, Reading A (staged)**: encoder + SOM trained self-supervised and
  decoupled from reward; the existing DQN then trains on that fixed or
  slowly-updated representation, replacing the GCN state encoding.

Two things from that session I am treating as constraints, not questions:
**no code yet**, and the honest check that the DQN already reaches 97% — so this
must be justified as solving a real limiter, not as a publishable upgrade.

## 12.2 What bd_csa changes about this problem

The simulator work is finished and validated. What it contributes:

**Data stops being scarce.** At 0.94 µs/env-step with 4096 batched environments:

| dataset | simulation wall time |
|---|---|
| 10⁴ configurations | ~1.6 min |
| 10⁵ configurations | ~16 min |
| 10⁶ configurations | ~2.6 h |

The prior pipeline's "few-thousand-image dataset" — the thing that motivated the
conv stem's locality bias in the first place — is no longer a constraint. I
argued this weakened the case for the conv stem, since plain ViT patchify's main
weakness is data efficiency.

**Decided: keep the conv stem.** The consequence is that the leakage noted in
§12.1 becomes something to *manage*, not something the data scale lets us avoid.
Concretely, with four 3×3 stride-2 layers the receptive field is

```
L1: RF = 3    L2: RF = 3 + 2·2 = 7    L3: RF = 7 + 2·4 = 15    L4: RF = 15 + 2·8 = 31
```

so tokens sit **16 px apart but each sees 31 px** — every token has already
absorbed ~7–8 px of each neighbouring patch before masking happens. At 256×256
over ±30 a a particle is only **8.5 px** across, so a masked particle near a
patch boundary is substantially visible through its neighbours' tokens. Two
mitigations worth building in from the start:

* **Shrink the stem's receptive field** toward patch-aligned (fewer layers or
  smaller kernels) so RF approaches 16 px.
* **Test for cheating**: a conv-stem model that beats plain patchify on
  *reconstruction* but not on downstream SOM quality is the signature.

**Physics labels come free.** Every frame in an HDF5 trajectory (§10.4a) carries
ψ₆, C₆, R_g, RC, λ, seed and time. These are for *evaluation only* — never
training signal — but they make the representation measurable rather than a
matter of taste.

**Runs are reproducible.** Seeds and the full `run.txt` are embedded; `replay()`
reproduces a trajectory to 0.0 nm. A dataset can be regenerated exactly.

## 12.3 Measured: over half your patches are empty

This bears directly on the masking-ratio question, and it is measurable now
rather than after training. Rasterising the shipped 300-particle configuration
at 256×256 with 16-px patches (256 patches):

| field of view | foreground pixels | patches that are **100% empty** |
|---|---|---|
| ±30 a | 26.0% | **52.7%** |
| ±25 a | 37.4% | 35.2% |
| ±22 a | 48.2% | 18.4% |

At the natural ±30 a framing, **more than half of all patches contain no
particle at all**. Random 75% masking therefore hides ~100 patches of which ~50
are pure background, and those are reconstructed trivially by predicting zeros.
The *effective* supervision is far weaker than "75%" suggests, and the gradient
signal is dominated by patches that teach nothing.

Three ways out, and they interact with the conv-stem question:

1. **Tighten the field of view** to ±22 a — cuts empty patches to 18%. But it
   crops the frame to the cluster, discarding R_g information that is physically
   meaningful, and the cluster grows (the λ=0 run reached 21.9 a and clipped a
   45 a box).
2. **Content-aware masking** — bias masking toward occupied patches. Departs
   from MAE-as-published; makes the reconstruction task uniformly informative.
3. **Raise the mask ratio** — the blunt instrument, and the one that compounds
   with conv-stem leakage as noted in your session.

**Decided: content-aware masking (2).** Still worth measuring the empty-patch
fraction across the *whole* dataset before fixing a ratio — dispersed states are
far emptier than crystalline ones, so a single global ratio may not serve both
ends of the trajectory.

### Training images are not visualisation images

**Decided: the model sees particles and nothing else.** No axes, ticks, labels,
colourbar, annotation header, or figure margins — every one of those is a
high-contrast artifact that a masked-reconstruction objective will happily spend
capacity on, and the annotation text is *especially* toxic because it encodes
ψ₆/C₆ numerically in the pixels the model is asked to reconstruct.

This means the R0 rasteriser is a **separate output path** from
`bd_csa.visualize`, not a faster version of it:

| | `visualize.plot_configuration` | R0 training rasteriser |
|---|---|---|
| purpose | human inspection | model input |
| chrome | axes, colourbar, annotation box | none — pure raster |
| colour | per-particle ψ₆ (viridis) | occupancy only; **no ψ₆ colouring**, which would feed the hand-crafted feature back in |
| output | PNG via matplotlib, ~0.3 s | numpy array, target ~1 ms |

## 12.4 Symmetry — what is and isn't invariant

> **Correction.** An earlier version of this section asserted that "the physics
> is invariant under global translation and rotation." That is **wrong**, and the
> error mattered: it would have licensed a rotation canonicalisation that
> destroys real information.

### The exact symmetry group is C4v, not SO(2)

The DEP trap *is* isotropic — `|E|² = 16(x²+y²)/dg²` depends only on radius. But
the dipole interaction uses the field **vector** `E = (−4x/dg, +4y/dg)`, a saddle
whose axes are pinned to the electrodes.

Rotating a configuration by φ leaves the dipole energy unchanged only if
`R_φ⁻¹ M R_φ = ±M`, where `M ∝ diag(1, −1)`. Conjugating `diag(1,−1)` by a
rotation turns the quadrupole axis by **2φ**, so this holds only when
`2φ ≡ 0 (mod π)`:

```
φ ∈ {0°, 90°, 180°, 270°}
```

At 90° the field picks up a global sign, and the dipole energy is *quadratic* in
E — `3(Eᵢ·r̂)(Eⱼ·r̂) − Eᵢ·Eⱼ` — so both sign flips cancel. Reflections about the
electrode axes follow by the same argument. DLVO and the Brownian term are
isotropic and do not reduce the group further.

**The exact symmetry is C4v: 8 elements.** A cluster rotated by 30° experiences
genuinely different dipole forces — it is a different physical state, not the
same state in a different pose.

Translation is not an exact symmetry either: the trap sits at the field null, so
displacing the cluster changes the DEP force. In practice the trap holds the
centroid near the origin, so the residual drift is small and centring discards
little.

### Rotation canonicalisation would fail even if it were licensed

The standard recipe — rotate to the gyration tensor's principal axis — is
numerically ill-conditioned here, because a trapped cluster is very nearly
circular. Measured on real configurations:

| configuration | anisotropy `(λ₁−λ₂)/(λ₁+λ₂)` | major axis |
|---|---|---|
| `start.txt` | 0.019 | 80.5° |
| λ=0, t=250 s | 0.032 | 25.0° |
| λ=0, t=500 s | 0.032 | 34.4° |
| λ=0, t=750 s | 0.011 | 120.7° |
| λ=0, t=1000 s | 0.035 | 89.9° |

With λ₁ ≈ λ₂ to within 1–3% the eigenvectors are effectively degenerate, so the
"principal axis" is mostly noise: it wanders **96°** across a run in which the
cluster's shape barely changes. Canonicalising on it would inject ~96° of
spurious rotation into a smooth trajectory, surfacing in the SOM as jumps
between physically adjacent states.

The ψ₆-phase alternative fails in the complementary regime — well-defined for
ordered states, undefined as |ψ₆| → 0, which is exactly the dispersed case
(the λ=0 run ended at ψ₆ = 0.042).

### Why MAE changes the augmentation calculus

Contrastive methods (SimCLR, DINO) make invariance *the objective*: the loss
explicitly pulls together two augmented views of one sample, so augmentation
**is** the learning signal and directly buys invariance.

MAE has no such mechanism. Its loss is pixel reconstruction of masked patches;
it never compares two views. Feed it rotated images and it learns to reconstruct
rotated images — it receives no signal that a rotated copy *should* embed
identically. Under MAE, augmentation is dataset expansion, not an invariance
mechanism. (The MAE paper also notes it needs only minimal augmentation, and
that heavy augmentation hurts, because masking already supplies the
regularisation.)

**Consequence at our data scale:** augmenting is a poor substitute for
generating. With 10⁵–10⁶ configurations available (§12.2), lattice orientations
are sampled densely by the dynamics anyway, and compute is better spent on more
*diverse trajectories* than on 8 copies of each frame. The C4v group is still
worth having — as an **evaluation probe** (below), not as training data.

### What to do

| degree of freedom | treatment | why |
|---|---|---|
| translation | **canonicalise** — centre on the centroid | unambiguous, continuous, no flip discontinuity. Append `\|centroid\|` as a scalar if the drift is wanted back. |
| continuous rotation | **leave alone** | not a symmetry; canonicalising would merge distinct states |
| C4v (8 elements) | **neither** — use for evaluation | exact equivalence, but MAE cannot convert it into invariance, and data is cheap |
| lattice orientation | **probably signal, not nuisance** — see below | the quadrupole breaks isotropy |

Note the conv stem (kept, per §12.2) is translation-*equivariant* by
construction, but the ViT's position embeddings are not — so centring still does
real work.

### The premise may be wrong, and it is testable

Calling lattice orientation a nuisance variable presumes the lab frame is
arbitrary. It is not: the quadrupole distinguishes the electrode axes, so the
crystal's orientation **relative to those axes** may be genuine physics — the
lattice may preferentially align with the field.

The λ-ramp experiment (§12.5) answers this directly at no extra cost: track the
global ψ₆ *phase* against the lab frame and test whether it is uniformly
distributed or concentrates near 0°/90° (mod 60°, since ψ₆ is 6-fold and the
field is 4-fold — an incommensurate pairing that makes the question more
interesting, not less). If it concentrates, orientation is signal and must not
be canonicalised or augmented away.

### How to verify, whichever route

1. Encode the 8 C4v-equivalent copies of a configuration; the embedding spread
   should be ≈0 if the invariance took.
2. Encode small *continuous* rotations; the embedding **should move**. If it does
   not, the model has learned an invariance the physics does not have — which
   would be a silent loss of information, not a success.

## 12.5 The open physics question is now answerable

Your session ended on: *do the assembly states form a continuous ordering
spectrum, or real discrete phases?* — with the note that the Siamese network
(A2) was needed because plain CNN classification blurred similar configurations,
suggesting fine-grained, possibly discontinuous structure.

That is now an experiment rather than an intuition. Sweep λ **up and then back
down**, holding at each value long enough to relax, and ask whether
(ψ₆, C₆, R_g) move continuously or jump, and whether the two branches
coincide. Repeat across seeds. Discontinuity or a genuine hysteresis loop is
evidence of real phase structure and argues for a growing SOM; smooth,
retraceable curves make a fixed grid defensible.

### λ range and spacing

**λ ∈ [0, 20]**, per theory — an earlier version of this section said 0 to 60,
which is outside the physical range. Values used in the related literature:

```
λ = 0.2209,  0.8744,  1.9674,  19.9373
```

Note `0.8744` is also the value sitting in `run.txt` line 44 — the Fortran reads
and discards it (λ comes from argv), but it independently confirms the scale.

Those four span nearly **two decades**, so the sweep is **logarithmically
spaced** by default. Measured, for 40 points per branch:

| spacing | sample index of each literature λ |
|---|---|
| linear 0–20 | **[0, 2, 4, 39]** — three of four in the first four samples |
| log 0.2–20 | **[1, 12, 19, 39]** — evenly spread |

Linear spacing would spend ~90% of the schedule on the high-λ end where little
changes. The four literature values are additionally **forced into the schedule
exactly**, on both branches, so results are directly comparable with prior work.

λ = 0 cannot appear in a log sweep; add it with `--include-lam 0` if the
no-field limit is wanted (that is the null control already run — see the λ=0
trajectory in `runs/`).

### Starting configuration

Starting from a **dispersed** state is better than starting from `start.txt`
(ψ₆ = 0.405, already partly ordered): an up-ramp from disorder watches ordering
happen, rather than beginning halfway through it. The λ=0 null-control run ended
fully melted and is the natural choice:

```sh
--start runs/relax_lam0_1000s.h5      # takes the last frame; --frame k for others
```

**Caveat worth knowing:** that final state has R_g = 31,386 nm, while the
mobility table covers 19,000–26,500 nm. Above the top of the table the R_g bin
index is clamped, so mobility is held at the table edge until the cluster
compacts back into range. This is the same clamping the legacy code did, not new
behaviour, but it means the earliest points of the up-ramp run on an
extrapolated mobility.

### Output

`scripts/lambda_ramp.py` writes one HDF5 with `lam`, a `branch` label
(up/down), and per-seed `psi6`, `psi6_phase`, `c6`, `rg`, `rc`, `positions`,
plus the embedded `run.txt` and the initial rotation angles. `--analyze` re-reads
it without re-running, and writes **`*_hysteresis.png`** — the four order
parameters against λ with the up and down branches drawn separately and a
±1 std band across seeds, so a loop is only meaningful if it clears the band.

**This is cheap and worth doing before any encoder exists**, because it decides
the SOM design and it validates the premise of the whole pipeline.

It also answers the orientation question in §12.4 at no extra cost: record the
ψ₆ *phase* alongside its magnitude and check whether the lattice aligns with the
electrode axes.

**Delivered as a script you run.** `scripts/lambda_ramp.py`, written and
smoke-tested on a few frames by me, executed by you — see §12.9.

## 12.6 What bd_csa must build

| # | deliverable | note |
|---|---|---|
| **R0** | Fast rasteriser | matplotlib is **~0.3 s/frame** — 10⁵ images would take 8.3 h against 16 min of simulation. Direct disk-stamping into numpy should reach ~1 ms. This is a hard blocker for any ViT training set. |
| **R1** | Dataset generator | random λ schedules (what an exploring agent produces), diverse seeds, deliberately melted and over-annealed starts. HDF5 with images + physics labels. **Split held-out data by trajectory, not by frame** — frames within a run are highly correlated and a random split leaks. |
| **R2** | λ-ramp experiment | §12.5 — continuous vs discrete phases |
| **R3** | Dataset statistics | empty-patch fraction across the full state distribution, to fix the masking ratio on evidence |

R0–R3 are all simulator-side and need no deep-learning stack.

## 12.7 Evaluation, when there is a model

The physics labels make this concrete:

1. **Linear probe** latent → (ψ₆, C₆, R_g, RC). Necessary, not sufficient: if it
   can't recover the classical order parameters it has lost what the baseline has.
2. **Beyond the baseline** — the real test. Construct pairs with *matched*
   (ψ₆, C₆) but different structure: one grain boundary versus dispersed
   defects. The λ=30 run produced exactly this (ψ₆ = 0.78 with a single
   misoriented grain). If the latent can't separate them, the pipeline has no
   justification over two scalars.
3. **Symmetry leakage** — encode rotated/translated copies, measure drift.
4. **SOM topology** — do neighbouring SOM cells hold physically similar states?
5. **Against the incumbent** — the comparison that matters is not against
   nothing, but against the **GCN state representation the DQN already uses at
   97%**.

## 12.8 Decisions (answered)

1. Is §12.1 a fair statement of the architecture?
	1. The architecture seems good. At least untill the RL checkpoint which we will elaborate later. 
2. Does the 10⁵–10⁶ image scale change your view on **conv stem vs plain ViT
   patch embedding**? (§12.2)
	1. I think we should stick to the conv stem idea. I see bennefits from that. 
3. Masking: content-aware, or tighter crop, or higher ratio? (§12.3)
	1. Content aware seems the way to go. Also for model purposes the images should be preprocessesd to get rid of any image artifacts not related to the particles itself. For examples, axis, annotations, etc.
4. Should I run the **λ-ramp experiment** (§12.5)? It is a few GPU-hours and
   answers the fixed-vs-growing SOM question with data.
	1. I would likt to run this but not to loose control about this thread so we could create a python script and leave the running up to me. 
5. Scope confirmation: build R0–R3 only, and leave MAE/SOM to you?
	1. We shoud go step by step together along the way. Unless you are running small test to new code I would like to be the one handling the code execution.

## 12.9 Working agreement

Agreed with you, and recorded here because it changes how these milestones get
delivered:

* **We go step by step together.** No multi-milestone sprints; each piece gets
  reviewed before the next starts.
* **You run the code.** I write it and may run *small tests of new code* — a
  couple of frames, a shape check, a timing probe — but anything that is a real
  experiment or a long run is delivered as a script for you to execute.
  `scripts/lambda_ramp.py` (§12.5) is the first instance.
* **No long background jobs.** Everything I start should finish inside a short
  test, or be handed over.
* The MAE/SOM side stays yours to implement, per the other session.

## 12.10 Dependencies

Nothing in §12.6 needs `torch`. The MAE/SOM side would need `torch`, plus
`minisom`/`somoclu` or a hand-rolled SOM, in an optional `[represent]`
requirements group — the simulator must stay installable without a
deep-learning stack.
