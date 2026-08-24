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
conv stem's locality bias in the first place — is no longer a constraint. **If
you can train on 10⁵–10⁶ images instead of a few thousand, the argument for the
conv stem weakens considerably**, since plain ViT patch embedding's weakness is
specifically data efficiency. That is worth deciding deliberately rather than
inheriting.

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

My read: this is an argument for (2), and it is worth measuring the empty-patch
fraction across the *whole* dataset (dispersed states will be far emptier than
crystalline ones) before fixing a ratio.

## 12.4 Symmetry — and why MAE changes the answer

The physics is invariant under global translation and rotation; pixels are not.
If the encoder spends capacity on cluster orientation, the SOM will organise
partly by orientation, which is exactly the kind of nuisance structure that
would make a topological map misleading.

The important wrinkle: **MAE deliberately uses minimal augmentation** — masking
*is* the pretext task, and heavy augmentation is not part of the recipe (unlike
DINO/contrastive, which depend on it). So "just add rotation augmentation"
fits DINO but sits awkwardly with MAE.

That points to **canonicalisation instead of augmentation** for the MAE route:
centre on the centroid (unambiguous, continuous) and optionally rotate to a
canonical frame. Rotation canonicalisation has a discontinuity when the frame
flips, which would show up as a spurious jump in the SOM — worth testing for
explicitly.

## 12.5 The open physics question is now answerable

Your session ended on: *do the assembly states form a continuous ordering
spectrum, or real discrete phases?* — with the note that the Siamese network
(A2) was needed because plain CNN classification blurred similar configurations,
suggesting fine-grained, possibly discontinuous structure.

That is now an experiment rather than an intuition. Generate **slow λ ramps** —
sweep λ continuously from 0 to 60 over a long trajectory — and ask whether
(ψ₆, C₆, R_g) move continuously or jump. Repeat across seeds. If order
parameters show hysteresis or discontinuity, that is evidence of genuine phase
structure and argues for a growing SOM over a fixed grid; if they move smoothly,
a fixed grid is defensible.

**This is cheap and worth doing before any encoder exists**, because it decides
the SOM design and it validates the premise of the whole pipeline.

## 12.6 What bd_csa must build

| # | deliverable | note |
|---|---|---|
| **R0** | Fast rasteriser | matplotlib is **~0.3 s/frame** — 10⁵ images would take 8.3 h against 16 min of simulation. Direct disk-stamping into numpy should reach ~1 ms. This is a hard blocker for any ViT training set. |
| **R1** | Dataset generator | random λ schedules (what an exploring agent produces), diverse seeds, deliberately melted and over-annealed starts. HDF5 with images + physics labels. **Split held-out data by trajectory, not by frame** — frames within a run are highly correlated and a random split leaks. |
| **R2** | λ-ramp experiment | §12.5 — continuous vs discrete phases |
| **R3** | Dataset statistics | empty-patch fraction across the full state distribution, to fix the masking ratio on evidence |

R0–R3 are all simulator-side, need no deep-learning stack, and are useful
regardless of which architecture you settle on. **I'd propose stopping there**
and leaving the MAE/SOM implementation to you, as you asked in the other session.

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

## 12.8 What I need from you

1. Is §12.1 a fair statement of the architecture?
2. Does the 10⁵–10⁶ image scale change your view on **conv stem vs plain ViT
   patch embedding**? (§12.2)
3. Masking: content-aware, or tighter crop, or higher ratio? (§12.3)
4. Should I run the **λ-ramp experiment** (§12.5)? It is a few GPU-hours and
   answers the fixed-vs-growing SOM question with data.
5. Scope confirmation: build R0–R3 only, and leave MAE/SOM to you?

## 12.9 Dependencies

Nothing in §12.6 needs `torch`. The MAE/SOM side would need `torch`, plus
`minisom`/`somoclu` or a hand-rolled SOM, in an optional `[represent]`
requirements group — the simulator must stay installable without a
deep-learning stack.
