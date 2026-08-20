# Golden inputs and measured reference values

Frozen inputs for the regression harness, copied byte-identical from
`legacy/fortran_bd/` (which is itself byte-identical to `SAC3/sac3/fortran_bd/`,
verified with `diff`).

| File | Role |
|---|---|
| `run.txt` | positional parameter file (shipped: `nstep = iprint = 10^6`) |
| `start.txt` | 300 initial particle positions, in multiples of `a` |
| `2dtabledssnp300.txt` | mobility table, 30 R_g bins x 50 distance bins |

Tests read from `data/`, never from `legacy/`, so runs cannot overwrite the
reference tree.

## Oracle

The legacy binary is the source of truth for differential testing. It is built
out-of-tree so `legacy/` stays pristine:

```sh
mkdir -p build/oracle && cp legacy/fortran_bd/*.f legacy/fortran_bd/makefile build/oracle/
cp data/run.txt data/start.txt data/2dtabledssnp300.txt build/oracle/
cd build/oracle && make
```

Builds clean and unmodified with `gfortran 13.3.0` on x86-64 Linux (the docs
report the same for `gfortran 16.1.0` on macOS/arm64).

The shipped `legacy/fortran_bd/bdpd` is a **Mach-O arm64** binary and does not
run here — always rebuild.

## Tier-1 golden values

Order parameters at `t = 0`, computed by `CONN6CALC` from `start.txt` before any
dynamics. **Verified invariant to both `lambda` and `seed`** (run with
`(30, -7)`, `(5, 12345)`, `(100, 1)` — identical to all printed digits), so this
is a deterministic fixture.

| Quantity | Value | Source |
|---|---|---|
| `psi6` | `0.40508` | `op1.txt` col 4 |
| `C6` | `4.28000` | `op1.txt` col 2 |
| `R_g` | `21015.90986` nm | `op1.txt` col 3 |
| `RC` | `0.76363` | `op1.txt` col 5 |

`R_g` was independently reproduced from `start.txt` to all five decimals by a
standalone double-precision centroid/variance calculation, and `RC` was
reproduced by hand from `(R_g, C6)` using the formula in
[03-order-parameters.md](../documentation/03-order-parameters.md) §3.4:

```
Ra  = 1 - (21015.90986 - 18526)/(26500 - 18526) = 0.687746
Crc = 4.28 / 5.6                                 = 0.764286
Wrc = 1/(exp(18*(Crc - 0.5)) + 1)                = 0.008517
RC  = Wrc*Ra + (1 - Wrc)*Crc                     = 0.763633   -> 0.76363  OK
```

### Correction to the documentation

[documentation/README.md](../documentation/README.md) and
[03-order-parameters.md](../documentation/03-order-parameters.md) state
`R_g = 21014.5 nm`, `C6 = 4.2667`, `RC = 0.76125` as the values for "the known
configuration". **Those do not correspond to the shipped `start.txt`** — the
real values are the table above. The discrepancy is in the configuration, not
the algorithm: the documented `RC` formula is correct and internally consistent
for its own `(R_g, C6)` pair, and reproduces the true value exactly when fed the
true inputs. The doc's numbers appear to come from a configuration one or more
episodes removed from the shipped one.

Use the measured values above as regression targets, not the documented ones.

## Performance baseline

Measured on this machine (x86-64, `gfortran 13.3 -O`, single core):

| Steps | Wall clock | Per step |
|---|---|---|
| 20,000 | 6.34 s | **317 us** |

Extrapolates to **≈5.3 min** for a full 10^6-step episode. The docs report
283 us/step (5.66 s / 20,000) on Apple M-series. Use ~312 us/step as the local
speedup denominator (6.23 s / 20,000, the steady-state repeat measurement).

## Toolchain notes

- **The CUDA toolkit is newer than the driver supports.** `nvcc` is 13.2 but the
  driver (580.178.04) reports CUDA 13.0. PTX JIT requires driver >= toolkit, so
  any build that embeds PTX fails at kernel launch with "the provided PTX was
  compiled with an unsupported toolchain". The build therefore targets
  `89-real`, emitting SASS for the local GPU with no PTX at all. On a different
  GPU pass `-DBD_CSA_CUDA_ARCH=<arch>-real`.
- `CMAKE_CUDA_ARCHITECTURES` must be set **before** `enable_language(CUDA)`;
  otherwise CMake's default wins (it picked sm_75 here, which cannot load on an
  sm_89 device and silently fell back to the failing PTX path).

## FP32 force error budget (tier 4)

Against the FP64 CPU reference, normalised to the RMS force over all particles
(1.90e-4 reduced units):

| metric | value |
|---|---|
| max error | 9.06e-05 |
| RMS error | 9.75e-06 |

Normalising instead by each particle's *own* force magnitude gives a misleading
1.4e-3, because some particles sit near a force null where the inward DEP trap
cancels the pair repulsion -- particle 175 carries 1/99 of the mean force. The
absolute error there is irrelevant to the dynamics.

Two optimisations were tried and **measured to make no difference**, so neither
is in the kernel:

| variant | RMS error |
|---|---|
| `__expf`, plain summation (kept) | 9.751e-06 |
| accurate `expf`, plain summation | 9.798e-06 |
| accurate `expf` + Kahan compensated summation | 9.812e-06 |

The budget is set by FP32 rounding within each pair term, not by the
exponential's accuracy nor by accumulation over the 299-term sum.
