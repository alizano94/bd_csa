#pragma once

#include <string>

namespace bd_csa {

// Physics choices that change results. The legacy code made each of these
// implicitly, usually by accident; here they are explicit so their effect can
// be measured in isolation. Defaults are the *fixed* behaviour.
struct PhysicsOptions {
  // Legacy recomputed mobility only when mod(step, iprint) == 0, which with the
  // shipped iprint == nstep meant *once per episode*. 1 = refresh every step.
  long mobility_update_interval = 1;

  // Ito drift term for position-dependent D. Requires smooth_mobility, since
  // grad.D of a piecewise-constant table is zero a.e. with spikes at bin edges.
  // Legacy omitted this; the omission was harmless only because mobility was
  // frozen. See documentation/07-porting-notes.md 7.3.
  bool enable_divD_drift = true;

  // C1 interpolation of the mobility table instead of nearest-bin lookup.
  bool smooth_mobility = true;

  // Legacy wrapped x,y into [-dg/2, dg/2] but the force loop has no minimum
  // image, so a wrap would silently corrupt forces. Correct physics for a
  // trapped finite cluster is no periodicity at all. See 7.8.
  bool periodic = false;

  // Replace the discontinuous Fhw contact branch (which is *weaker* than the
  // DLVO force it replaces) with a continuous capped repulsion. See 7.9.
  bool continuous_overlap = true;
};

// Immutable simulation parameters. Replaces the 15 COMMON blocks of the legacy
// code. Lengths are nm and times are ms once loaded, matching the Fortran's
// internal units (documentation/02-numerical-methods.md 2.2).
struct Config {
  // --- particle / run control -------------------------------------------
  int         np          = 300;
  long        nstep       = 1000000;
  long        iprint      = 1000000;
  int         istart      = 0;      // unused by the legacy code
  std::string par_in      = "start.txt";
  std::string par_out     = "bd_xyz";
  double      a           = 1435.0; // particle radius, nm
  double      tempr       = 20.0;   // degrees C
  double      phi         = 0.0;    // unused
  double      dt          = 0.1;    // ms
  double      t0          = 0.0;    // ms
  char        check       = 'n';

  // --- calibration (post-rescaling, see from_run_txt) --------------------
  double fac1 = 0.0;  // fac1_raw / a
  double fac2 = 0.0;  // fac2_raw * sqrt((273+tempr)/a) / sqrt(dt)
  double fac1_raw = 0.0;
  double fac2_raw = 0.0;

  // --- interactions ------------------------------------------------------
  double pwfactor = 0.0;   // unused
  double Fgrav    = 0.0;   // unused (system is 2-D)
  double rcut     = 0.0;   // nm, DLVO cutoff  (= 5a)
  double re       = 0.0;   // nm, dipole cutoff (= 5a)
  double kappa    = 143.5; // kappa*a, dimensionless
  double pfpp     = 0.0;   // bpp/kT/a, rescaled by *a
  double pfpw     = 0.0;   // unused
  double fcm      = -0.4667;
  double dg       = 0.0;   // nm, electrode gap (= 63.415a)
  double hlev     = 0.0;   // nm, levitation height (= a + a*h)
  double Fhw      = 0.417; // hard-coded at main.f:233

  // Boltzmann constant. The legacy hard-codes the pre-2019 value 1.380658e-23
  // (forces.f:55); the current CODATA value is 1.380649e-23. Switching costs a
  // uniform 6.5e-6 relative shift in every force -- physically meaningless next
  // to the 3e-4 fac1/fac2 calibration mismatch already present, but it would
  // put the differential test against the Fortran out of reach. Kept at the
  // legacy value so that test stays meaningful.
  double kb = 1.380658e-23;

  // Fortran defaults undecorated literals to REAL*4, so the legacy kb, the 1e18
  // unit scale and the field-correction polynomial coefficients all carry only
  // ~7 significant digits even though they are stored in double variables.
  // Enabling this reproduces that rounding bit-for-bit, which is what lets the
  // force differential reach 1e-12 and proves any residual disagreement is the
  // constants rather than the algebra. Off by default: the port uses
  // full-precision constants, a ~6e-8 relative change of no physical
  // significance.
  bool legacy_float_literals = false;

  // --- measurement -------------------------------------------------------
  double rmin      = 3780.0; // nm, conn6 neighbour cutoff
  double expbox[2] = {0.0, 0.0};
  double var       = 0.0;    // nm, synthetic measurement noise

  // --- polydispersity (shipped config is monodisperse) -------------------
  std::string polymono = "mono";
  std::string pdfile   = "raddist.txt";
  int         cyclenum = 1;

  // --- mobility table ----------------------------------------------------
  std::string rgdsfile   = "2dtabledssnp300.txt";
  double      rgdsmin    = 26500.0;
  double      delrgdsmin = -250.0;  // NOTE: negative stride
  int         rgdssbin   = 30;
  double      distmin    = 0.0;
  double      deldist    = 1435.0;
  int         distdssbin = 50;
  double      dssmin     = 0.10;
  double      dssmax     = 0.40;

  // --- control -----------------------------------------------------------
  double dpf          = 1.0;
  int    ecorrectflag = 1;

  PhysicsOptions physics{};

  // Absolute temperature as the legacy code computes it: 273 + T_C, *not*
  // 273.15 (documentation/05-io-formats.md).
  [[nodiscard]] double temperature_K() const { return 273.0 + tempr; }

  // Reduced-force scale, 1e18 * kB * T / a. Multiplies every force term.
  [[nodiscard]] double force_scale() const;

  // Bulk Stokes-Einstein diffusivity D0 = kT/(6*pi*eta*a), in nm^2/s.
  // Used only to cross-check fac1/fac2; the simulation itself uses fac1/fac2.
  [[nodiscard]] double D0_nm2_per_s(double eta_Pa_s = 0.890e-3) const;

  // Parse the positional run.txt. Mirrors main.f:74-156 read-for-read,
  // including the two label/value pairs that are read and discarded (lambda at
  // lines 43-44 and idummy at 61-62 -- both come from argv instead), then
  // applies the rescalings at main.f:170-177 and 223-225.
  static Config from_run_txt(const std::string& path);
};

}  // namespace bd_csa
