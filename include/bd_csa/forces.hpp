#pragma once

#include "bd_csa/config.hpp"

namespace bd_csa {

// How the dielectrophoretic trap evaluates grad|E|^2.
enum class DepGradient {
  // Closed form. Required on GPU: the legacy forward difference uses a step of
  // 1e-3 nm on a quantity of order 1, which in FP32 is pure cancellation noise.
  // Also removes the O(h) truncation bias and 3 EMAG evaluations per particle.
  kAnalytic,
  // Forward difference exactly as forces.f:306-320. Test-only: it reproduces
  // the legacy arithmetic path, including its ~2e-6 cancellation error.
  kFiniteDifference,
  // Omit the trap entirely, leaving DLVO + dipole. Used by the differential
  // test to isolate the well-conditioned pair forces.
  kNone,
};

struct ForceOptions {
  DepGradient dep_gradient = DepGradient::kAnalytic;
  // Reproduce the legacy contact branch (constant Fhw, which is *weaker* than
  // the DLVO force it replaces). Test-only; the default is the capped
  // continuous repulsion. Never exercised in practice -- minimum observed pair
  // separation is 2.052a.
  bool legacy_overlap = false;
};

// |E| at (x, y). The legacy EMAG computes the exact quadrupole solution and
// then throws it away (emag.f:30-52 are overwritten at line 57), so the live
// model is just the linear field 4*rho/dg times an empirical correction.
double emag(const Config& c, double x, double y);

// grad(|E|^2) in closed form, evaluated without dividing by rho so it stays
// finite at the origin.
void grad_emag_sq(const Config& c, double x, double y, double& gx, double& gy);

// Total force on every particle, in reduced units (1e18 * kB*T / a).
//
// Three contributions, matching forces.f:
//   - screened electrostatic (DLVO) repulsion, r < rcut, antisymmetric
//   - induced dipole-dipole, 2a < r < re, ASYMMETRIC: the dipole moments are
//     position dependent, so the force on i is not minus the force on j and the
//     two must be accumulated from separate expressions (felxnew/felxnew2)
//   - dielectrophoretic trap, per particle
//
// fx and fy are overwritten, not accumulated.
void compute_forces(const Config& c, double lambda, const double* x,
                    const double* y, double* fx, double* fy, int np,
                    const ForceOptions& opt = {});

}  // namespace bd_csa
