#pragma once

#include "bd_csa/config.hpp"

namespace bd_csa {

struct OrderParams {
  double psi6 = 0.0;  // [0,1] global hexatic order, RL observation 1
  double c6   = 0.0;  // [0,6] mean hexatic connectivity, RL observation 2
  double rg   = 0.0;  // nm, radius of gyration
  double rc   = 0.0;  // composite crystallinity, logged only
  int    n_in_window = 0;

  // Real and imaginary parts of the global average <psi6_i>, from which psi6 =
  // hypot(re, im). The PHASE carries the lattice orientation: psi6 is 6-fold,
  // so the hexatic director is atan2(im, re)/6, defined modulo 60 degrees.
  // Exposed because reimplementing psi6 elsewhere to recover the phase is a
  // duplication that silently diverges -- notably over the expbox measurement
  // window, which excludes particles that have drifted out of the cell.
  double psi6_re = 0.0;
  double psi6_im = 0.0;
};

// psi6, C6, R_g and RC exactly as conn6calc.f, with the 7.2 sparse-array defect
// fixed: particles passing the measurement window are compacted into a dense
// array and every loop uses that dense count consistently. For the shipped
// config (expbox = the full cell) no particle is ever excluded, so the fix is a
// no-op and results match the Fortran bit-for-bit.
//
// Note psi6 is |<psi6_i>| -- the phase-coherent global average, not <|psi6_i|>.
// A policy trained against one is not transferable to the other.
//
// The legacy synthetic measurement noise (var * gasdev) is omitted: var = 0 in
// the shipped config, so it only ever consumed RNG draws.
OrderParams compute_order_params(const Config& c, const double* x,
                                 const double* y, int np);

// Same computation, but also reporting the PER-PARTICLE local order.
//
// The global psi6 is |<psi6_i>| over the whole cluster; what this fills in is
// |psi6_i| for each particle individually -- how hexatically ordered that
// particle's own neighbourhood is, in [0,1]. The two answer different
// questions: a polycrystal of well-formed grains at random orientations has
// high per-particle |psi6_i| but low global psi6, because the phases cancel in
// the average. Visualising the per-particle field is exactly how you tell those
// cases apart.
//
// Buffers must have room for np entries each; either may be null.
//   psi6_local  |psi6_i| in [0,1]; 0 for a particle with no neighbours
//   neighbours  neighbour count within rmin, useful for spotting edge particles
//
// Particles outside the measurement window get 0 in both. Since the shipped
// expbox is the full cell that never happens in practice.
OrderParams compute_order_params_local(const Config& c, const double* x,
                                       const double* y, int np,
                                       double* psi6_local, int* neighbours);

// RC alone, for testing the composite against a known (R_g, C6) pair.
double compute_rc(double rg_nm, double c6);

}  // namespace bd_csa
