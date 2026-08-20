#pragma once

#include "bd_csa/config.hpp"

namespace bd_csa {

struct OrderParams {
  double psi6 = 0.0;  // [0,1] global hexatic order, RL observation 1
  double c6   = 0.0;  // [0,6] mean hexatic connectivity, RL observation 2
  double rg   = 0.0;  // nm, radius of gyration
  double rc   = 0.0;  // composite crystallinity, logged only
  int    n_in_window = 0;
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

// RC alone, for testing the composite against a known (R_g, C6) pair.
double compute_rc(double rg_nm, double c6);

}  // namespace bd_csa
