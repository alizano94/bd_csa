#pragma once

#include <cuda_runtime.h>

namespace bd_csa {
namespace cuda {

// Device-side mirror of the constants the kernels need. Kept as a plain POD so
// it can be passed by value into the launch.
struct DevConfig {
  int np;
  int mob_rows, mob_cols;
  double a, dg, re, rcut, kappa, pfpp, fcm, Fhw;
  double Fo_coef;    // 1e18*0.75*kb*T/a, multiplied by lambda at launch
  double Fpp_pref;   // 1e18*kb*T*kappa*pfpp/a
  double Fdep_coef;  // 2*1e18*kb*T/fcm, multiplied by lambda at launch
  double fac1_dt, fac2_dt, drift_coef;
  float inv_dg, inv_a;  // precomputed reciprocals for the inner loop
  float cut2_margin;    // (max(rcut,re) + 1 nm)^2, for the FP32 cutoff rejection
  float rlist2;         // (force cutoff + skin)^2, Verlet list radius
  double skin_half2;    // (skin/2)^2, the list-staleness threshold
  double rgdsmin, delrg, distmin, deldist, dssmin, dssmax;
  int ecorrectflag;
  int smooth_mobility, enable_drift, periodic, legacy_overlap;
  long mob_interval;
};

// --- Philox 4x32-10 -------------------------------------------------------
// Counter-based: the value is a pure function of the key, so every
// (env, particle, step) draws independently with no state to carry and no
// dependence on thread scheduling. This is what makes the batched kernel
// bitwise reproducible.
__device__ __forceinline__ void philox_normal2(unsigned long long seed,
                                               unsigned int env,
                                               unsigned int particle,
                                               unsigned long long step,
                                               float& z0, float& z1) {
  unsigned int c[4] = {(unsigned int)step, (unsigned int)(step >> 32), particle, env};
  unsigned int k[2] = {(unsigned int)seed, (unsigned int)(seed >> 32)};
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned int hi0 = __umulhi(0xD2511F53u, c[0]);
    const unsigned int lo0 = 0xD2511F53u * c[0];
    const unsigned int hi1 = __umulhi(0xCD9E8D57u, c[2]);
    const unsigned int lo1 = 0xCD9E8D57u * c[2];
    const unsigned int n0 = hi1 ^ c[1] ^ k[0];
    const unsigned int n1 = lo1;
    const unsigned int n2 = hi0 ^ c[3] ^ k[1];
    const unsigned int n3 = lo0;
    c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
    k[0] += 0x9E3779B9u;
    k[1] += 0xBB67AE85u;
  }
  const float u1 = (float)((double)c[0] + 0.5) * 2.3283064365386963e-10f;
  const float u2 = (float)((double)c[1] + 0.5) * 2.3283064365386963e-10f;
  const float r = sqrtf(-2.0f * logf(u1));
  float s, cs;
  __sincosf(6.28318530717958648f * u2, &s, &cs);
  z0 = r * cs;
  z1 = r * s;
}

// --- field ----------------------------------------------------------------
__device__ __forceinline__ float corr_f(float u) {
  return (((2.081e-7f * u + -1.539e-9f) * u + 8.341e-5f) * u + 1.961e-5f) * u +
         1.028f;
}
__device__ __forceinline__ float dcorr_f(float u) {
  return ((4.0f * 2.081e-7f * u + 3.0f * -1.539e-9f) * u + 2.0f * 8.341e-5f) * u +
         1.961e-5f;
}

// grad(|E|^2) in closed form. The legacy forward difference with h = 1e-3 nm is
// unusable here: in FP32 the two |E|^2 values are identical to the last bit, so
// the difference is exactly zero or pure noise. This is why the analytic form is
// mandatory on GPU rather than merely preferable.
__device__ __forceinline__ void grad_emag_sq_f(const DevConfig& c, float x,
                                               float y, float& gx, float& gy) {
  const float rho = sqrtf(x * x + y * y);
  const float u = rho * 1e-3f;
  const float f = c.ecorrectflag == 1 ? corr_f(u) : 1.0f;
  const float df = c.ecorrectflag == 1 ? dcorr_f(u) : 0.0f;
  const float A = 4.0f * f / (float)c.dg;
  const float dE = A + (4.0f * rho / (float)c.dg) * df * 1e-3f;
  gx = 2.0f * A * dE * x;
  gy = 2.0f * A * dE * y;
}

// --- mobility -------------------------------------------------------------
// tab already has the count==0 -> dssmax rule baked in on the host.
__device__ __forceinline__ float mob_resolved(const DevConfig& c,
                                              const float* tab, int row,
                                              int col) {
  if (row >= c.mob_rows) return (float)c.dssmin;
  row = row < 0 ? 0 : row;
  if (col < 0 || col >= c.mob_cols) return (float)c.dssmax;
  return tab[row * c.mob_cols + col];
}

__device__ __forceinline__ float mob_lookup(const DevConfig& c, const float* tab,
                                            double rg, double dist,
                                            float* grad_dist) {
  if (!c.smooth_mobility) {
    int rg_bin = (int)((rg - c.rgdsmin) / c.delrg) + 1;
    if (rg_bin <= 0) rg_bin = 1;
    const int d_bin = (int)((dist - c.distmin) / c.deldist) + 1;
    if (grad_dist) *grad_dist = 0.0f;
    if (rg_bin >= 1 && rg_bin <= c.mob_rows) {
      if (d_bin >= 1 && d_bin <= c.mob_cols)
        return tab[(rg_bin - 1) * c.mob_cols + (d_bin - 1)];
      return (float)c.dssmax;
    }
    return (float)c.dssmin;
  }

  const double rg_x = (rg - c.rgdsmin) / c.delrg - 0.5;
  const double d_x = (dist - c.distmin) / c.deldist - 0.5;
  const int r0 = (int)floor(rg_x);
  const int c0 = (int)floor(d_x);
  const float fr = (float)(rg_x - r0);
  const float fc = (float)(d_x - c0);
  const float wr = fr * fr * (3.0f - 2.0f * fr);
  const float wc = fc * fc * (3.0f - 2.0f * fc);

  const float v00 = mob_resolved(c, tab, r0, c0);
  const float v01 = mob_resolved(c, tab, r0, c0 + 1);
  const float v10 = mob_resolved(c, tab, r0 + 1, c0);
  const float v11 = mob_resolved(c, tab, r0 + 1, c0 + 1);

  const float lo = v00 + (v01 - v00) * wc;
  const float hi = v10 + (v11 - v10) * wc;
  if (grad_dist) {
    const float dwc = 6.0f * fc * (1.0f - fc);
    const float dlo = (v01 - v00) * dwc;
    const float dhi = (v11 - v10) * dwc;
    *grad_dist = (dlo + (dhi - dlo) * wr) / (float)c.deldist;
  }
  return lo + (hi - lo) * wr;
}

// --- pair force -----------------------------------------------------------
// Force exerted on `self` by `other`, in reduced units.
//
// The dipole interaction is NOT Newtonian: the induced moments depend on
// absolute position, so the force on the lower-indexed particle of a pair is
// not minus the force on the upper one. The legacy computes two separate
// expressions (forces.f:235-249) and so must this. To stay faithful we
// reconstruct the ordered pair (p = lower index, q = upper index) exactly as
// the Fortran loop would, then select the term belonging to `self`.
//
// PRECISION: the separation is passed in already differenced in FP64. Taking
// x[j] - x[i] in FP32 would be catastrophic -- coordinates are ~2e4 nm, so each
// carries ~1.2e-3 nm of representation error, and the DLVO exponential has a
// sensitivity of 0.1 per nm, turning that into a ~2e-4 relative force error.
// Differencing first in FP64 and converting the (small) result drops this to
// ~2e-5.
__device__ __forceinline__ void pair_force_on_self(
    const DevConfig& c, float lambda, float dx, float dy, float xs, float ys,
    float xo, float yo, bool self_is_lower, float& fx, float& fy) {
  const float r2 = dx * dx + dy * dy;
  if (r2 == 0.0f) { fx = 0.0f; fy = 0.0f; return; }
  // r and 1/r need different precision, and conflating them costs accuracy.
  //
  // r feeds the DLVO exponential, whose log-derivative is -kappa/a = -0.1 per
  // nm, so an error in r is amplified into the force. Computing r as
  // r2*rsqrtf(r2) was measured to nearly double the RMS force error
  // (9.75e-6 -> 1.80e-5) and pushed the max past its gate: rsqrtf carries ~2.4e-7,
  // which over a ~3000 nm separation is 7e-4 nm, i.e. ~7e-5 of relative force.
  // So r uses the correctly-rounded sqrtf.
  //
  // 1/r only ever scales a direction, entering linearly, so the fast
  // correctly-rounded reciprocal is fine there -- and hoisting it replaces seven
  // IEEE divisions with one.
  const float r = sqrtf(r2);
  const float inv_r = __frcp_rn(r);

  const float contact = 2.0f * (float)c.a;
  const float inv_dg = c.inv_dg;
  const float inv_a = c.inv_a;

  // Isotropic repulsion, antisymmetric, so it needs no role bookkeeping beyond
  // the direction of the separation.
  float Fss;
  if (r <= contact) {
    Fss = c.legacy_overlap ? (float)c.Fhw : (float)c.Fpp_pref;
  } else if (r < (float)c.rcut) {
    // __expf (fast SFU path) rather than expf. Measured: switching to the
    // accurate expf changes the RMS force error by 0.5% (9.751e-6 -> 9.798e-6),
    // i.e. not at all -- the error budget is set by per-term FP32 rounding
    // across the pair sum, not by the exponential.
    Fss = (float)c.Fpp_pref * __expf(-(float)c.kappa * (r - contact) * inv_a);
  } else {
    Fss = 0.0f;
  }

  // Reconstruct the ordered pair: rij always points from the lower index to the
  // upper one, matching rijtemp = r(j) - r(i) with i < j.
  const float rijx = self_is_lower ? dx : -dx;
  const float rijy = self_is_lower ? dy : -dy;
  const float xp = self_is_lower ? xs : xo;
  const float yp = self_is_lower ? ys : yo;
  const float xq = self_is_lower ? xo : xs;
  const float yq = self_is_lower ? yo : ys;

  float felx = 0.0f, fely = 0.0f;
  if (r > contact && r < (float)c.re) {
    const float Exp = -4.0f * xp * inv_dg;
    const float Eyp = 4.0f * yp * inv_dg;
    const float Exq = -4.0f * xq * inv_dg;
    const float Eyq = 4.0f * yq * inv_dg;

    const float F1 = Exp * Exq + Eyp * Eyq;
    const float F2 = (rijx * Exp + rijy * Eyp) * inv_r;  // projection of E_p
    const float F3 = (rijx * Exq + rijy * Eyq) * inv_r;  // projection of E_q

    const float t = 2.0f * (float)c.a * inv_r;
    const float pref = (float)c.Fo_coef * lambda * t * t * t * t;

    // Fixed-dipole part: antisymmetric under p<->q, i.e. Newtonian.
    const float common_x =
        F1 * rijx * inv_r + Exp * F3 + Exq * F2 - 5.0f * F2 * F3 * rijx * inv_r;
    const float common_y =
        F1 * rijy * inv_r + Eyp * F3 + Eyq * F2 - 5.0f * F2 * F3 * rijy * inv_r;

    const float g = 16.0f * inv_dg * inv_dg / 3.0f;
    if (self_is_lower) {
      // force on p = -(felxnew)
      felx = -(pref * (common_x + r * g * xq + F3 * 4.0f * rijx * inv_dg));
      fely = -(pref * (common_y + r * g * yq - F3 * 4.0f * rijy * inv_dg));
    } else {
      // force on q = +(felxnew2)
      felx = pref * (common_x - r * g * xp - F2 * 4.0f * rijx * inv_dg);
      fely = pref * (common_y - r * g * yp + F2 * 4.0f * rijy * inv_dg);
    }
  }

  const float ssx = Fss * rijx * inv_r;
  const float ssy = Fss * rijy * inv_r;
  if (self_is_lower) {
    fx = felx - ssx;
    fy = fely - ssy;
  } else {
    fx = felx + ssx;
    fy = fely + ssy;
  }
}

}  // namespace cuda
}  // namespace bd_csa
