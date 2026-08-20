#include "bd_csa/forces.hpp"

#include <cmath>

namespace bd_csa {
namespace {

// Round a literal the way a Fortran REAL*4 constant would be before promotion.
inline constexpr double f32(double v) {
  return static_cast<double>(static_cast<float>(v));
}

// Empirical quadrupole field correction, emag.f:60-61. Argument is rho/1000
// (i.e. rho in micrometres). The coefficients are REAL*4 literals in the
// Fortran, hence the optional rounding.
struct CorrCoef {
  double c4, c3, c2, c1, c0;
  static CorrCoef make(bool legacy) {
    if (legacy)
      return {f32(2.081e-7), f32(-1.539e-9), f32(8.341e-5), f32(1.961e-5),
              f32(1.028)};
    return {2.081e-7, -1.539e-9, 8.341e-5, 1.961e-5, 1.028};
  }
  [[nodiscard]] double f(double u) const {
    return (((c4 * u + c3) * u + c2) * u + c1) * u + c0;
  }
  [[nodiscard]] double df(double u) const {
    return ((4.0 * c4 * u + 3.0 * c3) * u + 2.0 * c2) * u + c1;
  }
};

// Legacy finite-difference step, forces.f:66.
constexpr double kDepStep = 1e-3;

}  // namespace

double emag(const Config& c, double x, double y) {
  const double rho = std::sqrt(x * x + y * y);
  double e = 4.0 * rho / c.dg;
  if (c.ecorrectflag == 1)
    e *= CorrCoef::make(c.legacy_float_literals).f(rho / 1000.0);
  return e;
}

void grad_emag_sq(const Config& c, double x, double y, double& gx, double& gy) {
  // |E| = A*rho with A = (4/dg)*f(rho/1000), so |E|^2 = (A*rho)^2 and
  //   d|E|^2/drho * (x/rho) = 2 * A * (d|E|/drho) * x
  // which removes the 1/rho singularity at the origin analytically.
  const double rho = std::sqrt(x * x + y * y);
  const double u = rho / 1000.0;
  const CorrCoef cc = CorrCoef::make(c.legacy_float_literals);
  const double f = (c.ecorrectflag == 1) ? cc.f(u) : 1.0;
  const double df = (c.ecorrectflag == 1) ? cc.df(u) : 0.0;

  const double A = 4.0 * f / c.dg;                       // |E| / rho
  const double dE = A + (4.0 * rho / c.dg) * df / 1000.0;  // d|E| / drho

  gx = 2.0 * A * dE * x;
  gy = 2.0 * A * dE * y;
}

void compute_forces(const Config& c, double lambda, const double* x,
                    const double* y, double* fx, double* fy, int np,
                    const ForceOptions& opt) {
  for (int i = 0; i < np; ++i) {
    fx[i] = 0.0;
    fy[i] = 0.0;
  }

  // Fortran evaluates left-to-right with promotion at each binary operator, so
  // in legacy mode the leading all-REAL*4 subexpressions (1e18*0.75 and 2*1e18)
  // are themselves computed in single precision before meeting a double.
  const bool L = c.legacy_float_literals;
  const double kb = L ? f32(c.kb) : c.kb;
  const double T = c.temperature_K();

  const double s_dipole = L ? f32(static_cast<double>(f32(1e18) * 0.75f))
                            : 1e18 * 0.75;                     // forces.f:62
  const double s_unit = L ? f32(1e18) : 1e18;                  // forces.f:111
  const double s_dep = L ? f32(static_cast<double>(2.0f * f32(1e18)))
                         : 2.0 * 1e18;                         // forces.f:335

  const double Fo = s_dipole * lambda * kb * T / c.a;
  const double Fpp_pref = s_unit * kb * T * c.kappa * c.pfpp / c.a;
  const double Fdep_pref = s_dep * kb * T * lambda / c.fcm;

  // Monodisperse: every radius is a, so contact is at 2a and the (radii/a)^3
  // DEP size factor is 1.
  const double contact = 2.0 * c.a;
  const double dlvo_at_contact = Fpp_pref;  // exp(0)

  for (int i = 0; i < np - 1; ++i) {
    // Dimensionless local field at i. Note the sign asymmetry: Ex has -4/dg,
    // Ey has +4/dg. It propagates into the gradient terms below.
    const double Exi = -4.0 * x[i] / c.dg;
    const double Eyi = 4.0 * y[i] / c.dg;

    for (int j = i + 1; j < np; ++j) {
      const double rijx = x[j] - x[i];
      const double rijy = y[j] - y[i];
      // No minimum image: the force loop never had one (forces.f:88-89 are
      // commented out), and the trap keeps the cluster far from the walls.
      const double r = std::sqrt(rijx * rijx + rijy * rijy);
      if (r == 0.0) continue;

      // --- pair repulsion, isotropic and antisymmetric ---
      double Fss;
      if (r <= contact) {
        Fss = opt.legacy_overlap ? c.Fhw : dlvo_at_contact;
      } else if (r < c.rcut) {
        Fss = Fpp_pref * std::exp(-c.kappa * (r - contact) / c.a);
      } else {
        Fss = 0.0;
      }

      // --- induced dipole-dipole, asymmetric ---
      double fel_xi = 0.0, fel_yi = 0.0, fel_xj = 0.0, fel_yj = 0.0;
      if (r > contact && r < c.re) {
        const double Exj = -4.0 * x[j] / c.dg;
        const double Eyj = 4.0 * y[j] / c.dg;

        const double F1 = Exi * Exj + Eyi * Eyj;
        const double F2 = (rijx * Exi + rijy * Eyi) / r;
        const double F3 = (rijx * Exj + rijy * Eyj) / r;

        const double pref = Fo * std::pow(2.0 * c.a / r, 4);

        // Fixed-dipole part, common to both particles.
        const double common_x =
            F1 * rijx / r + Exi * F3 + Exj * F2 - 5.0 * F2 * F3 * rijx / r;
        const double common_y =
            F1 * rijy / r + Eyi * F3 + Eyj * F2 - 5.0 * F2 * F3 * rijy / r;

        // Self-dipole gradient part. This is what makes the interaction
        // non-Newtonian: it depends on the absolute position of the *other*
        // particle, with opposite signs for i and j (forces.f:235-249).
        fel_xi = pref * (common_x + r * 16.0 * x[j] / (c.dg * c.dg) / 3.0 -
                         F3 * 4.0 * (-rijx) / c.dg);
        fel_yi = pref * (common_y + r * 16.0 * y[j] / (c.dg * c.dg) / 3.0 -
                         F3 * 4.0 * (rijy) / c.dg);
        fel_xj = pref * (common_x - r * 16.0 * x[i] / (c.dg * c.dg) / 3.0 +
                         F2 * 4.0 * (-rijx) / c.dg);
        fel_yj = pref * (common_y - r * 16.0 * y[i] / (c.dg * c.dg) / 3.0 +
                         F2 * 4.0 * (rijy) / c.dg);
      }

      const double sx = Fss * rijx / r;
      const double sy = Fss * rijy / r;

      fx[i] -= fel_xi + sx;
      fy[i] -= fel_yi + sy;
      fx[j] += fel_xj + sx;
      fy[j] += fel_yj + sy;
    }
  }

  // --- dielectrophoretic trap, one term per particle ---
  if (opt.dep_gradient == DepGradient::kNone) return;
  for (int i = 0; i < np; ++i) {
    double dE2x, dE2y;
    if (opt.dep_gradient == DepGradient::kAnalytic) {
      grad_emag_sq(c, x[i], y[i], dE2x, dE2y);
    } else {
      const double e0 = emag(c, x[i], y[i]);
      const double ex = emag(c, x[i] + kDepStep, y[i]);
      const double ey = emag(c, x[i], y[i] + kDepStep);
      dE2x = (ex * ex - e0 * e0) / kDepStep;
      dE2y = (ey * ey - e0 * e0) / kDepStep;
    }
    fx[i] += Fdep_pref * dE2x;
    fy[i] += Fdep_pref * dE2y;
  }
}

}  // namespace bd_csa
