#include "bd_csa/simulator.hpp"

#include <cmath>

#include "bd_csa/rng.hpp"

namespace bd_csa {

void SimulatorCpu::refresh_mobility(const State& s) {
  const int np = cfg_.np;

  // Centroid and R_g in double throughout. caldss.f left xmean/ymean/disttemp
  // undeclared, so they defaulted to REAL*4 and disagreed with the double
  // centroid conn6calc computed from the same coordinates (7.4).
  double xm = 0.0, ym = 0.0;
  for (int i = 0; i < np; ++i) {
    xm += s.x[i];
    ym += s.y[i];
  }
  xm /= np;
  ym /= np;

  double var = 0.0;
  for (int i = 0; i < np; ++i) {
    const double dx = s.x[i] - xm, dy = s.y[i] - ym;
    var += dx * dx + dy * dy;
  }
  const double rg = std::sqrt(var / np);

  const bool smooth = cfg_.physics.smooth_mobility;
  gdx_.assign(np, 0.0);
  gdy_.assign(np, 0.0);

  for (int i = 0; i < np; ++i) {
    const double dx = s.x[i] - xm, dy = s.y[i] - ym;
    const double dist = std::sqrt(dx * dx + dy * dy);

    if (!smooth) {
      mobility_[i] = table_.lookup_nearest(rg, dist);
      grad_d_[i] = 0.0;
      continue;
    }

    double dD_ddist = 0.0;
    mobility_[i] = table_.lookup_smooth(rg, dist, &dD_ddist);
    grad_d_[i] = dD_ddist;
    // Chain rule onto Cartesian components. At the centroid the radial
    // direction is undefined, but D is smooth there so the gradient vanishes.
    if (dist > 0.0) {
      gdx_[i] = dD_ddist * dx / dist;
      gdy_[i] = dD_ddist * dy / dist;
    }
  }
}

void SimulatorCpu::step(State& s, double lambda, long n_steps,
                        std::uint64_t seed, std::uint32_t env) {
  const int np = cfg_.np;
  const double dt = cfg_.dt;
  const double fac1_dt = cfg_.fac1 * dt;
  const double fac2_dt = cfg_.fac2 * dt;

  // Ito drift coefficient. Physically the term is grad(D)*dt with D = D_hat*D0;
  // since the random step is sqrt(2*D_hat*D0*dt) = sqrt(D_hat)*fac2*dt, we have
  // D0*dt = (fac2*dt)^2/2, so the drift is grad(D_hat) * (fac2*dt)^2/2 with no
  // need to reconstruct D0 or the viscosity.
  //
  // NOTE this includes only the local radial dependence of D_hat. D_hat also
  // depends on the cluster R_g, a collective coordinate; that contribution is a
  // many-body term and is not included here.
  const double drift_coef = 0.5 * fac2_dt * fac2_dt;
  const bool drift = cfg_.physics.enable_divD_drift && cfg_.physics.smooth_mobility;

  const long interval = cfg_.physics.mobility_update_interval > 0
                            ? cfg_.physics.mobility_update_interval
                            : n_steps + 1;

  for (long step = 0; step < n_steps; ++step) {
    if (step % interval == 0) refresh_mobility(s);

    compute_forces(cfg_, lambda, s.x.data(), s.y.data(), fx_.data(), fy_.data(),
                   np, force_opt_);

    for (int i = 0; i < np; ++i) {
      const double D = mobility_[i];
      double zx, zy;
      Philox::normal2(seed, env, static_cast<std::uint32_t>(i),
                      static_cast<std::uint64_t>(step), zx, zy);

      const double sqrtD = std::sqrt(D);
      double dx = D * fx_[i] * fac1_dt + sqrtD * zx * fac2_dt;
      double dy = D * fy_[i] * fac1_dt + sqrtD * zy * fac2_dt;

      if (drift) {
        dx += drift_coef * gdx_[i];
        dy += drift_coef * gdy_[i];
      }

      s.x[i] += dx;
      s.y[i] += dy;

      // Legacy wrapped into [-dg/2, dg/2] while the force loop had no minimum
      // image -- a wrap would have silently corrupted forces. Off by default;
      // the trap holds the cluster at R_g ~ 21 um inside a 91 um cell.
      if (cfg_.physics.periodic) {
        s.x[i] -= cfg_.dg * std::nearbyint(s.x[i] / cfg_.dg);
        s.y[i] -= cfg_.dg * std::nearbyint(s.y[i] / cfg_.dg);
      }
    }
  }
}

}  // namespace bd_csa
